"""Offline elapsed-time audit for row-based forward labels.

The audit preserves historical row-based targets and measures their actual UTC duration.
It never changes labels, chooses a model, produces trading signals, or connects to execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects instead of silently accepting the last duplicate."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    """Reject NaN and Infinity tokens, which are not valid RFC 8259 JSON."""
    raise ValueError(f"nonstandard_json_constant:{value}")


def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonstandard_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


from app.research_baseline_analysis import (
    _BASELINES,
    _label,
    _metric_summary,
    _parse_utc,
    _probability,
)
from app.research_results_summary import summarize_research_run

_TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}
_BUCKETS = (
    "nominal_or_within_tolerance",
    "extended_up_to_threshold",
    "long_extension_over_threshold",
)
_METRIC_BASELINES = {
    "constant_probability_0_5": "constant_probability_class_1",
    "matured_refit_prevalence": "matured_refit_prevalence_probability_class_1",
    "lagged_bar_direction": "lagged_direction_probability_class_1",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = _load_json_object(path.read_bytes(), source=str(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _distribution(values: Sequence[float], nominal_hours: float) -> dict[str, Any]:
    if not values:
        return {
            "sample_count": 0,
            "nominal_elapsed_hours": nominal_hours,
            "min_hours": None,
            "p50_hours": None,
            "p90_hours": None,
            "p95_hours": None,
            "p99_hours": None,
            "max_hours": None,
            "within_nominal_count": 0,
            "extended_beyond_nominal_count": 0,
            "within_nominal_fraction": None,
            "extended_beyond_nominal_fraction": None,
        }
    tolerance = 1e-6
    count_within = sum(value <= nominal_hours + tolerance for value in values)
    n = len(values)
    return {
        "sample_count": n,
        "nominal_elapsed_hours": round(nominal_hours, 6),
        "min_hours": round(min(values), 6),
        "p50_hours": round(float(_quantile(values, 0.50)), 6),
        "p90_hours": round(float(_quantile(values, 0.90)), 6),
        "p95_hours": round(float(_quantile(values, 0.95)), 6),
        "p99_hours": round(float(_quantile(values, 0.99)), 6),
        "max_hours": round(max(values), 6),
        "within_nominal_count": count_within,
        "extended_beyond_nominal_count": n - count_within,
        "within_nominal_fraction": count_within / n,
        "extended_beyond_nominal_fraction": (n - count_within) / n,
    }


def _bucket(elapsed_hours: float, nominal_hours: float, threshold_hours: float) -> str:
    if elapsed_hours <= nominal_hours + 1e-6:
        return "nominal_or_within_tolerance"
    if elapsed_hours <= threshold_hours:
        return "extended_up_to_threshold"
    return "long_extension_over_threshold"


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "model": None,
            "baselines": {},
            "improvements_baseline_minus_model": {},
        }
    labels = [int(row["label"]) for row in rows]
    model_probabilities = [float(row["model_probability"]) for row in rows]
    model_metrics = _metric_summary(labels, model_probabilities)
    baselines: dict[str, Any] = {}
    improvements: dict[str, Any] = {}
    for name, field in _METRIC_BASELINES.items():
        probabilities = [float(row["baseline_probabilities"][name]) for row in rows]
        metrics = _metric_summary(labels, probabilities)
        baselines[name] = metrics
        improvements[name] = {
            "log_loss_baseline_minus_model": metrics["log_loss"] - model_metrics["log_loss"],
            "brier_baseline_minus_model": metrics["brier_score"] - model_metrics["brier_score"],
        }
    return {
        "sample_count": len(rows),
        "model": model_metrics,
        "baselines": baselines,
        "improvements_baseline_minus_model": improvements,
    }


def analyze_label_horizon(
    matrix: Mapping[str, Any],
    ohlc_rows: Sequence[Mapping[str, Any]],
    *,
    extended_time_threshold_hours: float = 8.0,
    longest_examples_limit: int = 25,
) -> dict[str, Any]:
    """Audit the elapsed duration of the existing row-based forward-label target."""
    errors: list[str] = []
    warnings: list[str] = []
    symbol = str(matrix.get("symbol", "")).strip().upper()
    timeframe = str(matrix.get("timeframe", "")).strip().upper()
    horizon = matrix.get("horizon_bars")
    if not symbol or not timeframe:
        errors.append("matrix_symbol_or_timeframe_missing")
    if timeframe not in _TIMEFRAME_MINUTES:
        errors.append("matrix_timeframe_unsupported")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        errors.append("matrix_horizon_bars_invalid")
    if (
        isinstance(extended_time_threshold_hours, bool)
        or not isinstance(extended_time_threshold_hours, (int, float))
        or not math.isfinite(float(extended_time_threshold_hours))
        or extended_time_threshold_hours <= 0
    ):
        errors.append("extended_time_threshold_hours_invalid")
    if (
        isinstance(longest_examples_limit, bool)
        or not isinstance(longest_examples_limit, int)
        or not 0 <= longest_examples_limit <= 500
    ):
        errors.append("longest_examples_limit_invalid")
    if errors:
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_LABEL_HORIZON_ELAPSED_TIME_AUDIT",
            "status": "FAIL",
            "errors": errors,
            "warnings": warnings,
            "automatic_model_selection": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "execution_connected": False,
        }

    timestamps: list[datetime] = []
    try:
        if not ohlc_rows:
            raise ValueError("ohlc_rows_empty")
        for index, row in enumerate(ohlc_rows):
            timestamp = _parse_utc(row.get("timestamp"), "OHLC timestamp")
            row_symbol = str(row.get("symbol", "")).strip().upper()
            row_timeframe = str(row.get("timeframe", "")).strip().upper()
            if row_symbol and row_symbol != symbol:
                raise ValueError(f"ohlc_row_{index}_symbol_mismatch")
            if row_timeframe and row_timeframe != timeframe:
                raise ValueError(f"ohlc_row_{index}_timeframe_mismatch")
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError("ohlc_timestamps_must_be_strictly_increasing_unique")
            timestamps.append(timestamp)
    except (TypeError, ValueError, OverflowError) as exc:
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_LABEL_HORIZON_ELAPSED_TIME_AUDIT",
            "status": "FAIL",
            "errors": [f"ohlc_timestamp_validation_failed:{type(exc).__name__}:{exc}"],
            "warnings": [],
            "automatic_model_selection": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "execution_connected": False,
        }

    timestamp_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    nominal_hours = int(horizon) * _TIMEFRAME_MINUTES[timeframe] / 60.0
    threshold_hours = float(extended_time_threshold_hours)
    experiments = matrix.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        experiments = []
        errors.append("matrix_experiments_missing_or_empty")

    experiment_reports: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    unique_horizons: dict[datetime, float] = {}
    timestamp_occurrences: dict[tuple[str, datetime], int] = {}
    missing_timestamp_count = 0
    over_nominal_count = 0
    over_threshold_count = 0
    for experiment in experiments:
        if not isinstance(experiment, dict):
            errors.append("matrix_experiment_record_not_object")
            continue
        experiment_id = str(experiment.get("experiment_id", "unknown"))
        model_type = str(experiment.get("model_type", "unknown"))
        fold_results = experiment.get("fold_results")
        if not isinstance(fold_results, list) or not fold_results:
            errors.append(f"experiment_{experiment_id}_fold_results_missing")
            continue
        experiment_rows: list[dict[str, Any]] = []
        fold_reports: list[dict[str, Any]] = []
        for fold in fold_results:
            if not isinstance(fold, dict):
                errors.append(f"experiment_{experiment_id}_fold_record_not_object")
                continue
            fold_index = fold.get("fold_index")
            if isinstance(fold_index, bool) or not isinstance(fold_index, int):
                errors.append(f"experiment_{experiment_id}_fold_index_invalid")
                continue
            predictions = fold.get("test_predictions")
            if not isinstance(predictions, list) or not predictions:
                errors.append(f"experiment_{experiment_id}_fold_{fold_index}_predictions_missing")
                continue
            fold_rows: list[dict[str, Any]] = []
            for prediction_index, prediction in enumerate(predictions):
                try:
                    if not isinstance(prediction, dict):
                        raise ValueError("prediction_record_not_object")
                    prediction_timestamp = _parse_utc(
                        prediction.get("timestamp_utc"), "prediction timestamp"
                    )
                    prediction_row_index = timestamp_to_index.get(prediction_timestamp)
                    if prediction_row_index is None:
                        missing_timestamp_count += 1
                        raise ValueError("prediction_timestamp_missing_from_ohlc")
                    target_index = prediction_row_index + int(horizon)
                    if target_index >= len(timestamps):
                        raise ValueError("label_horizon_target_missing_from_ohlc")
                    elapsed_hours = (
                        timestamps[target_index] - prediction_timestamp
                    ).total_seconds() / 3600.0
                    if elapsed_hours <= 0:
                        raise ValueError("label_target_timestamp_not_after_prediction")
                    label = _label(prediction.get("label"), "label")
                    model_probability = _probability(
                        prediction.get("model_probability_class_1"),
                        "model_probability_class_1",
                    )
                    baseline_probabilities = {
                        name: _probability(prediction.get(field), field)
                        for name, field in _METRIC_BASELINES.items()
                    }
                    exposure_bucket = _bucket(elapsed_hours, nominal_hours, threshold_hours)
                    row = {
                        "timestamp": prediction_timestamp,
                        "target_timestamp": timestamps[target_index],
                        "elapsed_hours": elapsed_hours,
                        "label": label,
                        "model_probability": model_probability,
                        "baseline_probabilities": baseline_probabilities,
                        "bucket": exposure_bucket,
                        "experiment_id": experiment_id,
                        "model_type": model_type,
                        "fold_index": fold_index,
                        "prediction_row_index": prediction_row_index,
                        "target_row_index": target_index,
                    }
                    occurrence_key = (experiment_id, prediction_timestamp)
                    timestamp_occurrences[occurrence_key] = timestamp_occurrences.get(occurrence_key, 0) + 1
                    fold_rows.append(row)
                    experiment_rows.append(row)
                    all_rows.append(row)
                    unique_horizons[prediction_timestamp] = elapsed_hours
                    if elapsed_hours > nominal_hours + 1e-6:
                        over_nominal_count += 1
                    if elapsed_hours > threshold_hours:
                        over_threshold_count += 1
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    errors.append(
                        f"experiment_{experiment_id}_fold_{fold_index}_prediction_{prediction_index}_invalid:{exc}"
                    )
            if fold_rows:
                fold_rows.sort(key=lambda row: row["timestamp"])
                fold_elapsed = [row["elapsed_hours"] for row in fold_rows]
                fold_bucket_summaries = {
                    bucket: _summarize_rows([row for row in fold_rows if row["bucket"] == bucket])
                    for bucket in _BUCKETS
                }
                fold_reports.append({
                    "fold_index": fold_index,
                    "sample_count": len(fold_rows),
                    "prediction_start_utc": _timestamp_text(fold_rows[0]["timestamp"]),
                    "prediction_end_utc": _timestamp_text(fold_rows[-1]["timestamp"]),
                    "elapsed_time_distribution": _distribution(fold_elapsed, nominal_hours),
                    "counts_by_elapsed_bucket": {
                        bucket: sum(row["bucket"] == bucket for row in fold_rows)
                        for bucket in _BUCKETS
                    },
                    "metrics_by_elapsed_bucket": fold_bucket_summaries,
                })
        if experiment_rows:
            experiment_rows.sort(key=lambda row: row["timestamp"])
            elapsed_values = [row["elapsed_hours"] for row in experiment_rows]
            bucket_summaries = {
                bucket: _summarize_rows([row for row in experiment_rows if row["bucket"] == bucket])
                for bucket in _BUCKETS
            }
            fold_reports.sort(key=lambda row: row["fold_index"])
            repeated_timestamps = sum(
                count - 1 for (seen_experiment, _), count in timestamp_occurrences.items()
                if seen_experiment == experiment_id and count > 1
            )
            experiment_reports.append({
                "experiment_id": experiment_id,
                "model_type": model_type,
                "fold_count": len(fold_reports),
                "sample_count": len(experiment_rows),
                "repeated_timestamp_occurrence_count": repeated_timestamps,
                "elapsed_time_distribution": _distribution(elapsed_values, nominal_hours),
                "counts_by_elapsed_bucket": {
                    bucket: sum(row["bucket"] == bucket for row in experiment_rows)
                    for bucket in _BUCKETS
                },
                "metrics_by_elapsed_bucket": bucket_summaries,
                "folds": fold_reports,
            })

    if missing_timestamp_count:
        errors.append(f"prediction_timestamps_missing_from_ohlc:{missing_timestamp_count}")
    if not all_rows:
        errors.append("no_valid_held_out_predictions")
    if over_nominal_count:
        warnings.append("forward_label_elapsed_time_exceeds_nominal_bar_horizon")
    if over_threshold_count:
        warnings.append("long_elapsed_time_labels_detected_review_label_semantics")
    if any(report["repeated_timestamp_occurrence_count"] for report in experiment_reports):
        warnings.append("repeated_prediction_timestamps_within_experiment_review_fold_overlap")

    elapsed_all_rows = [row["elapsed_hours"] for row in all_rows]
    unique_elapsed = list(unique_horizons.values())
    longest_examples = sorted(
        all_rows,
        key=lambda row: row["elapsed_hours"],
        reverse=True,
    )[: int(longest_examples_limit)]
    examples = [{
        "experiment_id": row["experiment_id"],
        "model_type": row["model_type"],
        "fold_index": row["fold_index"],
        "prediction_timestamp_utc": _timestamp_text(row["timestamp"]),
        "target_timestamp_utc": _timestamp_text(row["target_timestamp"]),
        "elapsed_hours": round(row["elapsed_hours"], 6),
        "nominal_elapsed_hours": round(nominal_hours, 6),
        "elapsed_over_nominal_multiple": round(row["elapsed_hours"] / nominal_hours, 6),
        "elapsed_bucket": row["bucket"],
        "prediction_row_index": row["prediction_row_index"],
        "target_row_index": row["target_row_index"],
    } for row in longest_examples]
    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "scope": "GOLDCREST_LABEL_HORIZON_ELAPSED_TIME_AUDIT",
        "status": status,
        "run_id": matrix.get("run_id"),
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon_bars": horizon,
        "nominal_elapsed_hours": round(nominal_hours, 6),
        "extended_time_threshold_hours": threshold_hours,
        "ohlc_row_count": len(timestamps),
        "ohlc_first_timestamp_utc": _timestamp_text(timestamps[0]),
        "ohlc_last_timestamp_utc": _timestamp_text(timestamps[-1]),
        "prediction_model_row_count": len(all_rows),
        "unique_prediction_timestamp_count": len(unique_horizons),
        "repeated_prediction_timestamp_occurrence_count": sum(
            max(0, count - 1) for count in timestamp_occurrences.values()
        ),
        "elapsed_time_distribution_by_model_row": _distribution(elapsed_all_rows, nominal_hours),
        "elapsed_time_distribution_by_unique_timestamp": _distribution(unique_elapsed, nominal_hours),
        "counts_by_elapsed_bucket_model_rows": {
            bucket: sum(row["bucket"] == bucket for row in all_rows)
            for bucket in _BUCKETS
        },
        "prediction_rows_beyond_nominal_horizon_count": over_nominal_count,
        "prediction_rows_beyond_nominal_horizon_fraction": (
            over_nominal_count / len(all_rows) if all_rows else None
        ),
        "prediction_rows_over_extended_time_threshold_count": over_threshold_count,
        "prediction_rows_over_extended_time_threshold_fraction": (
            over_threshold_count / len(all_rows) if all_rows else None
        ),
        "longest_elapsed_horizon_examples": examples,
        "experiments": experiment_reports,
        "missing_prediction_timestamp_count": missing_timestamp_count,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "automatic_model_selection": False,
        "test_metrics_used_for_selection": False,
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "This diagnostic measures actual elapsed UTC time between each prediction timestamp "
            "and the existing row-index target at prediction_row_index + horizon_bars. It does not "
            "change the labels or historical run. Scores by elapsed-time bucket are descriptive, "
            "not a significance test, model-selection criterion, or profitability estimate."
        ),
    }


def analyze_research_run(
    manifest_path: str | Path,
    ohlc_csv_path: str | Path,
    *,
    extended_time_threshold_hours: float = 8.0,
    longest_examples_limit: int = 25,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    ohlc_file = Path(ohlc_csv_path).resolve()
    integrity = summarize_research_run(manifest_file)
    if integrity.get("status") == "FAIL":
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_LABEL_HORIZON_ELAPSED_TIME_AUDIT",
            "status": "FAIL",
            "manifest_path": str(manifest_file),
            "errors": ["research_run_integrity_check_failed", *integrity.get("errors", [])],
            "warnings": integrity.get("warnings", []),
            "automatic_model_selection": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "execution_connected": False,
        }
    try:
        manifest = _read_json(manifest_file)
        if not ohlc_file.is_file():
            raise ValueError("raw_ohlc_file_not_found")
        ohlc_digest = _sha256(ohlc_file)
        if ohlc_digest != manifest.get("raw_ohlc_sha256"):
            raise ValueError("raw_ohlc_sha256_does_not_match_pipeline_manifest")
        matrix_name = manifest.get("experiment_matrix_report")
        if not isinstance(matrix_name, str) or not matrix_name:
            raise ValueError("experiment_matrix_report_path_missing")
        matrix_file = Path(matrix_name)
        if not matrix_file.is_absolute():
            matrix_file = (manifest_file.parent / matrix_file).resolve()
        if not matrix_file.is_file():
            raise ValueError("experiment_matrix_report_not_found")
        matrix_digest = _sha256(matrix_file)
        if matrix_digest != manifest.get("experiment_matrix_report_sha256"):
            raise ValueError("experiment_matrix_report_sha256_mismatch")
        matrix = _read_json(matrix_file)
        if matrix.get("run_id") != manifest.get("run_id"):
            raise ValueError("experiment_matrix_run_id_mismatch")
        with ohlc_file.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        result = analyze_label_horizon(
            matrix,
            rows,
            extended_time_threshold_hours=extended_time_threshold_hours,
            longest_examples_limit=longest_examples_limit,
        )
        result["manifest_path"] = str(manifest_file)
        result["ohlc_path"] = str(ohlc_file)
        result["ohlc_sha256"] = ohlc_digest
        result["ohlc_hash_verified"] = True
        result["experiment_matrix_path"] = str(matrix_file)
        result["experiment_matrix_sha256"] = matrix_digest
        result["experiment_matrix_hash_verified"] = True
        result["research_run_validation_status"] = integrity.get("validation_status")
        result["research_run_provenance_status"] = integrity.get("provenance_audit_status")
        result["raw_ohlc_audit_status"] = integrity.get("raw_ohlc_audit_status")
        inherited_warnings = integrity.get("warnings", [])
        result["warnings"] = sorted(set(result.get("warnings", []) + inherited_warnings))
        if result["warnings"] and result["status"] == "PASS":
            result["status"] = "PASS_WITH_WARNINGS"
        return result
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as exc:
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_LABEL_HORIZON_ELAPSED_TIME_AUDIT",
            "status": "FAIL",
            "manifest_path": str(manifest_file),
            "ohlc_path": str(ohlc_file),
            "errors": [f"label_horizon_audit_input_invalid:{type(exc).__name__}:{exc}"],
            "warnings": integrity.get("warnings", []),
            "automatic_model_selection": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "execution_connected": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit real elapsed time for existing row-based forward labels (offline only)."
    )
    parser.add_argument("--manifest", required=True, help="Path to <run-id>_pipeline_manifest.json")
    parser.add_argument("--ohlc-csv", required=True, help="Exact raw OHLC CSV whose hash is recorded in the manifest")
    parser.add_argument("--output", help="Optional JSON report output path")
    parser.add_argument(
        "--extended-time-threshold-hours",
        type=float,
        default=8.0,
        help="Elapsed duration above which a label horizon is classed as long-extended (default: 8 hours)",
    )
    parser.add_argument(
        "--longest-examples-limit",
        type=int,
        default=25,
        help="Maximum longest-horizon examples to include (0-500, default: 25)",
    )
    args = parser.parse_args(argv)
    result = analyze_research_run(
        args.manifest,
        args.ohlc_csv,
        extended_time_threshold_hours=args.extended_time_threshold_hours,
        longest_examples_limit=args.longest_examples_limit,
    )
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 3 if result.get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline temporal-stability and label-horizon gap-sensitivity diagnostics.

This module evaluates how held-out probability metrics vary across walk-forward folds
and whether a label's forward horizon crosses unusually long OHLC timestamp gaps.
It is descriptive only: it does not rank/select models or connect to execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.research_baseline_analysis import (
    _brier,
    _label,
    _log_loss,
    _metric_summary,
    _parse_utc,
    _probability,
)
from app.research_data_audit import _is_weekend_closure_like_gap, _fixed_date_calendar_context
from app.research_results_summary import summarize_research_run

_BASELINES = {
    "constant_probability_0_5": "constant_probability_class_1",
    "matured_refit_prevalence": "matured_refit_prevalence_probability_class_1",
    "lagged_bar_direction": "lagged_direction_probability_class_1",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamps(
    ohlc_rows: Sequence[Mapping[str, Any]], *, symbol: str, timeframe: str
) -> list[datetime]:
    if not ohlc_rows:
        raise ValueError("ohlc_rows_empty")
    timestamps: list[datetime] = []
    for index, row in enumerate(ohlc_rows):
        timestamp = _parse_utc(row.get("timestamp"), "OHLC timestamp")
        row_symbol = str(row.get("symbol", "")).strip().upper()
        row_timeframe = str(row.get("timeframe", "")).strip().upper()
        if row_symbol and row_symbol != symbol.upper():
            raise ValueError(f"ohlc_row_{index}_symbol_mismatch")
        if row_timeframe and row_timeframe != timeframe.upper():
            raise ValueError(f"ohlc_row_{index}_timeframe_mismatch")
        if timestamps and timestamp <= timestamps[-1]:
            raise ValueError("ohlc_timestamps_must_be_strictly_increasing_unique")
        timestamps.append(timestamp)
    return timestamps


def _describe_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0, "model": None, "baselines": {}, "improvements": {}}
    labels = [int(row["label"]) for row in rows]
    model_probabilities = [float(row["model_probability"]) for row in rows]
    baseline_probabilities = {
        key: [float(row["baseline_probabilities"][field]) for row in rows]
        for key, field in _BASELINES.items()
    }
    model_metrics = _metric_summary(labels, model_probabilities)
    baselines: dict[str, Any] = {}
    improvements: dict[str, Any] = {}
    for name, probabilities in baseline_probabilities.items():
        baseline_metrics = _metric_summary(labels, probabilities)
        baselines[name] = baseline_metrics
        improvements[name] = {
            "log_loss_baseline_minus_model": baseline_metrics["log_loss"] - model_metrics["log_loss"],
            "brier_baseline_minus_model": baseline_metrics["brier_score"] - model_metrics["brier_score"],
        }
    return {
        "sample_count": len(rows),
        "model": model_metrics,
        "baselines": baselines,
        "improvements": improvements,
    }


def _fold_stability(fold_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for baseline_name in _BASELINES:
        for metric_name in ("log_loss_baseline_minus_model", "brier_baseline_minus_model"):
            values = [
                float(fold["baseline_comparisons"][baseline_name][metric_name])
                for fold in fold_reports
                if isinstance(fold.get("baseline_comparisons"), dict)
                and baseline_name in fold["baseline_comparisons"]
                and isinstance(fold[ "baseline_comparisons"][baseline_name].get(metric_name), (int, float))
            ]
            positive = sum(value > 0 for value in values)
            negative = sum(value < 0 for value in values)
            neutral = len(values) - positive - negative
            summary.setdefault(baseline_name, {})[metric_name] = {
                "folds_with_positive_improvement": positive,
                "folds_with_negative_improvement": negative,
                "folds_with_zero_improvement": neutral,
                "fold_count": len(values),
                "positive_fold_fraction": positive / len(values) if values else None,
                "mean_fold_improvement_unweighted": _mean(values),
                "minimum_fold_improvement": min(values) if values else None,
                "maximum_fold_improvement": max(values) if values else None,
                "interpretation": "Descriptive fold consistency only; folds may be dependent and this is not a significance test.",
            }
    return summary


def analyze_prediction_temporal_stability(
    matrix: Mapping[str, Any],
    ohlc_rows: Sequence[Mapping[str, Any]],
    *,
    gap_threshold_hours: float = 8.0,
) -> dict[str, Any]:
    """Calculate fold stability and test-label exposure to long timestamp gaps."""
    errors: list[str] = []
    warnings: list[str] = []
    symbol = str(matrix.get("symbol", "")).strip().upper()
    timeframe = str(matrix.get("timeframe", "")).strip().upper()
    horizon = matrix.get("horizon_bars")
    if not symbol or not timeframe:
        errors.append("matrix_symbol_or_timeframe_missing")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        errors.append("matrix_horizon_bars_invalid")
    if not isinstance(gap_threshold_hours, (int, float)) or isinstance(gap_threshold_hours, bool) or not math.isfinite(float(gap_threshold_hours)) or gap_threshold_hours <= 0:
        errors.append("gap_threshold_hours_invalid")
    if errors:
        return {
            "schema_version": 1, "scope": "GOLDCREST_TEMPORAL_STABILITY_GAP_SENSITIVITY",
            "status": "FAIL", "errors": errors, "warnings": warnings,
            "automatic_model_selection": False, "actionable": False, "execution_connected": False,
        }
    try:
        timestamps = _parse_timestamps(ohlc_rows, symbol=symbol, timeframe=timeframe)
    except (TypeError, ValueError, OverflowError) as exc:
        return {
            "schema_version": 1, "scope": "GOLDCREST_TEMPORAL_STABILITY_GAP_SENSITIVITY",
            "status": "FAIL", "errors": [f"ohlc_timestamp_validation_failed:{type(exc).__name__}:{exc}"],
            "warnings": [], "automatic_model_selection": False, "actionable": False,
            "execution_connected": False,
        }

    timestamp_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    gap_threshold_seconds = float(gap_threshold_hours) * 3600.0
    long_gaps: list[dict[str, Any]] = []
    for left, right in zip(timestamps, timestamps[1:]):
        elapsed = (right - left).total_seconds()
        if elapsed <= gap_threshold_seconds:
            continue
        weekend_like = _is_weekend_closure_like_gap(left, right)
        long_gaps.append({
            "from_utc": _timestamp_text(left),
            "to_utc": _timestamp_text(right),
            "gap_hours": round(elapsed / 3600.0, 3),
            "gap_category": "weekend_closure_like" if weekend_like else "non_weekend_long_gap",
            "calendar_context": _fixed_date_calendar_context(left, right),
        })
    if any(gap["gap_category"] == "non_weekend_long_gap" for gap in long_gaps):
        warnings.append("non_weekend_long_gaps_detected_review_market_calendar_and_provider_coverage")
    if not long_gaps:
        warnings.append("no_gaps_above_configured_threshold_found")

    experiments = matrix.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        errors.append("matrix_experiments_missing_or_empty")
        experiments = []

    results: list[dict[str, Any]] = []
    total_predictions = 0
    total_horizon_gap_exposed = 0
    missing_prediction_timestamps = 0
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
        fold_reports: list[dict[str, Any]] = []
        experiment_exposure_rows: list[dict[str, Any]] = []
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

            parsed_rows: list[dict[str, Any]] = []
            gap_crossing_count = 0
            for row_index, prediction in enumerate(predictions):
                try:
                    if not isinstance(prediction, dict):
                        raise ValueError("prediction_record_not_object")
                    timestamp = _parse_utc(prediction.get("timestamp_utc"), "prediction timestamp")
                    ohlc_index = timestamp_to_index.get(timestamp)
                    if ohlc_index is None:
                        missing_prediction_timestamps += 1
                        raise ValueError("prediction_timestamp_missing_from_ohlc")
                    target_index = ohlc_index + int(horizon)
                    if target_index >= len(timestamps):
                        raise ValueError("label_horizon_target_missing_from_ohlc")
                    label = _label(prediction.get("label"), "label")
                    model_probability = _probability(prediction.get("model_probability_class_1"), "model_probability_class_1")
                    baseline_probabilities = {
                        key: _probability(prediction.get(field), field)
                        for key, field in _BASELINES.items()
                    }
                    crossed = [
                        timestamps[pos + 1]
                        for pos in range(ohlc_index, target_index)
                        if (timestamps[pos + 1] - timestamps[pos]).total_seconds() > gap_threshold_seconds
                    ]
                    elapsed_hours = (timestamps[target_index] - timestamp).total_seconds() / 3600.0
                    row = {
                        "timestamp": timestamp,
                        "label": label,
                        "model_probability": model_probability,
                        "baseline_probabilities": baseline_probabilities,
                        "ohlc_index": ohlc_index,
                        "target_timestamp": timestamps[target_index],
                        "elapsed_hours": elapsed_hours,
                        "horizon_crosses_long_gap": bool(crossed),
                        "crossed_gap_count": len(crossed),
                        "crossed_gap_start_timestamps": [_timestamp_text(timestamps[pos]) for pos in range(ohlc_index, target_index)
                            if (timestamps[pos + 1] - timestamps[pos]).total_seconds() > gap_threshold_seconds],
                    }
                    parsed_rows.append(row)
                    experiment_exposure_rows.append(row)
                    if row["horizon_crosses_long_gap"]:
                        gap_crossing_count += 1
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    errors.append(f"experiment_{experiment_id}_fold_{fold_index}_prediction_{row_index}_invalid:{str(exc)}")
            if not parsed_rows:
                continue
            parsed_rows.sort(key=lambda row: row["timestamp"])
            fold_group_summaries = {
                "horizon_crosses_long_gap": _describe_group([row for row in parsed_rows if row["horizon_crosses_long_gap"]]),
                "horizon_does_not_cross_long_gap": _describe_group([row for row in parsed_rows if not row["horizon_crosses_long_gap"]]),
            }
            all_metrics = _describe_group(parsed_rows)
            fold_baseline_comparisons: dict[str, dict[str, float]] = {}
            for baseline_name, improvement in all_metrics["improvements"].items():
                fold_baseline_comparisons[baseline_name] = {
                    "log_loss_baseline_minus_model": improvement["log_loss_baseline_minus_model"],
                    "brier_baseline_minus_model": improvement["brier_baseline_minus_model"],
                }
            fold_reports.append({
                "fold_index": fold_index,
                "sample_count": len(parsed_rows),
                "test_start_utc": _timestamp_text(parsed_rows[0]["timestamp"]),
                "test_end_utc": _timestamp_text(parsed_rows[-1]["timestamp"]),
                "model_metrics": all_metrics["model"],
                "baseline_metrics": all_metrics["baselines"],
                "baseline_comparisons": fold_baseline_comparisons,
                "gap_exposure": {
                    "horizon_crosses_long_gap_count": gap_crossing_count,
                    "horizon_does_not_cross_long_gap_count": len(parsed_rows) - gap_crossing_count,
                    "metrics_by_group": fold_group_summaries,
                    "maximum_label_elapsed_hours": round(max(row["elapsed_hours"] for row in parsed_rows), 3),
                    "median_label_elapsed_hours": round(statistics.median(row["elapsed_hours"] for row in parsed_rows), 3),
                },
            })
            total_predictions += len(parsed_rows)
            total_horizon_gap_exposed += gap_crossing_count

        fold_reports.sort(key=lambda row: row["fold_index"])
        all_experiment_metrics = _describe_group(experiment_exposure_rows)
        exposure_metrics = {
            "horizon_crosses_long_gap": _describe_group([row for row in experiment_exposure_rows if row["horizon_crosses_long_gap"]]),
            "horizon_does_not_cross_long_gap": _describe_group([row for row in experiment_exposure_rows if not row["horizon_crosses_long_gap"]]),
        }
        results.append({
            "experiment_id": experiment_id,
            "model_type": model_type,
            "fold_count": len(fold_reports),
            "sample_count": len(experiment_exposure_rows),
            "overall_metrics": all_experiment_metrics,
            "fold_consistency": _fold_stability(fold_reports),
            "folds": fold_reports,
            "gap_exposure": {
                "horizon_crosses_long_gap_count": sum(bool(row["horizon_crosses_long_gap"]) for row in experiment_exposure_rows),
                "horizon_does_not_cross_long_gap_count": sum(not row["horizon_crosses_long_gap"] for row in experiment_exposure_rows),
                "metrics_by_group": exposure_metrics,
                "interpretation": "Forward-label horizon exposure is identified from the ordered OHLC rows. Differences between groups are descriptive and may be confounded by time-of-week, volatility, and sample size.",
            },
        })

    if missing_prediction_timestamps:
        errors.append(f"prediction_timestamps_missing_from_ohlc:{missing_prediction_timestamps}")
    if total_predictions == 0:
        errors.append("no_valid_held_out_predictions")
    status = "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS"
    return {
        "schema_version": 1,
        "scope": "GOLDCREST_TEMPORAL_STABILITY_GAP_SENSITIVITY",
        "status": status,
        "run_id": matrix.get("run_id"),
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon_bars": horizon,
        "ohlc_row_count": len(timestamps),
        "ohlc_first_timestamp_utc": _timestamp_text(timestamps[0]),
        "ohlc_last_timestamp_utc": _timestamp_text(timestamps[-1]),
        "gap_threshold_hours": float(gap_threshold_hours),
        "long_gap_count": len(long_gaps),
        "weekend_closure_like_gap_count": sum(gap["gap_category"] == "weekend_closure_like" for gap in long_gaps),
        "non_weekend_long_gap_count": sum(gap["gap_category"] == "non_weekend_long_gap" for gap in long_gaps),
        "holiday_context_gap_count": sum(bool(gap["calendar_context"]) for gap in long_gaps),
        "long_gap_examples": long_gaps[:100],
        "prediction_count": total_predictions,
        "prediction_horizon_crosses_long_gap_count": total_horizon_gap_exposed,
        "prediction_horizon_crosses_long_gap_fraction": (
            total_horizon_gap_exposed / total_predictions if total_predictions else None
        ),
        "missing_prediction_timestamp_count": missing_prediction_timestamps,
        "experiments": results,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "automatic_model_selection": False,
        "test_metrics_used_for_selection": False,
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "Descriptive walk-forward fold consistency and forward-label exposure to long OHLC gaps. "
            "A positive loss improvement favors the model over that baseline; fold consistency is not "
            "a significance test, and gap-exposed versus unexposed comparisons are not causal. "
            "Do not rank or select models from this report."
        ),
    }


def analyze_research_run(
    manifest_path: str | Path,
    ohlc_csv_path: str | Path,
    *,
    gap_threshold_hours: float = 8.0,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    ohlc_file = Path(ohlc_csv_path).resolve()
    integrity = summarize_research_run(manifest_file)
    if integrity.get("status") == "FAIL":
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_TEMPORAL_STABILITY_GAP_SENSITIVITY",
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
        result = analyze_prediction_temporal_stability(
            matrix, rows, gap_threshold_hours=gap_threshold_hours
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
            "scope": "GOLDCREST_TEMPORAL_STABILITY_GAP_SENSITIVITY",
            "status": "FAIL",
            "manifest_path": str(manifest_file),
            "ohlc_path": str(ohlc_file),
            "errors": [f"temporal_stability_input_invalid:{type(exc).__name__}:{exc}"],
            "warnings": integrity.get("warnings", []),
            "automatic_model_selection": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "execution_connected": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze walk-forward fold stability and long-gap exposure for held-out research forecasts."
    )
    parser.add_argument("--manifest", required=True, help="Path to <run-id>_pipeline_manifest.json")
    parser.add_argument("--ohlc-csv", required=True, help="Exact raw OHLC CSV whose hash is recorded in the manifest")
    parser.add_argument("--output", help="Optional JSON report output path")
    parser.add_argument("--gap-threshold-hours", type=float, default=8.0, help="Elapsed UTC gap threshold; default 8 hours")
    args = parser.parse_args(argv)
    result = analyze_research_run(
        args.manifest, args.ohlc_csv, gap_threshold_hours=args.gap_threshold_hours
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

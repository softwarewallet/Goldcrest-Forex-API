"""Paired comparison of an all-label baseline and elapsed-time-capped research run.

This audit verifies both runs' immutable provenance and common test windows, then compares
the baseline and capped model on the exact same eligible held-out timestamps. It is strictly
descriptive: it never ranks/selects a model or connects to broker/execution systems.
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

_LABEL_COLUMNS = {"label_elapsed_hours", "label_eligible"}
_CORE_DATA_COLUMNS = {
    "symbol", "timeframe", "timestamp", "open", "high", "low", "close",
    "label", "label_available_at",
}
_METRICS = ("sample_count", "prevalence", "log_loss", "brier_score",
            "accuracy_at_fixed_half", "mean_predicted_probability")
_DELTA_METRICS = ("log_loss", "brier_score", "accuracy_at_fixed_half",
                  "prevalence", "mean_predicted_probability")
_ARTIFACTS = (
    ("raw_ohlc_path", "raw_ohlc_sha256"),
    ("dataset_path", "dataset_sha256"),
    ("dataset_manifest_path", "dataset_manifest_sha256"),
    ("experiment_matrix_report", "experiment_matrix_report_sha256"),
    ("provenance_audit_path", "provenance_audit_sha256"),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_hash(payload: Mapping[str, Any], field: str) -> str:
    copy = dict(payload)
    copy.pop(field, None)
    encoded = json.dumps(copy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha256_bytes(encoded)


def _resolve_artifact(value: object, manifest_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest has a missing artifact path")
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        relative = (manifest_path.parent / candidate).resolve()
        if relative.is_file():
            return relative
    raise ValueError(f"artifact not found: {value}")


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_bound_run(manifest_file: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_file).expanduser().resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{manifest_path.name}: unsupported pipeline-manifest schema")
    for flag in ("actionable", "execution_connected", "automatic_winner_selection",
                 "test_metrics_used_for_selection"):
        if manifest.get(flag) is not False:
            raise ValueError(f"{manifest_path.name}: {flag} must be false")
    stored_hash = manifest.get("pipeline_manifest_sha256")
    if not isinstance(stored_hash, str) or stored_hash != _canonical_hash(manifest, "pipeline_manifest_sha256"):
        raise ValueError(f"{manifest_path.name}: pipeline manifest hash mismatch")

    artifacts: dict[str, Path] = {}
    values: dict[str, Any] = {}
    for path_field, hash_field in _ARTIFACTS:
        artifact = _resolve_artifact(manifest.get(path_field), manifest_path)
        actual_hash = _sha256_file(artifact)
        expected_hash = manifest.get(hash_field)
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            raise ValueError(f"{manifest_path.name}: {path_field} hash mismatch")
        artifacts[path_field] = artifact

    # Raw-data audit is part of newer manifests. Verify it whenever present.
    audit_value = manifest.get("raw_ohlc_audit_path")
    if audit_value:
        audit_path = _resolve_artifact(audit_value, manifest_path)
        audit_hash = _sha256_file(audit_path)
        if audit_hash != manifest.get("raw_ohlc_audit_sha256"):
            raise ValueError(f"{manifest_path.name}: raw OHLC audit hash mismatch")
        raw_audit = _read_json(audit_path)
        if raw_audit.get("snapshot_sha256") != _sha256_file(artifacts["raw_ohlc_path"]):
            raise ValueError(f"{manifest_path.name}: raw OHLC audit snapshot mismatch")
        if raw_audit.get("errors") != []:
            raise ValueError(f"{manifest_path.name}: raw OHLC audit contains hard errors")
        if raw_audit.get("actionable") is not False:
            raise ValueError(f"{manifest_path.name}: raw OHLC audit safety flag is invalid")
        values["raw_ohlc_audit"] = raw_audit

    dataset_manifest = _read_json(artifacts["dataset_manifest_path"])
    if dataset_manifest.get("output_dataset_sha256") != _sha256_file(artifacts["dataset_path"]):
        raise ValueError(f"{manifest_path.name}: dataset manifest/output hash mismatch")
    if dataset_manifest.get("input_snapshot_sha256") != _sha256_file(artifacts["raw_ohlc_path"]):
        raise ValueError(f"{manifest_path.name}: dataset manifest/raw snapshot hash mismatch")
    dataset_manifest_hash = dataset_manifest.get("manifest_sha256")
    if dataset_manifest_hash and dataset_manifest_hash != _canonical_hash(dataset_manifest, "manifest_sha256"):
        raise ValueError(f"{manifest_path.name}: dataset manifest hash mismatch")

    matrix = _read_json(artifacts["experiment_matrix_report"])
    if matrix.get("run_id") != manifest.get("run_id"):
        raise ValueError(f"{manifest_path.name}: experiment matrix run ID mismatch")
    if matrix.get("dataset_sha256") != _sha256_file(artifacts["dataset_path"]):
        raise ValueError(f"{manifest_path.name}: experiment matrix dataset hash mismatch")
    for flag in ("actionable", "automatic_winner_selection", "test_metrics_used_for_selection"):
        if matrix.get(flag) is not False:
            raise ValueError(f"{manifest_path.name}: experiment matrix {flag} must be false")
    matrix_hash = matrix.get("report_payload_sha256")
    if matrix_hash and matrix_hash != _canonical_hash(matrix, "report_payload_sha256"):
        raise ValueError(f"{manifest_path.name}: experiment matrix payload hash mismatch")

    provenance = _read_json(artifacts["provenance_audit_path"])
    if provenance.get("source_report_sha256") != _sha256_file(artifacts["experiment_matrix_report"]):
        raise ValueError(f"{manifest_path.name}: provenance audit source hash mismatch")
    if provenance.get("run_id") != manifest.get("run_id"):
        raise ValueError(f"{manifest_path.name}: provenance audit run ID mismatch")
    if provenance.get("status") == "FAIL" or manifest.get("provenance_audit_status") == "FAIL":
        raise ValueError(f"{manifest_path.name}: experiment-matrix provenance audit failed")
    if provenance.get("actionable") is not False or provenance.get("execution_connected") is not False:
        raise ValueError(f"{manifest_path.name}: provenance audit safety flags are invalid")

    mode = manifest.get("label_eligibility_mode", "all_labels_included")
    cap = manifest.get("max_label_elapsed_hours")
    dataset_eligibility = dataset_manifest.get("label_eligibility")
    if mode == "all_labels_included":
        if cap is not None:
            raise ValueError(f"{manifest_path.name}: baseline unexpectedly declares a label cap")
        if isinstance(dataset_eligibility, dict) and dataset_eligibility.get("mode") == "elapsed_time_cap":
            raise ValueError(f"{manifest_path.name}: baseline dataset is unexpectedly capped")
    elif mode == "elapsed_time_cap":
        if (isinstance(cap, bool) or not isinstance(cap, (int, float))
                or not math.isfinite(float(cap)) or float(cap) <= 0):
            raise ValueError(f"{manifest_path.name}: elapsed-time cap is invalid")
        if not isinstance(dataset_eligibility, dict):
            raise ValueError(f"{manifest_path.name}: capped dataset eligibility metadata is missing")
        if (dataset_eligibility.get("mode") != mode
                or dataset_eligibility.get("max_label_elapsed_hours") != cap
                or dataset_eligibility.get("labels_rewritten") is not False
                or dataset_eligibility.get("all_rows_preserved_for_fixed_temporal_folds") is not True):
            raise ValueError(f"{manifest_path.name}: cap metadata does not match pipeline manifest")
    else:
        raise ValueError(f"{manifest_path.name}: unsupported label eligibility mode {mode!r}")
    if matrix.get("label_eligibility_mode", "all_labels_included") != mode:
        raise ValueError(f"{manifest_path.name}: matrix eligibility mode mismatch")
    if matrix.get("max_label_elapsed_hours") != cap:
        raise ValueError(f"{manifest_path.name}: matrix elapsed-time cap mismatch")

    values.update({
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "artifacts": artifacts,
        "dataset_manifest": dataset_manifest,
        "matrix": matrix,
    })
    return values


def _read_dataset(path: Path) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not _CORE_DATA_COLUMNS.issubset(fields):
            raise ValueError(f"{path.name}: missing required dataset columns")
        rows = list(reader)
    by_time: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        timestamp = _iso(_utc(row.get("timestamp"), f"{path.name} row {index} timestamp"))
        if timestamp in by_time:
            raise ValueError(f"{path.name}: duplicate timestamp {timestamp}")
        row["_timestamp_utc"] = timestamp
        row["_label_available_utc"] = _iso(
            _utc(row.get("label_available_at"), f"{path.name} row {index} label_available_at")
        )
        by_time[timestamp] = row
    if not rows:
        raise ValueError(f"{path.name}: dataset has no rows")
    return fields, rows, by_time


def _eligible(row: Mapping[str, str], cap_hours: float) -> bool:
    text = row.get("label_eligible")
    elapsed_text = row.get("label_elapsed_hours")
    if text not in {"0", "1"} or elapsed_text is None:
        raise ValueError("capped dataset row lacks eligibility metadata")
    elapsed = float(elapsed_text)
    actual = (
        _utc(row["_label_available_utc"], "label availability")
        - _utc(row["_timestamp_utc"], "feature timestamp")
    ).total_seconds() / 3600.0
    if not math.isfinite(elapsed) or elapsed <= 0 or abs(elapsed - actual) > 1.1e-6:
        raise ValueError("declared label elapsed hours do not match target timestamps")
    expected = elapsed <= cap_hours + 1e-9
    if (text == "1") != expected:
        raise ValueError("label eligibility flag does not match configured elapsed-time cap")
    return text == "1"


def _metric_summary(rows: Sequence[Mapping[str, Any]], probability_field: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot calculate metrics for an empty prediction set")
    labels = [int(row["label"]) for row in rows]
    probabilities = [float(row[probability_field]) for row in rows]
    if any(label not in (0, 1) for label in labels):
        raise ValueError("prediction labels must be binary")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise ValueError("prediction probabilities must be finite and within [0, 1]")
    eps = 1e-15
    count = len(rows)
    brier = statistics.fmean((p - y) ** 2 for y, p in zip(labels, probabilities))
    log_loss = -statistics.fmean(
        y * math.log(max(eps, p)) + (1 - y) * math.log(max(eps, 1 - p))
        for y, p in zip(labels, probabilities)
    )
    return {
        "sample_count": count,
        "positive_count": sum(labels),
        "negative_count": count - sum(labels),
        "prevalence": statistics.fmean(labels),
        "log_loss": log_loss,
        "brier_score": brier,
        "accuracy_at_fixed_half": statistics.fmean(
            float((p >= 0.5) == bool(y)) for y, p in zip(labels, probabilities)
        ),
        "mean_predicted_probability": statistics.fmean(probabilities),
    }


def _check_recorded_metrics(actual: Mapping[str, Any], recomputed: Mapping[str, Any], label: str) -> None:
    for field in _METRICS:
        value = actual.get(field)
        expected = recomputed.get(field)
        if field == "sample_count":
            if value != expected:
                raise ValueError(f"{label}: recorded {field} does not match test predictions")
        elif (isinstance(value, bool) or not isinstance(value, (int, float))
              or not math.isfinite(float(value)) or abs(float(value) - float(expected)) > 1e-6):
            raise ValueError(f"{label}: recorded {field} does not match test predictions")


def _fold_map(experiment: Mapping[str, Any], run_label: str) -> dict[int, dict[str, Any]]:
    values = experiment.get("fold_results")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{run_label}: experiment fold_results is missing")
    result: dict[int, dict[str, Any]] = {}
    for row in values:
        if not isinstance(row, dict):
            raise ValueError(f"{run_label}: invalid fold record")
        index = row.get("fold_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in result:
            raise ValueError(f"{run_label}: invalid/duplicate fold index")
        result[index] = row
    return result


def _experiments_by_key(matrix: Mapping[str, Any], run_label: str) -> dict[tuple[str, str], dict[str, Any]]:
    values = matrix.get("experiments")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{run_label}: experiments missing")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in values:
        if not isinstance(row, dict):
            raise ValueError(f"{run_label}: invalid experiment record")
        key = (str(row.get("model_type", "")), str(row.get("feature_variant", "")))
        if not all(key) or key in result:
            raise ValueError(f"{run_label}: missing/duplicate model and feature-variant key")
        result[key] = row
    return result


def compare_label_eligibility_runs(
    baseline_manifest_path: str | Path,
    variant_manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify two compatible research runs and compute paired, same-timestamp comparisons."""
    try:
        baseline = _load_bound_run(baseline_manifest_path)
        variant = _load_bound_run(variant_manifest_path)
        base_manifest, variant_manifest = baseline["manifest"], variant["manifest"]
        base_matrix, variant_matrix = baseline["matrix"], variant["matrix"]

        if base_manifest.get("run_id") == variant_manifest.get("run_id"):
            raise ValueError("baseline and variant must use distinct run IDs")
        if variant_manifest.get("label_eligibility_mode", "all_labels_included") != "elapsed_time_cap":
            raise ValueError("variant must use elapsed_time_cap eligibility mode")
        if baseline["manifest"].get("label_horizon_bars") != variant_manifest.get("label_horizon_bars"):
            raise ValueError("runs have different label horizons")
        for field in ("symbol", "timeframe", "source", "source_version", "retrieved_at_utc",
                      "raw_ohlc_sha256", "raw_ohlc_audit_status", "class_1_means"):
            if base_manifest.get(field) != variant_manifest.get(field):
                raise ValueError(f"runs differ in required baseline metadata: {field}")
        if _sha256_file(baseline["artifacts"]["raw_ohlc_path"]) != _sha256_file(variant["artifacts"]["raw_ohlc_path"]):
            raise ValueError("runs do not use identical raw OHLC snapshots")
        for field in ("symbol", "timeframe", "label_horizon_bars", "label_definition",
                      "data_source", "source_version", "retrieved_at_utc", "terms_reference"):
            base_value = baseline["dataset_manifest"].get(field)
            variant_value = variant["dataset_manifest"].get(field)
            if base_value != variant_value:
                raise ValueError(f"dataset manifests differ on {field}")
        if base_manifest.get("raw_ohlc_audit_warnings") != variant_manifest.get("raw_ohlc_audit_warnings"):
            raise ValueError("runs have different raw OHLC audit warning sets")

        cap_hours = float(variant_manifest["max_label_elapsed_hours"])
        base_fields, base_rows, base_by_time = _read_dataset(baseline["artifacts"]["dataset_path"])
        variant_fields, variant_rows, variant_by_time = _read_dataset(variant["artifacts"]["dataset_path"])
        if [row["_timestamp_utc"] for row in base_rows] != [row["_timestamp_utc"] for row in variant_rows]:
            raise ValueError("baseline and variant dataset timestamp timelines differ")
        ignored = _LABEL_COLUMNS | {"_timestamp_utc", "_label_available_utc"}
        base_features = [name for name in base_fields if name not in _CORE_DATA_COLUMNS]
        variant_features = [name for name in variant_fields if name not in _CORE_DATA_COLUMNS | _LABEL_COLUMNS]
        if base_features != variant_features:
            raise ValueError("baseline and variant feature schemas differ")
        shared_columns = [name for name in base_fields if name in variant_fields]
        for index, (base_row, variant_row) in enumerate(zip(base_rows, variant_rows), start=2):
            for name in shared_columns:
                if base_row.get(name) != variant_row.get(name):
                    raise ValueError(f"dataset row {index}: shared column {name!r} differs between runs")
            _eligible(variant_row, cap_hours)
        expected_feature_names = base_manifest.get("feature_names")
        if expected_feature_names != variant_manifest.get("feature_names"):
            raise ValueError("pipeline manifests declare different feature-name schemas")
        if expected_feature_names != base_features:
            raise ValueError("dataset feature columns do not match pipeline feature names")

        base_windows = base_matrix.get("common_fold_test_windows")
        variant_windows = variant_matrix.get("common_fold_test_windows")
        if not isinstance(base_windows, list) or not isinstance(variant_windows, list):
            raise ValueError("experiment matrices lack common fold windows")
        def normalize_windows(rows: list[Any], name: str) -> dict[int, dict[str, Any]]:
            result: dict[int, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"{name}: invalid common fold window")
                index = row.get("fold_index")
                if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in result:
                    raise ValueError(f"{name}: invalid/duplicate common fold index")
                start = _iso(_utc(row.get("test_start_utc"), f"{name} test start"))
                end = _iso(_utc(row.get("test_end_utc"), f"{name} test end"))
                test_rows = row.get("test_rows")
                if isinstance(test_rows, bool) or not isinstance(test_rows, int) or test_rows < 1:
                    raise ValueError(f"{name}: invalid common test-row count")
                result[index] = {"test_start_utc": start, "test_end_utc": end, "test_rows": test_rows}
            return result
        base_window_map = normalize_windows(base_windows, "baseline")
        variant_window_map = normalize_windows(variant_windows, "variant")
        if base_window_map != variant_window_map:
            raise ValueError("runs do not use the same chronological fold/test windows")

        ordered_windows = sorted(
            (
                _utc(window["test_start_utc"], "fold test start"),
                _utc(window["test_end_utc"], "fold test end"),
                window["test_rows"],
            )
            for window in base_window_map.values()
        )
        overlapping_window_count = 0
        furthest_test_end: datetime | None = None
        for start, end, _ in ordered_windows:
            # Boundaries are inclusive; windows sharing a boundary timestamp overlap.
            if furthest_test_end is not None and start <= furthest_test_end:
                overlapping_window_count += 1
            if furthest_test_end is None or end > furthest_test_end:
                furthest_test_end = end
        test_observations_across_folds = sum(window["test_rows"] for window in base_window_map.values())
        unique_test_timestamps_across_windows = sum(
            any(start <= _utc(row["_timestamp_utc"], "dataset timestamp") <= end
                for start, end, _ in ordered_windows)
            for row in base_rows
        )

        base_experiments = _experiments_by_key(base_matrix, "baseline")
        variant_experiments = _experiments_by_key(variant_matrix, "variant")
        if set(base_experiments) != set(variant_experiments):
            raise ValueError("runs do not contain the same model/feature-variant experiments")
        if base_matrix.get("horizon_bars") != variant_matrix.get("horizon_bars"):
            raise ValueError("experiment matrices have different horizons")

        comparisons: list[dict[str, Any]] = []
        for experiment_key in sorted(base_experiments):
            base_exp = base_experiments[experiment_key]
            variant_exp = variant_experiments[experiment_key]
            model_type, feature_variant = experiment_key
            if base_exp.get("feature_names") != variant_exp.get("feature_names"):
                raise ValueError(f"{experiment_key}: feature names differ")
            if base_exp.get("feature_schema_sha256") != variant_exp.get("feature_schema_sha256"):
                raise ValueError(f"{experiment_key}: feature schema fingerprints differ")
            base_folds = _fold_map(base_exp, "baseline")
            variant_folds = _fold_map(variant_exp, "variant")
            if set(base_folds) != set(variant_folds) or set(base_folds) != set(base_window_map):
                raise ValueError(f"{experiment_key}: fold indices differ")
            paired_fold_results: list[dict[str, Any]] = []
            for fold_index in sorted(base_folds):
                base_fold, variant_fold = base_folds[fold_index], variant_folds[fold_index]
                window = base_window_map[fold_index]
                for field in ("test_start_utc", "test_end_utc"):
                    base_boundary = _iso(_utc(base_fold.get(field), f"baseline fold {field}"))
                    variant_boundary = _iso(_utc(variant_fold.get(field), f"variant fold {field}"))
                    if base_boundary != variant_boundary or base_boundary != window[field]:
                        raise ValueError(f"{experiment_key} fold {fold_index}: {field} differs")
                base_predictions = base_fold.get("test_predictions")
                variant_predictions = variant_fold.get("test_predictions")
                if not isinstance(base_predictions, list) or not isinstance(variant_predictions, list):
                    raise ValueError(f"{experiment_key} fold {fold_index}: test predictions are missing")
                def prediction_map(predictions: list[Any], label: str) -> dict[str, dict[str, Any]]:
                    result: dict[str, dict[str, Any]] = {}
                    for item in predictions:
                        if not isinstance(item, dict):
                            raise ValueError(f"{label}: invalid prediction record")
                        timestamp = _iso(_utc(item.get("timestamp_utc"), f"{label} timestamp"))
                        if timestamp in result:
                            raise ValueError(f"{label}: duplicate prediction timestamp")
                        if item.get("label") not in (0, 1) or isinstance(item.get("label"), bool):
                            raise ValueError(f"{label}: prediction label is invalid")
                        result[timestamp] = item
                    return result
                base_pred_map = prediction_map(base_predictions, "baseline")
                variant_pred_map = prediction_map(variant_predictions, "variant")
                if len(base_pred_map) != window["test_rows"]:
                    raise ValueError(f"{experiment_key} fold {fold_index}: baseline prediction count does not match test window")
                start, end = window["test_start_utc"], window["test_end_utc"]
                expected_full_timestamps = [
                    row["_timestamp_utc"] for row in base_rows if start <= row["_timestamp_utc"] <= end
                ]
                if sorted(base_pred_map) != expected_full_timestamps:
                    raise ValueError(f"{experiment_key} fold {fold_index}: baseline predictions do not cover the common test window")
                eligible_timestamps = [
                    timestamp for timestamp in expected_full_timestamps
                    if _eligible(variant_by_time[timestamp], cap_hours)
                ]
                if sorted(variant_pred_map) != eligible_timestamps:
                    raise ValueError(f"{experiment_key} fold {fold_index}: capped predictions do not match eligible test timestamps")
                for timestamp in eligible_timestamps:
                    if base_pred_map[timestamp].get("label") != variant_pred_map[timestamp].get("label"):
                        raise ValueError(f"{experiment_key} fold {fold_index}: label mismatch at {timestamp}")

                baseline_full_metrics = _metric_summary(base_predictions, "model_probability_class_1")
                baseline_paired_metrics = _metric_summary(
                    [base_pred_map[timestamp] for timestamp in eligible_timestamps],
                    "model_probability_class_1",
                )
                variant_paired_metrics = _metric_summary(variant_predictions, "model_probability_class_1")
                _check_recorded_metrics(base_fold.get("test_metrics", {}), baseline_full_metrics,
                                        f"{experiment_key} fold {fold_index} baseline metrics")
                _check_recorded_metrics(variant_fold.get("test_metrics", {}), variant_paired_metrics,
                                        f"{experiment_key} fold {fold_index} variant metrics")
                expected_excluded = len(base_pred_map) - len(eligible_timestamps)
                if (variant_fold.get("test_rows_available_before_eligibility") != len(base_pred_map)
                        or variant_fold.get("test_rows_excluded_ineligible") != expected_excluded):
                    raise ValueError(f"{experiment_key} fold {fold_index}: eligibility exclusion counts are inconsistent")

                delta = {
                    metric: round(float(variant_paired_metrics[metric]) - float(baseline_paired_metrics[metric]), 10)
                    for metric in _DELTA_METRICS
                }
                paired_fold_results.append({
                    "fold_index": fold_index,
                    "test_start_utc": start,
                    "test_end_utc": end,
                    "baseline_full_test_metrics": baseline_full_metrics,
                    "baseline_same_eligible_timestamps_metrics": baseline_paired_metrics,
                    "variant_eligible_test_metrics": variant_paired_metrics,
                    "paired_variant_minus_baseline_delta": delta,
                    "test_rows_available_before_eligibility": len(base_pred_map),
                    "test_rows_excluded_ineligible": expected_excluded,
                    "eligible_test_rows": len(eligible_timestamps),
                    "eligible_label_prevalence": variant_paired_metrics["prevalence"],
                    "actionable": False,
                })

            aggregate: dict[str, Any] = {}
            for partition_name in ("baseline_same_eligible_timestamps_metrics", "variant_eligible_test_metrics"):
                aggregate[partition_name] = {
                    metric: round(statistics.fmean(float(fold[partition_name][metric]) for fold in paired_fold_results), 10)
                    for metric in _METRICS
                }
            aggregate["paired_variant_minus_baseline_delta_mean_across_folds"] = {
                metric: round(statistics.fmean(float(fold["paired_variant_minus_baseline_delta"][metric])
                                               for fold in paired_fold_results), 10)
                for metric in _DELTA_METRICS
            }
            aggregate["test_rows_counted_across_folds"] = sum(fold["eligible_test_rows"] for fold in paired_fold_results)
            aggregate["interpretation"] = (
                "Arithmetic mean of fold-level metrics; folds can overlap and are not independent. "
                "Deltas are variant minus baseline evaluated on the same eligible timestamps. "
                "Negative log-loss/Brier deltas and positive accuracy deltas favor the capped variant descriptively."
            )
            comparisons.append({
                "model_type": model_type,
                "feature_variant": feature_variant,
                "feature_names": list(base_exp["feature_names"]),
                "fold_count": len(paired_fold_results),
                "fold_results": paired_fold_results,
                "mean_across_folds": aggregate,
                "automatically_selected": False,
                "actionable": False,
            })

        baseline_audit = baseline.get("raw_ohlc_audit", {})
        warnings: list[str] = []
        if baseline_audit.get("status") == "PASS_WITH_WARNINGS":
            warnings.append("raw_ohlc_audit_passed_with_warnings")
        if overlapping_window_count:
            warnings.append(
                f"overlapping_test_windows:{overlapping_window_count}; fold metrics are not independent"
            )
        overall = {
            "schema_version": 1,
            "scope": "GOLDCREST_PAIRED_LABEL_ELIGIBILITY_SENSITIVITY",
            "status": "PASS_WITH_WARNINGS" if warnings else "PASS",
            "baseline_run_id": base_manifest["run_id"],
            "variant_run_id": variant_manifest["run_id"],
            "symbol": base_manifest["symbol"],
            "timeframe": base_manifest["timeframe"],
            "label_horizon_bars": base_manifest["label_horizon_bars"],
            "class_1_means": base_manifest.get("class_1_means"),
            "max_label_elapsed_hours": cap_hours,
            "raw_ohlc_sha256": base_manifest["raw_ohlc_sha256"],
            "baseline_dataset_sha256": _sha256_file(baseline["artifacts"]["dataset_path"]),
            "variant_dataset_sha256": _sha256_file(variant["artifacts"]["dataset_path"]),
            "raw_ohlc_snapshots_identical": True,
            "dataset_timestamps_identical": True,
            "shared_dataset_labels_features_and_ohlc_identical": True,
            "common_fold_windows_identical": True,
            "test_window_overlap_count": overlapping_window_count,
            "test_windows_overlap": overlapping_window_count > 0,
            "test_observations_across_folds": test_observations_across_folds,
            "unique_test_timestamps_across_fold_windows": unique_test_timestamps_across_windows,
            "repeated_test_timestamp_observations": (
                test_observations_across_folds - unique_test_timestamps_across_windows
            ),
            "experiment_count": len(comparisons),
            "experiments": comparisons,
            "eligibility_summary": {
                "baseline_dataset_rows": len(base_rows),
                "variant_dataset_rows": len(variant_rows),
                "variant_eligible_rows": sum(_eligible(row, cap_hours) for row in variant_rows),
                "variant_ineligible_rows": sum(not _eligible(row, cap_hours) for row in variant_rows),
                "variant_eligible_fraction": statistics.fmean(_eligible(row, cap_hours) for row in variant_rows),
                "model_rows_excluded_across_experiments_and_folds": sum(
                    fold["test_rows_excluded_ineligible"]
                    for exp in comparisons for fold in exp["fold_results"]
                ),
                "model_rows_scored_across_experiments_and_folds": sum(
                    fold["eligible_test_rows"] for exp in comparisons for fold in exp["fold_results"]
                ),
            },
            "test_metrics_used_for_selection": False,
            "automatic_model_selection": False,
            "actionable": False,
            "execution_connected": False,
            "interpretation": (
                "Paired offline sensitivity report. Baseline and capped model metrics are compared on "
                "identical eligible held-out timestamps; baseline-full metrics are retained only as context. "
                "Metric deltas are descriptive, not causal or inferential evidence, and do not establish "
                "profitability. No model is ranked/selected or connected to execution."
            ),
            "warnings": warnings,
            "errors": [],
        }
        return overall
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_PAIRED_LABEL_ELIGIBILITY_SENSITIVITY",
            "status": "FAIL",
            "errors": [f"{type(exc).__name__}:{exc}"],
            "warnings": [],
            "test_metrics_used_for_selection": False,
            "automatic_model_selection": False,
            "actionable": False,
            "execution_connected": False,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare an all-label baseline with an elapsed-time-capped offline research run."
    )
    parser.add_argument("--baseline-manifest", required=True, help="Immutable all-label baseline pipeline manifest")
    parser.add_argument("--variant-manifest", required=True, help="Pipeline manifest produced with --max-label-elapsed-hours")
    parser.add_argument("--output", required=True, help="New JSON path for the paired comparison report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = compare_label_eligibility_runs(args.baseline_manifest, args.variant_manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "baseline_run_id": report.get("baseline_run_id"),
        "variant_run_id": report.get("variant_run_id"),
        "experiment_count": report.get("experiment_count", 0),
        "output": str(output),
        "errors": report.get("errors", []),
        "warnings": report.get("warnings", []),
        "actionable": False,
    }, indent=2))
    return 0 if report["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

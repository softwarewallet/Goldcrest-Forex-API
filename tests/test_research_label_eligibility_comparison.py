from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.research_label_eligibility_comparison import (
    compare_label_eligibility_runs,
    _metric_summary,
)


def _canonical_hash(payload: dict, field: str, *, ensure_ascii: bool = True) -> str:
    value = dict(payload)
    value.pop(field, None)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=ensure_ascii, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_predictions(capped: bool) -> list[dict]:
    predictions = [
        {"timestamp_utc": "2026-01-01T00:00:00Z", "label": 0,
         "model_probability_class_1": 0.2},
        {"timestamp_utc": "2026-01-01T01:00:00Z", "label": 1,
         "model_probability_class_1": 0.8},
        {"timestamp_utc": "2026-01-02T01:00:00Z", "label": 1,
         "model_probability_class_1": 0.6},
    ]
    if capped:
        return [predictions[0], {
            "timestamp_utc": predictions[2]["timestamp_utc"],
            "label": predictions[2]["label"],
            "model_probability_class_1": 0.55,
        }]
    return predictions


def _build_run(root: Path, run_id: str, *, capped: bool, raw_salt: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "raw.csv"
    raw.write_text(
        "timestamp,open,high,low,close\n"
        "2025-12-31T23:00:00Z,1.0,1.1,0.9,1.0\n"
        "2026-01-01T00:00:00Z,1.0,1.1,0.9,1.01\n"
        "2026-01-01T01:00:00Z,1.01,1.11,0.91,1.02\n"
        "2026-01-02T01:00:00Z,1.02,1.12,0.92,1.03\n"
        "2026-01-02T02:00:00Z,1.03,1.13,0.93,1.04\n"
        "2026-01-02T03:00:00Z,1.04,1.14,0.94,1.05\n"
        + (f"# {raw_salt}\n" if raw_salt else ""),
        encoding="utf-8",
    )
    raw_hash = _digest(raw)

    data = root / f"{run_id}_dataset.csv"
    base_rows = [
        {"symbol": "EURUSD", "timeframe": "M15", "timestamp": "2026-01-01T00:00:00Z",
         "open": "1.0", "high": "1.1", "low": "0.9", "close": "1.01", "label": "0",
         "label_available_at": "2026-01-01T01:00:00Z", "feature_x": "0.1",
         "label_elapsed_hours": "1.0", "label_eligible": "1"},
        {"symbol": "EURUSD", "timeframe": "M15", "timestamp": "2026-01-01T01:00:00Z",
         "open": "1.01", "high": "1.11", "low": "0.91", "close": "1.02", "label": "1",
         "label_available_at": "2026-01-02T01:00:00Z", "feature_x": "0.2",
         "label_elapsed_hours": "24.0", "label_eligible": "0"},
        {"symbol": "EURUSD", "timeframe": "M15", "timestamp": "2026-01-02T01:00:00Z",
         "open": "1.02", "high": "1.12", "low": "0.92", "close": "1.03", "label": "1",
         "label_available_at": "2026-01-02T02:00:00Z", "feature_x": "0.3",
         "label_elapsed_hours": "1.0", "label_eligible": "1"},
        {"symbol": "EURUSD", "timeframe": "M15", "timestamp": "2026-01-02T02:00:00Z",
         "open": "1.03", "high": "1.13", "low": "0.93", "close": "1.04", "label": "1",
         "label_available_at": "2026-01-02T03:00:00Z", "feature_x": "0.4",
         "label_elapsed_hours": "1.0", "label_eligible": "1"},
    ]
    columns = list(base_rows[0])
    if not capped:
        columns = [name for name in columns if name not in {"label_elapsed_hours", "label_eligible"}]
    with data.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in base_rows:
            writer.writerow({key: value for key, value in row.items() if key in columns})
    data_hash = _digest(data)

    retrieved = "2026-01-03T00:00:00Z"
    dataset_manifest = {
        "schema_version": 1,
        "scope": "OFFLINE_RESEARCH_DATASET_PREPARATION",
        "actionable": False,
        "symbol": "EURUSD",
        "timeframe": "M15",
        "data_source": "cTrader Open API (DEMO historical snapshot)",
        "source_version": "fixture-export-v1",
        "retrieved_at_utc": retrieved,
        "terms_reference": "https://example.invalid/provider-terms",
        "input_snapshot_sha256": raw_hash,
        "output_dataset_sha256": data_hash,
        "input_rows": 6,
        "output_rows": 4,
        "feature_names": ["feature_x"],
        "feature_backend": "fixture",
        "label_definition": "Binary forward-close direction over 1 bars; class 1 means UP; equality maps to class 0.",
        "label_horizon_bars": 1,
        "automatic_sorting": False,
        "automatic_imputation": False,
        "automatic_model_selection": False,
    }
    if capped:
        dataset_manifest["label_eligibility"] = {
            "mode": "elapsed_time_cap",
            "max_label_elapsed_hours": 1.0,
            "eligible_rows": 3,
            "ineligible_rows": 1,
            "all_rows_preserved_for_fixed_temporal_folds": True,
            "labels_rewritten": False,
            "features_changed": False,
            "automatic_model_selection": False,
        }
    dataset_manifest["manifest_sha256"] = _canonical_hash(
        dataset_manifest, "manifest_sha256", ensure_ascii=False
    )
    dataset_manifest_path = root / f"{run_id}_dataset_manifest.json"
    dataset_manifest_file_hash = _write_json(dataset_manifest_path, dataset_manifest)

    predictions = _fixture_predictions(capped)
    test_metrics = _metric_summary(predictions, "model_probability_class_1")
    fold = {
        "fold_index": 0,
        "train_end_utc": "2025-12-31T20:00:00Z",
        "validation_start_utc": "2025-12-31T21:00:00Z",
        "validation_end_utc": "2025-12-31T22:00:00Z",
        "test_start_utc": "2026-01-01T00:00:00Z",
        "test_end_utc": "2026-01-02T01:00:00Z",
        "best_params": {"fixture": True},
        "trials_requested": 1,
        "trials_completed": 1,
        "test_metrics": test_metrics,
        "test_predictions": predictions,
        "test_rows_available_before_eligibility": 3,
        "test_rows_excluded_ineligible": 1 if capped else 0,
        "actionable": False,
    }
    matrix = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset_sha256": data_hash,
        "symbol": "EURUSD",
        "timeframe": "M15",
        "feature_names": ["feature_x"],
        "model_types": ["xgboost"],
        "fold_count": 1,
        "horizon_bars": 1,
        "class_1_means": "UP",
        "label_eligibility_mode": "elapsed_time_cap" if capped else "all_labels_included",
        "max_label_elapsed_hours": 1.0 if capped else None,
        "experiments": [{
            "experiment_id": f"{run_id}-xgboost-all_features",
            "model_type": "xgboost",
            "feature_variant": "all_features",
            "feature_names": ["feature_x"],
            "feature_count": 1,
            "feature_schema_sha256": "a" * 64,
            "fold_count": 1,
            "fold_results": [fold],
            "test_metrics_used_for_tuning": False,
            "used_common_fold_plan": True,
            "actionable": False,
        }],
        "common_fold_test_windows": [{
            "fold_index": 0,
            "test_start_utc": "2026-01-01T00:00:00Z",
            "test_end_utc": "2026-01-02T01:00:00Z",
            "test_rows": 3,
        }],
        "test_metrics_used_for_selection": False,
        "automatic_winner_selection": False,
        "actionable": False,
        "report_warning": "fixture only",
    }
    matrix["report_payload_sha256"] = _canonical_hash(matrix, "report_payload_sha256")
    matrix_path = root / f"{run_id}_experiment_matrix.json"
    matrix_file_hash = _write_json(matrix_path, matrix)

    provenance_path = root / f"{run_id}_provenance_audit.json"
    provenance = {
        "schema_version": 1, "run_id": run_id, "status": "PASS",
        "source_report_sha256": matrix_file_hash,
        "actionable": False, "execution_connected": False,
    }
    provenance_hash = _write_json(provenance_path, provenance)

    raw_audit_path = root / f"{run_id}_raw_audit.json"
    raw_audit = {
        "schema_version": 1, "snapshot_sha256": raw_hash, "status": "PASS",
        "errors": [], "warnings": [], "actionable": False, "execution_connected": False,
    }
    raw_audit_hash = _write_json(raw_audit_path, raw_audit)

    pipeline_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "scope": "GOLDCREST_OFFLINE_RESEARCH_PIPELINE",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "source": "cTrader Open API (DEMO historical snapshot)",
        "source_version": "fixture-export-v1",
        "retrieved_at_utc": retrieved,
        "raw_ohlc_path": str(raw.resolve()),
        "raw_ohlc_sha256": raw_hash,
        "raw_ohlc_audit_path": str(raw_audit_path.resolve()),
        "raw_ohlc_audit_sha256": raw_audit_hash,
        "raw_ohlc_audit_status": "PASS",
        "raw_ohlc_audit_warnings": [],
        "dataset_path": str(data.resolve()),
        "dataset_sha256": data_hash,
        "dataset_manifest_path": str(dataset_manifest_path.resolve()),
        "dataset_manifest_sha256": dataset_manifest_file_hash,
        "feature_names": ["feature_x"],
        "label_horizon_bars": 1,
        "class_1_means": "UP",
        "max_label_elapsed_hours": 1.0 if capped else None,
        "label_eligibility_mode": "elapsed_time_cap" if capped else "all_labels_included",
        "label_eligibility_metadata": dataset_manifest.get("label_eligibility"),
        "experiment_matrix_report": str(matrix_path.resolve()),
        "experiment_matrix_report_sha256": matrix_file_hash,
        "provenance_audit_path": str(provenance_path.resolve()),
        "provenance_audit_sha256": provenance_hash,
        "provenance_audit_status": "PASS",
        "actionable": False,
        "execution_connected": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    }
    pipeline_manifest["pipeline_manifest_sha256"] = _canonical_hash(
        pipeline_manifest, "pipeline_manifest_sha256"
    )
    manifest_path = root / f"{run_id}_pipeline_manifest.json"
    _write_json(manifest_path, pipeline_manifest)
    return manifest_path


def test_compares_models_on_exact_same_eligible_held_out_timestamps(tmp_path):
    baseline = _build_run(tmp_path / "baseline", "baseline-run", capped=False)
    variant = _build_run(tmp_path / "variant", "cap-run", capped=True)

    report = compare_label_eligibility_runs(baseline, variant)

    assert report["status"] == "PASS"
    assert report["raw_ohlc_snapshots_identical"] is True
    assert report["dataset_timestamps_identical"] is True
    assert report["common_fold_windows_identical"] is True
    assert report["eligibility_summary"]["baseline_dataset_rows"] == 4
    assert report["eligibility_summary"]["variant_eligible_rows"] == 3
    assert report["eligibility_summary"]["variant_ineligible_rows"] == 1
    experiment = report["experiments"][0]
    fold = experiment["fold_results"][0]
    assert fold["test_rows_available_before_eligibility"] == 3
    assert fold["test_rows_excluded_ineligible"] == 1
    assert fold["eligible_test_rows"] == 2
    assert fold["baseline_same_eligible_timestamps_metrics"]["sample_count"] == 2
    assert fold["variant_eligible_test_metrics"]["sample_count"] == 2
    assert fold["paired_variant_minus_baseline_delta"]["prevalence"] == 0.0
    assert experiment["automatically_selected"] is False
    assert report["actionable"] is False
    assert report["execution_connected"] is False


def test_comparison_fails_closed_when_raw_snapshots_differ(tmp_path):
    baseline = _build_run(tmp_path / "baseline", "baseline-run", capped=False)
    variant = _build_run(tmp_path / "variant", "cap-run", capped=True, raw_salt="different snapshot")

    report = compare_label_eligibility_runs(baseline, variant)

    assert report["status"] == "FAIL"
    assert any("raw_ohlc_sha256" in error for error in report["errors"])
    assert report["actionable"] is False


def test_comparison_fails_when_shared_dataset_feature_differs(tmp_path):
    baseline = _build_run(tmp_path / "baseline", "baseline-run", capped=False)
    variant = _build_run(tmp_path / "variant", "cap-run", capped=True)

    # Tamper with a shared feature, then rebuild the artifact hashes so the comparator
    # reaches the semantic equality check instead of stopping at simple hash mismatch.
    variant_manifest_path = Path(variant)
    pipeline = json.loads(variant_manifest_path.read_text(encoding="utf-8"))
    dataset_path = Path(pipeline["dataset_path"])
    text = dataset_path.read_text(encoding="utf-8").replace(",0.2,24.0,0", ",0.25,24.0,0")
    dataset_path.write_text(text, encoding="utf-8")
    new_dataset_hash = _digest(dataset_path)
    pipeline["dataset_sha256"] = new_dataset_hash
    dm_path = Path(pipeline["dataset_manifest_path"])
    dm = json.loads(dm_path.read_text(encoding="utf-8"))
    dm["output_dataset_sha256"] = new_dataset_hash
    dm["manifest_sha256"] = _canonical_hash(dm, "manifest_sha256", ensure_ascii=False)
    pipeline["dataset_manifest_sha256"] = _write_json(dm_path, dm)
    matrix_path = Path(pipeline["experiment_matrix_report"])
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["dataset_sha256"] = new_dataset_hash
    matrix["report_payload_sha256"] = _canonical_hash(matrix, "report_payload_sha256")
    pipeline["experiment_matrix_report_sha256"] = _write_json(matrix_path, matrix)
    provenance_path = Path(pipeline["provenance_audit_path"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_report_sha256"] = _digest(matrix_path)
    pipeline["provenance_audit_sha256"] = _write_json(provenance_path, provenance)
    pipeline["pipeline_manifest_sha256"] = _canonical_hash(pipeline, "pipeline_manifest_sha256")
    variant_manifest_path.write_text(
        json.dumps(pipeline, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report = compare_label_eligibility_runs(baseline, variant)

    assert report["status"] == "FAIL"
    assert any("shared column" in error for error in report["errors"])

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from app.research_results_summary import summarize_research_run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_run(tmp_path: Path) -> Path:
    run_id = "summary-fixture-01"
    matrix_path = tmp_path / f"{run_id}_experiment_matrix.json"
    validation_path = tmp_path / f"{run_id}_validation_report.json"

    test_windows = [
        {
            "fold_index": 0,
            "test_start_utc": "2026-01-01T00:45:00Z",
            "test_end_utc": "2026-01-01T01:00:00Z",
            "test_rows": 2,
        },
        {
            "fold_index": 1,
            "test_start_utc": "2026-01-01T02:00:00Z",
            "test_end_utc": "2026-01-01T02:15:00Z",
            "test_rows": 2,
        },
    ]
    metrics = [
        {
            "sample_count": 2,
            "positive_count": 1,
            "negative_count": 1,
            "prevalence": 0.5,
            "brier_score": 0.20,
            "log_loss": 0.60,
            "accuracy_at_fixed_half": 0.5,
            "mean_predicted_probability": 0.51,
            "expected_calibration_error": 0.08,
        },
        {
            "sample_count": 2,
            "positive_count": 1,
            "negative_count": 1,
            "prevalence": 0.5,
            "brier_score": 0.22,
            "log_loss": 0.62,
            "accuracy_at_fixed_half": 0.5,
            "mean_predicted_probability": 0.49,
            "expected_calibration_error": 0.09,
        },
    ]
    fold_results = []
    boundaries = [
        ("2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z", "2026-01-01T00:30:00Z"),
        ("2026-01-01T01:15:00Z", "2026-01-01T01:30:00Z", "2026-01-01T01:45:00Z"),
    ]
    for index, window in enumerate(test_windows):
        train_end, validation_start, validation_end = boundaries[index]
        fold_results.append({
            "fold_index": index,
            "train_end_utc": train_end,
            "validation_start_utc": validation_start,
            "validation_end_utc": validation_end,
            "test_start_utc": window["test_start_utc"],
            "test_end_utc": window["test_end_utc"],
            "best_params": {"max_depth": 3},
            "trials_requested": 2,
            "trials_completed": 2,
            "test_metrics_used_for_tuning": False,
            "test_metrics": metrics[index],
        })

    matrix = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset_sha256": "a" * 64,
        "symbol": "EURUSD",
        "timeframe": "M15",
        "model_types": ["xgboost"],
        "fold_count": 2,
        "common_fold_test_windows": test_windows,
        "test_metrics_used_for_selection": False,
        "automatic_winner_selection": False,
        "actionable": False,
        "experiments": [{
            "experiment_id": f"{run_id}-xgboost-all_features",
            "model_type": "xgboost",
            "feature_variant": "all_features",
            "feature_names": ["return_1"],
            "feature_schema_sha256": "b" * 64,
            "fold_count": 2,
            "test_metrics_used_for_tuning": False,
            "actionable": False,
            "fold_results": fold_results,
        }],
    }
    _write_json(matrix_path, matrix)
    _write_json(validation_path, {
        "run_id": run_id,
        "status": "PASS_WITH_WARNINGS",
        "errors": [],
        "warnings": ["integrity:raw_ohlc_audit_passed_with_warnings"],
        "actionable": False,
        "execution_connected": False,
    })

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "symbol": "EURUSD",
        "timeframe": "M15",
        "fold_count": 2,
        "experiment_count": 1,
        "experiment_matrix_report": str(matrix_path.resolve()),
        "experiment_matrix_report_sha256": _sha(matrix_path),
        "validation_report_path": str(validation_path.resolve()),
        "raw_ohlc_audit_status": "PASS_WITH_WARNINGS",
        "raw_ohlc_audit_warnings": [
            "large_time_gaps_detected_review_market_closures_and_data_coverage"
        ],
        "provenance_audit_status": "PASS",
        "provenance_audit_errors": [],
        "provenance_audit_warnings": [],
        "actionable": False,
        "execution_connected": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest_path = tmp_path / f"{run_id}_pipeline_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_compact_summary_reports_held_out_metrics_and_constant_half_reference(tmp_path):
    manifest = _build_run(tmp_path)

    result = summarize_research_run(manifest)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["validation_status"] == "PASS_WITH_WARNINGS"
    assert result["errors"] == []
    assert result["experiment_matrix_hash_verified"] is True
    assert result["models"][0]["ranked_or_selected"] is False
    assert result["models"][0]["metrics"]["log_loss"]["mean_across_folds"] == 0.61
    assert result["models"][0]["metrics"]["brier_score"]["mean_across_folds"] == 0.21
    assert result["models"][0]["metrics"]["log_loss"]["folds_below_constant_half_baseline"] == 2
    assert result["models"][0]["log_loss_improvement_vs_constant_0_5"] == round(math.log(2) - 0.61, 8)
    assert result["actionable"] is False
    assert result["execution_connected"] is False
    assert result["automatic_model_selection"] is False


def test_compact_summary_fails_when_experiment_matrix_hash_changes(tmp_path):
    manifest = _build_run(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    matrix_path = Path(payload["experiment_matrix_report"])
    matrix_path.write_text(matrix_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = summarize_research_run(manifest)

    assert result["status"] == "FAIL"
    assert "experiment_matrix_report_sha256_mismatch" in result["errors"]


def test_compact_summary_fails_when_pipeline_manifest_fingerprint_changes(tmp_path):
    manifest = _build_run(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["actionable"] = True
    _write_json(manifest, payload)

    result = summarize_research_run(manifest)

    assert result["status"] == "FAIL"
    assert "pipeline_manifest_sha256_mismatch" in result["errors"]
    assert "pipeline_safety_flags_invalid" in result["errors"]

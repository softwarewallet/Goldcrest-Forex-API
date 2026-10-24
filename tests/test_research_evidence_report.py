from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app import research_evidence_report as evidence


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_run(tmp_path: Path, monkeypatch, *, with_optional: bool = True) -> Path:
    run_id = "evidence-run-001"
    manifest_path = tmp_path / f"{run_id}_pipeline_manifest.json"
    matrix_path = tmp_path / f"{run_id}_experiment_matrix.json"
    matrix_path.write_text('{"run_id":"evidence-run-001","matrix":true}\n', encoding="utf-8")
    matrix_hash = _sha(matrix_path)

    provenance_path = tmp_path / f"{run_id}_provenance_audit.json"
    _write_json(provenance_path, {
        "schema_version": 1,
        "run_id": run_id,
        "status": "PASS",
        "experiment_count": 2,
        "errors": [],
        "warnings": [],
        "actionable": False,
        "execution_connected": False,
    })
    temporal_path = tmp_path / f"{run_id}_temporal_calibration_stability.json"
    _write_json(temporal_path, {
        "schema_version": 1,
        "scope": "GOLDCREST_TEMPORAL_CALIBRATION_STABILITY_AUDIT",
        "run_id": run_id,
        "experiment_matrix_sha256": matrix_hash,
        "experiment_matrix_payload_hash_verified": True,
        "status": "PASS",
        "comparison_count": 1,
        "comparisons": [{
            "model_name": "example-model", "partition": "test", "period_count": 4,
            "first_period_ece": 0.02, "last_period_ece": 0.03,
            "first_to_last_ece_delta": 0.01, "ece_min": 0.02, "ece_max": 0.03,
            "status": "NO_CLEAR_ENDPOINT_SEPARATION",
        }],
        "errors": [],
        "warnings": [],
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "test_metrics_used_for_selection": False,
    })

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "scope": "GOLDCREST_OFFLINE_RESEARCH_PIPELINE",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "label_horizon_bars": 4,
        "raw_ohlc_path": str(tmp_path / "raw.csv"),
        "raw_ohlc_sha256": "1" * 64,
        "dataset_path": str(tmp_path / "dataset.csv"),
        "dataset_sha256": "2" * 64,
        "dataset_manifest_path": str(tmp_path / "dataset.manifest.json"),
        "dataset_manifest_sha256": "3" * 64,
        "experiment_matrix_report": str(matrix_path),
        "experiment_matrix_report_sha256": matrix_hash,
        "provenance_audit_path": str(provenance_path),
        "provenance_audit_sha256": _sha(provenance_path),
        "provenance_audit_status": "PASS",
        "temporal_calibration_audit_path": str(temporal_path),
        "temporal_calibration_audit_sha256": _sha(temporal_path),
        "temporal_calibration_audit_status": "PASS",
        "actionable": False,
        "execution_connected": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    }
    _write_json(manifest_path, manifest)

    validation_path = tmp_path / f"{run_id}_validation_report.json"
    _write_json(validation_path, {
        "schema_version": 1,
        "scope": "GOLDCREST_COMPLETED_RUN_VALIDATION",
        "run_id": run_id,
        "manifest_path": str(manifest_path.resolve()),
        "status": "PASS",
        "dataset_quality": {"status": "PASS", "row_count": 100, "feature_count": 4},
        "experiment_quality": {"status": "PASS", "experiment_count": 2, "fold_count": 7},
        "runtime_preflight": {"status": "PASS"},
        "errors": [],
        "warnings": [],
        "actionable": False,
        "execution_connected": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    })

    if with_optional:
        baseline_path = tmp_path / f"{run_id}_baseline_analysis.json"
        _write_json(baseline_path, {
            "schema_version": 1,
            "scope": "GOLDCREST_TIME_SERIES_BASELINE_ANALYSIS",
            "status": "PASS",
            "run_id": run_id,
            "symbol": "EURUSD",
            "timeframe": "M15",
            "manifest_path": str(manifest_path.resolve()),
            "experiment_matrix_sha256": matrix_hash,
            "fold_count": 7,
            "experiments": [{
                "experiment_id": "example-xgb",
                "model_type": "xgboost",
                "feature_variant": "all_features",
                "fold_count": 7,
                "test_prediction_count_counted_by_fold": 700,
                "model_metrics": {"log_loss": 0.55, "brier_score": 0.19},
                "probability_loss_comparisons": [{
                    "baseline": "constant_probability_0_5",
                    "metrics": {
                        "log_loss": {
                            "improvement_baseline_minus_model": 0.1,
                            "confidence_interval_95": [0.01, 0.2],
                            "interval_evidence": "positive_interval_excludes_zero",
                            "folds_compared": 7,
                            "folds_with_positive_mean_improvement": 6,
                            "folds_with_negative_mean_improvement": 1,
                        }
                    },
                }],
            }],
            "errors": [],
            "warnings": [],
            "actionable": False,
            "execution_connected": False,
            "automatic_winner_selection": False,
            "test_metrics_used_for_selection": False,
        })
        regime_path = tmp_path / f"{run_id}_regime_analysis.json"
        _write_json(regime_path, {
            "schema_version": 1,
            "scope": "GOLDCREST_VOLATILITY_REGIME_ANALYSIS",
            "status": "PASS",
            "run_id": run_id,
            "symbol": "EURUSD",
            "timeframe": "M15",
            "manifest_path": str(manifest_path.resolve()),
            "experiment_matrix_sha256": matrix_hash,
            "volatility_window_bars": 20,
            "fold_thresholds": [{"fold_index": 0}],
            "held_out_predictions_without_volatility": 0,
            "experiments": [{
                "experiment_id": "example-xgb", "model_type": "xgboost",
                "feature_variant": "all_features",
                "regime_metrics": {
                    "low_volatility": {
                        "sample_count": 100, "fold_count": 7,
                        "model": {"log_loss": 0.5, "brier_score": 0.18},
                        "loss_improvements_baseline_minus_model": {},
                        "paired_bootstrap_uncertainty": {},
                    }
                },
            }],
            "errors": [], "warnings": [],
            "actionable": False, "execution_connected": False,
            "automatic_model_selection": False, "test_metrics_used_for_selection": False,
        })
        gap_path = tmp_path / f"{run_id}_temporal_stability.json"
        _write_json(gap_path, {
            "schema_version": 1,
            "scope": "GOLDCREST_TEMPORAL_STABILITY_GAP_SENSITIVITY",
            "status": "PASS",
            "run_id": run_id,
            "symbol": "EURUSD",
            "timeframe": "M15",
            "manifest_path": str(manifest_path.resolve()),
            "experiment_matrix_sha256": matrix_hash,
            "long_gap_count": 3,
            "weekend_closure_like_gap_count": 2,
            "non_weekend_long_gap_count": 1,
            "holiday_context_gap_count": 1,
            "prediction_count": 1400,
            "unique_prediction_timestamp_count": 700,
            "prediction_horizon_crosses_long_gap_count": 12,
            "prediction_horizon_crosses_long_gap_fraction": 12 / 1400,
            "experiments": [{
                "experiment_id": "example-xgb", "model_type": "xgboost",
                "fold_count": 7, "sample_count": 700,
                "fold_consistency": {"log_loss": {"fold_count": 7}},
                "gap_exposure": {
                    "horizon_crosses_long_gap_count": 6,
                    "horizon_does_not_cross_long_gap_count": 694,
                    "horizon_crosses_weekend_gap_count": 5,
                    "horizon_crosses_non_weekend_gap_count": 1,
                    "horizon_crosses_holiday_context_gap_count": 1,
                },
            }],
            "errors": [], "warnings": [],
            "actionable": False, "execution_connected": False,
            "automatic_model_selection": False, "test_metrics_used_for_selection": False,
        })

    monkeypatch.setattr(evidence, "audit_research_run_manifest", lambda path: {
        "status": "PASS", "checked_artifact_count": 7, "errors": [], "warnings": [],
    })
    return manifest_path


def test_consolidated_evidence_report_binds_and_summarizes_sources(tmp_path, monkeypatch):
    manifest = _build_run(tmp_path, monkeypatch)

    result = evidence.build_research_evidence_report(manifest)

    assert result["status"] == "PASS"
    assert result["run_id"] == "evidence-run-001"
    assert result["source_matrix_sha256"]
    assert result["actionable"] is False
    assert result["execution_connected"] is False
    assert result["automatic_model_selection"] is False
    assert result["evidence_coverage"]["available_optional_section_count"] == 3
    assert result["summary"]["validation"]["dataset_row_count"] == 100
    assert result["summary"]["temporal_calibration"]["comparison_count"] == 1
    assert result["summary"]["baseline_analysis"]["experiment_count"] == 1
    assert result["summary"]["baseline_analysis"]["experiments"][0]["ranked_or_selected"] is False
    assert result["summary"]["regime_analysis"]["fold_threshold_count"] == 1
    assert result["summary"]["temporal_stability"]["long_gap_count"] == 3


def test_consolidated_report_flags_source_matrix_mismatch_and_unsafe_flag(tmp_path, monkeypatch):
    manifest_path = _build_run(tmp_path, monkeypatch)
    baseline_path = tmp_path / "evidence-run-001_baseline_analysis.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["experiment_matrix_sha256"] = "0" * 64
    baseline["actionable"] = True
    _write_json(baseline_path, baseline)

    result = evidence.build_research_evidence_report(manifest_path)

    assert result["status"] == "FAIL"
    assert "baseline_analysis_source_matrix_hash_mismatch" in result["errors"]
    assert "baseline_analysis_actionable_must_be_false" in result["errors"]


def test_missing_optional_analysis_is_warning_not_failure(tmp_path, monkeypatch):
    manifest_path = _build_run(tmp_path, monkeypatch, with_optional=False)

    result = evidence.build_research_evidence_report(manifest_path)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["evidence_coverage"]["missing_optional_sections"] == [
        "baseline_analysis", "regime_analysis", "temporal_stability"
    ]
    assert "baseline_analysis_not_provided" in result["warnings"]


def test_explicitly_requested_missing_analysis_fails_closed(tmp_path, monkeypatch):
    manifest_path = _build_run(tmp_path, monkeypatch, with_optional=False)

    result = evidence.build_research_evidence_report(
        manifest_path, baseline_analysis_path=tmp_path / "missing.json"
    )

    assert result["status"] == "FAIL"
    assert "baseline_analysis_artifact_missing" in result["errors"]


def test_writer_refuses_to_overwrite_existing_evidence_report(tmp_path, monkeypatch):
    manifest = _build_run(tmp_path, monkeypatch)
    output = tmp_path / "evidence.json"
    output.write_text("preserve", encoding="utf-8")

    try:
        evidence.write_research_evidence_report(manifest, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("writer must refuse an existing report")
    assert output.read_text(encoding="utf-8") == "preserve"


def test_legacy_run_accepts_explicit_temporal_calibration_receipt(tmp_path, monkeypatch):
    manifest_path = _build_run(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    temporal_path = Path(manifest["temporal_calibration_audit_path"])
    manifest.pop("temporal_calibration_audit_path")
    manifest.pop("temporal_calibration_audit_sha256")
    manifest.pop("temporal_calibration_audit_status")
    _write_json(manifest_path, manifest)

    result = evidence.build_research_evidence_report(
        manifest_path, temporal_calibration_path=temporal_path
    )

    assert result["status"] == "PASS"
    assert result["sections"]["temporal_calibration"]["status"] == "PASS"
    assert result["summary"]["temporal_calibration"]["matrix_payload_hash_verified"] is True

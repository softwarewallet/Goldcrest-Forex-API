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
    fold_results = []
    for fold_index in range(7):
        fold_results.append({
            "fold_index": fold_index,
            "train_end_utc": f"2026-01-{fold_index + 1:02d}T00:00:00Z",
            "validation_start_utc": f"2026-02-{fold_index + 1:02d}T00:00:00Z",
            "validation_end_utc": f"2026-03-{fold_index + 1:02d}T00:00:00Z",
            "test_start_utc": f"2026-04-{fold_index + 1:02d}T00:00:00Z",
            "test_end_utc": f"2026-04-{fold_index + 1:02d}T12:00:00Z",
            "test": {
                "sample_count": 10,
                "brier_score": 0.20 + 0.01 * fold_index,
                "log_loss": 0.50 + 0.02 * fold_index,
                "accuracy_at_fixed_half": 0.50 + 0.01 * fold_index,
                "expected_calibration_error": 0.10 + 0.005 * fold_index,
            },
            "test_metrics": {
                "sample_count": 10,
                "positive_count": 5,
                "negative_count": 5,
                "brier_score": 0.20 + 0.01 * fold_index,
                "log_loss": 0.50 + 0.02 * fold_index,
                "accuracy_at_fixed_half": 0.50 + 0.01 * fold_index,
                "expected_calibration_error": 0.10 + 0.005 * fold_index,
            },
            "actionable": False,
        })
    matrix = {
        "schema_version": 1,
        "run_id": "evidence-run-001",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "fold_count": 7,
        "common_fold_test_windows": [],
        "experiments": [
            {
                "experiment_id": "example-xgb",
                "model_type": "xgboost",
                "fold_results": fold_results,
                "actionable": False,
                "test_metrics_used_for_tuning": False,
            },
            {
                "experiment_id": "example-lgbm",
                "model_type": "lightgbm",
                "fold_results": [
                    {
                        **fold,
                        "test": {
                            **fold["test"],
                            "brier_score": 0.19 + 0.008 * fold["fold_index"],
                            "log_loss": 0.48 + 0.015 * fold["fold_index"],
                            "expected_calibration_error": 0.08 + 0.004 * fold["fold_index"],
                        },
                    }
                    for fold in fold_results
                ],
                "actionable": False,
                "test_metrics_used_for_tuning": False,
            },
        ],
        "actionable": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    }
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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

    gate_path = tmp_path / f"{run_id}_model_evaluation_gate.json"
    _write_json(gate_path, {
        "schema_version": 1,
        "scope": "GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE",
        "status": "PASS",
        "run_id": run_id,
        "symbol": "EURUSD",
        "timeframe": "M15",
        "source_matrix_sha256": matrix_hash,
        "experiment_count": 2,
        "folds_checked": 14,
        "fold_metric_objects_checked": 14,
        "errors": [],
        "warnings": [],
        "actionable": False,
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
        "model_evaluation_gate_path": str(gate_path),
        "model_evaluation_gate_sha256": _sha(gate_path),
        "model_evaluation_gate_status": "PASS",
        "model_evaluation_gate_errors": [],
        "model_evaluation_gate_warnings": [],
        "strict_model_evaluation_gate": False,
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

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert any(item.endswith("_not_reported") for item in result["warnings"])
    assert result["run_id"] == "evidence-run-001"
    assert result["source_matrix_sha256"]
    assert result["actionable"] is False
    assert result["execution_connected"] is False
    assert result["automatic_model_selection"] is False
    assert result["evidence_coverage"]["available_optional_section_count"] == 3
    assert result["summary"]["validation"]["dataset_row_count"] == 100
    assert result["summary"]["temporal_calibration"]["comparison_count"] == 1
    assert result["sections"]["model_evaluation_gate"]["status"] == "PASS"
    assert result["summary"]["model_evaluation_gate"]["folds_checked"] == 14
    assert "model_evaluation_gate" in result["evidence_coverage"]["available_sections"]
    assert result["summary"]["baseline_analysis"]["experiment_count"] == 1
    assert result["summary"]["baseline_analysis"]["experiments"][0]["ranked_or_selected"] is False
    assert result["summary"]["regime_analysis"]["fold_threshold_count"] == 1
    assert result["summary"]["temporal_stability"]["long_gap_count"] == 3
    oos = result["summary"]["oos_period_robustness"]
    assert oos["status"] == "PERIOD_DIAGNOSTIC_AVAILABLE"
    assert oos["common_test_fold_indices"] == list(range(7))
    assert oos["model_selection_performed"] is False
    assert len(oos["candidates"]) == 2
    early_period = oos["candidates"][0]["periods"][0]
    assert early_period["period"] == "EARLY"
    assert early_period["ece_fold_count"] == 3
    assert early_period["median_test_expected_calibration_error"] is not None
    interpretation = result["evidence_interpretation"]["chronological_oos_robustness"]
    assert interpretation["status"] == "PERIOD_DIAGNOSTIC_AVAILABLE"
    assert interpretation["test_metrics_used_for_selection"] is False


def test_oos_period_diagnostic_fails_closed_when_matrix_hash_mismatches(tmp_path, monkeypatch):
    manifest_path = _build_run(tmp_path, monkeypatch)
    matrix_path = tmp_path / "evidence-run-001_experiment_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["run_id"] = "other-run"
    _write_json(matrix_path, matrix)

    result = evidence.build_research_evidence_report(manifest_path)

    assert result["status"] == "FAIL"
    assert "oos_period_robustness_source_matrix_hash_mismatch" in result["errors"]
    assert result["summary"]["oos_period_robustness"]["status"] == "SOURCE_MATRIX_HASH_MISMATCH"
    assert result["evidence_interpretation"]["chronological_oos_robustness"]["test_metrics_used_for_selection"] is False


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

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["sections"]["temporal_calibration"]["status"] == "PASS"
    assert result["summary"]["temporal_calibration"]["matrix_payload_hash_verified"] is True



def test_evidence_interpretation_withholds_interval_classification_when_fold_timestamps_repeat():
    baseline = {
        "status": "PASS_WITH_WARNINGS",
        "warnings": ["held_out_timestamps_repeat_across_folds; forecast errors may be dependent across fold fits"],
        "experiments": [{
            "experiment_id": "model-a",
            "probability_loss_comparisons": [{
                "baseline": "constant_probability_0_5",
                "metrics": {
                    "log_loss": {
                        "improvement_baseline_minus_model": 0.01,
                        "confidence_interval_95": [0.002, 0.02],
                    },
                    "brier_score": {
                        "improvement_baseline_minus_model": 0.01,
                        "confidence_interval_95": [0.001, 0.02],
                    },
                },
            }],
        }],
    }

    result = evidence.build_evidence_interpretation(
        baseline=baseline, calibration=None, regimes=None, temporal_stability=None
    )

    comparison = result["baseline_comparison"]
    assert comparison["status"] == "DEPENDENCE_LIMITED_INTERVALS_NOT_CLASSIFIED"
    assert comparison["intervals_dependence_limited"] == 2
    assert comparison["comparisons_evaluated"] == 0
    assert comparison["positive_point_estimates"] == 2


def test_evidence_interpretation_detects_inconclusive_baselines_and_upward_ece():
    baseline = {
        "status": "PASS_WITH_WARNINGS",
        "experiments": [{
            "experiment_id": "model-a",
            "probability_loss_comparisons": [
                {"baseline": "constant_probability_0_5", "metrics": {
                    "log_loss": {
                        "improvement_baseline_minus_model": 0.001,
                        "confidence_interval_95": [-0.002, 0.003],
                    },
                    "brier_score": {
                        "improvement_baseline_minus_model": -0.001,
                        "confidence_interval_95": [-0.003, 0.001],
                    },
                }},
            ],
        }],
    }
    calibration = {
        "status": "PASS",
        "comparisons": [
            {"model_name": "model-a", "partition": "test", "first_to_last_ece_delta": 0.02},
            {"model_name": "model-b", "partition": "test", "first_to_last_ece_delta": 0.01},
        ],
    }

    result = evidence.build_evidence_interpretation(
        baseline=baseline, calibration=calibration, regimes=None, temporal_stability=None
    )

    assert result["baseline_comparison"]["status"] == "NO_CLEAR_INTERVAL_SEPARATION_FROM_TESTED_BASELINES"
    assert result["baseline_comparison"]["intervals_including_zero"] == 2
    assert result["baseline_comparison"]["positive_point_estimates"] == 1
    assert result["temporal_calibration"]["status"] == "UPWARD_ECE_DRIFT_OBSERVED_IN_ALL_COMPARISONS"
    assert result["temporal_calibration"]["upward_ece_changes"] == 2
    assert result["promotion_readiness"] == "NOT_ESTABLISHED_BY_THIS_REPORT"
    assert result["automatic_model_selection"] is False


def test_evidence_interpretation_identifies_mixed_regime_improvements():
    regimes = {
        "status": "PASS_WITH_WARNINGS",
        "experiments": [{
            "experiment_id": "model-a",
            "model_type": "xgboost",
            "regime_metrics": {
                "low_volatility": {"loss_improvements_baseline_minus_model": {
                    "constant_probability_0_5": {"log_loss": -0.01, "brier_score": -0.02},
                }},
                "mid_volatility": {"loss_improvements_baseline_minus_model": {
                    "constant_probability_0_5": {"log_loss": 0.01, "brier_score": 0.02},
                }},
                "high_volatility": {"loss_improvements_baseline_minus_model": {
                    "constant_probability_0_5": {"log_loss": -0.005, "brier_score": -0.01},
                }},
            },
        }],
    }

    result = evidence.build_evidence_interpretation(
        baseline=None, calibration=None, regimes=regimes, temporal_stability=None
    )

    assert result["volatility_regime_consistency"]["status"] == "MIXED_BY_VOLATILITY_REGIME_OBSERVED"
    assert result["volatility_regime_consistency"]["mixed_comparison_group_count"] == 2
    assert all(
        row["consistency"] == "MIXED_BY_VOLATILITY_REGIME"
        for row in result["volatility_regime_consistency"]["comparisons"]
    )


def test_evidence_interpretation_summarizes_regime_bootstrap_intervals():
    def regime(improvement_log, interval_log, improvement_brier, interval_brier):
        return {
            "loss_improvements_baseline_minus_model": {
                "constant_probability_0_5": {
                    "log_loss": improvement_log,
                    "brier_score": improvement_brier,
                },
            },
            "paired_bootstrap_uncertainty": {
                "constant_probability_0_5": {
                    "log_loss": {
                        "confidence_interval_95": interval_log,
                        "bootstrap_resamples": 500,
                    },
                    "brier_score": {
                        "confidence_interval_95": interval_brier,
                        "bootstrap_resamples": 500,
                    },
                },
            },
        }

    regimes = {
        "status": "PASS_WITH_WARNINGS",
        "experiments": [{
            "experiment_id": "model-a",
            "model_type": "xgboost",
            "regime_metrics": {
                "low_volatility": regime(0.01, [0.001, 0.03], 0.01, [-0.01, 0.01]),
                "mid_volatility": regime(-0.01, [-0.01, 0.02], -0.01, [-0.02, 0.01]),
                "high_volatility": regime(0.02, [-0.04, -0.001], 0.02, [-0.03, 0.04]),
            },
        }],
    }

    result = evidence.build_evidence_interpretation(
        baseline=None, calibration=None, regimes=regimes, temporal_stability=None
    )
    summary = result["volatility_regime_consistency"]
    uncertainty = summary["paired_bootstrap_uncertainty"]
    log_loss = next(
        row for row in summary["comparisons"] if row["metric"] == "log_loss"
    )["paired_bootstrap_uncertainty"]

    assert uncertainty["status"] == "MIXED_INTERVAL_EVIDENCE"
    assert uncertainty["intervals_evaluated"] == 6
    assert uncertainty["intervals_excluding_zero_positive"] == 1
    assert uncertainty["intervals_excluding_zero_negative"] == 1
    assert uncertainty["intervals_including_zero"] == 4
    assert uncertainty["intervals_unavailable"] == 0
    assert log_loss["status"] == "MIXED_INTERVAL_EVIDENCE"
    assert log_loss["intervals_evaluated"] == 3
    assert log_loss["intervals_excluding_zero_positive"] == 1
    assert log_loss["intervals_excluding_zero_negative"] == 1
    assert log_loss["intervals_including_zero"] == 1
    assert all(row["bootstrap_resamples"] == 500 for row in log_loss["regimes"])


def test_evidence_interpretation_marks_missing_regime_intervals_unavailable():
    regimes = {
        "status": "PASS",
        "experiments": [{
            "experiment_id": "model-a",
            "model_type": "xgboost",
            "regime_metrics": {
                "low_volatility": {
                    "loss_improvements_baseline_minus_model": {
                        "constant_probability_0_5": {"log_loss": 0.01}
                    }
                },
                "mid_volatility": {
                    "loss_improvements_baseline_minus_model": {
                        "constant_probability_0_5": {"log_loss": 0.02}
                    }
                },
            },
        }],
    }
    result = evidence.build_evidence_interpretation(
        baseline=None, calibration=None, regimes=regimes, temporal_stability=None
    )
    uncertainty = result["volatility_regime_consistency"]["paired_bootstrap_uncertainty"]
    assert uncertainty["status"] == "UNCERTAINTY_UNAVAILABLE"
    assert uncertainty["intervals_evaluated"] == 0
    assert uncertainty["intervals_unavailable"] == 2


def test_consolidated_report_withholds_oos_periods_for_invalid_chronology(tmp_path, monkeypatch):
    manifest = _build_run(tmp_path, monkeypatch)
    matrix_path = tmp_path / "evidence-run-001_experiment_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    for experiment in matrix["experiments"]:
        fold_rows = experiment["fold_results"]
        fold_rows[0]["test_start_utc"] = "2026-04-02T00:00:00Z"
        fold_rows[1]["test_start_utc"] = "2026-04-01T00:00:00Z"
    _write_json(matrix_path, matrix)

    # Make the manifest and the source matrix agree on the new immutable artifact hash.
    manifest_path = manifest
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["experiment_matrix_report_sha256"] = _sha(matrix_path)
    manifest_payload = dict(data)
    manifest_payload.pop("pipeline_manifest_sha256", None)
    data["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, data)

    result = evidence.build_research_evidence_report(manifest_path)
    oos = result["evidence_interpretation"]["chronological_oos_robustness"]

    assert oos["status"] == "CHRONOLOGY_INVALID"
    assert oos["chronology_integrity"]["status"] == "FOLD_ORDER_TIMESTAMP_MISMATCH"
    assert oos["candidates"] == []
    assert "oos_period_robustness_chronology_invalid_period_summaries_withheld" in result["warnings"]


def test_evidence_interpretation_exposes_unadjusted_multiplicity_context():
    baseline = {
        "status": "PASS",
        "experiments": [{
            "experiment_id": "model-a",
            "probability_loss_comparisons": [{
                "baseline": "constant_probability_0_5",
                "metrics": {
                    "log_loss": {
                        "improvement_baseline_minus_model": 0.01,
                        "confidence_interval_95": [0.001, 0.02],
                    },
                    "brier_score": {
                        "improvement_baseline_minus_model": 0.02,
                        "confidence_interval_95": [0.001, 0.03],
                    },
                },
            }],
        }],
    }
    calibration = {
        "status": "PASS",
        "comparisons": [
            {"model_name": "model-a", "partition": "test", "first_to_last_ece_delta": 0.01},
            {"model_name": "model-b", "partition": "test", "first_to_last_ece_delta": 0.02},
        ],
    }

    result = evidence.build_evidence_interpretation(
        baseline=baseline, calibration=calibration, regimes=None, temporal_stability=None
    )

    context = result["multiplicity_context"]
    assert context["status"] == "MULTIPLE_DIAGNOSTIC_COMPARISONS_UNADJUSTED"
    assert context["diagnostic_family_count"] == 2
    assert context["total_diagnostic_comparison_count"] == 4
    assert context["adjustment_applied"] is False
    assert context["confirmatory_inference_supported"] is False
    assert context["families"]["baseline_probability_loss_intervals"]["comparison_count"] == 2
    assert context["families"]["temporal_calibration_test_period_changes"]["comparison_count"] == 2


def test_consolidated_report_includes_cross_diagnostic_interpretation(tmp_path, monkeypatch):
    manifest = _build_run(tmp_path, monkeypatch)

    result = evidence.build_research_evidence_report(manifest)

    interpretation = result["evidence_interpretation"]
    assert interpretation["research_conclusion"] == "DESCRIPTIVE_ONLY_NO_MODEL_SELECTED"
    assert interpretation["automatic_model_selection"] is False
    assert interpretation["test_metrics_used_for_selection"] is False
    assert interpretation["actionable"] is False
    assert interpretation["execution_connected"] is False
    assert interpretation["baseline_comparison"]["status"] == "POSITIVE_INTERVAL_EVIDENCE_PRESENT"
    assert interpretation["temporal_calibration"]["status"] == "UPWARD_ECE_DRIFT_OBSERVED_IN_ALL_COMPARISONS"


def test_nonfinite_evidence_numbers_are_treated_as_unavailable():
    assert evidence._number(float("nan")) is None
    assert evidence._number(float("inf")) is None
    assert evidence._number(True) is None


def test_oos_period_diagnostic_uses_legacy_metrics_and_matrix_window_timestamps(tmp_path, monkeypatch):
    manifest_path = _build_run(tmp_path, monkeypatch)
    matrix_path = tmp_path / "evidence-run-001_experiment_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

    common_windows = []
    for fold_index in range(7):
        common_windows.append({
            "fold_index": fold_index,
            "test_start_utc": f"2026-04-{fold_index + 1:02d}T00:00:00Z",
            "test_end_utc": f"2026-04-{fold_index + 1:02d}T12:00:00Z",
            "test_rows": 10,
        })
    matrix["common_fold_test_windows"] = common_windows

    for experiment in matrix["experiments"]:
        for fold in experiment["fold_results"]:
            # Simulate an older matrix schema that records held-out metrics under
            # test_metrics and keeps canonical timestamps at matrix level.
            fold.pop("test", None)
            fold.pop("test_start_utc", None)
            fold.pop("test_start", None)
    _write_json(matrix_path, matrix)

    audit, errors, warnings = evidence._derive_oos_period_robustness(
        matrix_path,
        expected_run_id="evidence-run-001",
        expected_matrix_sha256=_sha(matrix_path),
    )

    assert errors == []
    assert audit["status"] == "PERIOD_DIAGNOSTIC_AVAILABLE"
    assert audit["candidate_count"] == 2
    assert audit["common_test_fold_indices"] == list(range(7))
    assert audit["chronology_integrity"]["status"] == "VERIFIED_TEST_START_ORDER"
    assert audit["candidate_fold_coverage"][0]["test_metric_source_counts"] == {
        "test": 0,
        "test_metrics": 7,
        "missing": 0,
    }


def test_evidence_interpretation_keeps_candidate_count_when_periods_are_unavailable():
    diagnostic = {
        "status": "NO_COMPLETE_METRIC_FOLDS",
        "candidate_count": 3,
        "common_fold_index_count": 5,
        "complete_test_fold_count": 0,
        "period_group_count": 0,
        "candidate_fold_coverage": [
            {
                "candidate_name": "candidate-a",
                "fold_count": 5,
                "folds_with_both_required_metrics": 0,
            },
        ],
        "common_test_fold_indices": [],
        "candidates": [],
    }

    result = evidence.build_evidence_interpretation(
        baseline=None,
        calibration=None,
        regimes=None,
        temporal_stability=None,
        oos_period_robustness=diagnostic,
    )["chronological_oos_robustness"]

    assert result["status"] == "NO_COMPLETE_METRIC_FOLDS"
    assert result["candidate_count"] == 3
    assert result["summary_candidate_count"] == 0
    assert result["common_fold_index_count"] == 5
    assert result["complete_test_fold_count"] == 0
    assert result["candidate_fold_coverage"][0]["candidate_name"] == "candidate-a"


def test_consolidated_report_detects_manifest_receipt_summary_mismatch(tmp_path, monkeypatch):
    manifest_path = _build_run(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Deliberately make the manifest's copied receipt summary stale while
    # leaving the hash-bound receipt unchanged.
    manifest["model_evaluation_gate_status"] = "FAIL"
    manifest["model_evaluation_gate_errors"] = ["stale-manifest-error"]
    _write_json(manifest_path, manifest)

    result = evidence.build_research_evidence_report(manifest_path)

    assert result["status"] == "FAIL"
    assert "model_evaluation_gate_manifest_status_mismatch" in result["errors"]
    assert "model_evaluation_gate_manifest_errors_mismatch" in result["errors"]


def test_consolidated_report_accepts_matching_manifest_receipt_summary(tmp_path, monkeypatch):
    manifest_path = _build_run(tmp_path, monkeypatch)

    result = evidence.build_research_evidence_report(manifest_path)

    assert "model_evaluation_gate_manifest_status_mismatch" not in result["errors"]
    assert "model_evaluation_gate_manifest_errors_mismatch" not in result["errors"]



def test_artifact_record_reports_partial_identity_metadata_as_warning(tmp_path):
    artifact = tmp_path / "receipt.json"
    _write_json(artifact, {
        "scope": "GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE",
        "run_id": "run-1",
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "actionable": False,
    })
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    record, report, errors, warnings = evidence._artifact_record(
        section_name="model_evaluation_gate",
        path=artifact,
        expected_scope="GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE",
        expected_run_id="run-1",
        expected_symbol="EURUSD",
        expected_timeframe="M15",
        expected_manifest=manifest,
        expected_matrix_sha256=None,
    )

    assert report is not None
    assert errors == []
    assert record["identity_metadata"]["status"] == "PARTIAL"
    assert record["identity_metadata"]["fields_missing"] == [
        "symbol", "timeframe", "manifest_path"
    ]
    assert "model_evaluation_gate_symbol_not_reported" in warnings
    assert "model_evaluation_gate_timeframe_not_reported" in warnings
    assert "model_evaluation_gate_manifest_path_not_reported" in warnings


def test_artifact_record_marks_complete_identity_metadata(tmp_path):
    artifact = tmp_path / "receipt.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    _write_json(artifact, {
        "scope": "GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE",
        "run_id": "run-1",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "manifest_path": str(manifest.resolve()),
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "actionable": False,
    })

    record, report, errors, warnings = evidence._artifact_record(
        section_name="model_evaluation_gate",
        path=artifact,
        expected_scope="GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE",
        expected_run_id="run-1",
        expected_symbol="EURUSD",
        expected_timeframe="M15",
        expected_manifest=manifest,
        expected_matrix_sha256=None,
    )

    assert report is not None
    assert errors == []
    assert warnings == []
    assert record["identity_metadata"]["status"] == "COMPLETE"
    assert record["identity_metadata"]["fields_missing"] == []


def test_verify_matrix_payload_hash_accepts_canonical_payload(tmp_path):
    path = tmp_path / "matrix.json"
    payload = {
        "schema_version": 1,
        "run_id": "run-1",
        "experiments": [],
        "actionable": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["report_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _write_json(path, payload)

    result = evidence._verify_matrix_payload_hash(path)

    assert result["status"] == "VERIFIED"
    assert result["computed_payload_sha256"] == result["declared_payload_sha256"]


def test_verify_matrix_payload_hash_detects_payload_tampering(tmp_path):
    path = tmp_path / "matrix.json"
    payload = {
        "schema_version": 1,
        "run_id": "run-1",
        "experiments": [],
        "actionable": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["report_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload["run_id"] = "tampered-run"
    _write_json(path, payload)

    result = evidence._verify_matrix_payload_hash(path)

    assert result["status"] == "HASH_MISMATCH"


def test_verify_matrix_payload_hash_marks_legacy_payload_unverified(tmp_path):
    path = tmp_path / "legacy-matrix.json"
    _write_json(path, {"schema_version": 1, "run_id": "run-1", "experiments": []})

    result = evidence._verify_matrix_payload_hash(path)

    assert result["status"] == "LEGACY_UNVERIFIED"
    assert result["declared_payload_sha256"] is None



def test_artifact_record_detects_declared_comparison_count_mismatch(tmp_path):
    artifact = tmp_path / "calibration.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    _write_json(artifact, {
        "scope": "GOLDCREST_TEMPORAL_CALIBRATION_STABILITY_AUDIT",
        "run_id": "run-1",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "manifest_path": str(manifest.resolve()),
        "status": "PASS",
        "comparison_count": 2,
        "comparisons": [{"model_name": "model-a", "partition": "test"}],
        "errors": [],
        "warnings": [],
        "actionable": False,
    })

    record, report, errors, warnings = evidence._artifact_record(
        section_name="temporal_calibration",
        path=artifact,
        expected_scope="GOLDCREST_TEMPORAL_CALIBRATION_STABILITY_AUDIT",
        expected_run_id="run-1",
        expected_symbol="EURUSD",
        expected_timeframe="M15",
        expected_manifest=manifest,
        expected_matrix_sha256=None,
    )

    assert report is not None
    assert record["status"] == "FAIL"
    assert record["declared_count_integrity"]["status"] == "MISMATCH"
    assert "temporal_calibration_comparison_count_mismatch" in errors


def test_artifact_record_verifies_declared_comparison_count(tmp_path):
    artifact = tmp_path / "calibration.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    _write_json(artifact, {
        "scope": "GOLDCREST_TEMPORAL_CALIBRATION_STABILITY_AUDIT",
        "run_id": "run-1",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "manifest_path": str(manifest.resolve()),
        "status": "PASS",
        "comparison_count": 1,
        "comparisons": [{"model_name": "model-a", "partition": "test"}],
        "errors": [],
        "warnings": [],
        "actionable": False,
    })

    record, report, errors, warnings = evidence._artifact_record(
        section_name="temporal_calibration",
        path=artifact,
        expected_scope="GOLDCREST_TEMPORAL_CALIBRATION_STABILITY_AUDIT",
        expected_run_id="run-1",
        expected_symbol="EURUSD",
        expected_timeframe="M15",
        expected_manifest=manifest,
        expected_matrix_sha256=None,
    )

    assert report is not None
    assert record["declared_count_integrity"]["status"] == "VERIFIED"
    assert errors == []
    assert warnings == []



def test_consolidated_report_exposes_integrity_summary_inventory(tmp_path, monkeypatch):
    manifest_path = _build_run(tmp_path, monkeypatch)

    result = evidence.build_research_evidence_report(manifest_path)

    summary = result["integrity_summary"]
    assert summary["artifact_section_count"] == len(result["sections"])
    assert summary["identity_metadata_partial_section_count"] >= 1
    assert summary["matrix_payload_integrity_status"] == "LEGACY_UNVERIFIED"
    assert summary["has_integrity_errors"] is False
    assert summary["has_warnings"] is True
    assert summary["interpretation"].startswith("Aggregated integrity inventory only.")



def test_cli_strict_warnings_returns_distinct_nonzero_code(tmp_path, monkeypatch, capsys):
    manifest_path = _build_run(tmp_path, monkeypatch)
    output_path = tmp_path / "strict-evidence.json"

    exit_code = evidence.main([
        "--manifest", str(manifest_path),
        "--output", str(output_path),
        "--strict-warnings",
    ])

    assert exit_code == 4
    assert output_path.is_file()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert report["warnings"]


def test_cli_without_strict_warnings_keeps_warning_exit_code_zero(tmp_path, monkeypatch, capsys):
    manifest_path = _build_run(tmp_path, monkeypatch)
    output_path = tmp_path / "non-strict-evidence.json"

    exit_code = evidence.main([
        "--manifest", str(manifest_path),
        "--output", str(output_path),
    ])

    assert exit_code == 0
    assert output_path.is_file()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS_WITH_WARNINGS"



def test_artifact_record_rejects_malformed_diagnostic_fields(tmp_path):
    artifact = tmp_path / "receipt.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    _write_json(artifact, {
        "scope": "GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE",
        "run_id": "run-1",
        "status": "PASS",
        "errors": "this-should-be-a-list",
        "warnings": ["valid warning", 17, ""],
        "actionable": False,
    })

    record, report, errors, warnings = evidence._artifact_record(
        section_name="model_evaluation_gate",
        path=artifact,
        expected_scope="GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE",
        expected_run_id="run-1",
        expected_symbol=None,
        expected_timeframe=None,
        expected_manifest=manifest,
        expected_matrix_sha256=None,
    )

    assert report is not None
    assert record["status"] == "FAIL"
    assert record["diagnostic_schema"]["status"] == "INVALID"
    assert "model_evaluation_gate_errors_must_be_list" in errors
    assert "model_evaluation_gate_warnings_item_1_invalid" in errors
    assert "model_evaluation_gate_warnings_item_2_invalid" in errors


def test_artifact_record_marks_well_formed_diagnostic_fields_valid(tmp_path):
    artifact = tmp_path / "receipt.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    _write_json(artifact, {
        "scope": "GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE",
        "run_id": "run-1",
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "actionable": False,
    })

    record, report, errors, warnings = evidence._artifact_record(
        section_name="model_evaluation_gate",
        path=artifact,
        expected_scope="GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE",
        expected_run_id="run-1",
        expected_symbol=None,
        expected_timeframe=None,
        expected_manifest=manifest,
        expected_matrix_sha256=None,
    )

    assert report is not None
    assert record["diagnostic_schema"]["status"] == "VALID"
    assert errors == []

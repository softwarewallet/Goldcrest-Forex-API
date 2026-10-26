from __future__ import annotations

import json

import pytest

from app.research_provenance_auditor import audit_experiment_matrix, audit_experiment_matrix_file


def sample_report():
    return {
        "schema_version": 1,
        "run_id": "eurusd-m15-run-01",
        "dataset_sha256": "a" * 64,
        "symbol": "EUR/USD",
        "timeframe": "M15",
        "model_types": ["xgboost", "lightgbm"],
        "fold_count": 1,
        "common_fold_test_windows": [{
            "fold_index": 0,
            "test_start_utc": "2026-01-02T00:00:00Z",
            "test_end_utc": "2026-01-02T12:00:00Z",
        }],
        "test_metrics_used_for_selection": False,
        "automatic_winner_selection": False,
        "actionable": False,
        "experiments": [
            {
                "experiment_id": model,
                "model_type": model,
                "feature_variant": "all_features",
                "feature_names": ["return_1", "rsi_14"],
                "feature_schema_sha256": "b" * 64,
                "fold_count": 1,
                "test_metrics_used_for_tuning": False,
                "actionable": False,
                "fold_results": [{
                    "fold_index": 0,
                    "test_start_utc": "2026-01-02T00:00:00Z",
                    "test_end_utc": "2026-01-02T12:00:00Z",
                    "best_params": {"max_depth": 3},
                    "trials_requested": 5,
                    "trials_completed": 5,
                }],
            }
            for model in ("xgboost", "lightgbm")
        ],
    }


def test_audit_accepts_matching_shared_fold_provenance():
    result = audit_experiment_matrix(sample_report())
    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert result["actionable"] is False
    assert result["execution_connected"] is False


def test_audit_rejects_different_test_windows_between_models():
    payload = sample_report()
    payload["experiments"][1]["fold_results"][0]["test_end_utc"] = "2026-01-02T13:00:00Z"
    result = audit_experiment_matrix(payload)
    assert result["status"] == "FAIL"
    assert "experiment_1_test_windows_do_not_match_common_plan" in result["errors"]


def test_audit_rejects_test_selection_and_invalid_feature_fingerprint():
    payload = sample_report()
    payload["experiments"][0]["test_metrics_used_for_tuning"] = True
    payload["experiments"][1]["feature_schema_sha256"] = "not-a-hash"
    result = audit_experiment_matrix(payload)
    assert "experiment_0_test_metrics_used_for_tuning_must_be_false" in result["errors"]
    assert "experiment_1_invalid_feature_schema_sha256" in result["errors"]


def test_file_audit_writes_receipt_with_source_hash(tmp_path):
    source = tmp_path / "matrix.json"
    destination = tmp_path / "audit.json"
    source.write_text(json.dumps(sample_report()), encoding="utf-8")
    result = audit_experiment_matrix_file(source, destination)
    stored = json.loads(destination.read_text(encoding="utf-8"))
    assert stored["source_report_sha256"] == result["source_report_sha256"]
    assert stored["status"] == "PASS"


@pytest.mark.parametrize("raw", [
    b'{"schema_version":1,"schema_version":1}',
    b'{"metric":NaN}',
    b'{"metric":Infinity}',
    b'{"metric":-Infinity}',
])
def test_file_auditor_rejects_ambiguous_or_nonstandard_json(tmp_path, raw):
    path = tmp_path / "experiment-matrix.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        audit_experiment_matrix_file(path)

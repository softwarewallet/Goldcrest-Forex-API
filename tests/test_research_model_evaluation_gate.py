from __future__ import annotations

import json

import pytest
from pathlib import Path

from app.research_model_evaluation_gate import audit_model_evaluation_report


def _report() -> dict:
    return {
        "schema_version": 1,
        "run_id": "fixture-evaluation-gate",
        "fold_count": 1,
        "test_metrics_used_for_selection": False,
        "automatic_winner_selection": False,
        "actionable": False,
        "experiments": [{
            "experiment_id": "fixture-xgb",
            "actionable": False,
            "test_metrics_used_for_tuning": False,
            "fold_results": [{
                "fold_index": 0,
                "train_start_utc": "2026-01-01T00:00:00Z",
                "train_end_utc": "2026-01-01T01:00:00Z",
                "validation_start_utc": "2026-01-01T01:15:00Z",
                "validation_end_utc": "2026-01-01T01:30:00Z",
                "test_start_utc": "2026-01-01T01:45:00Z",
                "test_end_utc": "2026-01-01T02:00:00Z",
                "test_metrics_used_for_tuning": False,
                "test_metrics": {
                    "sample_count": 100,
                    "positive_count": 40,
                    "negative_count": 60,
                    "prevalence": 0.4,
                    "brier_score": 0.21,
                    "log_loss": 0.61,
                    "accuracy_at_fixed_half": 0.62,
                    "mean_predicted_probability": 0.43,
                    "expected_calibration_error": 0.07,
                },
            }],
        }],
    }


def _write(path: Path, report: dict) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_evaluation_gate_accepts_consistent_non_actionable_report(tmp_path: Path) -> None:
    result = audit_model_evaluation_report(_write(tmp_path / "report.json", _report()))
    assert result["status"] == "PASS"
    assert result["experiment_count"] == 1
    assert result["folds_checked"] == 1
    assert result["fold_metric_objects_checked"] == 1
    assert result["actionable"] is False
    assert result["scope"] == "GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE"
    assert result["run_id"] == "fixture-evaluation-gate"
    assert result["source_matrix_sha256"]
    assert result["test_metrics_used_for_selection"] is False
    assert result["automatic_winner_selection"] is False


def test_evaluation_gate_rejects_test_metric_selection(tmp_path: Path) -> None:
    report = _report()
    report["test_metrics_used_for_selection"] = True
    report["experiments"][0]["fold_results"][0]["test_metrics_used_for_tuning"] = True
    result = audit_model_evaluation_report(_write(tmp_path / "report.json", report))
    assert result["status"] == "FAIL"
    codes = {item["code"] for item in result["errors"]}
    assert "unsafe_or_unrecorded_selection_flag" in codes
    assert "test_metrics_used_for_tuning" in codes


def test_evaluation_gate_rejects_inconsistent_class_counts_and_prevalence(tmp_path: Path) -> None:
    report = _report()
    metrics = report["experiments"][0]["fold_results"][0]["test_metrics"]
    metrics["negative_count"] = 61
    metrics["prevalence"] = 0.5
    result = audit_model_evaluation_report(_write(tmp_path / "report.json", report))
    assert result["status"] == "FAIL"
    codes = {item["code"] for item in result["errors"]}
    assert "class_counts_do_not_sum" in codes
    assert "prevalence_inconsistent" in codes


def test_evaluation_gate_rejects_invalid_metric_ranges_and_duplicate_folds(tmp_path: Path) -> None:
    report = _report()
    fold = report["experiments"][0]["fold_results"][0]
    fold["test_metrics"]["brier_score"] = 1.2
    report["experiments"][0]["fold_results"].append(dict(fold))
    report["fold_count"] = 2
    result = audit_model_evaluation_report(_write(tmp_path / "report.json", report))
    assert result["status"] == "FAIL"
    codes = {item["code"] for item in result["errors"]}
    assert "metric_out_of_range" in codes
    assert "duplicate_fold_index" in codes


def test_evaluation_gate_rejects_naive_timestamps_and_reversed_windows(tmp_path: Path) -> None:
    report = _report()
    fold = report["experiments"][0]["fold_results"][0]
    fold["test_start_utc"] = "2026-01-01T01:00:00"
    fold["validation_end_utc"] = "2026-01-01T01:00:00Z"
    result = audit_model_evaluation_report(_write(tmp_path / "report.json", report))
    assert result["status"] == "FAIL"
    codes = {item["code"] for item in result["errors"]}
    assert "invalid_fold_timestamp" in codes
    assert "invalid_fold_window" in codes


def test_evaluation_gate_rejects_cross_experiment_fold_window_mismatch(tmp_path: Path) -> None:
    report = _report()
    second_experiment = {
        **report["experiments"][0],
        "experiment_id": "fixture-lightgbm",
        "fold_results": [
            {
                **report["experiments"][0]["fold_results"][0],
                "test_start_utc": "2026-01-01T01:46:00Z",
            }
        ],
    }
    report["experiments"].append(second_experiment)
    result = audit_model_evaluation_report(_write(tmp_path / "report.json", report))
    assert result["status"] == "FAIL"
    assert "common_fold_evaluation_mismatch" in {
        item["code"] for item in result["errors"]
    }


def test_evaluation_gate_accepts_identical_common_fold_windows(tmp_path: Path) -> None:
    report = _report()
    report["experiments"].append({
        **report["experiments"][0],
        "experiment_id": "fixture-lightgbm",
    })
    result = audit_model_evaluation_report(_write(tmp_path / "report.json", report))
    assert result["status"] == "PASS"


def test_evaluation_gate_reports_unreadable_json(tmp_path: Path) -> None:
    source = tmp_path / "broken.json"
    source.write_text("{", encoding="utf-8")
    result = audit_model_evaluation_report(source)
    assert result["status"] == "FAIL"
    assert result["errors"][0]["code"] == "report_unreadable"


@pytest.mark.parametrize(
    "payload",
    [
        '{"a": 1, "a": 2}',
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": -Infinity}',
    ],
)
def test_json_reader_rejects_ambiguous_or_nonstandard_json(tmp_path, payload):
    path = tmp_path / "malformed.json"
    path.write_text(payload, encoding="utf-8")
    result = audit_model_evaluation_report(path)
    assert result["status"] == "FAIL"
    assert result["errors"][0]["code"] == "report_unreadable"

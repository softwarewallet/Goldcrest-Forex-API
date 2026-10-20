import copy
import hashlib
import json

import pytest

from app.experiment_report import build_experiment_report, write_experiment_report
from app.research_report_inspection import inspect_experiment_report


def _report():
    return build_experiment_report(
        manifest={"run_id": "inspect-run", "dataset_id": "synthetic-fixture"},
        fold_results={
            "candidate": [
                {
                    "fold_index": 1,
                    "validation": {"sample_count": 10, "brier_score": 0.18},
                    "test": {"sample_count": 10, "brier_score": 0.22},
                    "actionable": False,
                },
                {
                    "fold_index": 0,
                    "validation": {"sample_count": 12, "brier_score": 0.17},
                    "test": {"sample_count": 12, "brier_score": 0.21},
                    "actionable": False,
                },
            ],
            "prevalence": [
                {
                    "fold_index": 0,
                    "validation": {"sample_count": 12, "brier_score": 0.25},
                    "test": {"sample_count": 12, "brier_score": 0.26},
                    "actionable": False,
                }
            ],
        },
        paired_comparisons=[
            {"sample_count": 12, "candidate_name": "candidate", "baseline_name": "prevalence",
             "comparison": {"brier_difference": -0.04}, "actionable": False}
        ],
        exclusions={"unlabeled": 3, "ambiguous": 1},
        evaluation_contract={"label": "caller-supplied-binary-label"},
        warnings=["Synthetic fixture; no market-performance inference."],
    )


def test_inspection_checks_hash_and_preserves_per_fold_partitions(tmp_path):
    report = _report()
    path = write_experiment_report(report, tmp_path / "report.json")

    summary = inspect_experiment_report(path)

    assert summary["verified"] is True
    assert summary["run_id"] == "inspect-run"
    assert summary["actionable"] is False
    assert summary["interpretation"]["pooled_test_metrics"] is False
    candidate = next(row for row in summary["models"] if row["model_name"] == "candidate")
    assert [fold["fold_index"] for fold in candidate["folds"]] == [0, 1]
    assert candidate["folds"][0]["validation"]["brier_score"] == 0.17
    assert candidate["folds"][0]["test"]["brier_score"] == 0.21
    assert summary["exclusions"] == {"ambiguous": 1, "unlabeled": 3}
    assert len(summary["paired_comparisons"]) == 1


def test_inspection_rejects_modified_report_even_when_json_is_valid(tmp_path):
    report = _report()
    path = write_experiment_report(report, tmp_path / "report.json")
    modified = json.loads(path.read_text(encoding="utf-8"))
    modified["warnings"].append("tampered after fingerprinting")
    path.write_text(json.dumps(modified), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        inspect_experiment_report(path)


def test_inspection_rejects_actionable_report_even_with_recomputed_fingerprint(tmp_path):
    report = _report()
    report["interpretation"]["actionable"] = True
    payload = dict(report)
    payload.pop("report_sha256")
    report["report_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="actionable=false"):
        inspect_experiment_report(path)


def test_inspection_rejects_nonfinite_json_constants(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"run_id":"x","value":NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="strict JSON"):
        inspect_experiment_report(path)


def test_inspection_rejects_duplicate_fold_indices_after_valid_fingerprint(tmp_path):
    report = _report()
    report["fold_results"]["candidate"][1]["fold_index"] = 1
    payload = dict(report)
    payload.pop("report_sha256")
    report["report_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate fold index"):
        inspect_experiment_report(path)

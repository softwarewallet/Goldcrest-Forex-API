import hashlib
import json

import pytest

from app.experiment_report import build_experiment_report, write_experiment_report
from app.research_report_inspection import inspect_experiment_report
from app.research_reproducibility import reproducibility_manifest


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
             "fold_index": 0, "partition": "test",
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
    assert len(summary["fold_stability"]) == 1
    assert summary["fold_stability"][0]["fold_indices"] == [0]
    assert summary["fold_stability"][0]["metrics"]["brier_score"]["candidate_wins"] == 1


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
    report["fold_results"]["candidate"][1]["fold_index"] = 0
    payload = dict(report)
    payload.pop("report_sha256")
    report["report_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate fold index"):
        inspect_experiment_report(path)



def test_inspection_verifies_nested_reproducibility_manifest(tmp_path):
    class Model:
        model_type = "fixture"
        model_version = "v1"

        def __init__(self):
            self.depth = 4

    def factory():
        return Model()

    report = build_experiment_report(
        manifest={
            "run_id": "nested-repro-run",
            "reproducibility": reproducibility_manifest({"candidate": factory}, {"candidate": factory()}),
        },
        fold_results={
            "candidate": [{
                "fold_index": 0,
                "validation": {"sample_count": 2, "brier_score": 0.2},
                "test": {"sample_count": 2, "brier_score": 0.2},
                "actionable": False,
            }]
        },
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert inspect_experiment_report(path)["verified"] is True

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["manifest"]["reproducibility"]["initial_model_configuration"]["candidate"]["public_configuration"]["depth"] = 999
    payload = dict(tampered)
    payload.pop("report_sha256")
    tampered["report_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="reproducibility manifest fingerprint mismatch"):
        inspect_experiment_report(path)



def _rewrite_fingerprint(report):
    payload = dict(report)
    payload.pop("report_sha256", None)
    report["report_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return report


def test_inspection_rejects_comparison_with_missing_model_fold(tmp_path):
    report = _report()
    report["paired_comparisons"][0]["baseline_name"] = "unknown-model"
    _rewrite_fingerprint(report)
    path = tmp_path / "dangling-comparison.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="baseline_name must reference a reported model"):
        inspect_experiment_report(path)


def test_inspection_rejects_paired_sample_count_mismatch(tmp_path):
    report = _report()
    report["paired_comparisons"][0]["sample_count"] = 11
    _rewrite_fingerprint(report)
    path = tmp_path / "mismatched-comparison.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="sample_count does not match"):
        inspect_experiment_report(path)


def test_inspection_rejects_duplicate_comparison_identity(tmp_path):
    report = _report()
    report["paired_comparisons"].append(dict(report["paired_comparisons"][0]))
    _rewrite_fingerprint(report)
    path = tmp_path / "duplicate-comparison.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate paired comparison identity"):
        inspect_experiment_report(path)



def test_inspection_rejects_mismatched_paired_observation_fingerprints(tmp_path):
    report = _report()
    for fold in report["fold_results"]["candidate"]:
        fold["test"]["observation_identity_sha256"] = "a" * 64
    for fold in report["fold_results"]["prevalence"]:
        fold["test"]["observation_identity_sha256"] = "b" * 64
    report["paired_comparisons"][0]["observation_identity_sha256"] = "a" * 64
    _rewrite_fingerprint(report)
    path = tmp_path / "mismatched-identities.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate and baseline observation identity fingerprints do not match"):
        inspect_experiment_report(path)


def test_inspection_rejects_comparison_fingerprint_not_matching_fold_metrics(tmp_path):
    report = _report()
    for fold in report["fold_results"]["candidate"]:
        fold["test"]["observation_identity_sha256"] = "a" * 64
    for fold in report["fold_results"]["prevalence"]:
        fold["test"]["observation_identity_sha256"] = "a" * 64
    report["paired_comparisons"][0]["observation_identity_sha256"] = "c" * 64
    _rewrite_fingerprint(report)
    path = tmp_path / "comparison-identity-mismatch.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="paired comparison observation identity fingerprint does not match"):
        inspect_experiment_report(path)


def test_inspection_rejects_selection_audit_that_uses_different_validation_metrics(tmp_path):
    report = _report()
    report["model_selection_audit"]["selected_candidate"] = "prevalence"
    _rewrite_fingerprint(report)
    path = tmp_path / "tampered-selection-audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="model_selection_audit does not match validation-only fold metrics"):
        inspect_experiment_report(path)



def test_inspection_recomputes_block_length_sensitivity_audit(tmp_path):
    report = _report()
    sensitivity = [
        {"block_length": 1, "brier_difference_ci_low": -0.06, "brier_difference_ci_high": -0.01,
         "log_loss_difference_ci_low": -0.12, "log_loss_difference_ci_high": -0.02,
         "brier_direction": "CANDIDATE_LOWER", "log_loss_direction": "CANDIDATE_LOWER"},
        {"block_length": 2, "brier_difference_ci_low": -0.04, "brier_difference_ci_high": 0.01,
         "log_loss_difference_ci_low": -0.09, "log_loss_difference_ci_high": 0.02,
         "brier_direction": "INCLUDES_ZERO", "log_loss_direction": "INCLUDES_ZERO"},
    ]
    report["paired_comparisons"][0]["comparison"] = {"block_length": 1,
        "brier_difference_ci_low": -0.06, "brier_difference_ci_high": -0.01,
        "log_loss_difference_ci_low": -0.12, "log_loss_difference_ci_high": -0.02,
        "block_length_sensitivity": sensitivity}
    _rewrite_fingerprint(report)
    path = tmp_path / "tampered-block-sensitivity-audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="block_length_sensitivity_audit does not match"):
        inspect_experiment_report(path)

from __future__ import annotations

import hashlib

import pytest

from app.research_walk_forward_robustness_audit import audit_walk_forward_robustness


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def valid_manifest():
    records = []
    for seed in (11, 22, 33):
        for fold, offset in (("fold_1", 0.0), ("fold_2", 0.02)):
            records.append({
                "run_id": f"seed{seed}_{fold}",
                "seed": seed,
                "fold_id": fold,
                "dataset_sha256": sha(b"dataset"),
                "metrics": {
                    "balanced_accuracy": 0.60 + offset + seed / 10000,
                    "log_loss": 0.55 - offset + seed / 10000,
                },
            })
    return {
        "expected_folds": ["fold_1", "fold_2"],
        "metric_directions": {
            "balanced_accuracy": "maximize",
            "log_loss": "minimize",
        },
        "evaluations": records,
    }


def run_audit(manifest, minimum_unique_seeds=3):
    return audit_walk_forward_robustness(
        manifest,
        manifest_sha256=sha(b"manifest"),
        minimum_unique_seeds=minimum_unique_seeds,
    )


def test_complete_repeated_seed_fold_bundle_passes_descriptively():
    report = run_audit(valid_manifest())

    assert report["status"] == "PASS_WITH_WARNINGS"
    assert any("small_seed_count" in warning for warning in report["warnings"])
    assert report["evaluations_supplied"] == 6
    assert report["evaluations_validated"] == 6
    assert report["unique_seed_count"] == 3
    assert report["metric_summaries"]["balanced_accuracy"]["observations"] == 6
    assert set(report["per_seed_summaries"]) == {"11", "22", "33"}
    assert set(report["per_fold_summaries"]) == {"fold_1", "fold_2"}
    assert report["statistical_significance_claimed"] is False
    assert report["execution_connected"] is False
    assert report["automatic_model_selection"] is False


def test_dataset_fingerprint_mismatch_fails_closed():
    manifest = valid_manifest()
    manifest["evaluations"][0]["dataset_sha256"] = sha(b"different dataset")

    report = run_audit(manifest)

    assert report["status"] == "FAIL"
    assert any(error["code"] == "dataset_fingerprint_mismatch" for error in report["errors"])


def test_duplicate_run_and_seed_fold_records_are_detected():
    manifest = valid_manifest()
    duplicate = dict(manifest["evaluations"][0])
    duplicate["run_id"] = manifest["evaluations"][1]["run_id"]
    duplicate["seed"] = manifest["evaluations"][1]["seed"]
    duplicate["fold_id"] = manifest["evaluations"][1]["fold_id"]
    manifest["evaluations"].append(duplicate)

    report = run_audit(manifest)
    codes = {error["code"] for error in report["errors"]}

    assert "duplicate_run_id" in codes
    assert "duplicate_seed_fold" in codes


def test_missing_expected_fold_for_one_seed_fails():
    manifest = valid_manifest()
    manifest["evaluations"] = [
        item for item in manifest["evaluations"]
        if not (item["seed"] == 22 and item["fold_id"] == "fold_2")
    ]

    report = run_audit(manifest)

    assert report["status"] == "FAIL"
    assert any(error["code"] == "seed_fold_coverage_incomplete" for error in report["errors"])


def test_insufficient_seed_count_fails_even_if_metrics_are_valid():
    manifest = valid_manifest()
    manifest["evaluations"] = [
        item for item in manifest["evaluations"] if item["seed"] in (11, 22)
    ]

    report = run_audit(manifest)

    assert report["status"] == "FAIL"
    assert any(error["code"] == "insufficient_unique_seeds" for error in report["errors"])


def test_nonfinite_metric_and_inconsistent_metric_sets_are_detected():
    manifest = valid_manifest()
    manifest["evaluations"][0]["metrics"]["balanced_accuracy"] = float("nan")
    manifest["evaluations"][1]["metrics"].pop("log_loss")

    report = run_audit(manifest)
    codes = {error["code"] for error in report["errors"]}

    assert "metric_value_not_finite_number:balanced_accuracy" in codes
    assert "metric_set_mismatch" in codes


def test_unknown_metric_direction_and_invalid_hash_are_detected():
    manifest = valid_manifest()
    manifest["metric_directions"]["balanced_accuracy"] = "guess"
    manifest["evaluations"][0]["dataset_sha256"] = "invalid"

    report = run_audit(manifest)
    codes = {error["code"] for error in report["errors"]}

    assert "metric_direction_invalid" in codes
    assert "dataset_sha256_invalid" in codes


def test_missing_expected_fold_declaration_is_warning():
    manifest = valid_manifest()
    manifest.pop("expected_folds")

    report = run_audit(manifest)

    assert report["status"] == "PASS_WITH_WARNINGS"
    assert "expected_folds_not_declared; fold completeness cannot be certified" in report["warnings"]


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="minimum_unique_seeds_must_be_at_least_two"):
        audit_walk_forward_robustness(
            valid_manifest(),
            manifest_sha256=sha(b"manifest"),
            minimum_unique_seeds=1,
        )
    with pytest.raises(ValueError, match="manifest_sha256_invalid"):
        audit_walk_forward_robustness(valid_manifest(), manifest_sha256="bad")

from __future__ import annotations

import hashlib

import pytest

from app.research_pipeline_leakage_boundary_audit import audit_pipeline_leakage_boundaries


def digest() -> str:
    return hashlib.sha256(b"manifest fixture").hexdigest()


def valid_manifest() -> dict:
    return {
        "splits": {
            "train": {"start_index": 0, "end_index": 69},
            "validation": {"start_index": 70, "end_index": 84},
            "test": {"start_index": 85, "end_index": 99},
        },
        "preprocessing": [
            {
                "name": "standard_scaler",
                "fit_start_index": 0,
                "fit_end_index": 69,
                "applied_to_splits": ["train", "validation", "test"],
                "fit_only_on_train": True,
            }
        ],
        "label_horizon_bars": 3,
        "labels": [
            {"source_index": 70, "target_index": 73, "split": "validation"},
            {"source_index": 85, "target_index": 88, "split": "test"},
        ],
    }


def test_valid_pipeline_manifest_passes_with_safety_flags():
    report = audit_pipeline_leakage_boundaries(valid_manifest(), source_sha256=digest())

    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert report["preprocessing_results"][0]["passed"] is True
    assert all(item["passed"] for item in report["label_results"])
    assert report["actionable"] is False
    assert report["inferential"] is False
    assert report["model_training_performed"] is False
    assert report["execution_connected"] is False
    assert report["automatic_repair_performed"] is False


def test_preprocessing_fitted_on_validation_rows_fails_closed():
    manifest = valid_manifest()
    manifest["preprocessing"][0]["fit_end_index"] = 84

    report = audit_pipeline_leakage_boundaries(manifest, source_sha256=digest())

    assert report["status"] == "FAIL"
    assert any(error["code"] == "fit_uses_rows_after_training_boundary" for error in report["errors"])


def test_label_target_crossing_split_boundary_is_detected():
    manifest = valid_manifest()
    manifest["labels"][0] = {
        "source_index": 83,
        "target_index": 86,
        "split": "validation",
    }

    report = audit_pipeline_leakage_boundaries(manifest, source_sha256=digest())

    assert report["status"] == "FAIL"
    assert any(error["code"] == "label_target_crosses_split_boundary" for error in report["errors"])


def test_overlapping_splits_and_short_label_horizon_are_detected():
    manifest = valid_manifest()
    manifest["splits"]["validation"]["start_index"] = 69
    manifest["labels"][0]["target_index"] = 71

    report = audit_pipeline_leakage_boundaries(manifest, source_sha256=digest())

    codes = {error["code"] for error in report["errors"]}
    assert "split_overlap_or_nonchronological" in codes
    assert "label_horizon_shorter_than_declared" in codes


def test_invalid_manifest_and_source_hash_are_rejected():
    with pytest.raises(ValueError, match="manifest_must_be_object"):
        audit_pipeline_leakage_boundaries([], source_sha256=digest())
    with pytest.raises(ValueError, match="source_sha256_invalid"):
        audit_pipeline_leakage_boundaries(valid_manifest(), source_sha256="bad")


def test_missing_optional_evidence_is_reported_as_warning_not_assumed_safe():
    manifest = {"splits": {"train": {"start_index": 0, "end_index": 9}}}

    report = audit_pipeline_leakage_boundaries(manifest, source_sha256=digest())

    assert report["status"] == "PASS_WITH_WARNINGS"
    assert "no_preprocessing_fit_records_supplied" in report["warnings"]
    assert any("no_label_records_supplied" in warning for warning in report["warnings"])

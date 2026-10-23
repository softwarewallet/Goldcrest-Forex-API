import hashlib

import pytest

from app.research_experiment_fingerprint import build_experiment_fingerprint, verify_experiment_fingerprint


def _manifest():
    return {
        "manifest_sha256": "f" * 64,
        "feature_implementation_sha256": "a" * 64,
        "feature_schema_sha256": "b" * 64,
        "feature_backend": "numpy_fallback",
        "actionable": False,
    }


def _build(configuration=None):
    return build_experiment_fingerprint(
        configuration=configuration or {"random_seed": 7, "model_types": ["xgboost", "lightgbm"]},
        raw_ohlc_sha256=hashlib.sha256(b"raw").hexdigest(),
        dataset_sha256=hashlib.sha256(b"dataset").hexdigest(),
        dataset_manifest=_manifest(),
    )


def test_fingerprint_is_deterministic_for_identical_inputs():
    first = _build()
    second = _build()
    assert first == second
    assert verify_experiment_fingerprint(first)["status"] == "PASS"
    assert first["actionable"] is False
    assert first["execution_connected"] is False


def test_fingerprint_changes_when_experiment_configuration_changes():
    first = _build({"random_seed": 7, "model_types": ["xgboost", "lightgbm"]})
    changed = _build({"random_seed": 8, "model_types": ["xgboost", "lightgbm"]})
    assert first["experiment_fingerprint_sha256"] != changed["experiment_fingerprint_sha256"]


def test_fingerprint_verifier_detects_receipt_tampering():
    receipt = _build()
    receipt["configuration"]["random_seed"] = 999
    result = verify_experiment_fingerprint(receipt)
    assert result["status"] == "FAIL"
    assert "experiment_fingerprint_sha256_mismatch" in result["errors"]


def test_fingerprint_rejects_actionable_dataset_manifest():
    manifest = _manifest()
    manifest["actionable"] = True
    with pytest.raises(ValueError, match="non-actionable"):
        build_experiment_fingerprint(
            configuration={"random_seed": 7},
            raw_ohlc_sha256=hashlib.sha256(b"raw").hexdigest(),
            dataset_sha256=hashlib.sha256(b"dataset").hexdigest(),
            dataset_manifest=manifest,
        )

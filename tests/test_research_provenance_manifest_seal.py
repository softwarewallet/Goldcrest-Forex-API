from __future__ import annotations

from copy import deepcopy

import pytest

from app.research_provenance_manifest_seal import (
    canonical_manifest_bytes,
    create_manifest_seal,
    verify_manifest_seal,
)


def manifest():
    return {
        "audit_name": "research_artifact_provenance_manifest",
        "source_revision": "a" * 40,
        "artifact_count": 1,
        "artifacts": [{"artifact_id": "metrics", "path": "metrics.csv", "sha256": "b" * 64}],
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "automatic_model_selection": False,
        "model_training_performed": False,
        "execution_connected": False,
        "automatic_repair_performed": False,
        "deployment_authorized": False,
    }


def test_seal_round_trip_passes():
    payload = manifest()
    seal = create_manifest_seal(payload)
    report = verify_manifest_seal(payload, seal)
    assert report["status"] == "PASS"
    assert report["manifest_sha256"] == seal["manifest_sha256"]
    assert report["deployment_authorized"] is False


def test_canonicalization_is_independent_of_object_key_order():
    payload = manifest()
    reversed_payload = dict(reversed(list(payload.items())))
    assert canonical_manifest_bytes(payload) == canonical_manifest_bytes(reversed_payload)
    assert create_manifest_seal(payload)["manifest_sha256"] == create_manifest_seal(reversed_payload)["manifest_sha256"]


def test_manifest_edit_after_sealing_is_detected():
    payload = manifest()
    seal = create_manifest_seal(payload)
    changed = deepcopy(payload)
    changed["artifacts"][0]["sha256"] = "c" * 64
    report = verify_manifest_seal(changed, seal)
    assert report["status"] == "FAIL"
    assert any(item.get("field") == "manifest_sha256" for item in report["errors"])


def test_seal_metadata_tampering_is_detected():
    payload = manifest()
    seal = create_manifest_seal(payload)
    seal["artifact_count"] = 99
    report = verify_manifest_seal(payload, seal)
    assert report["status"] == "FAIL"
    assert any(item.get("field") == "artifact_count" for item in report["errors"])


def test_missing_explicit_false_safety_flag_fails():
    payload = manifest()
    seal = create_manifest_seal(payload)
    del seal["execution_connected"]
    report = verify_manifest_seal(payload, seal)
    assert report["status"] == "FAIL"
    assert any(item.get("field") == "execution_connected" for item in report["errors"])


def test_invalid_manifest_revision_is_rejected():
    payload = manifest()
    payload["source_revision"] = "short"
    with pytest.raises(ValueError, match="full_git_sha"):
        create_manifest_seal(payload)



@pytest.mark.parametrize("flag", [
    "actionable",
    "inferential",
    "statistical_significance_claimed",
    "automatic_model_selection",
    "model_training_performed",
    "execution_connected",
    "automatic_repair_performed",
    "deployment_authorized",
])
def test_unsafe_or_missing_manifest_safety_flag_cannot_be_sealed(flag):
    payload = manifest()
    payload[flag] = True
    with pytest.raises(ValueError, match="safety_flag_not_explicitly_false"):
        create_manifest_seal(payload)


def test_missing_manifest_safety_flag_cannot_be_sealed():
    payload = manifest()
    del payload["deployment_authorized"]
    with pytest.raises(ValueError, match="safety_flag_not_explicitly_false"):
        create_manifest_seal(payload)

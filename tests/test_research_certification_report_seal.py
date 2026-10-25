from __future__ import annotations

from copy import deepcopy

import pytest

from app.research_certification_report_seal import (
    canonical_report_bytes,
    create_report_seal,
    verify_report_seal,
)


def report():
    return {
        "schema_version": 1,
        "audit_name": "research_unified_provenance_certification",
        "status": "PASS",
        "readiness": "CERTIFIED_CURRENT",
        "source_revision": "a" * 40,
        "errors": [],
        "artifact_count": 2,
        "deployment_authorized": False,
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "automatic_model_selection": False,
        "model_training_performed": False,
        "execution_connected": False,
        "automatic_repair_performed": False,
    }


def test_report_seal_round_trip_passes():
    payload = report()
    seal = create_report_seal(payload)
    verified = verify_report_seal(payload, seal)
    assert verified["status"] == "PASS"
    assert verified["report_sha256"] == seal["report_sha256"]
    assert verified["deployment_authorized"] is False


def test_canonicalization_ignores_dictionary_key_order():
    payload = report()
    reordered = dict(reversed(list(payload.items())))
    assert canonical_report_bytes(payload) == canonical_report_bytes(reordered)
    assert create_report_seal(payload)["report_sha256"] == create_report_seal(reordered)["report_sha256"]


def test_report_edit_after_sealing_is_detected():
    payload = report()
    seal = create_report_seal(payload)
    changed = deepcopy(payload)
    changed["artifact_count"] = 99
    verified = verify_report_seal(changed, seal)
    assert verified["status"] == "FAIL"
    assert any(item.get("field") == "report_sha256" for item in verified["errors"])


def test_seal_metadata_tampering_is_detected():
    payload = report()
    seal = create_report_seal(payload)
    seal["report_status"] = "FAIL"
    verified = verify_report_seal(payload, seal)
    assert verified["status"] == "FAIL"
    assert any(item.get("field") == "report_status" for item in verified["errors"])


def test_missing_explicit_false_safety_flag_fails():
    payload = report()
    seal = create_report_seal(payload)
    del seal["deployment_authorized"]
    verified = verify_report_seal(payload, seal)
    assert verified["status"] == "FAIL"
    assert any(item.get("field") == "deployment_authorized" for item in verified["errors"])


@pytest.mark.parametrize("status", ["UNKNOWN", None, 3])
def test_invalid_report_status_is_rejected(status):
    payload = report()
    payload["status"] = status
    with pytest.raises(ValueError, match="status_invalid"):
        create_report_seal(payload)


def test_invalid_report_revision_is_rejected():
    payload = report()
    payload["source_revision"] = "not-a-sha"
    with pytest.raises(ValueError, match="revision_invalid"):
        create_report_seal(payload)

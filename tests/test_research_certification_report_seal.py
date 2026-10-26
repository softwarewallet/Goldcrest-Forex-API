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
def test_unsafe_or_missing_report_safety_flag_cannot_be_sealed(flag):
    payload = report()
    payload[flag] = True
    with pytest.raises(ValueError, match="safety_flag_not_explicitly_false"):
        create_report_seal(payload)


def test_missing_report_safety_flag_cannot_be_sealed():
    payload = report()
    del payload["deployment_authorized"]
    with pytest.raises(ValueError, match="safety_flag_not_explicitly_false"):
        create_report_seal(payload)


def test_passing_report_with_errors_cannot_be_sealed():
    payload = report()
    payload["errors"] = [{"code": "something_failed"}]
    with pytest.raises(ValueError, match="must_not_contain_errors"):
        create_report_seal(payload)

@pytest.mark.parametrize("raw", [b'{"metric":NaN}', b'{"metric":Infinity}', b'{"metric":-Infinity}', b'{"value":1,"value":2}'])
def test_certification_json_readers_reject_nonstandard_json(tmp_path, raw):
    from app.research_certification_report_seal import _read_object
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        _read_object(path)

def test_malformed_report_warnings_cannot_be_sealed():
    payload = report()
    payload["warnings"] = {"not": "a-list"}
    with pytest.raises(ValueError, match="warnings_invalid"):
        create_report_seal(payload)


@pytest.mark.parametrize("warnings", [[""], [None], ["valid", 3]])
def test_report_warnings_must_be_nonempty_strings(warnings):
    payload = report()
    payload["warnings"] = warnings
    with pytest.raises(ValueError, match="warnings_invalid"):
        create_report_seal(payload)

def test_explicit_null_report_warnings_cannot_be_sealed():
    payload = report()
    payload["warnings"] = None
    with pytest.raises(ValueError, match="warnings_invalid"):
        create_report_seal(payload)



def test_scope_limitations_tampering_is_detected():
    payload = report()
    seal = create_report_seal(payload)
    seal["scope_limitations"] = ["All claims are independently authenticated."]
    verified = verify_report_seal(payload, seal)
    assert verified["status"] == "FAIL"
    assert any(item.get("field") == "scope_limitations" for item in verified["errors"])


def test_unexpected_seal_metadata_is_rejected():
    payload = report()
    seal = create_report_seal(payload)
    seal["producer_verified"] = True
    verified = verify_report_seal(payload, seal)
    assert verified["status"] == "FAIL"
    assert any(item.get("code") == "REPORT_SEAL_UNEXPECTED_FIELD"
               and item.get("field") == "producer_verified" for item in verified["errors"])



@pytest.mark.parametrize(
    "status,warnings",
    [
        ("PASS", ["unexpected warning"]),
        ("PASS_WITH_WARNINGS", []),
    ],
)
def test_report_status_must_match_warning_collection(status, warnings):
    payload = report()
    payload["status"] = status
    payload["warnings"] = warnings
    with pytest.raises(ValueError):
        create_report_seal(payload)


@pytest.mark.parametrize("errors", ["not-a-list", [None], ["unstructured error"]])
def test_report_errors_must_be_structured_objects(errors):
    payload = report()
    payload["status"] = "FAIL"
    payload["errors"] = errors
    with pytest.raises(ValueError, match="errors_must_be_list_of_objects"):
        create_report_seal(payload)


def test_seal_writer_does_not_clobber_target_created_after_staging(monkeypatch, tmp_path):
    import app.research_certification_report_seal as module

    target = tmp_path / "certification.seal.json"
    competitor_payload = '{"owner":"concurrent-writer"}\\n'

    def concurrent_writer_then_link_failure(source, destination):
        from pathlib import Path
        Path(destination).write_text(competitor_payload, encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr(module.os, "link", concurrent_writer_then_link_failure)
    with pytest.raises(ValueError, match="output_already_exists"):
        module._write_json_no_clobber(target, '{"status":"PASS"}\\n')

    assert target.read_text(encoding="utf-8") == competitor_payload
    assert not list(tmp_path.glob("*.tmp"))

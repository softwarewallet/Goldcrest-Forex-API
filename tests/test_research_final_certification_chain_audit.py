from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.research_final_certification_chain_audit import (
    _write_report_no_clobber,
    audit_final_certification_chain,
    main,
)


FLAGS = {
    "actionable": False,
    "inferential": False,
    "statistical_significance_claimed": False,
    "automatic_model_selection": False,
    "model_training_performed": False,
    "execution_connected": False,
    "automatic_repair_performed": False,
    "deployment_authorized": False,
}


def fixtures():
    readiness = {
        "audit_name": "research_reproducibility_readiness",
        "status": "PASS",
        "plan_id": "plan-123",
        "plan_sha256": "a" * 64,
        "registry_id": "registry-456",
        "errors": [],
        "warnings": [],
        **FLAGS,
    }
    bundle = {
        "audit_name": "research_evaluation_certification_evidence_bundle",
        "status": "PASS",
        "plan_id": "plan-123",
        "plan_sha256": "a" * 64,
        "registry_id": "registry-456",
        "bundle_sha256": "b" * 64,
        "report_count": 6,
        "errors": [],
        **FLAGS,
    }
    verification = {
        "audit_name": "research_evaluation_certification_evidence_bundle_verification",
        "status": "PASS",
        "readiness": "EVIDENCE_BUNDLE_CONTENT_VERIFIED",
        "bundle_sha256": "b" * 64,
        "reports_declared": 6,
        "reports_verified": 6,
        "errors": [],
        **FLAGS,
    }
    return readiness, bundle, verification


def test_complete_chain_passes():
    report = audit_final_certification_chain(*fixtures())
    assert report["status"] == "PASS"
    assert report["readiness"] == "FINAL_CERTIFICATION_CHAIN_VERIFIED"
    assert report["reports_verified"] == 6
    assert report["deployment_authorized"] is False


def test_plan_identity_mismatch_fails():
    readiness, bundle, verification = fixtures()
    bundle["plan_sha256"] = "c" * 64
    report = audit_final_certification_chain(readiness, bundle, verification)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "plan_sha256_mismatch_between_readiness_and_bundle" for e in report["errors"])


def test_verification_must_reference_exact_bundle_digest():
    readiness, bundle, verification = fixtures()
    verification["bundle_sha256"] = "d" * 64
    report = audit_final_certification_chain(readiness, bundle, verification)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "verification_bundle_digest_mismatch" for e in report["errors"])


def test_incomplete_bundle_verification_fails():
    readiness, bundle, verification = fixtures()
    verification["reports_verified"] = 5
    report = audit_final_certification_chain(readiness, bundle, verification)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "not_all_bundle_reports_verified" for e in report["errors"])


def test_safety_flags_must_be_explicitly_false():
    readiness, bundle, verification = fixtures()
    del readiness["execution_connected"]
    report = audit_final_certification_chain(readiness, bundle, verification)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "safety_flag_not_explicitly_false" and e["report"] == "readiness" for e in report["errors"])


def test_readiness_warnings_are_preserved():
    readiness, bundle, verification = fixtures()
    readiness["status"] = "PASS_WITH_WARNINGS"
    report = audit_final_certification_chain(readiness, bundle, verification)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert "readiness_passed_with_warnings" in report["warnings"]
    assert "reproducibility_readiness_has_warnings" in report["warnings"]


def test_upstream_error_list_blocks_chain():
    readiness, bundle, verification = fixtures()
    verification["errors"] = [{"code": "tampered"}]
    report = audit_final_certification_chain(readiness, bundle, verification)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "input_report_contains_errors" and e["report"] == "verification" for e in report["errors"])


def test_digest_fields_require_lowercase_sha256_hex():
    readiness, bundle, verification = fixtures()
    bundle["bundle_sha256"] = "g" * 64
    verification["bundle_sha256"] = "g" * 64

    report = audit_final_certification_chain(readiness, bundle, verification)

    assert report["status"] == "FAIL"
    assert any(error["code"] == "bundle_digest_invalid" for error in report["errors"])


def test_plan_identity_fields_must_be_present_and_well_formed():
    readiness, bundle, verification = fixtures()
    readiness["plan_id"] = None
    readiness["plan_sha256"] = "g" * 64
    bundle["plan_id"] = None
    bundle["plan_sha256"] = "g" * 64

    report = audit_final_certification_chain(readiness, bundle, verification)

    assert report["status"] == "FAIL"
    assert any(error["code"] == "plan_id_invalid" for error in report["errors"])
    assert any(error["code"] == "plan_digest_invalid" for error in report["errors"])


def test_boolean_values_are_not_accepted_as_report_counts():
    readiness, bundle, verification = fixtures()
    bundle["report_count"] = True
    verification["reports_declared"] = True
    verification["reports_verified"] = True

    report = audit_final_certification_chain(readiness, bundle, verification)

    assert report["status"] == "FAIL"
    codes = {error["code"] for error in report["errors"]}
    assert "bundle_report_count_invalid" in codes
    assert "verified_declared_report_count_invalid" in codes
    assert "verified_report_count_invalid" in codes


def test_diagnostic_fields_must_be_lists():
    readiness, bundle, verification = fixtures()
    readiness["warnings"] = "not-a-list"

    report = audit_final_certification_chain(readiness, bundle, verification)

    assert report["status"] == "FAIL"
    assert any(error["code"] == "input_warnings_schema_invalid" for error in report["errors"])


def test_missing_plan_and_registry_identity_cannot_pass_by_matching_nulls():
    readiness, bundle, verification = fixtures()
    readiness.pop("plan_id")
    bundle.pop("plan_id")
    readiness.pop("registry_id")
    bundle.pop("registry_id")

    report = audit_final_certification_chain(readiness, bundle, verification)

    assert report["status"] == "FAIL"
    assert any(error["code"] == "plan_id_invalid" for error in report["errors"])
    assert any(error["code"] == "registry_id_invalid" for error in report["errors"])


def _write_cli_inputs(tmp_path):
    readiness, bundle, verification = fixtures()
    paths = [
        tmp_path / "readiness.json",
        tmp_path / "bundle.json",
        tmp_path / "verification.json",
    ]
    for path, payload in zip(paths, (readiness, bundle, verification)):
        path.write_text(json.dumps(payload), encoding="utf-8")
    return paths


def test_final_chain_cli_refuses_output_aliasing_an_input(tmp_path, capsys):
    readiness, bundle, verification = _write_cli_inputs(tmp_path)
    original = readiness.read_bytes()

    result = main([
        "--readiness-report", str(readiness),
        "--evidence-bundle", str(bundle),
        "--bundle-verification-report", str(verification),
        "--output", str(readiness),
    ])

    assert result == 2
    assert readiness.read_bytes() == original
    assert "output_must_not_alias_input" in capsys.readouterr().err


def test_final_chain_cli_preserves_existing_output(tmp_path, capsys):
    readiness, bundle, verification = _write_cli_inputs(tmp_path)
    output = tmp_path / "final.json"
    output.write_text("preserve", encoding="utf-8")

    result = main([
        "--readiness-report", str(readiness),
        "--evidence-bundle", str(bundle),
        "--bundle-verification-report", str(verification),
        "--output", str(output),
    ])

    assert result == 2
    assert output.read_text(encoding="utf-8") == "preserve"
    assert "output_already_exists" in capsys.readouterr().err


def test_final_chain_report_publication_does_not_clobber_concurrent_output(tmp_path, monkeypatch):
    import app.research_final_certification_chain_audit as module

    output = tmp_path / "final.json"
    real_link = module.os.link

    def create_destination_then_link(source, destination):
        Path(destination).write_text("created concurrently", encoding="utf-8")
        return real_link(source, destination)

    monkeypatch.setattr(module.os, "link", create_destination_then_link)
    with pytest.raises(FileExistsError):
        _write_report_no_clobber(output, {"status": "PASS"}, [])
    assert output.read_text(encoding="utf-8") == "created concurrently"
    assert list(tmp_path.glob(".final.json.*.tmp")) == []

def test_final_certification_reader_rejects_duplicate_json_keys(tmp_path):
    from app.research_final_certification_chain_audit import _read_object

    report_path = tmp_path / "readiness.json"
    report_path.write_text(
        '{"audit_name":"research_reproducibility_readiness",'
        '"audit_name":"shadowed-name","status":"PASS"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate_json_key:audit_name"):
        _read_object(report_path)

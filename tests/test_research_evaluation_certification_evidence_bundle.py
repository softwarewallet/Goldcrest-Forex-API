from __future__ import annotations

import json

import pytest

from app.research_evaluation_certification_evidence_bundle import (
    REPORT_SPECS,
    build_evidence_bundle,
    verify_evidence_bundle,
)


FALSE_FLAGS = {
    "actionable": False,
    "inferential": False,
    "statistical_significance_claimed": False,
    "automatic_model_selection": False,
    "model_training_performed": False,
    "execution_connected": False,
    "automatic_repair_performed": False,
    "deployment_authorized": False,
}


def fixture(tmp_path):
    reports = {}
    for role, (audit_name, _) in REPORT_SPECS.items():
        payload = {
            "audit_name": audit_name,
            "status": "PASS",
            "plan_id": "plan-1",
            "plan_sha256": "a" * 64,
            "registry_id": "registry-1",
            "errors": [],
            **FALSE_FLAGS,
        }
        if role == "plan":
            payload.pop("deployment_authorized")
            payload["expected_runs"] = 2
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        reports[role] = path
    return reports


def test_build_and_verify_bundle(tmp_path):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    assert bundle["status"] == "PASS"
    assert len(bundle["bundle_sha256"]) == 64
    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)
    assert report["status"] == "PASS"
    assert report["reports_declared"] == len(REPORT_SPECS)
    assert report["reports_verified"] == len(REPORT_SPECS)
    assert report["deployment_authorized"] is False


def test_tampered_report_fails_verification(tmp_path):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    paths["metric_semantics"].write_text('{"audit_name":"tampered"}', encoding="utf-8")
    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "bundle_report_sha256_mismatch" and e["role"] == "metric_semantics" for e in report["errors"])


def test_tampered_bundle_digest_fails(tmp_path):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    bundle["registry_id"] = "other"
    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "bundle_sha256_mismatch" for e in report["errors"])


def test_inconsistent_plan_identity_refused_at_build(tmp_path):
    paths = fixture(tmp_path)
    payload = json.loads(paths["row_window_lineage"].read_text())
    payload["plan_sha256"] = "b" * 64
    paths["row_window_lineage"].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="plan_sha256_mismatch:row_window_lineage"):
        build_evidence_bundle(paths)


def test_missing_required_role_refused(tmp_path):
    paths = fixture(tmp_path)
    paths.pop("registry")
    with pytest.raises(ValueError, match="report_roles_must_match_required_roles"):
        build_evidence_bundle(paths)


def test_unsafe_bundle_path_fails(tmp_path):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    bundle["reports"]["plan"]["path"] = "../plan.json"
    # Bundle digest must also be stale because the inventory was edited.
    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "bundle_sha256_mismatch" for e in report["errors"])
    assert any(e["code"] == "bundle_report_path_invalid" and e["role"] == "plan" for e in report["errors"])


def test_warning_status_is_supported(tmp_path):
    paths = fixture(tmp_path)
    payload = json.loads(paths["registry"].read_text())
    payload["status"] = "PASS_WITH_WARNINGS"
    paths["registry"].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    bundle = build_evidence_bundle(paths)
    assert bundle["status"] == "PASS"
    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)
    assert report["status"] == "PASS"

@pytest.mark.parametrize("raw", [b'{"metric":NaN}', b'{"metric":Infinity}', b'{"metric":-Infinity}', b'{"value":1,"value":2}'])
def test_certification_json_readers_reject_nonstandard_json(tmp_path, raw):
    from app.research_evaluation_certification_evidence_bundle import _read_object
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        _read_object(path)

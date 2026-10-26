from __future__ import annotations

import hashlib
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
            payload.pop("errors")  # deterministic plan reports do not declare an errors array
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
    payload["warnings"] = ["fixture warning"]
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


def test_builder_refuses_upstream_report_with_errors(tmp_path):
    paths = fixture(tmp_path)
    payload = json.loads(paths["metric_semantics"].read_text())
    payload["errors"] = [{"code": "metric_mismatch"}]
    paths["metric_semantics"].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="report_contains_errors:metric_semantics"):
        build_evidence_bundle(paths)


def test_verifier_fails_closed_on_report_errors_even_if_bundle_was_built(tmp_path):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    payload = json.loads(paths["metric_semantics"].read_text())
    payload["errors"] = [{"code": "metric_mismatch"}]
    paths["metric_semantics"].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    from app.research_evaluation_certification_evidence_bundle import _sha, _canonical
    entry = bundle["reports"]["metric_semantics"]
    raw = paths["metric_semantics"].read_bytes()
    entry["sha256"] = _sha(raw)
    entry["size_bytes"] = len(raw)
    entry["identity"] = {
        "audit_name": payload["audit_name"], "status": payload["status"],
        "plan_id": payload["plan_id"], "plan_sha256": payload["plan_sha256"],
        "registry_id": payload["registry_id"], "deployment_authorized": payload.get("deployment_authorized"),
    }
    body = dict(bundle); body.pop("bundle_sha256", None)
    bundle["bundle_sha256"] = _sha(_canonical(body))
    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "bundle_report_contains_errors" and e["role"] == "metric_semantics" for e in report["errors"])

@pytest.mark.parametrize(
    ("role", "field", "error_code"),
    [
        ("registry", "warnings", "report_warnings_schema_invalid:registry"),
        ("plan", "errors", "report_errors_schema_invalid:plan"),
    ],
)
def test_builder_rejects_explicit_null_optional_diagnostics(tmp_path, role, field, error_code):
    paths = fixture(tmp_path)
    payload = json.loads(paths[role].read_text())
    payload[field] = None
    paths[role].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=error_code):
        build_evidence_bundle(paths)


def test_verifier_rejects_explicit_null_warnings_in_upstream_report(tmp_path):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    payload = json.loads(paths["registry"].read_text())
    payload["warnings"] = None
    paths["registry"].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    from app.research_evaluation_certification_evidence_bundle import _sha, _canonical
    entry = bundle["reports"]["registry"]
    raw = paths["registry"].read_bytes()
    entry["sha256"] = _sha(raw)
    entry["size_bytes"] = len(raw)
    entry["identity"] = {
        "audit_name": payload["audit_name"], "status": payload["status"],
        "plan_id": payload["plan_id"], "plan_sha256": payload["plan_sha256"],
        "registry_id": payload["registry_id"], "deployment_authorized": payload.get("deployment_authorized"),
    }
    body = dict(bundle)
    body.pop("bundle_sha256", None)
    bundle["bundle_sha256"] = _sha(_canonical(body))

    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "bundle_report_warnings_schema_invalid" and error["role"] == "registry"
        for error in report["errors"]
    )



def _refresh_bundle_digest(bundle):
    body = dict(bundle)
    body.pop("bundle_sha256", None)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    bundle["bundle_sha256"] = hashlib.sha256(canonical).hexdigest()


def test_verifier_binds_top_level_plan_identity_to_plan_report(tmp_path):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    bundle["plan_id"] = "different-plan"
    _refresh_bundle_digest(bundle)

    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)

    assert report["status"] == "FAIL"
    assert any(item["code"] == "bundle_report_plan_binding_mismatch" and item["role"] == "plan" for item in report["errors"])


def test_verifier_binds_top_level_registry_identity_to_registry_report(tmp_path):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    bundle["registry_id"] = "different-registry"
    _refresh_bundle_digest(bundle)

    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)

    assert report["status"] == "FAIL"
    assert any(item["code"] == "bundle_report_registry_binding_mismatch" and item["role"] == "registry" for item in report["errors"])


def test_builder_rejects_invalid_plan_identity(tmp_path):
    paths = fixture(tmp_path)
    plan_path = paths["plan"]
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["plan_sha256"] = "not-a-digest"
    plan_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="plan_sha256_invalid:plan"):
        build_evidence_bundle(paths)


def test_builder_rejects_missing_registry_identity(tmp_path):
    paths = fixture(tmp_path)
    registry_path = paths["registry"]
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["registry_id"] = "   "
    registry_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="registry_id_invalid:registry"):
        build_evidence_bundle(paths)


@pytest.mark.parametrize(
    ("role", "field", "value", "error_code"),
    [
        ("plan", "plan_id", " plan-1 ", "plan_id_invalid:plan"),
        ("registry", "registry_id", " registry-1 ", "registry_id_invalid:registry"),
    ],
)
def test_builder_rejects_whitespace_padded_identity_strings(tmp_path, role, field, value, error_code):
    paths = fixture(tmp_path)
    payload = json.loads(paths[role].read_text(encoding="utf-8"))
    payload[field] = value
    paths[role].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=error_code):
        build_evidence_bundle(paths)


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("plan_id", " plan-1 ", "bundle_plan_id_invalid"),
        ("registry_id", " registry-1 ", "bundle_registry_id_invalid"),
    ],
)
def test_verifier_rejects_whitespace_padded_top_level_identities(tmp_path, field, value, error_code):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    bundle[field] = value
    _refresh_bundle_digest(bundle)

    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)

    assert report["status"] == "FAIL"
    assert any(error["code"] == error_code for error in report["errors"])



def test_builder_rejects_status_warning_mismatch(tmp_path):
    paths = fixture(tmp_path)
    payload = json.loads(paths["registry"].read_text())
    payload["status"] = "PASS_WITH_WARNINGS"
    payload.pop("warnings", None)
    paths["registry"].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="report_status_warning_mismatch:registry"):
        build_evidence_bundle(paths)


def test_verifier_rejects_status_warning_mismatch_after_bundle_reseal(tmp_path):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    payload = json.loads(paths["registry"].read_text())
    payload["status"] = "PASS_WITH_WARNINGS"
    payload.pop("warnings", None)
    paths["registry"].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    from app.research_evaluation_certification_evidence_bundle import _sha, _canonical
    entry = bundle["reports"]["registry"]
    raw = paths["registry"].read_bytes()
    entry["sha256"] = _sha(raw)
    entry["size_bytes"] = len(raw)
    entry["identity"] = {
        "audit_name": payload["audit_name"], "status": payload["status"],
        "plan_id": payload["plan_id"], "plan_sha256": payload["plan_sha256"],
        "registry_id": payload["registry_id"], "deployment_authorized": payload.get("deployment_authorized"),
    }
    body = dict(bundle)
    body.pop("bundle_sha256", None)
    bundle["bundle_sha256"] = _sha(_canonical(body))

    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(
        item["code"] == "bundle_report_status_warning_mismatch"
        and item["role"] == "registry"
        for item in report["errors"]
    )


def test_builder_rejects_unhashable_report_status(tmp_path):
    paths = fixture(tmp_path)
    payload = json.loads(paths["registry"].read_text(encoding="utf-8"))
    payload["status"] = ["PASS"]
    paths["registry"].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="report_not_passing:registry"):
        build_evidence_bundle(paths)


def test_verifier_rejects_unhashable_report_status_without_raising(tmp_path):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    payload = json.loads(paths["registry"].read_text(encoding="utf-8"))
    payload["status"] = {"value": "PASS"}
    paths["registry"].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    from app.research_evaluation_certification_evidence_bundle import _sha, _canonical
    entry = bundle["reports"]["registry"]
    raw = paths["registry"].read_bytes()
    entry["sha256"] = _sha(raw)
    entry["size_bytes"] = len(raw)
    entry["identity"] = {
        "audit_name": payload["audit_name"], "status": payload["status"],
        "plan_id": payload["plan_id"], "plan_sha256": payload["plan_sha256"],
        "registry_id": payload["registry_id"], "deployment_authorized": payload.get("deployment_authorized"),
    }
    _refresh_bundle_digest(bundle)

    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "bundle_report_identity_or_status_mismatch" and error["role"] == "registry"
        for error in report["errors"]
    )


@pytest.mark.parametrize(
    ("declared_count", "expected_code"),
    [
        (True, "bundle_report_count_invalid"),
        ("6", "bundle_report_count_invalid"),
        (0, "bundle_report_count_invalid"),
        (5, "bundle_report_count_mismatch"),
        (7, "bundle_report_count_mismatch"),
    ],
)
def test_verifier_rejects_invalid_or_inconsistent_declared_report_count(
    tmp_path, declared_count, expected_code
):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    bundle["report_count"] = declared_count
    _refresh_bundle_digest(bundle)

    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)

    assert report["status"] == "FAIL"
    assert any(error["code"] == expected_code for error in report["errors"])



@pytest.mark.parametrize("bad_path", ["../plan.json", "nested/plan.json", " plan.json ", r"nested\\plan.json", ".", ".."])
def test_verifier_rejects_noncanonical_report_basenames(tmp_path, bad_path):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    bundle["reports"]["plan"]["path"] = bad_path
    _refresh_bundle_digest(bundle)

    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "bundle_report_path_invalid" and error["role"] == "plan"
        for error in report["errors"]
    )


def test_builder_rejects_symlinked_report_input(tmp_path):
    paths = fixture(tmp_path)
    alias = tmp_path / "plan-alias.json"
    alias.symlink_to(paths["plan"].name)
    paths["plan"] = alias

    with pytest.raises(ValueError, match="report_symlink_not_allowed"):
        build_evidence_bundle(paths)


def test_verifier_rejects_symlinked_report_even_when_target_is_inside_root(tmp_path):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    alias = tmp_path / "plan-alias.json"
    alias.symlink_to(paths["plan"].name)
    bundle["reports"]["plan"]["path"] = alias.name
    _refresh_bundle_digest(bundle)

    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "bundle_report_symlink_not_allowed" and error["role"] == "plan"
        for error in report["errors"]
    )



@pytest.mark.parametrize(
    ("role", "field", "bad_value", "error"),
    [
        ("plan", "plan_id", "../plan", "plan_id_invalid:plan"),
        ("plan", "plan_id", "métrics", "plan_id_invalid:plan"),
        ("plan", "plan_id", "x" * 129, "plan_id_invalid:plan"),
        ("registry", "registry_id", "registry/name", "registry_id_invalid:registry"),
        ("registry", "registry_id", " registry-1 ", "registry_id_invalid:registry"),
        ("registry", "registry_id", "x" * 129, "registry_id_invalid:registry"),
    ],
)
def test_builder_rejects_noncanonical_plan_and_registry_ids(tmp_path, role, field, bad_value, error):
    paths = fixture(tmp_path)
    payload = json.loads(paths[role].read_text(encoding="utf-8"))
    payload[field] = bad_value
    paths[role].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        build_evidence_bundle(paths)


@pytest.mark.parametrize(
    ("field", "bad_value", "error_code"),
    [
        ("plan_id", "../plan", "bundle_plan_id_invalid"),
        ("plan_id", "métrics", "bundle_plan_id_invalid"),
        ("plan_id", "x" * 129, "bundle_plan_id_invalid"),
        ("registry_id", "registry/name", "bundle_registry_id_invalid"),
        ("registry_id", "x" * 129, "bundle_registry_id_invalid"),
    ],
)
def test_verifier_rejects_noncanonical_top_level_identity_values(tmp_path, field, bad_value, error_code):
    paths = fixture(tmp_path)
    bundle = build_evidence_bundle(paths)
    bundle[field] = bad_value
    _refresh_bundle_digest(bundle)

    report = verify_evidence_bundle(bundle, evidence_root=tmp_path)

    assert report["status"] == "FAIL"
    assert any(error["code"] == error_code for error in report["errors"])

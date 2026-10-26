from __future__ import annotations

from copy import deepcopy

import pytest

from app.research_freshness_certification_gate import gate_freshness_certification, _read_object

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


def freshness():
    return {
        "audit_name": "research_artifact_freshness_audit",
        "status": "PASS",
        "artifact_count": 2,
        "current_count": 2,
        "regeneration_required_count": 0,
        "artifacts": [
            {"artifact_id": "plan", "classification": "CURRENT", "needs_regeneration": False},
            {"artifact_id": "metrics", "classification": "CURRENT", "needs_regeneration": False},
        ],
        "errors": [],
        **FLAGS,
    }


def certification(status="PASS"):
    return {
        "audit_name": "research_final_certification_chain",
        "status": status,
        "errors": [],
        "warnings": ["fixture warning"] if status == "PASS_WITH_WARNINGS" else [],
        **FLAGS,
    }


def test_current_artifacts_and_passing_certification_pass_gate():
    report = gate_freshness_certification(freshness(), certification())
    assert report["status"] == "PASS"
    assert report["readiness"] == "CERTIFIED_AND_CURRENT"
    assert report["deployment_authorized"] is False


def test_stale_artifact_blocks_final_certification():
    stale = freshness()
    stale["status"] = "STALE"
    stale["current_count"] = 1
    stale["regeneration_required_count"] = 1
    stale["artifacts"][1]["classification"] = "REGENERATION_REQUIRED"
    stale["artifacts"][1]["needs_regeneration"] = True
    report = gate_freshness_certification(stale, certification())
    assert report["status"] == "FAIL"
    codes = {item["code"] for item in report["errors"]}
    assert "input_report_not_passing" in codes
    assert "research_artifacts_require_regeneration" in codes
    assert "freshness_inventory_contains_noncurrent_artifact" in codes


def test_failed_final_certification_blocks_gate():
    report = gate_freshness_certification(freshness(), certification("FAIL"))
    assert report["status"] == "FAIL"
    assert any(item["code"] == "input_report_not_passing" for item in report["errors"])


def test_warning_certification_propagates_warning():
    report = gate_freshness_certification(freshness(), certification("PASS_WITH_WARNINGS"))
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert "final_certification_chain_has_warnings" in report["warnings"]


def test_missing_safety_flag_fails_closed():
    unsafe = freshness()
    unsafe["deployment_authorized"] = True
    report = gate_freshness_certification(unsafe, certification())
    assert report["status"] == "FAIL"
    assert any(item["code"] == "safety_flag_not_explicitly_false" for item in report["errors"])


def test_inconsistent_inventory_counts_fail_closed():
    malformed = freshness()
    malformed["artifact_count"] = 3
    report = gate_freshness_certification(malformed, certification())
    assert report["status"] == "FAIL"
    assert any(item["code"] == "freshness_artifact_counts_inconsistent" for item in report["errors"])


def test_inputs_are_not_mutated():
    fresh = freshness()
    final = certification()
    before_fresh, before_final = deepcopy(fresh), deepcopy(final)
    gate_freshness_certification(fresh, final)
    assert fresh == before_fresh
    assert final == before_final


@pytest.mark.parametrize("raw", [
    b'{"status":"PASS","status":"FAIL"}',
    b'{"metric":NaN}',
    b'{"metric":Infinity}',
    b'{"metric":-Infinity}',
])
def test_json_reader_rejects_duplicate_keys_and_nonstandard_constants(tmp_path, raw):
    path = tmp_path / "malformed.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        _read_object(path)

def test_missing_errors_array_fails_closed():
    fresh = freshness()
    fresh.pop("errors")
    report = gate_freshness_certification(fresh, certification())
    assert report["status"] == "FAIL"
    assert any(
        item["code"] == "input_errors_schema_invalid" and item["report"] == "freshness"
        for item in report["errors"]
    )


def test_malformed_empty_errors_value_fails_closed():
    cert = certification()
    cert["errors"] = {}
    report = gate_freshness_certification(freshness(), cert)
    assert report["status"] == "FAIL"
    assert any(
        item["code"] == "input_errors_schema_invalid" and item["report"] == "certification"
        for item in report["errors"]
    )


def test_malformed_warnings_value_fails_closed():
    fresh = freshness()
    fresh["warnings"] = {"unexpected": "object"}
    report = gate_freshness_certification(fresh, certification())
    assert report["status"] == "FAIL"
    assert any(item["code"] == "input_warnings_schema_invalid" for item in report["errors"])

def test_explicit_null_warnings_fails_closed():
    fresh = freshness()
    fresh["warnings"] = None
    report = gate_freshness_certification(fresh, certification())
    assert report["status"] == "FAIL"
    assert any(item["code"] == "input_warnings_schema_invalid" for item in report["errors"])



@pytest.mark.parametrize(
    ("report_name", "bad_status"),
    [("freshness", ["PASS"]), ("certification", {"value": "PASS"})],
)
def test_unhashable_status_values_fail_closed(report_name, bad_status):
    fresh = freshness()
    cert = certification()
    (fresh if report_name == "freshness" else cert)["status"] = bad_status

    report = gate_freshness_certification(fresh, cert)

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "input_report_not_passing" and error["report"] == report_name
        for error in report["errors"]
    )


@pytest.mark.parametrize(
    ("status", "warnings"),
    [("PASS", ["unexpected warning"]), ("PASS_WITH_WARNINGS", [])],
)
def test_gate_rejects_status_warning_mismatch(status, warnings):
    fresh = freshness()
    fresh["status"] = status
    fresh["warnings"] = warnings

    report = gate_freshness_certification(fresh, certification())

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "input_status_warning_mismatch" and error["report"] == "freshness"
        for error in report["errors"]
    )


def test_gate_rejects_certification_pass_with_warnings_but_no_warning_status():
    cert = certification("PASS")
    cert["warnings"] = ["unexpected warning"]

    report = gate_freshness_certification(freshness(), cert)

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "input_status_warning_mismatch" and error["report"] == "certification"
        for error in report["errors"]
    )


def test_gate_requires_warning_for_pass_with_warnings_certification():
    cert = certification("PASS_WITH_WARNINGS")
    cert["warnings"] = []

    report = gate_freshness_certification(freshness(), cert)

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "input_status_warning_mismatch" and error["report"] == "certification"
        for error in report["errors"]
    )

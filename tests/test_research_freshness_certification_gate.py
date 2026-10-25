from __future__ import annotations

from copy import deepcopy

from app.research_freshness_certification_gate import gate_freshness_certification

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

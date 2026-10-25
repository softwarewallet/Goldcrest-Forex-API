from __future__ import annotations

import hashlib

from app.research_reproducibility_readiness_audit import audit_reproducibility_readiness


FALSE_FLAGS = {
    "actionable": False,
    "inferential": False,
    "statistical_significance_claimed": False,
    "execution_connected": False,
    "model_training_performed": False,
    "automatic_model_selection": False,
    "automatic_repair_performed": False,
}


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def reports():
    dataset = sha(b"dataset")
    config = sha(b"config")
    commit = "a" * 40
    plan = {
        "audit_name": "research_experiment_plan",
        "status": "PASS",
        "plan_id": "plan-a",
        "plan_sha256": sha(b"plan"),
        "expected_runs": 2,
        "dataset_sha256": dataset,
        "config_sha256": config,
        "code_commit_sha": commit,
        **FALSE_FLAGS,
    }
    registry = {
        "audit_name": "research_experiment_registry",
        "status": "PASS",
        "registry_id": "registry-a",
        "runs_supplied": 2,
        "runs_validated": 2,
        "run_results": [
            {"run_id": "r1", "status": "PASS", "dataset_sha256": dataset, "config_sha256": config, "code_commit_sha": commit},
            {"run_id": "r2", "status": "PASS", "dataset_sha256": dataset, "config_sha256": config, "code_commit_sha": commit},
        ],
        **FALSE_FLAGS,
    }
    reconciliation = {
        "audit_name": "research_planned_observed_reconciliation",
        "status": "PASS",
        "plan_id": "plan-a",
        "plan_sha256": plan["plan_sha256"],
        "registry_id": "registry-a",
        "planned_runs": 2,
        "observed_runs": 2,
        "matched_runs": 2,
        "missing_run_ids": [],
        "errors": [],
        **FALSE_FLAGS,
    }
    evidence = {
        "audit_name": "research_evidence_pack",
        "status": "PASS",
        "pack_id": "pack-a",
        "errors": [],
        "dataset_fingerprints_reconciled": [dataset],
        **FALSE_FLAGS,
    }
    return plan, registry, reconciliation, evidence


def test_complete_reproducibility_chain_passes():
    report = audit_reproducibility_readiness(*reports())
    assert report["status"] == "PASS"
    assert report["readiness"] == "REPRODUCIBILITY_EVIDENCE_RECONCILED"
    assert report["matched_runs"] == 2
    assert report["deployment_authorized"] is False


def test_plan_fingerprint_mismatch_fails_closed():
    plan, registry, reconciliation, evidence = reports()
    reconciliation["plan_sha256"] = sha(b"other-plan")
    report = audit_reproducibility_readiness(plan, registry, reconciliation, evidence)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "reconciliation_plan_fingerprint_mismatch" for item in report["errors"])


def test_run_provenance_mismatch_fails_closed():
    plan, registry, reconciliation, evidence = reports()
    registry["run_results"][0]["dataset_sha256"] = sha(b"wrong-dataset")
    report = audit_reproducibility_readiness(plan, registry, reconciliation, evidence)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "run_provenance_mismatch" for item in report["errors"])


def test_missing_run_provenance_fails_closed():
    plan, registry, reconciliation, evidence = reports()
    del registry["run_results"][0]["code_commit_sha"]
    report = audit_reproducibility_readiness(plan, registry, reconciliation, evidence)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "run_provenance_missing" for item in report["errors"])


def test_upstream_warning_is_preserved():
    plan, registry, reconciliation, evidence = reports()
    registry["status"] = "PASS_WITH_WARNINGS"
    report = audit_reproducibility_readiness(plan, registry, reconciliation, evidence)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert "registry_passed_with_warnings" in report["warnings"]


def test_unsafe_upstream_flag_fails_closed():
    plan, registry, reconciliation, evidence = reports()
    evidence["execution_connected"] = True
    report = audit_reproducibility_readiness(plan, registry, reconciliation, evidence)
    assert report["status"] == "FAIL"
    assert any(
        item["code"] == "safety_flag_not_explicitly_false"
        and item["report"] == "evidence_pack"
        and item["flag"] == "execution_connected"
        for item in report["errors"]
    )

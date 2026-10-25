from __future__ import annotations

import hashlib

from app.research_unified_provenance_certification import run_unified_certification
from app.research_provenance_manifest_seal import create_manifest_seal

REV = "a" * 40
CONFIG_HASH = hashlib.sha256(b"config").hexdigest()
DATASET_HASH = hashlib.sha256(b"dataset").hexdigest()
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


def manifest(revision=REV):
    return {
        "audit_name": "research_artifact_provenance_manifest",
        "source_revision": revision,
        "config": {"sha256": CONFIG_HASH},
        "dataset": {"sha256": DATASET_HASH},
        "artifact_count": 1,
        "artifacts": [{
            "artifact_id": "metrics",
            "path": "metrics.csv",
            "sha256": "c" * 64,
            "size_bytes": 8,
            "kind": "file",
            "source_revision": revision,
            "config_sha256": CONFIG_HASH,
            "dataset_sha256": DATASET_HASH,
            "depends_on": [],
        }],
        **FLAGS,
    }


def call(monkeypatch, tmp_path, provenance=None, gate_status="PASS", repo_path=None):
    import app.research_unified_provenance_certification as module

    provenance_report = {
        "status": "PASS",
        "artifact_count": 1,
        "current_count": 1,
        "stale_or_invalid_count": 0,
        "errors": [],
        **FLAGS,
    }
    if provenance and provenance.get("source_revision") != REV:
        provenance_report["status"] = "FAIL"
        provenance_report["stale_or_invalid_count"] = 1
    monkeypatch.setattr(module, "verify_provenance", lambda *a, **k: provenance_report)
    monkeypatch.setattr(module, "fingerprint_path", lambda path: CONFIG_HASH if str(path).endswith("config.json") else DATASET_HASH)
    monkeypatch.setattr(module, "audit_artifact_freshness", lambda *a, **k: {
        "audit_name": "research_artifact_freshness_audit",
        "status": "PASS" if gate_status != "FAIL" else "STALE",
        "artifact_count": 1,
        "current_count": 1 if gate_status != "FAIL" else 0,
        "regeneration_required_count": 0 if gate_status != "FAIL" else 1,
        "artifacts": [{"artifact_id": "metrics", "classification": "CURRENT" if gate_status != "FAIL" else "REGENERATION_REQUIRED", "needs_regeneration": gate_status == "FAIL"}],
        "errors": [],
        **FLAGS,
    })
    monkeypatch.setattr(module, "run_final_certification", lambda *a, **k: (
        {"audit_name": "evidence_bundle_verification", "status": "PASS", "reports_verified": 6, "errors": [], **FLAGS},
        {"audit_name": "research_final_certification_chain", "status": "PASS", "readiness": "FINAL_CERTIFICATION_CHAIN_VERIFIED", "errors": [], **FLAGS},
    ))
    monkeypatch.setattr(module, "gate_freshness_certification", lambda fresh, cert: {
        "status": gate_status,
        "errors": [] if gate_status == "PASS" else [{"code": "gate_failed"}],
        "warnings": [],
    })
    selected_manifest = provenance or manifest()
    return module.run_unified_certification(
        {"audit_name": "readiness"}, {"audit_name": "bundle"}, selected_manifest,
        provenance_seal=create_manifest_seal(selected_manifest),
        evidence_root=tmp_path, artifact_root=tmp_path,
        current_source_revision=REV, repo_path=repo_path, config_path=tmp_path / "config.json",
        dataset_path=tmp_path / "dataset",
    )


def test_unified_certification_passes_when_all_components_pass(monkeypatch, tmp_path):
    report = call(monkeypatch, tmp_path)
    assert report["status"] == "PASS"
    assert report["readiness"] == "CERTIFIED_CURRENT"
    assert report["provenance_verification"]["status"] == "PASS"
    assert report["provenance_manifest_seal_verification"]["status"] == "PASS"
    assert report["freshness_audit"]["status"] == "PASS"
    assert report["deployment_authorized"] is False


def test_provenance_failure_blocks_overall_certification(monkeypatch, tmp_path):
    report = call(monkeypatch, tmp_path, provenance=manifest(revision="b" * 40))
    assert report["status"] == "FAIL"
    assert any(item["code"] == "PROVENANCE_VERIFICATION_FAILED" for item in report["errors"])


def test_freshness_gate_failure_blocks_overall_certification(monkeypatch, tmp_path):
    report = call(monkeypatch, tmp_path, gate_status="FAIL")
    assert report["status"] == "FAIL"
    assert any(item["code"] == "FRESHNESS_CERTIFICATION_GATE_FAILED" for item in report["errors"])


def test_unified_report_retains_all_component_reports(monkeypatch, tmp_path):
    report = call(monkeypatch, tmp_path)
    assert {
        "provenance_verification", "bundle_verification", "final_certification_chain",
        "freshness_audit", "freshness_certification_gate",
    }.issubset(report)


def test_git_clean_tree_revision_is_used_and_reported(monkeypatch, tmp_path):
    import app.research_unified_provenance_certification as module

    git_report = {
        "audit_name": "research_git_source_provenance",
        "status": "PASS",
        "source_revision": REV,
        "working_tree_clean": True,
        "dirty_path_count": 0,
        "dirty_paths": [],
        "errors": [],
        **FLAGS,
    }
    monkeypatch.setattr(module, "inspect_git_provenance", lambda path: git_report)
    report = call(monkeypatch, tmp_path, repo_path=tmp_path / "repo")
    assert report["status"] == "PASS"
    assert report["source_revision"] == REV
    assert report["git_source_provenance"]["working_tree_clean"] is True


def test_git_dirty_tree_blocks_overall_certification(monkeypatch, tmp_path):
    import app.research_unified_provenance_certification as module

    git_report = {
        "audit_name": "research_git_source_provenance",
        "status": "FAIL",
        "source_revision": REV,
        "working_tree_clean": False,
        "dirty_path_count": 1,
        "dirty_paths": ["app/changed.py"],
        "errors": [{"code": "WORKING_TREE_DIRTY", "dirty_path_count": 1}],
        **FLAGS,
    }
    monkeypatch.setattr(module, "inspect_git_provenance", lambda path: git_report)
    report = call(monkeypatch, tmp_path, repo_path=tmp_path / "repo")
    assert report["status"] == "FAIL"
    assert any(item["code"] == "GIT_SOURCE_PROVENANCE_FAILED" for item in report["errors"])
    assert report["git_source_provenance"]["dirty_paths"] == ["app/changed.py"]
    assert report["deployment_authorized"] is False


def test_git_inspection_error_blocks_overall_certification(monkeypatch, tmp_path):
    import app.research_unified_provenance_certification as module

    def fail(path):
        raise ValueError("not_a_git_repository")

    monkeypatch.setattr(module, "inspect_git_provenance", fail)
    report = call(monkeypatch, tmp_path, repo_path=tmp_path / "not-a-repo")
    assert report["status"] == "FAIL"
    assert any(item["code"] == "GIT_SOURCE_PROVENANCE_FAILED" for item in report["errors"])
    assert report["git_source_provenance"]["status"] == "NOT_CHECKED"


def test_missing_provenance_seal_blocks_certification(monkeypatch, tmp_path):
    import app.research_unified_provenance_certification as module
    from app.research_provenance_manifest_seal import create_manifest_seal

    selected_manifest = manifest()
    report = module.run_unified_certification(
        {"audit_name": "readiness"}, {"audit_name": "bundle"}, selected_manifest,
        evidence_root=tmp_path, artifact_root=tmp_path, current_source_revision=REV,
        config_path=tmp_path / "config.json", dataset_path=tmp_path / "dataset",
    )
    assert report["status"] == "FAIL"
    assert report["provenance_manifest_seal_verification"]["status"] == "FAIL"
    assert any(item["code"] == "PROVENANCE_MANIFEST_SEAL_FAILED" for item in report["errors"])

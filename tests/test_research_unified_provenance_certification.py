from __future__ import annotations

import hashlib

from app.research_unified_provenance_certification import run_unified_certification
from app.research_provenance_manifest_seal import create_manifest_seal
from app.research_artifact_inventory_spec_seal import create_inventory_spec_seal

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


def call(monkeypatch, tmp_path, provenance=None, gate_status="PASS", repo_path=None, include_seal=True, inventory_override=None, include_inventory_seal=True, fingerprint_error=False, final_certification_error=False, provenance_error=False):
    import app.research_unified_provenance_certification as module

    if repo_path is None:
        repo_path = tmp_path / "repo"
        monkeypatch.setattr(module, "inspect_git_provenance", lambda path: {
            "audit_name": "research_git_source_provenance",
            "status": "PASS",
            "source_revision": REV,
            "working_tree_clean": True,
            "dirty_path_count": 0,
            "dirty_paths": [],
            "errors": [],
            **FLAGS,
        })

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
    if provenance_error:
        monkeypatch.setattr(module, "verify_provenance", lambda *a, **k: (_ for _ in ()).throw(ValueError("invalid_provenance_manifest")))
    else:
        monkeypatch.setattr(module, "verify_provenance", lambda *a, **k: provenance_report)
    if fingerprint_error:
        monkeypatch.setattr(module, "fingerprint_path", lambda path: (_ for _ in ()).throw(FileNotFoundError("missing_input")) if str(path).endswith("config.json") else DATASET_HASH)
    else:
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
    if final_certification_error:
        monkeypatch.setattr(module, "run_final_certification", lambda *a, **k: (_ for _ in ()).throw(ValueError("invalid_evidence_bundle")))
    else:
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
    inventory_spec = inventory_override or {
        "schema_version": 1,
        "audit_name": "research_artifact_inventory_specification",
        "required_artifacts": [{"artifact_id": item["artifact_id"], "path": item["path"]} for item in selected_manifest["artifacts"]],
    }
    return module.run_unified_certification(
        {"audit_name": "readiness"}, {"audit_name": "bundle"}, selected_manifest,
        provenance_seal=create_manifest_seal(selected_manifest) if include_seal else None,
        inventory_spec=inventory_spec,
        inventory_seal=create_inventory_spec_seal(inventory_spec) if include_inventory_seal else None,
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
    assert report["artifact_inventory_completeness"]["inventory_spec_sha256"] == report["artifact_inventory_spec_seal_verification"]["inventory_spec_sha256"]
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
    assert any(item["code"] == "FRESHNESS_CERTIFICATION_GATE_FAILED" for item in report["errors"])


def test_unified_report_retains_all_component_reports(monkeypatch, tmp_path):
    report = call(monkeypatch, tmp_path)
    assert {
        "artifact_inventory_spec_seal_verification", "artifact_inventory_completeness", "provenance_manifest_seal_verification", "provenance_verification", "bundle_verification", "final_certification_chain",
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
    report = call(monkeypatch, tmp_path, include_seal=False)
    assert report["status"] == "FAIL"
    assert report["provenance_manifest_seal_verification"]["status"] == "FAIL"
    assert any(item["code"] == "PROVENANCE_MANIFEST_SEAL_FAILED" for item in report["errors"])


def test_inventory_incompleteness_blocks_unified_certification(monkeypatch, tmp_path):
    incomplete = {
        "schema_version": 1,
        "audit_name": "research_artifact_inventory_specification",
        "required_artifacts": [{"artifact_id": "metrics", "path": "metrics.csv"},
                               {"artifact_id": "extra", "path": "extra.json"}],
    }
    report = call(monkeypatch, tmp_path, inventory_override=incomplete)
    assert report["status"] == "FAIL"
    assert report["artifact_inventory_completeness"]["status"] == "FAIL"
    assert any(item["code"] == "ARTIFACT_INVENTORY_COMPLETENESS_FAILED" for item in report["errors"])
    assert report["deployment_authorized"] is False


def test_missing_inventory_spec_seal_blocks_certification(monkeypatch, tmp_path):
    report = call(monkeypatch, tmp_path, include_inventory_seal=False)
    assert report["status"] == "FAIL"
    assert report["artifact_inventory_spec_seal_verification"]["status"] == "FAIL"
    assert any(item["code"] == "INVENTORY_SPEC_SEAL_FAILED" for item in report["errors"])


def test_cli_passes_all_seals_and_inventory_spec_to_unified_runner(monkeypatch, tmp_path):
    import sys
    import app.research_unified_provenance_certification as module

    output = tmp_path / "out" / "certification.json"
    monkeypatch.setattr(sys, "argv", [
        "research_unified_provenance_certification",
        "--readiness-report", str(tmp_path / "readiness.json"),
        "--evidence-bundle", str(tmp_path / "bundle.json"),
        "--evidence-root", str(tmp_path),
        "--provenance-manifest", str(tmp_path / "provenance.json"),
        "--provenance-seal", str(tmp_path / "provenance-seal.json"),
        "--inventory-spec", str(tmp_path / "inventory.json"),
        "--inventory-seal", str(tmp_path / "inventory-seal.json"),
        "--artifact-root", str(tmp_path),
        "--repo-path", str(tmp_path),
        "--config-path", str(tmp_path / "config.json"),
        "--dataset-path", str(tmp_path / "dataset"),
        "--output", str(output),
        "--report-seal-output", str(tmp_path / "out" / "certification.seal.json"),
    ])
    import pathlib
    monkeypatch.setattr(module, "_read_object", lambda path: {"source": pathlib.Path(path).name})
    captured = {}
    def fake_runner(*args, **kwargs):
        captured.update(kwargs)
        return {
            "audit_name": "research_unified_provenance_certification",
            "source_revision": REV,
            "status": "PASS", "readiness": "CERTIFIED_CURRENT", "artifact_count": 1,
            "stale_or_invalid_artifact_count": 0,
            "provenance_manifest_seal_verification": {"status": "PASS"},
            "artifact_inventory_completeness": {"status": "PASS"},
            "artifact_inventory_spec_seal_verification": {"status": "PASS"},
            "regeneration_required_count": 0, "errors": [], "deployment_authorized": False,
            **FLAGS,
        }
    monkeypatch.setattr(module, "run_unified_certification", fake_runner)
    assert module.main() == 0
    assert captured["provenance_seal"]["source"] == "provenance-seal.json"
    assert captured["inventory_spec"]["source"] == "inventory.json"
    assert captured["inventory_seal"]["source"] == "inventory-seal.json"
    assert output.is_file()
    seal_path = tmp_path / "out" / "certification.seal.json"
    assert seal_path.is_file()
    import json
    from app.research_certification_report_seal import verify_report_seal
    assert verify_report_seal(json.loads(output.read_text()), json.loads(seal_path.read_text()))["status"] == "PASS"



def test_missing_repo_path_blocks_certification(monkeypatch, tmp_path):
    import app.research_unified_provenance_certification as module

    monkeypatch.setattr(module, "run_final_certification", lambda *a, **k: (
        {"audit_name": "bundle_verification", "status": "PASS", "errors": []},
        {"audit_name": "final_chain", "status": "PASS", "errors": []},
    ))
    selected_manifest = manifest()
    inventory_spec = {
        "schema_version": 1,
        "audit_name": "research_artifact_inventory_specification",
        "required_artifacts": [{"artifact_id": "metrics", "path": "metrics.csv"}],
    }
    report = module.run_unified_certification(
        {"audit_name": "readiness"}, {"audit_name": "bundle"}, selected_manifest,
        provenance_seal=create_manifest_seal(selected_manifest),
        inventory_spec=inventory_spec,
        inventory_seal=create_inventory_spec_seal(inventory_spec),
        evidence_root=tmp_path, artifact_root=tmp_path,
        current_source_revision=REV, repo_path=None,
        config_path=tmp_path / "config.json", dataset_path=tmp_path / "dataset",
    )
    assert report["status"] == "FAIL"
    assert any(item["code"] == "GIT_SOURCE_PROVENANCE_FAILED" for item in report["errors"])
    assert report["git_source_provenance"]["status"] == "NOT_CHECKED"
    assert report["deployment_authorized"] is False


def test_git_report_with_invalid_revision_blocks_certification(monkeypatch, tmp_path):
    import app.research_unified_provenance_certification as module

    monkeypatch.setattr(module, "inspect_git_provenance", lambda path: {
        "audit_name": "research_git_source_provenance",
        "status": "PASS",
        "source_revision": "not-a-sha",
        "working_tree_clean": True,
        "dirty_path_count": 0,
        "dirty_paths": [],
        "errors": [],
        **FLAGS,
    })
    report = call(monkeypatch, tmp_path, repo_path=tmp_path / "repo")
    assert report["status"] == "FAIL"
    assert any(item["code"] == "GIT_SOURCE_PROVENANCE_FAILED" for item in report["errors"])



def test_missing_current_config_fails_closed_with_consolidated_report(monkeypatch, tmp_path):
    report = call(monkeypatch, tmp_path, fingerprint_error=True)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "CURRENT_INPUT_FINGERPRINT_FAILED" and item["input"] == "config"
               for item in report["errors"])
    assert report["deployment_authorized"] is False



def test_invalid_inventory_spec_returns_consolidated_failure(monkeypatch, tmp_path):
    invalid_spec = {"audit_name": "wrong_spec", "schema_version": 999, "required_artifacts": []}
    report = call(monkeypatch, tmp_path, inventory_override=invalid_spec, include_inventory_seal=False)
    assert report["status"] == "FAIL"
    assert report["artifact_inventory_completeness"]["status"] == "FAIL"
    assert any(item["code"] == "INVENTORY_AUDIT_EXECUTION_FAILED"
               for item in report["artifact_inventory_completeness"]["errors"])


def test_invalid_provenance_manifest_returns_consolidated_failure(monkeypatch, tmp_path):
    report = call(monkeypatch, tmp_path, provenance={"audit_name": "wrong_manifest", "artifacts": [{"artifact_id": "metrics", "path": "metrics.csv"}]}, include_seal=False, provenance_error=True)
    assert report["status"] == "FAIL"
    assert report["provenance_verification"]["status"] == "FAIL"
    assert any(item["code"] == "PROVENANCE_AUDIT_EXECUTION_FAILED"
               for item in report["provenance_verification"]["errors"])


def test_final_certification_exception_returns_consolidated_failure(monkeypatch, tmp_path):
    report = call(monkeypatch, tmp_path, final_certification_error=True)
    assert report["status"] == "FAIL"
    assert report["final_certification_chain"]["status"] == "FAIL"
    assert any(item["code"] == "FINAL_CERTIFICATION_EXECUTION_FAILED"
               for item in report["final_certification_chain"]["errors"])
    assert any(item["code"] == "FINAL_CERTIFICATION_CHAIN_FAILED" for item in report["errors"])

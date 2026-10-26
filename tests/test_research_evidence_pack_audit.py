from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app import research_evidence_pack_audit
from app.research_evidence_pack_audit import audit_research_evidence_pack


FALSE_FLAGS = {
    "actionable": False,
    "inferential": False,
    "execution_connected": False,
    "model_training_performed": False,
    "automatic_model_selection": False,
    "automatic_repair_performed": False,
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_pack(tmp_path, payloads, **extra):
    entries = []
    for role, payload in payloads:
        name = f"{role}.json"
        raw = json.dumps(payload, sort_keys=True).encode()
        (tmp_path / name).write_bytes(raw)
        entries.append({"role": role, "path": name, "expected_sha256": sha(raw)})
    manifest = {
        "schema_version": 1,
        "pack_id": "test-pack",
        "reports": entries,
        **extra,
    }
    return manifest


def registry_report():
    return {
        "status": "PASS",
        "runs_supplied": 2,
        "runs_validated": 2,
        "run_results": [
            {"run_id": "run-a", "status": "PASS"},
            {"run_id": "run-b", "status": "PASS"},
        ],
        **FALSE_FLAGS,
    }


def workflow_report(dataset_digest=None):
    return {
        "status": "PASS",
        "artifacts_checked": 7,
        "missing_required_audits": [],
        "dataset_fingerprints": {dataset_digest or sha(b"dataset"): ["backtest_result_integrity"]},
        **FALSE_FLAGS,
    }


def audit(tmp_path, manifest):
    raw = json.dumps(manifest, sort_keys=True).encode()
    return audit_research_evidence_pack(
        manifest, manifest_sha256=sha(raw), base_dir=tmp_path
    )


def test_valid_pack_reconciles_reports_and_run_ids(tmp_path):
    manifest = write_pack(
        tmp_path,
        [
            ("experiment_registry", registry_report()),
            ("validation_workflow", workflow_report()),
        ],
    )
    report = audit(tmp_path, manifest)
    assert report["status"] == "PASS"
    assert report["readiness"] == "EVIDENCE_INTEGRITY_VALIDATED"
    assert report["run_ids_reconciled"] == ["run-a", "run-b"]
    assert report["execution_connected"] is False
    assert report["automatic_model_selection"] is False


def test_report_checksum_mismatch_fails_closed(tmp_path):
    manifest = write_pack(
        tmp_path,
        [
            ("experiment_registry", registry_report()),
            ("validation_workflow", workflow_report()),
        ],
    )
    manifest["reports"][0]["expected_sha256"] = "0" * 64
    report = audit(tmp_path, manifest)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "report_sha256_mismatch" for item in report["errors"])


def test_missing_required_role_fails_closed(tmp_path):
    manifest = write_pack(tmp_path, [("experiment_registry", registry_report())])
    report = audit(tmp_path, manifest)
    assert report["status"] == "FAIL"
    assert any(
        item.get("role") == "validation_workflow"
        and item["code"] == "required_report_role_missing"
        for item in report["errors"]
    )


def test_dataset_fingerprint_mismatch_fails_closed(tmp_path):
    manifest = write_pack(
        tmp_path,
        [
            ("experiment_registry", registry_report()),
            ("validation_workflow", workflow_report(sha(b"dataset-a"))),
            ("extra_workflow", workflow_report(sha(b"dataset-b"))),
        ],
        required_roles=["experiment_registry", "validation_workflow"],
    )
    report = audit(tmp_path, manifest)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "dataset_fingerprint_mismatch" for item in report["errors"])


def test_path_traversal_fails_closed(tmp_path):
    manifest = {
        "schema_version": 1,
        "pack_id": "unsafe",
        "reports": [
            {
                "role": "experiment_registry",
                "path": "../outside.json",
                "expected_sha256": sha(b"outside"),
            }
        ],
    }
    report = audit(tmp_path, manifest)
    assert report["status"] == "FAIL"
    assert any(
        item["code"] == "report_path_must_be_relative_and_confined"
        for item in report["errors"]
    )


def test_non_false_safety_flag_fails_closed(tmp_path):
    unsafe = registry_report()
    unsafe["execution_connected"] = True
    manifest = write_pack(
        tmp_path,
        [
            ("experiment_registry", unsafe),
            ("validation_workflow", workflow_report()),
        ],
    )
    report = audit(tmp_path, manifest)
    assert report["status"] == "FAIL"
    assert any(
        item["code"] == "safety_flag_not_explicitly_false:execution_connected"
        for item in report["errors"]
    )


def test_failed_upstream_report_fails_closed(tmp_path):
    registry = registry_report()
    registry["status"] = "FAIL"
    manifest = write_pack(
        tmp_path,
        [
            ("experiment_registry", registry),
            ("validation_workflow", workflow_report()),
        ],
    )
    report = audit(tmp_path, manifest)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "upstream_report_not_passing" for item in report["errors"])



def test_boolean_values_are_not_accepted_as_coverage_counts(tmp_path):
    registry = registry_report()
    registry["runs_supplied"] = True
    registry["runs_validated"] = True
    workflow = workflow_report()
    workflow["artifacts_checked"] = True
    manifest = write_pack(
        tmp_path,
        [("experiment_registry", registry), ("validation_workflow", workflow)],
    )

    report = audit(tmp_path, manifest)

    codes = {item["code"] for item in report["errors"]}
    assert "registry_run_coverage_incomplete" in codes
    assert "validation_artifact_coverage_missing" in codes


def test_required_roles_cannot_disable_core_evidence_roles(tmp_path):
    manifest = write_pack(
        tmp_path,
        [("experiment_registry", registry_report()), ("validation_workflow", workflow_report())],
        required_roles=["experiment_comparison"],
    )

    with pytest.raises(ValueError, match="required_roles_must_include_core_roles"):
        audit(tmp_path, manifest)


def _write_manifest_file(tmp_path):
    manifest = write_pack(
        tmp_path,
        [("experiment_registry", registry_report()), ("validation_workflow", workflow_report())],
    )
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_evidence_pack_cli_publishes_report_without_overwriting(tmp_path, capsys):
    manifest_path = _write_manifest_file(tmp_path)
    output = tmp_path / "evidence-audit.json"

    result = research_evidence_pack_audit.main(["--input", str(manifest_path), "--output", str(output)])

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_evidence_pack_cli_refuses_output_aliasing_manifest(tmp_path, capsys):
    manifest_path = _write_manifest_file(tmp_path)
    original = manifest_path.read_bytes()

    result = research_evidence_pack_audit.main(["--input", str(manifest_path), "--output", str(manifest_path)])

    assert result == 2
    assert manifest_path.read_bytes() == original
    assert "output_must_not_alias_input" in capsys.readouterr().out


def test_evidence_pack_cli_does_not_clobber_concurrent_output(tmp_path, monkeypatch, capsys):
    manifest_path = _write_manifest_file(tmp_path)
    output = tmp_path / "evidence-audit.json"
    real_link = research_evidence_pack_audit.os.link

    def create_destination_then_link(source, destination):
        Path(destination).write_text("created concurrently", encoding="utf-8")
        return real_link(source, destination)

    monkeypatch.setattr(research_evidence_pack_audit.os, "link", create_destination_then_link)
    result = research_evidence_pack_audit.main(["--input", str(manifest_path), "--output", str(output)])

    assert result == 2
    assert output.read_text(encoding="utf-8") == "created concurrently"
    assert "FileExistsError" in capsys.readouterr().out
    assert list(tmp_path.glob(".evidence-audit.json.*.tmp")) == []

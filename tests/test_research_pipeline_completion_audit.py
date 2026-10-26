from __future__ import annotations

import hashlib
import json

from app.research_pipeline_completion_audit import audit_pipeline_completion, main


FLAGS = {
    "actionable": False,
    "execution_connected": False,
    "automatic_winner_selection": False,
    "test_metrics_used_for_selection": False,
}


def _write(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def fixture(tmp_path, evaluation_status="PASS"):
    fingerprint = _write(tmp_path / "fingerprint.json", {"fingerprint": "fixture"})
    evaluation = _write(tmp_path / "evaluation.json", {
        "audit_name": "research_model_evaluation_gate",
        "status": evaluation_status,
        "errors": [] if evaluation_status == "PASS" else [{"code": "metric_gate_failed"}],
        "warnings": [],
        "actionable": False,
    })
    manifest = {
        "schema_version": 1,
        "run_id": "eurusd-m15-test",
        "experiment_fingerprint_path": str(fingerprint.resolve()),
        "model_evaluation_gate_path": str(evaluation.resolve()),
        "model_evaluation_gate_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        "actionable": False,
        "execution_connected": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    manifest_path = _write(tmp_path / "manifest.json", manifest)
    validation = _write(tmp_path / "validation.json", {
        "scope": "GOLDCREST_COMPLETED_RUN_VALIDATION",
        "status": "PASS",
        "run_id": manifest["run_id"],
        "manifest_path": str(manifest_path.resolve()),
        "errors": [],
        "warnings": [],
        "actionable": False,
        "execution_connected": False,
    })
    reproducibility = _write(tmp_path / "reproducibility.json", {
        "audit_name": "research_experiment_reproducibility_audit",
        "status": "PASS",
        "errors": [],
        "pipeline_manifest_path": str(manifest_path.resolve()),
        "fingerprint_path": str(fingerprint.resolve()),
        "fingerprint_file_sha256": hashlib.sha256(fingerprint.read_bytes()).hexdigest(),
        "actionable": False,
        "execution_connected": False,
    })
    return manifest_path, validation, reproducibility, evaluation, fingerprint


def test_complete_pipeline_reports_pass_integrity_gate(tmp_path):
    manifest, validation, reproducibility, _, _ = fixture(tmp_path)
    report = audit_pipeline_completion(manifest, validation, reproducibility)
    assert report["status"] == "PASS"
    assert report["readiness"] == "RUN_INTEGRITY_VERIFIED"
    assert report["model_qualified"] is False
    assert report["deployment_authorized"] is False


def test_reproducibility_failure_blocks_completion(tmp_path):
    manifest, validation, reproducibility, _, _ = fixture(tmp_path)
    payload = json.loads(reproducibility.read_text())
    payload["status"] = "FAIL"
    payload["errors"] = ["dataset_hash_mismatch"]
    _write(reproducibility, payload)
    report = audit_pipeline_completion(manifest, validation, reproducibility)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "reproducibility_audit_not_passing" for item in report["errors"])


def test_manifest_tampering_blocks_completion(tmp_path):
    manifest, validation, reproducibility, _, _ = fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["run_id"] = "changed-after-sealing"
    _write(manifest, payload)
    report = audit_pipeline_completion(manifest, validation, reproducibility)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "pipeline_manifest_sha256_mismatch" for item in report["errors"])


def test_tampered_model_evaluation_receipt_blocks_integrity_gate(tmp_path):
    manifest, validation, reproducibility, evaluation, _ = fixture(tmp_path)
    _write(evaluation, {"audit_name": "tampered", "status": "PASS"})
    report = audit_pipeline_completion(manifest, validation, reproducibility)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "model_evaluation_gate_missing_unreadable_or_tampered" for item in report["errors"])


def test_failed_model_evaluation_is_reported_but_does_not_claim_model_qualified(tmp_path):
    manifest, validation, reproducibility, _, _ = fixture(tmp_path, evaluation_status="FAIL")
    report = audit_pipeline_completion(manifest, validation, reproducibility)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert report["model_evaluation_status"] == "FAIL"
    assert report["model_qualified"] is False
    assert "model_evaluation_gate_did_not_pass" in report["warnings"]


def test_non_strict_validation_findings_are_preserved_as_warning(tmp_path):
    manifest, validation, reproducibility, _, _ = fixture(tmp_path)
    payload = json.loads(validation.read_text())
    payload["status"] = "FAIL"
    payload["errors"] = ["dataset_quality_issue"]
    _write(validation, payload)
    report = audit_pipeline_completion(manifest, validation, reproducibility)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert report["validation_status"] == "FAIL"
    assert "validation_report_contains_quality_or_integrity_findings" in report["warnings"]


def test_changed_fingerprint_blocks_completion(tmp_path):
    manifest, validation, reproducibility, _, fingerprint = fixture(tmp_path)
    fingerprint.write_text('{"tampered":true}\n', encoding="utf-8")
    report = audit_pipeline_completion(manifest, validation, reproducibility)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "reproducibility_fingerprint_hash_mismatch" for item in report["errors"])



def test_completion_audit_cli_writes_report_and_returns_success(tmp_path, capsys):
    manifest, validation, reproducibility, _, _ = fixture(tmp_path)
    output = tmp_path / "reports" / "completion.json"

    result = main([
        "--manifest", str(manifest),
        "--validation-report", str(validation),
        "--reproducibility-audit", str(reproducibility),
        "--output", str(output),
    ])

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["model_qualified"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_completion_audit_cli_refuses_output_aliasing_input(tmp_path, capsys):
    manifest, validation, reproducibility, _, _ = fixture(tmp_path)
    original = manifest.read_bytes()

    result = main([
        "--manifest", str(manifest),
        "--validation-report", str(validation),
        "--reproducibility-audit", str(reproducibility),
        "--output", str(manifest),
    ])

    assert result == 2
    assert manifest.read_bytes() == original
    assert "output_must_not_alias_input" in capsys.readouterr().err


def test_completion_audit_cli_refuses_to_overwrite_existing_output(tmp_path, capsys):
    manifest, validation, reproducibility, _, _ = fixture(tmp_path)
    output = tmp_path / "completion.json"
    output.write_text("preserve", encoding="utf-8")

    result = main([
        "--manifest", str(manifest),
        "--validation-report", str(validation),
        "--reproducibility-audit", str(reproducibility),
        "--output", str(output),
    ])

    assert result == 2
    assert output.read_text(encoding="utf-8") == "preserve"
    assert "output_already_exists" in capsys.readouterr().err


def test_completion_audit_cli_writes_failure_receipt_and_returns_nonzero(tmp_path):
    manifest, validation, reproducibility, _, _ = fixture(tmp_path)
    payload = json.loads(reproducibility.read_text(encoding="utf-8"))
    payload["status"] = "FAIL"
    payload["errors"] = ["fingerprint_mismatch"]
    _write(reproducibility, payload)
    output = tmp_path / "completion-failed.json"

    result = main([
        "--manifest", str(manifest),
        "--validation-report", str(validation),
        "--reproducibility-audit", str(reproducibility),
        "--output", str(output),
    ])

    assert result == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "FAIL"


def test_completion_audit_cli_does_not_clobber_concurrent_output(tmp_path, monkeypatch, capsys):
    import app.research_pipeline_completion_audit as module

    manifest, validation, reproducibility, _, _ = fixture(tmp_path)
    output = tmp_path / "completion.json"
    real_link = module.os.link

    def create_destination_then_link(source, destination):
        Path(destination).write_text("created concurrently", encoding="utf-8")
        return real_link(source, destination)

    monkeypatch.setattr(module.os, "link", create_destination_then_link)
    result = main([
        "--manifest", str(manifest),
        "--validation-report", str(validation),
        "--reproducibility-audit", str(reproducibility),
        "--output", str(output),
    ])

    assert result == 2
    assert output.read_text(encoding="utf-8") == "created concurrently"
    assert "FileExistsError" in capsys.readouterr().err
    assert list(tmp_path.glob(".completion.json.*.tmp")) == []

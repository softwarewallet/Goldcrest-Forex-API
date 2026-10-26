from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.research_pipeline_completion_audit import audit_pipeline_completion, main, _read_object_with_hash


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
    manifest, validation, reproducibility, evaluation, _ = fixture(tmp_path)
    report = audit_pipeline_completion(manifest, validation, reproducibility)
    assert report["status"] == "PASS"
    assert report["readiness"] == "RUN_INTEGRITY_VERIFIED"
    assert report["model_qualified"] is False
    assert report["deployment_authorized"] is False
    assert report["validation_report_sha256"] == hashlib.sha256(validation.read_bytes()).hexdigest()
    assert report["reproducibility_audit_sha256"] == hashlib.sha256(reproducibility.read_bytes()).hexdigest()
    assert report["model_evaluation_gate_sha256"] == hashlib.sha256(evaluation.read_bytes()).hexdigest()
    assert report["checks"]["validation_report_file_hash"] == "PASS"
    assert report["checks"]["reproducibility_audit_file_hash"] == "PASS"


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



def test_report_hash_is_for_the_exact_bytes_parsed(tmp_path):
    path = tmp_path / "report.json"
    raw = b'{"status":"PASS","run_id":"snapshot-test"}\n'
    path.write_bytes(raw)

    payload, digest = _read_object_with_hash(path)

    assert payload == {"status": "PASS", "run_id": "snapshot-test"}
    assert digest == hashlib.sha256(raw).hexdigest()



def test_unparseable_required_report_does_not_claim_hash_check_pass(tmp_path):
    manifest, validation, reproducibility, _, _ = fixture(tmp_path)
    validation.write_text("{malformed json", encoding="utf-8")

    report = audit_pipeline_completion(manifest, validation, reproducibility)

    assert report["status"] == "FAIL"
    assert report["checks"]["validation_report_readable"] == "FAIL"
    assert report["checks"]["validation_report_file_hash"] == "FAIL"
    assert any(item["code"] == "required_report_hash_unavailable" for item in report["errors"])

def test_duplicate_json_key_in_required_report_fails_closed(tmp_path):
    manifest, validation, reproducibility, _, _ = fixture(tmp_path)
    original = manifest.read_text(encoding="utf-8")
    assert '"run_id":' in original
    manifest.write_text(
        original.replace(
            '"run_id": "eurusd-m15-test",',
            '"run_id": "eurusd-m15-test",\n  "run_id": "shadow-run",',
            1,
        ),
        encoding="utf-8",
    )

    report = audit_pipeline_completion(manifest, validation, reproducibility)

    assert report["status"] == "FAIL"
    assert report["checks"]["pipeline_manifest_readable"] == "FAIL"
    assert any(item["code"] == "required_report_unreadable" for item in report["errors"])
    assert report["deployment_authorized"] is False


def test_completion_reader_rejects_nonstandard_json_constants(tmp_path):
    from app.research_pipeline_completion_audit import _read_object_with_hash

    report_path = tmp_path / "nonstandard.json"
    report_path.write_text('{"status":"PASS","metric":NaN}', encoding="utf-8")

    try:
        _read_object_with_hash(report_path)
    except ValueError as exc:
        assert "nonstandard_json_constant:NaN" in str(exc)
    else:
        raise AssertionError("non-standard JSON constant must be rejected")

def test_missing_validation_errors_array_is_integrity_failure(tmp_path):
    manifest, validation, reproducibility, _, _ = fixture(tmp_path)
    payload = json.loads(validation.read_text(encoding="utf-8"))
    payload.pop("errors")
    _write(validation, payload)

    report = audit_pipeline_completion(manifest, validation, reproducibility)

    assert report["status"] == "FAIL"
    assert any(item["code"] == "validation_report_errors_schema_invalid" for item in report["errors"])


def test_malformed_validation_warnings_array_is_integrity_failure(tmp_path):
    manifest, validation, reproducibility, _, _ = fixture(tmp_path)
    payload = json.loads(validation.read_text(encoding="utf-8"))
    payload["warnings"] = "not-a-list"
    _write(validation, payload)

    report = audit_pipeline_completion(manifest, validation, reproducibility)

    assert report["status"] == "FAIL"
    assert any(item["code"] == "validation_report_warnings_schema_invalid" for item in report["errors"])


def test_missing_evaluation_errors_array_is_integrity_failure(tmp_path):
    manifest, validation, reproducibility, evaluation, _ = fixture(tmp_path)
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    payload.pop("errors")
    _write(evaluation, payload)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["model_evaluation_gate_sha256"] = hashlib.sha256(evaluation.read_bytes()).hexdigest()
    _write(manifest, manifest_payload)
    # Recompute the self-hash after updating the expected evaluation digest.
    manifest_payload.pop("pipeline_manifest_sha256", None)
    manifest_payload["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    _write(manifest, manifest_payload)
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    validation_payload["manifest_path"] = str(manifest.resolve())
    _write(validation, validation_payload)

    report = audit_pipeline_completion(manifest, validation, reproducibility)

    assert report["status"] == "FAIL"
    assert any(item["code"] == "model_evaluation_gate_errors_schema_invalid" for item in report["errors"])


def test_malformed_evaluation_warnings_array_is_integrity_failure(tmp_path):
    manifest, validation, reproducibility, evaluation, _ = fixture(tmp_path)
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    payload["warnings"] = {"not": "a-list"}
    _write(evaluation, payload)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["model_evaluation_gate_sha256"] = hashlib.sha256(evaluation.read_bytes()).hexdigest()
    manifest_payload.pop("pipeline_manifest_sha256", None)
    manifest_payload["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    _write(manifest, manifest_payload)
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    validation_payload["manifest_path"] = str(manifest.resolve())
    _write(validation, validation_payload)

    report = audit_pipeline_completion(manifest, validation, reproducibility)

    assert report["status"] == "FAIL"
    assert any(item["code"] == "model_evaluation_gate_warnings_schema_invalid" for item in report["errors"])


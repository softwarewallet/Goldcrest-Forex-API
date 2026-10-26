from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.research_release_readiness import audit_research_release_readiness


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def build_inputs(tmp_path: Path):
    reference = "https://provider.example/legal/terms"
    source = tmp_path / "terms-source.txt"
    source.write_text("Terms evidence fixture.\n", encoding="utf-8")
    digest = hashlib.sha256(reference.encode("utf-8")).hexdigest()
    manifest_path = tmp_path / "fixture-run_pipeline_manifest.json"
    write_json(manifest_path, {
        "schema_version": 1, "run_id": "fixture-run",
        "terms_reference": reference, "terms_reference_sha256": digest,
        "actionable": False, "execution_connected": False,
        "automatic_winner_selection": False, "test_metrics_used_for_selection": False,
        "automatic_model_selection": False, "deployment_authorized": False,
    })
    evidence_path = tmp_path / "terms-review.json"
    write_json(evidence_path, {
        "schema_version": 1, "evidence_type": "provider_terms_review",
        "provider": "Test provider", "terms_reference": reference,
        "terms_reference_sha256": digest, "source_evidence_path": str(source),
        "source_evidence_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "review_status": "VERIFIED", "applicability": "CONFIRMED",
        "reviewed_by": "reviewer-fixture", "reviewed_at_utc": "2026-10-26T12:00:00Z",
    })
    return manifest_path, evidence_path


def test_gate_fails_closed_without_complete_pipeline_artifacts(tmp_path):
    manifest, evidence = build_inputs(tmp_path)
    result = audit_research_release_readiness(manifest, evidence)
    assert result["status"] == "FAIL"
    assert result["readiness"] == "NOT_RELEASE_READY"
    assert result["actionable"] is False
    assert result["execution_connected"] is False
    assert result["model_qualified"] is False
    assert result["deployment_authorized"] is False
    assert "cross_artifact_integrity_failed" in result["errors"]
    assert "pipeline_completion_receipt_missing" in result["errors"]


def test_gate_detects_provider_evidence_hash_mismatch(tmp_path):
    manifest, evidence_path = build_inputs(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["source_evidence_sha256"] = "0" * 64
    write_json(evidence_path, evidence)
    result = audit_research_release_readiness(manifest, evidence_path)
    assert result["status"] == "FAIL"
    assert "provider_terms_source_evidence_hash_mismatch" in result["errors"]


def test_gate_rejects_non_uri_terms_reference(tmp_path):
    manifest, evidence_path = build_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["terms_reference"] = "terms not yet verified"
    payload["terms_reference_sha256"] = hashlib.sha256(
        payload["terms_reference"].encode("utf-8")
    ).hexdigest()
    write_json(manifest, payload)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["terms_reference"] = payload["terms_reference"]
    evidence["terms_reference_sha256"] = payload["terms_reference_sha256"]
    write_json(evidence_path, evidence)
    result = audit_research_release_readiness(manifest, evidence_path)
    assert result["status"] == "FAIL"
    assert "provider_terms_reference_must_be_http_or_https_uri" in result["errors"]

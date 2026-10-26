from __future__ import annotations

import hashlib
import json

import pytest

from app.research_artifact_bundle_certification_audit import audit_research_artifact_bundle


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_report(tmp_path, name="feature_temporal_integrity", status="PASS"):
    payload = {
        "audit_name": name,
        "status": status,
        "errors": [],
        "warnings": [],
        "dataset_sha256": sha(b"dataset"),
        "actionable": False,
        "automatic_model_selection": False,
        "inferential": False,
        "model_training_performed": False,
        "execution_connected": False,
    }
    path = tmp_path / f"{name}.json"
    raw = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(raw)
    return path, payload, raw


def test_valid_bundle_passes_and_records_report_fingerprint(tmp_path):
    path, _, raw = write_report(tmp_path)
    manifest = {
        "artifacts": [{"audit_name": "feature_temporal_integrity", "path": path.name}],
        "required_audits": ["feature_temporal_integrity"],
        "expected_dataset_sha256": sha(b"dataset"),
    }

    report = audit_research_artifact_bundle(
        manifest,
        manifest_sha256=sha(b"manifest"),
        base_dir=tmp_path,
    )

    assert report["status"] == "PASS"
    assert report["artifacts_checked"] == 1
    assert report["artifact_results"][0]["report_sha256"] == sha(raw)
    assert report["errors"] == []
    assert report["execution_connected"] is False


def test_upstream_failure_and_dataset_mismatch_fail_certification(tmp_path):
    path, payload, _ = write_report(tmp_path, status="FAIL")
    payload["errors"] = ["fixture failure"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "artifacts": [{"audit_name": "feature_temporal_integrity", "path": path.name}],
        "expected_dataset_sha256": sha(b"other dataset"),
    }

    report = audit_research_artifact_bundle(
        manifest,
        manifest_sha256=sha(b"manifest"),
        base_dir=tmp_path,
    )

    codes = {item["code"] for item in report["errors"]}
    assert report["status"] == "FAIL"
    assert "upstream_audit_failed" in codes
    assert "upstream_report_contains_errors" in codes
    assert "dataset_sha256_mismatch_or_missing" in codes


def test_missing_required_audit_is_detected(tmp_path):
    path, _, _ = write_report(tmp_path)
    report = audit_research_artifact_bundle(
        {
            "artifacts": [{"audit_name": "feature_temporal_integrity", "path": path.name}],
            "required_audits": ["feature_temporal_integrity", "feature_backend_parity"],
        },
        manifest_sha256=sha(b"manifest"),
        base_dir=tmp_path,
    )

    assert report["status"] == "FAIL"
    assert report["missing_required_audits"] == ["feature_backend_parity"]


def test_path_traversal_and_duplicate_paths_are_rejected(tmp_path):
    path, _, _ = write_report(tmp_path)
    manifest = {
        "artifacts": [
            {"audit_name": "feature_temporal_integrity", "path": "../outside.json"},
            {"audit_name": "feature_backend_parity", "path": path.name},
            {"audit_name": "other_audit", "path": path.name},
        ]
    }
    report = audit_research_artifact_bundle(
        manifest,
        manifest_sha256=sha(b"manifest"),
        base_dir=tmp_path,
    )

    codes = {item["code"] for item in report["errors"]}
    assert "artifact_path_must_be_relative_and_confined" in codes
    assert "duplicate_artifact_path" in codes


def test_report_digest_mismatch_and_non_false_safety_flag_fail(tmp_path):
    path, payload, _ = write_report(tmp_path)
    payload["execution_connected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "artifacts": [{
            "audit_name": "feature_temporal_integrity",
            "path": path.name,
            "expected_report_sha256": sha(b"not the report"),
        }]
    }
    report = audit_research_artifact_bundle(
        manifest,
        manifest_sha256=sha(b"manifest"),
        base_dir=tmp_path,
    )

    codes = {item["code"] for item in report["errors"]}
    assert "report_sha256_mismatch" in codes
    assert "safety_flag_not_false:execution_connected" in codes


def test_invalid_manifest_and_hash_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="manifest_must_be_object"):
        audit_research_artifact_bundle([], manifest_sha256=sha(b"manifest"), base_dir=tmp_path)
    with pytest.raises(ValueError, match="manifest_sha256_invalid"):
        audit_research_artifact_bundle({"artifacts": []}, manifest_sha256="bad", base_dir=tmp_path)

@pytest.mark.parametrize("raw", [b'{"metric":NaN}', b'{"metric":Infinity}', b'{"metric":-Infinity}', b'{"value":1,"value":2}'])
def test_certification_json_readers_reject_nonstandard_json(tmp_path, raw):
    from app.research_artifact_bundle_certification_audit import _load_json_object
    with pytest.raises(ValueError):
        _load_json_object(raw, source="fixture.json")

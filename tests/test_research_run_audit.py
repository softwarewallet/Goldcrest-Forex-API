from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.research_run_audit import audit_research_run_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_valid_run(tmp_path: Path) -> Path:
    raw = tmp_path / "raw.csv"
    raw.write_text("timestamp,open,high,low,close\n2026-01-01T00:00:00Z,1,1,1,1\n", encoding="utf-8")
    raw_audit = tmp_path / "raw.audit.json"
    _write_json(raw_audit, {
        "status": "PASS",
        "snapshot_sha256": _sha(raw),
        "errors": [],
        "warnings": [],
        "actionable": False,
        "execution_connected": False,
    })
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,rsi_14\nEURUSD,M15,2026-01-01T00:00:00Z,1,1,1,1,1,2026-01-01T01:00:00Z,50\n", encoding="utf-8")
    dataset_manifest = tmp_path / "dataset.manifest.json"
    _write_json(dataset_manifest, {"output_dataset_sha256": _sha(dataset)})

    report = tmp_path / "run_experiment_matrix.json"
    _write_json(report, {
        "schema_version": 1,
        "run_id": "run-001",
        "dataset_sha256": _sha(dataset),
        "actionable": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    })
    audit = tmp_path / "run_provenance_audit.json"
    _write_json(audit, {
        "schema_version": 1,
        "run_id": "run-001",
        "source_report_sha256": _sha(report),
        "status": "PASS",
        "actionable": False,
        "execution_connected": False,
    })

    manifest = {
        "schema_version": 1,
        "run_id": "run-001",
        "raw_ohlc_path": str(raw),
        "raw_ohlc_sha256": _sha(raw),
        "raw_ohlc_audit_path": str(raw_audit),
        "raw_ohlc_audit_sha256": _sha(raw_audit),
        "raw_ohlc_audit_status": "PASS",
        "dataset_path": str(dataset),
        "dataset_sha256": _sha(dataset),
        "dataset_manifest_path": str(dataset_manifest),
        "dataset_manifest_sha256": _sha(dataset_manifest),
        "experiment_matrix_report": str(report),
        "experiment_matrix_report_sha256": _sha(report),
        "provenance_audit_path": str(audit),
        "provenance_audit_sha256": _sha(audit),
        "provenance_audit_status": "PASS",
        "actionable": False,
        "execution_connected": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest_path = tmp_path / "run-001_pipeline_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_audit_accepts_consistent_research_run(tmp_path):
    result = audit_research_run_manifest(_build_valid_run(tmp_path))
    assert result["status"] == "PASS"
    assert result["checked_artifact_count"] == 6
    assert result["errors"] == []
    assert result["actionable"] is False
    assert result["execution_connected"] is False


def test_audit_verifies_temporal_calibration_receipt_hash_and_matrix_binding(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix_path = Path(manifest["experiment_matrix_report"])
    matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix_payload.pop("report_payload_sha256", None)
    matrix_payload["report_payload_sha256"] = hashlib.sha256(
        json.dumps(matrix_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(matrix_path, matrix_payload)
    manifest["experiment_matrix_report_sha256"] = _sha(matrix_path)

    provenance_path = Path(manifest["provenance_audit_path"])
    provenance_payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_payload["source_report_sha256"] = _sha(matrix_path)
    _write_json(provenance_path, provenance_payload)
    manifest["provenance_audit_sha256"] = _sha(provenance_path)

    receipt_path = tmp_path / "run-001_temporal_calibration_stability.json"
    receipt = {
        "schema_version": 1,
        "scope": "GOLDCREST_TEMPORAL_CALIBRATION_STABILITY_AUDIT",
        "run_id": "run-001",
        "experiment_matrix_sha256": _sha(matrix_path),
        "experiment_matrix_payload_hash_verified": True,
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "comparisons": [],
        "comparison_count": 0,
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "test_metrics_used_for_selection": False,
    }
    _write_json(receipt_path, receipt)
    manifest.update({
        "temporal_calibration_audit_path": str(receipt_path),
        "temporal_calibration_audit_sha256": _sha(receipt_path),
        "temporal_calibration_audit_status": "PASS",
    })
    manifest.pop("pipeline_manifest_sha256", None)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _write_json(manifest_path, manifest)

    valid = audit_research_run_manifest(manifest_path)
    assert valid["status"] == "PASS"
    assert "temporal_calibration_audit_path" in valid["checked_artifacts"]

    receipt_path.write_text(receipt_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered = audit_research_run_manifest(manifest_path)
    assert tampered["status"] == "FAIL"
    assert "temporal_calibration_audit_path_sha256_mismatch" in tampered["errors"]


def test_audit_detects_tampered_dataset_and_manifest_fingerprint(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    Path(manifest["dataset_path"]).write_text("tampered\n", encoding="utf-8")

    result = audit_research_run_manifest(manifest_path)
    assert result["status"] == "FAIL"
    assert "dataset_path_sha256_mismatch" in result["errors"]
    assert "pipeline_manifest_sha256_mismatch" not in result["errors"]


def test_audit_rejects_actionable_manifest(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["actionable"] = True
    canonical = dict(manifest)
    canonical.pop("pipeline_manifest_sha256", None)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)

    result = audit_research_run_manifest(manifest_path)
    assert result["status"] == "FAIL"
    assert "actionable_must_be_false" in result["errors"]


def test_audit_reports_unreadable_manifest_without_traceback(tmp_path):
    missing = tmp_path / "missing.json"
    result = audit_research_run_manifest(missing)
    assert result["status"] == "FAIL"
    assert result["errors"] == ["manifest_unreadable:FileNotFoundError"]



def test_audit_detects_tampered_raw_ohlc_audit_receipt(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_path = Path(manifest["raw_ohlc_audit_path"])
    audit_path.write_text(audit_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = audit_research_run_manifest(manifest_path)

    assert result["status"] == "FAIL"
    assert "raw_ohlc_audit_path_sha256_mismatch" in result["errors"]

import csv
import hashlib
import json
from pathlib import Path

from app.research_artifact_certification import certify_pipeline_manifest
from app.research_experiment_fingerprint import build_experiment_fingerprint


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload, field):
    payload[field] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _fixture(tmp_path, *, immature_label=False):
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "symbol,timeframe,timestamp,open,high,low,close\n"
        "EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.csv"
    available = "2026-01-01T00:00:00Z" if immature_label else "2026-01-01T01:00:00Z"
    dataset.write_text(
        "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,feature_1\n"
        f"EURUSD,M15,2026-01-01T00:00:00Z,1,1.1,0.9,1.05,1,{available},0.2\n",
        encoding="utf-8",
    )
    data_manifest = {
        "actionable": False,
        "input_snapshot_sha256": _digest(raw),
        "output_dataset_sha256": _digest(dataset),
        "feature_implementation_sha256": "a" * 64,
        "feature_schema_sha256": "b" * 64,
        "feature_backend": "numpy_fallback",
    }
    _canonical_hash(data_manifest, "manifest_sha256")
    (tmp_path / "dataset-manifest.json").write_text(json.dumps(data_manifest), encoding="utf-8")
    fingerprint = build_experiment_fingerprint(
        configuration={"symbol": "EURUSD", "timeframe": "M15", "seed": 1},
        raw_ohlc_sha256=_digest(raw),
        dataset_sha256=_digest(dataset),
        dataset_manifest=data_manifest,
    )
    fingerprint_path = tmp_path / "fingerprint.json"
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    quality = {"status": "PASS", "source_sha256": _digest(raw), "actionable": False}
    (tmp_path / "quality.json").write_text(json.dumps(quality), encoding="utf-8")
    matrix = {"actionable": False, "experiments": []}
    (tmp_path / "matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    provenance = {
        "status": "PASS", "source_report_sha256": _digest(tmp_path / "matrix.json"),
        "actionable": False,
    }
    (tmp_path / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

    manifest = {
        "run_id": "cert-test",
        "raw_ohlc_path": "raw.csv",
        "raw_ohlc_sha256": _digest(raw),
        "dataset_path": "dataset.csv",
        "dataset_sha256": _digest(dataset),
        "dataset_manifest_path": "dataset-manifest.json",
        "dataset_manifest_sha256": _digest(tmp_path / "dataset-manifest.json"),
        "data_quality_audit_path": "quality.json",
        "data_quality_audit_sha256": _digest(tmp_path / "quality.json"),
        "data_quality_audit_status": "PASS",
        "experiment_matrix_report": "matrix.json",
        "experiment_matrix_report_sha256": _digest(tmp_path / "matrix.json"),
        "provenance_audit_path": "provenance.json",
        "provenance_audit_sha256": _digest(tmp_path / "provenance.json"),
        "provenance_audit_status": "PASS",
        "experiment_fingerprint_path": "fingerprint.json",
        "experiment_fingerprint_file_sha256": _digest(fingerprint_path),
        "experiment_fingerprint_sha256": fingerprint["experiment_fingerprint_sha256"],
        "actionable": False,
        "execution_connected": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    }
    _canonical_hash(manifest, "pipeline_manifest_sha256")
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path, dataset


def test_certifier_passes_consistent_artifacts_and_mature_labels(tmp_path):
    manifest, _ = _fixture(tmp_path)
    result = certify_pipeline_manifest(manifest)
    assert result["status"] == "PASS"
    assert result["dataset_temporal_checks"]["rows"] == 1
    assert result["errors"] == []


def test_certifier_detects_dataset_hash_tampering(tmp_path):
    manifest, dataset = _fixture(tmp_path)
    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    result = certify_pipeline_manifest(manifest)
    assert result["status"] == "FAIL"
    assert "artifact_sha256_mismatch:dataset" in result["errors"]


def test_certifier_rejects_label_available_at_sample_time(tmp_path):
    manifest, _ = _fixture(tmp_path, immature_label=True)
    result = certify_pipeline_manifest(manifest)
    assert result["status"] == "FAIL"
    assert "dataset_label_maturity_not_after_sample_time" in result["errors"]

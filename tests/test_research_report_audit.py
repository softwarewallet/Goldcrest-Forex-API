from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.research_report_audit import audit_research_pipeline


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict, hash_field: str | None = None) -> None:
    value = dict(payload)
    if hash_field:
        value[hash_field] = canonical_hash(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_artifacts(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "ohlc.csv"
    raw.write_text("timestamp,open,high,low,close\n2026-01-01T00:00:00Z,1,1.1,0.9,1.05\n", encoding="utf-8")
    dataset = root / "dataset.csv"
    dataset.write_text("timestamp,label,feature\n2026-01-01T00:00:00Z,1,0.2\n", encoding="utf-8")

    dataset_manifest_path = root / "dataset.manifest.json"
    write_json(dataset_manifest_path, {
        "schema_version": 1, "actionable": False, "output_dataset_sha256": sha(dataset),
    }, "manifest_sha256")

    fold = {
        "fold_index": 0,
        "test_start_utc": "2026-01-02T00:00:00Z",
        "test_end_utc": "2026-01-03T00:00:00Z",
        "test_metrics_used_for_tuning": False,
        "test_metrics": {
            "sample_count": 12, "brier_score": 0.21, "log_loss": 0.61,
            "accuracy_at_fixed_half": 0.67, "expected_calibration_error": 0.08,
        },
    }
    matrix_path = root / "matrix.json"
    write_json(matrix_path, {
        "schema_version": 1, "dataset_sha256": sha(dataset), "fold_count": 1,
        "common_fold_test_windows": [{
            "fold_index": 0, "test_start_utc": fold["test_start_utc"],
            "test_end_utc": fold["test_end_utc"], "test_rows": 12,
        }],
        "experiments": [{"experiment_id": "xgb-full", "fold_results": [fold]}],
        "actionable": False, "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    }, "report_payload_sha256")

    manifest_path = root / "run_pipeline_manifest.json"
    write_json(manifest_path, {
        "schema_version": 1,
        "raw_ohlc_path": str(raw), "raw_ohlc_sha256": sha(raw),
        "dataset_path": str(dataset), "dataset_sha256": sha(dataset),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": sha(dataset_manifest_path),
        "experiment_matrix_report": str(matrix_path),
        "experiment_matrix_report_sha256": sha(matrix_path),
        "actionable": False, "execution_connected": False,
        "automatic_winner_selection": False, "test_metrics_used_for_selection": False,
    }, "pipeline_manifest_sha256")
    return manifest_path


def test_audit_passes_for_consistent_non_actionable_artifacts(tmp_path):
    manifest = make_artifacts(tmp_path)
    report = audit_research_pipeline(manifest)
    assert report["audit_passed"] is True
    assert report["checks_failed"] == 0
    assert report["actionable"] is False
    assert report["execution_connected"] is False
    assert report["fold_metric_summary"]["brier_score"]["mean"] == 0.21


def test_audit_detects_tampered_raw_data(tmp_path):
    manifest = make_artifacts(tmp_path)
    (tmp_path / "ohlc.csv").write_text("tampered\n", encoding="utf-8")
    report = audit_research_pipeline(manifest)
    assert report["audit_passed"] is False
    assert "raw_ohlc_path_sha256" in report["failed_checks"]


def test_audit_detects_manifest_hash_tampering(tmp_path):
    manifest = make_artifacts(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["execution_connected"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    report = audit_research_pipeline(manifest)
    assert report["audit_passed"] is False
    assert "pipeline_manifest_hash" in report["failed_checks"]
    assert "research_only_flags" in report["failed_checks"]


def test_audit_detects_inconsistent_matrix_fold_windows(tmp_path):
    manifest = make_artifacts(tmp_path)
    matrix_path = tmp_path / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["experiments"][0]["fold_results"][0]["test_start_utc"] = "2026-01-09T00:00:00Z"
    write_json(matrix_path, {k: v for k, v in matrix.items() if k != "report_payload_sha256"}, "report_payload_sha256")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["experiment_matrix_report_sha256"] = sha(matrix_path)
    write_json(manifest, {k: v for k, v in payload.items() if k != "pipeline_manifest_sha256"}, "pipeline_manifest_sha256")
    report = audit_research_pipeline(manifest)
    assert report["audit_passed"] is False
    assert "common_test_windows" in report["failed_checks"]

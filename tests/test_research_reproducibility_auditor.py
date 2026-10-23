import hashlib
import json
from pathlib import Path

from app.research_experiment_fingerprint import build_experiment_fingerprint
from app.research_reproducibility_auditor import audit_experiment_reproducibility


def _write_run(tmp_path: Path):
    raw = tmp_path / "raw.csv"
    raw.write_text("raw-data\n", encoding="utf-8")
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("dataset-data\n", encoding="utf-8")
    dataset_manifest_path = tmp_path / "dataset.manifest.json"
    dataset_manifest = {
        "manifest_sha256": "unused-manifest-claim",
        "feature_implementation_sha256": "a" * 64,
        "feature_schema_sha256": "b" * 64,
        "feature_backend": "numpy_fallback",
        "actionable": False,
    }
    dataset_manifest_path.write_text(json.dumps(dataset_manifest), encoding="utf-8")
    fp = build_experiment_fingerprint(
        configuration={"seed": 11, "model_types": ["xgboost"]},
        raw_ohlc_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        dataset_manifest=dataset_manifest,
    )
    fp_path = tmp_path / "fingerprint.json"
    fp_path.write_text(json.dumps(fp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pipeline = {
        "run_id": "test-run",
        "raw_ohlc_sha256": fp["raw_ohlc_sha256"],
        "dataset_sha256": fp["dataset_sha256"],
        "dataset_manifest_path": str(dataset_manifest_path),
        "experiment_fingerprint_path": str(fp_path),
        "experiment_fingerprint_file_sha256": hashlib.sha256(fp_path.read_bytes()).hexdigest(),
        "experiment_fingerprint_sha256": fp["experiment_fingerprint_sha256"],
        "actionable": False,
        "execution_connected": False,
    }
    canonical = json.dumps(pipeline, sort_keys=True, separators=(",", ":"), allow_nan=False)
    pipeline["pipeline_manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    pipe_path = tmp_path / "pipeline.json"
    pipe_path.write_text(json.dumps(pipeline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return fp_path, pipe_path


def test_audit_passes_for_consistent_reproducibility_artifacts(tmp_path):
    fp_path, pipe_path = _write_run(tmp_path)
    result = audit_experiment_reproducibility(fp_path, pipe_path)
    assert result["status"] == "PASS"
    assert result["actionable"] is False
    assert result["execution_connected"] is False


def test_audit_detects_fingerprint_file_tampering(tmp_path):
    fp_path, pipe_path = _write_run(tmp_path)
    fingerprint = json.loads(fp_path.read_text(encoding="utf-8"))
    fingerprint["configuration"]["seed"] = 999
    fp_path.write_text(json.dumps(fingerprint), encoding="utf-8")
    result = audit_experiment_reproducibility(fp_path, pipe_path)
    assert result["status"] == "FAIL"
    assert "experiment_fingerprint_sha256_mismatch" in result["errors"]
    assert "pipeline_fingerprint_file_hash_mismatch" in result["errors"]


def test_audit_detects_pipeline_link_mismatch(tmp_path):
    fp_path, pipe_path = _write_run(tmp_path)
    pipeline = json.loads(pipe_path.read_text(encoding="utf-8"))
    pipeline["experiment_fingerprint_sha256"] = "0" * 64
    pipeline.pop("pipeline_manifest_sha256")
    pipeline["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(pipeline, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    pipe_path.write_text(json.dumps(pipeline), encoding="utf-8")
    result = audit_experiment_reproducibility(fp_path, pipe_path)
    assert result["status"] == "FAIL"
    assert "pipeline_fingerprint_hash_mismatch" in result["errors"]


def test_unicode_manifest_hash_uses_same_canonicalization(tmp_path):
    fp_path, pipe_path = _write_run(tmp_path)
    pipeline = json.loads(pipe_path.read_text(encoding="utf-8"))
    pipeline["note"] = "\u00e9"
    pipeline.pop("pipeline_manifest_sha256", None)
    pipeline["pipeline_manifest_sha256"] = hashlib.sha256(json.dumps(pipeline, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    pipe_path.write_text(json.dumps(pipeline, ensure_ascii=False), encoding="utf-8")
    result = audit_experiment_reproducibility(fp_path, pipe_path)
    assert "pipeline_manifest_sha256_mismatch" not in result["errors"]

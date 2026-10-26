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
        "symbol": "EURUSD",
        "timeframe": "M15",
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
        "symbol": "EURUSD",
        "timeframe": "M15",
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


def _write_completion_receipt(manifest_path: Path, *, status="PASS", manifest_hash=None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = {
        "schema_version": 1,
        "audit_name": "research_pipeline_completion_audit",
        "status": status,
        "run_id": manifest["run_id"],
        "pipeline_manifest_sha256": manifest_hash or manifest["pipeline_manifest_sha256"],
        "errors": [],
        "warnings": ["model_evaluation_gate_did_not_pass"] if status == "PASS_WITH_WARNINGS" else [],
        "checks": {
            "pipeline_manifest_self_hash": "PASS",
            "reproducibility_audit_status": "PASS",
            "model_evaluation_gate_file_hash": "PASS",
            "validation_report_status": "PASS",
        },
        "actionable": False,
        "execution_connected": False,
        "model_qualified": False,
        "deployment_authorized": False,
    }
    path = manifest_path.parent / f"{manifest['run_id']}_pipeline_completion_audit.json"
    _write_json(path, receipt)
    return path


def test_run_audit_verifies_pipeline_completion_receipt(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    receipt_path = _write_completion_receipt(manifest_path)

    result = audit_research_run_manifest(manifest_path)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["checked_artifact_count"] == 7
    assert "pipeline_completion_audit_path" in result["checked_artifacts"]
    assert result["errors"] == []
    assert "pipeline_completion_audit_source_hashes_unavailable" in result["warnings"]


def test_run_audit_rejects_failed_pipeline_completion_receipt(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    _write_completion_receipt(manifest_path, status="FAIL")

    result = audit_research_run_manifest(manifest_path)

    assert result["status"] == "FAIL"
    assert "pipeline_completion_audit_not_passing" in result["errors"]


def test_run_audit_rejects_completion_receipt_bound_to_another_manifest(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    _write_completion_receipt(manifest_path, manifest_hash="f" * 64)

    result = audit_research_run_manifest(manifest_path)

    assert result["status"] == "FAIL"
    assert "pipeline_completion_audit_manifest_hash_mismatch" in result["errors"]


def test_run_audit_rejects_completion_receipt_that_claims_model_qualification(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    receipt_path = _write_completion_receipt(manifest_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["model_qualified"] = True
    _write_json(receipt_path, receipt)

    result = audit_research_run_manifest(manifest_path)

    assert result["status"] == "FAIL"
    assert "pipeline_completion_audit_model_qualified_must_be_false" in result["errors"]


def test_run_audit_preserves_completion_warning_status(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    _write_completion_receipt(manifest_path, status="PASS_WITH_WARNINGS")

    result = audit_research_run_manifest(manifest_path)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert "pipeline_completion_audit_passed_with_warnings" in result["warnings"]


def _write_v2_completion_receipt(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    validation_path = manifest_path.parent / f"{run_id}_validation_report.json"
    reproducibility_path = manifest_path.parent / f"{run_id}_reproducibility_audit.json"
    evaluation_path = manifest_path.parent / f"{run_id}_model_evaluation_gate.json"
    _write_json(validation_path, {"status": "PASS", "run_id": run_id, "actionable": False})
    _write_json(reproducibility_path, {
        "schema_version": 1,
        "audit_name": "research_experiment_reproducibility_audit",
        "status": "PASS",
        "errors": [],
        "checks": {},
        "pipeline_manifest_path": str(manifest_path.resolve()),
        "fingerprint_path": None,
        "fingerprint_file_sha256": None,
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
    })
    _write_json(evaluation_path, {
        "schema_version": 1,
        "scope": "GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE",
        "status": "PASS",
        "run_id": run_id,
        "symbol": manifest["symbol"],
        "timeframe": manifest["timeframe"],
        "source_matrix_sha256": _sha(Path(manifest["experiment_matrix_report"])),
        "experiment_count": 1,
        "folds_checked": 1,
        "fold_metric_objects_checked": 1,
        "errors": [],
        "warnings": [],
        "actionable": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    })

    manifest.update({
        "reproducibility_audit_path": str(reproducibility_path),
        "model_evaluation_gate_path": str(evaluation_path),
        "model_evaluation_gate_sha256": _sha(evaluation_path),
        "model_evaluation_gate_status": "PASS",
    })
    manifest.pop("pipeline_manifest_sha256", None)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _write_json(manifest_path, manifest)

    receipt = {
        "schema_version": 2,
        "audit_name": "research_pipeline_completion_audit",
        "status": "PASS",
        "run_id": run_id,
        "pipeline_manifest_sha256": manifest["pipeline_manifest_sha256"],
        "validation_report_sha256": _sha(validation_path),
        "reproducibility_audit_sha256": _sha(reproducibility_path),
        "model_evaluation_gate_sha256": _sha(evaluation_path),
        "errors": [],
        "warnings": [],
        "checks": {
            "pipeline_manifest_self_hash": "PASS",
            "reproducibility_audit_status": "PASS",
            "model_evaluation_gate_file_hash": "PASS",
            "validation_report_status": "PASS",
        },
        "actionable": False,
        "execution_connected": False,
        "model_qualified": False,
        "deployment_authorized": False,
    }
    _write_json(manifest_path.parent / f"{run_id}_pipeline_completion_audit.json", receipt)
    return validation_path


def test_run_audit_verifies_completion_receipt_source_hashes(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    validation_path = _write_v2_completion_receipt(manifest_path)

    valid = audit_research_run_manifest(manifest_path)
    assert valid["status"] == "PASS"
    assert valid["errors"] == []

    validation_path.write_text(validation_path.read_text(encoding="utf-8") + "\\n", encoding="utf-8")
    tampered = audit_research_run_manifest(manifest_path)
    assert tampered["status"] == "FAIL"
    assert "pipeline_completion_audit_validation_report_source_hash_mismatch" in tampered["errors"]



def test_run_audit_detects_tampering_of_each_completion_source_report(tmp_path):
    for label in ("validation_report", "reproducibility_audit", "model_evaluation_gate"):
        case_dir = tmp_path / label
        case_dir.mkdir()
        manifest_path = _build_valid_run(case_dir)
        _write_v2_completion_receipt(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if label == "validation_report":
            source_path = case_dir / f"{manifest['run_id']}_validation_report.json"
        else:
            source_path = Path(manifest[f"{label}_path"])
        source_path.write_bytes(source_path.read_bytes() + bytes([10]))

        result = audit_research_run_manifest(manifest_path)

        assert result["status"] == "FAIL"
        assert f"pipeline_completion_audit_{label}_source_hash_mismatch" in result["errors"]



def test_run_audit_verifies_point_in_time_invariance_receipt_and_source_binding(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_path = tmp_path / "run-001_point_in_time_feature_invariance.json"
    receipt = {
        "schema_version": 1,
        "audit_name": "research_point_in_time_feature_invariance_audit",
        "scope": "OFFLINE_OHLC_PREFIX_INVARIANCE",
        "status": "PASS",
        "source_sha256": manifest["raw_ohlc_sha256"],
        "rows": 48,
        "feature_count": 15,
        "cutoff_count": 4,
        "feature_values_compared": 500,
        "errors": [],
        "actionable": False,
        "execution_connected": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
    }
    _write_json(receipt_path, receipt)
    manifest.update({
        "point_in_time_feature_invariance_audit_path": str(receipt_path),
        "point_in_time_feature_invariance_audit_sha256": _sha(receipt_path),
        "point_in_time_feature_invariance_audit_status": "PASS",
    })
    manifest.pop("pipeline_manifest_sha256", None)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _write_json(manifest_path, manifest)

    valid = audit_research_run_manifest(manifest_path)
    assert valid["status"] == "PASS"
    assert "point_in_time_feature_invariance_audit_path" in valid["checked_artifacts"]

    receipt_path.write_text(receipt_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered = audit_research_run_manifest(manifest_path)
    assert tampered["status"] == "FAIL"
    assert "point_in_time_feature_invariance_audit_path_sha256_mismatch" in tampered["errors"]


def _attach_feature_source_receipt(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = Path(manifest["dataset_path"])
    dataset_manifest_path = Path(manifest["dataset_manifest_path"])
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    schema_hash = "a" * 64
    implementation_hash = "b" * 64
    receipt_path = manifest_path.parent / "run_feature_source_audit.json"
    receipt = {
        "schema_version": 1,
        "audit_name": "research_feature_source_provenance_audit",
        "status": "PASS",
        "dataset_sha256": _sha(dataset_path),
        "manifest_sha256_declared": dataset_manifest.get("manifest_sha256"),
        "feature_implementation_sha256_declared": implementation_hash,
        "feature_schema_sha256_declared": schema_hash,
        "feature_count": 1,
        "dataset_rows": 1,
        "errors": [],
        "warnings": [],
        "actionable": False,
        "execution_connected": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
    }
    _write_json(receipt_path, receipt)
    manifest.update({
        "feature_source_audit_path": str(receipt_path),
        "feature_source_audit_sha256": _sha(receipt_path),
        "feature_source_audit_status": "PASS",
        "feature_schema_sha256": schema_hash,
        "feature_implementation_sha256": implementation_hash,
        "feature_count": 1,
    })
    manifest.pop("pipeline_manifest_sha256", None)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)
    return receipt_path


def test_run_audit_verifies_feature_source_provenance_receipt(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    receipt_path = _attach_feature_source_receipt(manifest_path)

    result = audit_research_run_manifest(manifest_path)

    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert "feature_source_audit_path" in result["checked_artifacts"]

    receipt_path.write_bytes(receipt_path.read_bytes() + b"\\n")
    tampered = audit_research_run_manifest(manifest_path)
    assert tampered["status"] == "FAIL"
    assert "feature_source_audit_path_sha256_mismatch" in tampered["errors"]


def test_run_audit_rejects_feature_source_receipt_bound_to_another_dataset(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    receipt_path = _attach_feature_source_receipt(manifest_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["dataset_sha256"] = "f" * 64
    _write_json(receipt_path, receipt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_source_audit_sha256"] = _sha(receipt_path)
    manifest.pop("pipeline_manifest_sha256", None)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)

    result = audit_research_run_manifest(manifest_path)

    assert result["status"] == "FAIL"
    assert "feature_source_audit_dataset_sha256_mismatch" in result["errors"]


def test_run_audit_rejects_feature_source_receipt_that_claims_actionability(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    receipt_path = _attach_feature_source_receipt(manifest_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["execution_connected"] = True
    _write_json(receipt_path, receipt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_source_audit_sha256"] = _sha(receipt_path)
    manifest.pop("pipeline_manifest_sha256", None)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)

    result = audit_research_run_manifest(manifest_path)

    assert result["status"] == "FAIL"
    assert "feature_source_audit_execution_connected_must_be_false" in result["errors"]


def _attach_experiment_fingerprint(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_manifest_path = Path(manifest["dataset_manifest_path"])
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest["feature_implementation_sha256"] = "b" * 64
    dataset_manifest["feature_schema_sha256"] = "a" * 64
    dataset_manifest.pop("manifest_sha256", None)
    dataset_manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(dataset_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(dataset_manifest_path, dataset_manifest)
    manifest["dataset_manifest_sha256"] = _sha(dataset_manifest_path)
    dataset_manifest.pop("manifest_sha256", None)
    manifest_content_hash = hashlib.sha256(
        json.dumps(dataset_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    dataset_manifest_file = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    receipt = {
        "schema_version": 1,
        "scope": "OFFLINE_RESEARCH_REPRODUCIBILITY",
        "configuration": {"symbol": "EURUSD", "timeframe": "M15"},
        "raw_ohlc_sha256": manifest["raw_ohlc_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_manifest_file_sha256": _sha(dataset_manifest_path),
        "dataset_manifest_content_sha256": manifest_content_hash,
        "feature_implementation_sha256": "b" * 64,
        "feature_schema_sha256": "a" * 64,
        "feature_backend": "test",
        "implementation_file_sha256": {"app/causal_ta_features.py": "c" * 64},
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
    }
    receipt["experiment_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    path = manifest_path.parent / "run_experiment_fingerprint.json"
    _write_json(path, receipt)
    manifest.update({
        "experiment_fingerprint_path": str(path),
        "experiment_fingerprint_file_sha256": _sha(path),
        "experiment_fingerprint_sha256": receipt["experiment_fingerprint_sha256"],
        "feature_schema_sha256": "a" * 64,
        "feature_implementation_sha256": "b" * 64,
    })
    manifest.pop("pipeline_manifest_sha256", None)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)
    return path


def test_run_audit_verifies_experiment_fingerprint_receipt_and_data_binding(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    fingerprint_path = _attach_experiment_fingerprint(manifest_path)

    valid = audit_research_run_manifest(manifest_path)
    assert valid["status"] == "PASS"
    assert valid["errors"] == []
    assert "experiment_fingerprint_path" in valid["checked_artifacts"]

    fingerprint_path.write_bytes(fingerprint_path.read_bytes() + b"\\n")
    tampered = audit_research_run_manifest(manifest_path)
    assert tampered["status"] == "FAIL"
    assert "experiment_fingerprint_file_sha256_mismatch" in tampered["errors"]


def test_run_audit_rejects_experiment_fingerprint_for_different_dataset(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    fingerprint_path = _attach_experiment_fingerprint(manifest_path)
    receipt = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    receipt["dataset_sha256"] = "f" * 64
    receipt.pop("experiment_fingerprint_sha256", None)
    receipt["experiment_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(fingerprint_path, receipt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment_fingerprint_file_sha256"] = _sha(fingerprint_path)
    manifest["experiment_fingerprint_sha256"] = receipt["experiment_fingerprint_sha256"]
    manifest.pop("pipeline_manifest_sha256", None)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)

    result = audit_research_run_manifest(manifest_path)
    assert result["status"] == "FAIL"
    assert "experiment_fingerprint_dataset_hash_mismatch" in result["errors"]


def _attach_model_evaluation_gate(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix_path = Path(manifest["experiment_matrix_report"])
    receipt_path = manifest_path.parent / "run_model_evaluation_gate.json"
    receipt = {
        "schema_version": 1,
        "scope": "GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE",
        "status": "PASS",
        "run_id": manifest["run_id"],
        "symbol": manifest["symbol"],
        "timeframe": manifest["timeframe"],
        "source_matrix_sha256": _sha(matrix_path),
        "experiment_count": 1,
        "folds_checked": 1,
        "fold_metric_objects_checked": 1,
        "errors": [],
        "warnings": [],
        "actionable": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    }
    _write_json(receipt_path, receipt)
    manifest.update({
        "model_evaluation_gate_path": str(receipt_path),
        "model_evaluation_gate_sha256": _sha(receipt_path),
        "model_evaluation_gate_status": "PASS",
    })
    manifest.pop("pipeline_manifest_sha256", None)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)
    return receipt_path


def test_run_audit_verifies_model_evaluation_gate_source_binding(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    receipt_path = _attach_model_evaluation_gate(manifest_path)

    valid = audit_research_run_manifest(manifest_path)
    assert valid["status"] == "PASS"
    assert valid["errors"] == []
    assert "model_evaluation_gate_path" in valid["checked_artifacts"]

    receipt_path.write_bytes(receipt_path.read_bytes() + b"\\n")
    tampered = audit_research_run_manifest(manifest_path)
    assert tampered["status"] == "FAIL"
    assert "model_evaluation_gate_path_sha256_mismatch" in tampered["errors"]


def test_run_audit_rejects_model_evaluation_gate_bound_to_other_matrix(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    receipt_path = _attach_model_evaluation_gate(manifest_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_matrix_sha256"] = "f" * 64
    _write_json(receipt_path, receipt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_evaluation_gate_sha256"] = _sha(receipt_path)
    manifest.pop("pipeline_manifest_sha256", None)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)

    result = audit_research_run_manifest(manifest_path)
    assert result["status"] == "FAIL"
    assert "model_evaluation_gate_source_matrix_hash_mismatch" in result["errors"]


def _attach_reproducibility_receipt(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_path = manifest_path.parent / "run_reproducibility_audit.json"
    receipt = {
        "schema_version": 1,
        "audit_name": "research_experiment_reproducibility_audit",
        "status": "PASS",
        "errors": [],
        "checks": {},
        "pipeline_manifest_path": str(manifest_path.resolve()),
        "fingerprint_path": None,
        "fingerprint_file_sha256": None,
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
    }
    _write_json(receipt_path, receipt)
    manifest["reproducibility_audit_path"] = str(receipt_path)
    manifest.pop("pipeline_manifest_sha256", None)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)
    return receipt_path


def test_run_audit_verifies_reproducibility_receipt_without_completion_receipt(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    receipt_path = _attach_reproducibility_receipt(manifest_path)

    result = audit_research_run_manifest(manifest_path)

    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert "reproducibility_audit_path" in result["checked_artifacts"]


def test_run_audit_rejects_reproducibility_receipt_bound_to_another_manifest(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    receipt_path = _attach_reproducibility_receipt(manifest_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["pipeline_manifest_path"] = str(tmp_path / "different_manifest.json")
    _write_json(receipt_path, receipt)

    result = audit_research_run_manifest(manifest_path)

    assert result["status"] == "FAIL"
    assert "reproducibility_audit_pipeline_manifest_path_mismatch" in result["errors"]


def _upgrade_dataset_manifest_integrity(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = Path(manifest["raw_ohlc_path"])
    dataset_path = Path(manifest["dataset_path"])
    dataset_manifest_path = Path(manifest["dataset_manifest_path"])
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest["input_snapshot_sha256"] = _sha(raw_path)
    dataset_manifest["output_dataset_sha256"] = _sha(dataset_path)
    dataset_manifest["feature_schema_sha256"] = "a" * 64
    dataset_manifest["feature_implementation_sha256"] = "b" * 64
    dataset_manifest.pop("manifest_sha256", None)
    dataset_manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(dataset_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(dataset_manifest_path, dataset_manifest)
    manifest["dataset_manifest_sha256"] = _sha(dataset_manifest_path)
    manifest["feature_schema_sha256"] = dataset_manifest["feature_schema_sha256"]
    manifest["feature_implementation_sha256"] = dataset_manifest["feature_implementation_sha256"]
    manifest.pop("pipeline_manifest_sha256", None)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)
    return dataset_manifest_path


def test_run_audit_verifies_dataset_manifest_self_hash_and_raw_binding(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    _upgrade_dataset_manifest_integrity(manifest_path)

    valid = audit_research_run_manifest(manifest_path)
    assert valid["status"] == "PASS"
    assert valid["errors"] == []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_manifest_path = Path(manifest["dataset_manifest_path"])
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest["input_snapshot_sha256"] = "f" * 64
    dataset_manifest.pop("manifest_sha256", None)
    dataset_manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(dataset_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(dataset_manifest_path, dataset_manifest)
    manifest["dataset_manifest_sha256"] = _sha(dataset_manifest_path)
    manifest.pop("pipeline_manifest_sha256", None)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)

    result = audit_research_run_manifest(manifest_path)
    assert result["status"] == "FAIL"
    assert "dataset_manifest_input_snapshot_hash_mismatch" in result["errors"]


def test_run_audit_rejects_invalid_dataset_manifest_self_hash(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    dataset_manifest_path = _upgrade_dataset_manifest_integrity(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest["manifest_sha256"] = "0" * 64
    _write_json(dataset_manifest_path, dataset_manifest)
    manifest["dataset_manifest_sha256"] = _sha(dataset_manifest_path)
    manifest.pop("pipeline_manifest_sha256", None)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)

    result = audit_research_run_manifest(manifest_path)
    assert result["status"] == "FAIL"
    assert "dataset_manifest_self_hash_mismatch" in result["errors"]


def test_run_audit_rejects_model_evaluation_gate_with_wrong_symbol(tmp_path):
    manifest_path = _build_valid_run(tmp_path)
    receipt_path = _attach_model_evaluation_gate(manifest_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["symbol"] = "GBPUSD"
    _write_json(receipt_path, receipt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_evaluation_gate_sha256"] = _sha(receipt_path)
    manifest.pop("pipeline_manifest_sha256", None)
    manifest["pipeline_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)

    result = audit_research_run_manifest(manifest_path)
    assert result["status"] == "FAIL"
    assert "model_evaluation_gate_symbol_mismatch" in result["errors"]


def test_audit_detects_artifact_mutation_after_snapshot(tmp_path, monkeypatch):
    manifest_path = _build_valid_run(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = Path(manifest["dataset_path"])
    original_read_bytes = Path.read_bytes
    reads = {"dataset": 0}

    def mutate_on_final_dataset_read(path):
        if path.resolve() == dataset_path.resolve():
            reads["dataset"] += 1
            if reads["dataset"] == 2:
                dataset_path.write_text("mutated after snapshot\\n", encoding="utf-8")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", mutate_on_final_dataset_read)

    result = audit_research_run_manifest(manifest_path)

    assert result["status"] == "FAIL"
    assert any(
        error == f"artifact_changed_during_audit:{dataset_path.name}"
        for error in result["errors"]
    )
    assert result["actionable"] is False
    assert result["execution_connected"] is False

from __future__ import annotations

import hashlib
import json

import pytest

from app.research_validation_workflow_audit import (
    REQUIRED_AUDITS,
    audit_research_validation_workflow,
    _load_json_object,
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def payload_for(name: str) -> dict:
    base = {
        "schema_version": 1, "audit_name": name, "status": "PASS",
        "errors": [], "warnings": [], "dataset_sha256": sha(b"same dataset"),
        "actionable": False, "inferential": False, "execution_connected": False,
        "model_training_performed": False, "automatic_model_selection": False,
        "automatic_repair_performed": False,
    }
    if name == "feature_temporal_integrity":
        base.update({
            "prefix_invariance": {"all_passed": True},
            "future_perturbation_invariance": {"all_passed": True},
            "chronological_partition_isolation": {"purge_gaps_sufficient_for_horizon": True},
        })
    elif name == "feature_backend_parity":
        base.update({"comparison_performed": True, "all_features_passed": True})
    elif name == "pipeline_leakage_boundary":
        base.update({"split_results": {"train": {}}, "preprocessing_results": [], "label_results": []})
    elif name == "research_artifact_bundle_certification":
        base.update({"missing_required_audits": [], "artifacts_checked": 4})
    elif name == "walk_forward_robustness":
        base.update({"evaluations_supplied": 4, "evaluations_validated": 4,
                     "unique_seed_count": 3, "observed_folds": ["1", "2"]})
    elif name == "research_cohort_comparability_and_distribution_shift":
        base.update({"runs_supplied": 4, "runs_validated": 4, "cohort_definition": {"symbol": "EURUSD"}})
        base.pop("dataset_sha256")
    elif name == "backtest_result_integrity":
        base.update({"trades_supplied": 10, "trades_validated": 10,
                     "net_pnl_recomputed_from_valid_trades": 42.5})
    return base


def make_manifest(tmp_path, names=REQUIRED_AUDITS):
    entries = []
    for name in names:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload_for(name)), encoding="utf-8")
        entries.append({"audit_name": name, "path": path.name})
    return {"artifacts": entries}


def run_audit(tmp_path, manifest):
    raw = json.dumps(manifest, sort_keys=True).encode()
    return audit_research_validation_workflow(
        manifest, manifest_sha256=sha(raw), base_dir=tmp_path
    )


def test_full_research_validation_bundle_passes(tmp_path):
    report = run_audit(tmp_path, make_manifest(tmp_path))
    assert report["status"] == "PASS"
    assert report["readiness"] == "READY"
    assert report["artifacts_checked"] == len(REQUIRED_AUDITS)
    assert report["errors"] == []
    assert report["execution_connected"] is False
    assert report["actionable"] is False


def test_missing_required_audit_fails_closed(tmp_path):
    report = run_audit(tmp_path, make_manifest(tmp_path, REQUIRED_AUDITS[:-1]))
    assert report["status"] == "FAIL"
    assert "NOT_READY" == report["readiness"]
    assert any(e["code"] == "required_audit_missing" for e in report["errors"])


def test_report_digest_mismatch_fails_closed(tmp_path):
    manifest = make_manifest(tmp_path)
    manifest["artifacts"][0]["expected_report_sha256"] = "0" * 64
    report = run_audit(tmp_path, manifest)
    assert any(e["code"] == "report_sha256_mismatch" for e in report["errors"])


def test_upstream_fail_and_missing_safety_flag_fail_closed(tmp_path):
    manifest = make_manifest(tmp_path)
    path = tmp_path / manifest["artifacts"][0]["path"]
    payload = json.loads(path.read_text())
    payload["status"] = "FAIL"
    payload["execution_connected"] = True
    path.write_text(json.dumps(payload))
    report = run_audit(tmp_path, manifest)
    codes = {e["code"] for e in report["errors"]}
    assert "upstream_audit_failed" in codes
    assert "safety_flag_not_explicitly_false:execution_connected" in codes


def test_dataset_fingerprint_mismatch_fails_closed(tmp_path):
    manifest = make_manifest(tmp_path)
    path = tmp_path / manifest["artifacts"][1]["path"]
    payload = json.loads(path.read_text())
    payload["dataset_sha256"] = sha(b"different dataset")
    path.write_text(json.dumps(payload))
    report = run_audit(tmp_path, manifest)
    assert any(e["code"] == "dataset_fingerprint_mismatch_across_reports" for e in report["errors"])


def test_path_traversal_and_duplicate_names_fail_closed(tmp_path):
    manifest = make_manifest(tmp_path)
    manifest["artifacts"].append({"audit_name": manifest["artifacts"][0]["audit_name"], "path": "../outside.json"})
    report = run_audit(tmp_path, manifest)
    codes = {e["code"] for e in report["errors"]}
    assert "duplicate_audit_name" in codes
    assert "artifact_path_must_be_relative_and_confined" in codes


def test_incomplete_backtest_ledger_is_not_ready(tmp_path):
    manifest = make_manifest(tmp_path)
    name = "backtest_result_integrity"
    path = tmp_path / f"{name}.json"
    payload = json.loads(path.read_text())
    payload["trades_validated"] = payload["trades_supplied"] - 1
    path.write_text(json.dumps(payload))
    report = run_audit(tmp_path, manifest)
    assert any(e["code"] == "backtest_trade_ledger_incomplete" for e in report["errors"])


def test_manifest_rejects_duplicate_required_audits(tmp_path):
    manifest = make_manifest(tmp_path)
    manifest["required_audits"] = ["feature_temporal_integrity", "feature_temporal_integrity"]
    with pytest.raises(ValueError, match="required_audits_must_be_unique"):
        run_audit(tmp_path, manifest)


@pytest.mark.parametrize("raw", [
    b'{"status":"PASS","status":"FAIL"}',
    b'{"metric":NaN}',
    b'{"metric":Infinity}',
    b'{"metric":-Infinity}',
])
def test_json_loader_rejects_duplicate_keys_and_nonstandard_constants(raw):
    with pytest.raises(ValueError):
        _load_json_object(raw, source="test-manifest.json")


@pytest.mark.parametrize("status", [["PASS"], {"status": "PASS"}, None, 7])
def test_validation_workflow_malformed_status_fails_closed(tmp_path, status):
    manifest = make_manifest(tmp_path)
    path = tmp_path / manifest["artifacts"][0]["path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = status
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = run_audit(tmp_path, manifest)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "status_unknown_or_missing" for item in report["errors"])

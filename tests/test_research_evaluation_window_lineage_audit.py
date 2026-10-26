from __future__ import annotations

import csv
import pytest
import hashlib
import json
from datetime import datetime

from app.research_evaluation_window_lineage_audit import audit_evaluation_window_lineage
from app.research_experiment_plan import build_experiment_plan


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fixture(tmp_path, timestamps=("2026-01-01T00:00:00Z", "2026-01-31T23:45:00Z")):
    plan = build_experiment_plan({
        "schema_version": 1,
        "plan_id": "lineage-window",
        "context": {"symbol": "EURUSD", "timeframe": "M15"},
        "config_sha256": sha(b"config"),
        "dataset_sha256": sha(b"dataset"),
        "code_commit_sha": "a" * 40,
        "seeds": [9],
        "folds": ["fold-1"],
        "evaluation_windows": [{
            "window_id": "eval-jan",
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-02-01T00:00:00Z",
        }],
    })
    run = plan["run_matrix"][0]
    (tmp_path / "outputs").mkdir()
    rows_path = tmp_path / "outputs" / "evaluation_rows.csv"
    with rows_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "close"])
        for i, timestamp in enumerate(timestamps):
            writer.writerow([timestamp, 1.1 + i / 1000])
    rows_raw = rows_path.read_bytes()
    metrics_path = tmp_path / "outputs" / "metrics.json"
    metrics_raw = b'{"net_pnl": 4.2}\n'
    metrics_path.write_bytes(metrics_raw)
    first = min(datetime.fromisoformat(t.replace("Z", "+00:00")) for t in timestamps).isoformat().replace("+00:00", "Z")
    last = max(datetime.fromisoformat(t.replace("Z", "+00:00")) for t in timestamps).isoformat().replace("+00:00", "Z")
    manifest = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "evaluation_evidence": {
            "dataset_sha256": run["dataset_sha256"],
            "evaluation_code_sha256": sha(b"evaluation-code"),
            "evaluation_rows_path": "outputs/evaluation_rows.csv",
            "evaluation_rows_sha256": sha(rows_raw),
            "timestamp_column": "timestamp",
            "evaluated_row_count": len(timestamps),
            "first_evaluated_timestamp": first,
            "last_evaluated_timestamp": last,
            "metrics_artifact_path": "outputs/metrics.json",
            "metrics_artifact_sha256": sha(metrics_raw),
        },
    }
    raw = json.dumps(manifest, sort_keys=True).encode()
    (tmp_path / "manifest.json").write_bytes(raw)
    registry = {
        "audit_name": "research_experiment_registry",
        "registry_id": "registry-lineage-test",
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "run_results": [{
            "run_id": run["run_id"],
            "status": "PASS",
            "manifest_path": "manifest.json",
            "manifest_sha256": sha(raw),
        }],
    }
    return plan, registry


def test_valid_row_lineage_passes(tmp_path):
    plan, registry = fixture(tmp_path)
    report = audit_evaluation_window_lineage(plan, registry, registry_root=tmp_path)
    assert report["status"] == "PASS"
    assert report["runs_checked"] == 1
    assert report["evaluation_rows_checked"] == 2
    assert report["metrics_artifacts_checked"] == 1
    assert report["deployment_authorized"] is False


def test_row_outside_window_fails(tmp_path):
    plan, registry = fixture(tmp_path, timestamps=("2025-12-31T23:59:59Z", "2026-01-31T23:45:00Z"))
    report = audit_evaluation_window_lineage(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "evaluation_row_outside_declared_window" for e in report["errors"])


def test_end_boundary_is_exclusive(tmp_path):
    plan, registry = fixture(tmp_path, timestamps=("2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"))
    report = audit_evaluation_window_lineage(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "evaluation_row_outside_declared_window" for e in report["errors"])


def test_dataset_digest_mismatch_fails(tmp_path):
    plan, registry = fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_evidence"]["dataset_sha256"] = sha(b"wrong dataset")
    raw = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(raw)
    registry["run_results"][0]["manifest_sha256"] = sha(raw)
    report = audit_evaluation_window_lineage(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "evaluation_dataset_digest_mismatch" for e in report["errors"])


def test_tampered_evaluation_rows_fail(tmp_path):
    plan, registry = fixture(tmp_path)
    (tmp_path / "outputs" / "evaluation_rows.csv").write_text("timestamp,close\n2026-01-15T00:00:00Z,1.2\n")
    report = audit_evaluation_window_lineage(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "evaluation_rows_sha256_mismatch_or_invalid" for e in report["errors"])


def test_metrics_artifact_tampering_fails(tmp_path):
    plan, registry = fixture(tmp_path)
    (tmp_path / "outputs" / "metrics.json").write_text('{"net_pnl":999}\n')
    report = audit_evaluation_window_lineage(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "metrics_artifact_sha256_mismatch_or_invalid" for e in report["errors"])


def test_missing_evidence_fails_closed(tmp_path):
    plan, registry = fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["evaluation_evidence"]
    raw = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(raw)
    registry["run_results"][0]["manifest_sha256"] = sha(raw)
    report = audit_evaluation_window_lineage(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "evaluation_evidence_missing" for e in report["errors"])


def test_unsafe_row_path_fails(tmp_path):
    plan, registry = fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_evidence"]["evaluation_rows_path"] = "../outside.csv"
    raw = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(raw)
    registry["run_results"][0]["manifest_sha256"] = sha(raw)
    report = audit_evaluation_window_lineage(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "evaluation_rows_path_must_be_confined" for e in report["errors"])


@pytest.mark.parametrize("raw", [
    b'{"schema_version":1,"schema_version":1}',
    b'{"evaluation_evidence":{"row_count":NaN}}',
    b'{"evaluation_evidence":{"row_count":Infinity}}',
    b'{"evaluation_evidence":{"row_count":-Infinity}}',
])
def test_manifest_strict_json_rejects_ambiguous_or_nonstandard_values(tmp_path, raw):
    plan, registry = fixture(tmp_path)
    (tmp_path / "manifest.json").write_bytes(raw)
    registry["run_results"][0]["manifest_sha256"] = sha(raw)
    report = audit_evaluation_window_lineage(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(error["code"] == "manifest_json_invalid" for error in report["errors"])

def test_registry_missing_errors_array_fails_closed(tmp_path):
    plan, registry = fixture(tmp_path)
    registry.pop("errors")
    with pytest.raises(ValueError, match="registry_report_errors_schema_invalid"):
        audit_evaluation_window_lineage(plan, registry, registry_root=tmp_path)


def test_registry_report_with_errors_fails_closed(tmp_path):
    plan, registry = fixture(tmp_path)
    registry["errors"] = [{"code": "upstream_integrity_error"}]
    with pytest.raises(ValueError, match="registry_report_contains_errors"):
        audit_evaluation_window_lineage(plan, registry, registry_root=tmp_path)


@pytest.mark.parametrize("warnings", [None, {"not": "a-list"}, [""], [None], ["valid", 1]])
def test_registry_malformed_warnings_fail_closed(tmp_path, warnings):
    plan, registry = fixture(tmp_path)
    registry["warnings"] = warnings
    with pytest.raises(ValueError, match="registry_report_warnings_schema_invalid"):
        audit_evaluation_window_lineage(plan, registry, registry_root=tmp_path)

def test_registry_warnings_are_propagated(tmp_path):
    plan, registry = fixture(tmp_path)
    registry["status"] = "PASS_WITH_WARNINGS"
    registry["warnings"] = ["registry_notice"]
    report = audit_evaluation_window_lineage(plan, registry, registry_root=tmp_path)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert "registry_report_has_warnings" in report["warnings"]


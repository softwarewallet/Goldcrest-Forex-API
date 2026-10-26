from __future__ import annotations

import hashlib
import json

import pytest

from app.research_evaluation_output_integrity_audit import audit_evaluation_output_integrity
from app.research_experiment_plan import build_experiment_plan


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def make_plan():
    return build_experiment_plan({
        "schema_version": 1,
        "plan_id": "window-integrity",
        "context": {"symbol": "EURUSD", "timeframe": "M15"},
        "config_sha256": sha(b"config"),
        "dataset_sha256": sha(b"dataset"),
        "code_commit_sha": "a" * 40,
        "seeds": [7],
        "folds": ["fold-1"],
        "evaluation_windows": [{
            "window_id": "eval-a",
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-02-01T00:00:00Z",
        }],
    })


def make_manifest(plan, run_id, artifact_path, artifact_bytes, *, window=None, artifacts=True):
    run = plan["run_matrix"][0]
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "evaluation_window": window or dict(run["evaluation_window"]),
        "metrics": {"net_pnl": 12.5, "max_drawdown": 3.0},
        "artifacts": ([{"path": artifact_path, "sha256": sha(artifact_bytes)}] if artifacts else []),
    }
    return payload


def registry_for(tmp_path, plan, *, window=None, artifacts=True, artifact_bytes=b"metrics-csv\n12.5\n"):
    run_id = plan["run_matrix"][0]["run_id"]
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "metrics.csv").write_bytes(artifact_bytes)
    manifest = make_manifest(plan, run_id, "results/metrics.csv", artifact_bytes, window=window, artifacts=artifacts)
    manifest_raw = json.dumps(manifest, sort_keys=True).encode()
    (tmp_path / "manifest.json").write_bytes(manifest_raw)
    report = {
        "audit_name": "research_experiment_registry",
        "registry_id": "registry-integrity-test",
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "run_results": [{
            "run_id": run_id,
            "manifest_path": "manifest.json",
            "manifest_sha256": sha(manifest_raw),
            "status": "PASS",
        }],
    }
    return report


def test_valid_window_metrics_and_artifact_pass(tmp_path):
    plan = make_plan()
    registry = registry_for(tmp_path, plan)
    report = audit_evaluation_output_integrity(plan, registry, registry_root=tmp_path)
    assert report["status"] == "PASS"
    assert report["planned_runs_checked"] == 1
    assert report["artifacts_checked"] == 1
    assert report["finite_metric_values_checked"] == 2
    assert report["deployment_authorized"] is False


def test_window_mismatch_fails_closed(tmp_path):
    plan = make_plan()
    registry = registry_for(tmp_path, plan, window={
        "start": "2025-12-01T00:00:00Z",
        "end": "2026-02-01T00:00:00Z",
    })
    report = audit_evaluation_output_integrity(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(error["code"] == "evaluation_window_does_not_match_plan" for error in report["errors"])


def test_missing_output_artifact_fails_closed(tmp_path):
    plan = make_plan()
    registry = registry_for(tmp_path, plan, artifacts=False)
    report = audit_evaluation_output_integrity(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(error["code"] == "output_artifact_evidence_missing" for error in report["errors"])


def test_artifact_checksum_mismatch_fails_closed(tmp_path):
    plan = make_plan()
    registry = registry_for(tmp_path, plan)
    (tmp_path / "results" / "metrics.csv").write_bytes(b"tampered")
    report = audit_evaluation_output_integrity(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(error["code"] == "artifact_sha256_mismatch_or_invalid" for error in report["errors"])


def test_non_finite_metric_fails_closed(tmp_path):
    plan = make_plan()
    registry = registry_for(tmp_path, plan)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["metrics"]["net_pnl"] = float("nan")
    raw = json.dumps(manifest, sort_keys=True).encode()
    (tmp_path / "manifest.json").write_bytes(raw)
    registry["run_results"][0]["manifest_sha256"] = sha(raw)
    report = audit_evaluation_output_integrity(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(error["code"] in {"evaluation_metric_invalid", "manifest_json_invalid"} for error in report["errors"])


def test_manifest_path_traversal_fails_closed(tmp_path):
    plan = make_plan()
    registry = registry_for(tmp_path, plan)
    registry["run_results"][0]["manifest_path"] = "../manifest.json"
    report = audit_evaluation_output_integrity(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(error["code"] == "manifest_path_must_be_confined" for error in report["errors"])


def test_missing_planned_run_fails_closed(tmp_path):
    plan = make_plan()
    registry = registry_for(tmp_path, plan)
    registry["run_results"] = []
    report = audit_evaluation_output_integrity(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(error["code"] == "planned_run_missing_from_registry" for error in report["errors"])


def test_allow_no_artifacts_is_explicit_opt_out(tmp_path):
    plan = make_plan()
    registry = registry_for(tmp_path, plan, artifacts=False)
    report = audit_evaluation_output_integrity(
        plan, registry, registry_root=tmp_path, require_output_artifact=False
    )
    assert report["status"] == "PASS"
    assert report["require_output_artifact"] is False


@pytest.mark.parametrize("raw", [
    b'{"schema_version":1,"schema_version":1}',
    b'{"metrics":{"net_pnl":NaN}}',
    b'{"metrics":{"net_pnl":Infinity}}',
    b'{"metrics":{"net_pnl":-Infinity}}',
])
def test_manifest_strict_json_rejects_ambiguous_or_nonstandard_values(tmp_path, raw):
    plan = make_plan()
    registry = registry_for(tmp_path, plan)
    (tmp_path / "manifest.json").write_bytes(raw)
    registry["run_results"][0]["manifest_sha256"] = sha(raw)
    report = audit_evaluation_output_integrity(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(error["code"] == "manifest_json_invalid" for error in report["errors"])

def test_registry_missing_errors_fails_closed(tmp_path):
    plan = make_plan()
    registry = registry_for(tmp_path, plan)
    registry.pop("errors")
    with pytest.raises(ValueError, match="registry_report_errors_schema_invalid"):
        audit_evaluation_output_integrity(plan, registry, registry_root=tmp_path)


def test_registry_nonempty_errors_fails_closed(tmp_path):
    plan = make_plan()
    registry = registry_for(tmp_path, plan)
    registry["errors"] = [{"code": "upstream_error"}]
    with pytest.raises(ValueError, match="registry_report_contains_errors"):
        audit_evaluation_output_integrity(plan, registry, registry_root=tmp_path)


@pytest.mark.parametrize("warnings", [None, {"bad": "shape"}, [""], [None], ["ok", 1]])
def test_registry_malformed_warnings_fails_closed(tmp_path, warnings):
    plan = make_plan()
    registry = registry_for(tmp_path, plan)
    registry["warnings"] = warnings
    with pytest.raises(ValueError, match="registry_report_warnings_schema_invalid"):
        audit_evaluation_output_integrity(plan, registry, registry_root=tmp_path)


def test_registry_warnings_are_propagated(tmp_path):
    plan = make_plan()
    registry = registry_for(tmp_path, plan)
    registry["status"] = "PASS_WITH_WARNINGS"
    registry["warnings"] = ["registry_notice"]
    report = audit_evaluation_output_integrity(plan, registry, registry_root=tmp_path)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert "registry_report_has_warnings" in report["warnings"]


from __future__ import annotations

import csv
import hashlib
import json

import pytest

from app.research_experiment_plan import build_experiment_plan
from app.research_metric_semantic_recalculation_audit import audit_metric_semantics, _load_json_object


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fixture(tmp_path):
    plan = build_experiment_plan({
        "schema_version": 1,
        "plan_id": "metric-semantics",
        "context": {"symbol": "EURUSD", "timeframe": "M15"},
        "config_sha256": sha(b"config"),
        "dataset_sha256": sha(b"dataset"),
        "code_commit_sha": "a" * 40,
        "seeds": [1],
        "folds": ["fold-1"],
        "evaluation_windows": [{"window_id": "eval-1", "start": "2026-01-01T00:00:00Z", "end": "2026-02-01T00:00:00Z"}],
    })
    run = plan["run_matrix"][0]
    (tmp_path / "outputs").mkdir()
    rows_path = tmp_path / "outputs" / "rows.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "return"])
        writer.writerow(["2026-01-01T00:00:00Z", "1.5"])
        writer.writerow(["2026-01-02T00:00:00Z", "-0.5"])
        writer.writerow(["2026-01-03T00:00:00Z", "2.0"])
    rows_raw = rows_path.read_bytes()
    metric_path = tmp_path / "outputs" / "metrics.json"
    metric_raw = json.dumps({"metrics": {"count": 3, "mean_return": 1.0, "sum_return": 3.0, "median_return": 1.5}}, sort_keys=True).encode()
    metric_path.write_bytes(metric_raw)
    manifest = {
        "run_id": run["run_id"],
        "evaluation_evidence": {
            "dataset_sha256": run["dataset_sha256"],
            "evaluation_code_sha256": sha(b"evaluation-code"),
            "evaluation_rows_path": "outputs/rows.csv",
            "evaluation_rows_sha256": sha(rows_raw),
            "metrics_artifact_path": "outputs/metrics.json",
            "metrics_artifact_sha256": sha(metric_raw),
            "metric_recalculation_contract": {
                "version": "1",
                "metrics": {
                    "count": {"kind": "row_count"},
                    "mean_return": {"kind": "mean", "column": "return"},
                    "sum_return": {"kind": "sum", "column": "return"},
                    "median_return": {"kind": "median", "column": "return"},
                },
            },
        },
    }
    manifest_raw = json.dumps(manifest, sort_keys=True).encode()
    (tmp_path / "manifest.json").write_bytes(manifest_raw)
    registry = {
        "audit_name": "research_experiment_registry",
        "registry_id": "registry-metric-test",
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "run_results": [{
            "run_id": run["run_id"],
            "status": "PASS",
            "manifest_path": "manifest.json",
            "manifest_sha256": sha(manifest_raw),
        }],
    }
    return plan, registry


def rewrite_manifest(tmp_path, registry, edit):
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    raw = json.dumps(manifest, sort_keys=True).encode()
    path.write_bytes(raw)
    registry["run_results"][0]["manifest_sha256"] = sha(raw)


def test_recalculates_declared_metrics(tmp_path):
    plan, registry = fixture(tmp_path)
    report = audit_metric_semantics(plan, registry, registry_root=tmp_path)
    assert report["status"] == "PASS"
    assert report["runs_checked"] == 1
    assert report["metrics_recalculated"] == 4
    assert report["deployment_authorized"] is False


def test_metric_value_mismatch_fails(tmp_path):
    plan, registry = fixture(tmp_path)
    path = tmp_path / "outputs" / "metrics.json"
    raw = json.dumps({"metrics": {"count": 3, "mean_return": 999.0, "sum_return": 3.0, "median_return": 1.5}}, sort_keys=True).encode()
    path.write_bytes(raw)
    rewrite_manifest(tmp_path, registry, lambda m: m["evaluation_evidence"].update(metrics_artifact_sha256=sha(raw)))
    report = audit_metric_semantics(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "metric_recalculation_mismatch" for e in report["errors"])


def test_missing_contract_fails_closed(tmp_path):
    plan, registry = fixture(tmp_path)
    rewrite_manifest(tmp_path, registry, lambda m: m["evaluation_evidence"].pop("metric_recalculation_contract"))
    report = audit_metric_semantics(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "metric_recalculation_contract_missing_or_invalid" for e in report["errors"])


def test_unsupported_metric_kind_fails(tmp_path):
    plan, registry = fixture(tmp_path)
    rewrite_manifest(tmp_path, registry, lambda m: m["evaluation_evidence"]["metric_recalculation_contract"]["metrics"].update(bad={"kind": "sharpe_ratio", "column": "return"}))
    report = audit_metric_semantics(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "metric_contract_kind_or_column_invalid" for e in report["errors"])


def test_tampered_rows_fail_before_recalculation(tmp_path):
    plan, registry = fixture(tmp_path)
    (tmp_path / "outputs" / "rows.csv").write_text("timestamp,return\n2026-01-01T00:00:00Z,100\n")
    report = audit_metric_semantics(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "evaluation_rows_sha256_mismatch_or_invalid" for e in report["errors"])


def test_unsafe_rows_path_fails(tmp_path):
    plan, registry = fixture(tmp_path)
    rewrite_manifest(tmp_path, registry, lambda m: m["evaluation_evidence"].update(evaluation_rows_path="../outside.csv"))
    report = audit_metric_semantics(plan, registry, registry_root=tmp_path)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "evaluation_rows_path_must_be_confined" for e in report["errors"])


@pytest.mark.parametrize("raw", [
    b'{"status":"PASS","status":"FAIL"}', b'{"metric":NaN}',
    b'{"metric":Infinity}', b'{"metric":-Infinity}',
])
def test_strict_json_loader_rejects_ambiguous_inputs(raw):
    with pytest.raises(ValueError):
        _load_json_object(raw, source="metric-manifest.json")

def test_registry_missing_errors_fails_closed(tmp_path):
    plan, registry = fixture(tmp_path)
    registry.pop("errors")
    with pytest.raises(ValueError, match="registry_report_errors_schema_invalid"):
        audit_metric_semantics(plan, registry, registry_root=tmp_path)


def test_registry_nonempty_errors_fails_closed(tmp_path):
    plan, registry = fixture(tmp_path)
    registry["errors"] = [{"code": "upstream_error"}]
    with pytest.raises(ValueError, match="registry_report_contains_errors"):
        audit_metric_semantics(plan, registry, registry_root=tmp_path)


@pytest.mark.parametrize("warnings", [{"bad": "shape"}, [""], [None], ["ok", 1]])
def test_registry_malformed_warnings_fails_closed(tmp_path, warnings):
    plan, registry = fixture(tmp_path)
    registry["warnings"] = warnings
    with pytest.raises(ValueError, match="registry_report_warnings_schema_invalid"):
        audit_metric_semantics(plan, registry, registry_root=tmp_path)


def test_registry_warnings_are_propagated(tmp_path):
    plan, registry = fixture(tmp_path)
    registry["status"] = "PASS_WITH_WARNINGS"
    registry["warnings"] = ["registry_notice"]
    report = audit_metric_semantics(plan, registry, registry_root=tmp_path)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert "registry_report_has_warnings" in report["warnings"]


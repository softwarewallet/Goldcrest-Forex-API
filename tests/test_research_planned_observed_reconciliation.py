from __future__ import annotations

import hashlib

import pytest

from app.research_experiment_plan import build_experiment_plan
from app.research_planned_observed_reconciliation import reconcile_planned_observed_runs


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def plan_report():
    return build_experiment_plan({
        "schema_version": 1,
        "plan_id": "eurusd-reconciliation",
        "context": {"symbol": "EURUSD", "timeframe": "M15"},
        "config_sha256": digest(b"config"),
        "dataset_sha256": digest(b"dataset"),
        "code_commit_sha": "a" * 40,
        "seeds": [7],
        "folds": ["fold-1", "fold-2"],
        "evaluation_windows": [
            {"window_id": "window-a", "start": "2026-01-01T00:00:00Z", "end": "2026-02-01T00:00:00Z"}
        ],
    })


def registry_report(plan, *, include_all=True):
    matrix = plan["run_matrix"] if include_all else plan["run_matrix"][:1]
    results = []
    for item in matrix:
        results.append({
            "run_id": item["run_id"],
            "seed": item["seed"],
            "fold_id": item["fold_id"],
            "window_id": item["window_id"],
            "evaluation_window": item["evaluation_window"],
            "context": item["context"],
            "config_sha256": item["config_sha256"],
            "dataset_sha256": item["dataset_sha256"],
            "code_commit_sha": item["code_commit_sha"],
            "status": "PASS",
        })
    return {
        "audit_name": "research_experiment_registry",
        "registry_id": "registry-test",
        "status": "PASS",
        "runs_supplied": len(results),
        "runs_validated": len(results),
        "run_results": results,
    }


def test_full_plan_reconciles_against_registry():
    plan = plan_report()
    report = reconcile_planned_observed_runs(plan, registry_report(plan))
    assert report["status"] == "PASS"
    assert report["readiness"] == "PLANNED_OBSERVED_RUNS_RECONCILED"
    assert report["planned_runs"] == 2
    assert report["observed_runs"] == 2
    assert report["matched_runs"] == 2
    assert report["missing_run_ids"] == []
    assert report["execution_connected"] is False


def test_missing_planned_run_fails_closed():
    plan = plan_report()
    report = reconcile_planned_observed_runs(plan, registry_report(plan, include_all=False))
    assert report["status"] == "FAIL"
    assert len(report["missing_run_ids"]) == 1
    assert any(item["code"] == "planned_run_missing" for item in report["errors"])


def test_unexpected_run_is_warning_not_silent_acceptance():
    plan = plan_report()
    registry = registry_report(plan)
    extra = dict(registry["run_results"][0])
    extra["run_id"] = "exploratory-run"
    registry["run_results"].append(extra)
    registry["runs_supplied"] += 1
    registry["runs_validated"] += 1
    report = reconcile_planned_observed_runs(plan, registry)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert report["unexpected_run_ids"] == ["exploratory-run"]
    assert any("unexpected_observed_run" in warning for warning in report["warnings"])


def test_dimension_mismatch_fails_closed():
    plan = plan_report()
    registry = registry_report(plan)
    registry["run_results"][0]["dataset_sha256"] = digest(b"other-dataset")
    report = reconcile_planned_observed_runs(plan, registry)
    assert report["status"] == "FAIL"
    assert any(
        item["code"] == "planned_observed_dimension_mismatch"
        and item["field"] == "dataset_sha256"
        for item in report["errors"]
    )


def test_unvalidated_observed_run_fails_closed():
    plan = plan_report()
    registry = registry_report(plan)
    registry["run_results"][0]["status"] = "FAIL"
    registry["runs_validated"] -= 1
    report = reconcile_planned_observed_runs(plan, registry)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "observed_run_not_validated" for item in report["errors"])


def test_nonpassing_registry_report_is_rejected():
    plan = plan_report()
    registry = registry_report(plan)
    registry["status"] = "FAIL"
    with pytest.raises(ValueError, match="registry_report_not_passing"):
        reconcile_planned_observed_runs(plan, registry)

@pytest.mark.parametrize(
    "raw",
    [
        b'{"audit_name":"research_experiment_plan","audit_name":"shadowed"}',
        b'{"metric":NaN}',
        b'{"metric":Infinity}',
        b'{"metric":-Infinity}',
    ],
)
def test_reconciliation_reader_rejects_ambiguous_or_nonstandard_json(tmp_path, raw):
    from app.research_planned_observed_reconciliation import _read_object

    path = tmp_path / "input.json"
    path.write_bytes(raw)

    with pytest.raises(ValueError):
        _read_object(path)


def test_reconciliation_reader_requires_object_root(tmp_path):
    from app.research_planned_observed_reconciliation import _read_object

    path = tmp_path / "input.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="json_root_must_be_object"):
        _read_object(path)


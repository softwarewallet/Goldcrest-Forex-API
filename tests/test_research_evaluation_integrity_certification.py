from __future__ import annotations

import pytest

from copy import deepcopy

from app.research_evaluation_integrity_certification import certify_evaluation_integrity


FALSE_FLAGS = {
    "actionable": False,
    "inferential": False,
    "statistical_significance_claimed": False,
    "automatic_model_selection": False,
    "model_training_performed": False,
    "execution_connected": False,
    "automatic_repair_performed": False,
    "deployment_authorized": False,
}


def fixtures():
    plan = {
        "audit_name": "research_experiment_plan",
        "status": "PASS",
        "plan_id": "plan-1",
        "plan_sha256": "a" * 64,
        "expected_runs": 2,
        "run_matrix": [{"run_id": "r1"}, {"run_id": "r2"}],
        **FALSE_FLAGS,
    }
    registry = {
        "audit_name": "research_experiment_registry",
        "status": "PASS",
        "registry_id": "registry-1",
        "errors": [],
        **FALSE_FLAGS,
    }
    output = {
        "audit_name": "research_evaluation_output_integrity",
        "status": "PASS",
        "plan_id": "plan-1",
        "plan_sha256": "a" * 64,
        "registry_id": "registry-1",
        "planned_runs_checked": 2,
        "missing_planned_runs": [],
        "errors": [],
        **FALSE_FLAGS,
    }
    window = {
        "audit_name": "research_evaluation_window_lineage",
        "status": "PASS",
        "plan_id": "plan-1",
        "plan_sha256": "a" * 64,
        "registry_id": "registry-1",
        "runs_checked": 2,
        "evaluation_rows_checked": 100,
        "warnings": [],
        "errors": [],
        **FALSE_FLAGS,
    }
    metrics = {
        "audit_name": "research_metric_semantic_recalculation",
        "status": "PASS",
        "plan_id": "plan-1",
        "plan_sha256": "a" * 64,
        "registry_id": "registry-1",
        "runs_checked": 2,
        "metrics_recalculated": 8,
        "warnings": [],
        "errors": [],
        **FALSE_FLAGS,
    }
    return plan, registry, output, window, metrics


def run(values):
    return certify_evaluation_integrity(*values)


def test_all_audits_pass_certification():
    report = run(fixtures())
    assert report["status"] == "PASS"
    assert report["readiness"] == "EVALUATION_INTEGRITY_CERTIFIED"
    assert report["evaluation_rows_checked"] == 100
    assert report["metrics_recalculated"] == 8
    assert report["deployment_authorized"] is False


def test_plan_fingerprint_mismatch_fails():
    values = fixtures()
    values[3]["plan_sha256"] = "b" * 64
    report = run(values)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "plan_sha256_mismatch" and e["report"] == "row_window_lineage" for e in report["errors"])


def test_upstream_error_fails_closed():
    values = fixtures()
    values[4]["errors"] = [{"code": "metric_recalculation_mismatch"}]
    report = run(values)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "upstream_report_contains_errors" and e["report"] == "metric_semantics" for e in report["errors"])


def test_incomplete_run_coverage_fails():
    values = fixtures()
    values[2]["planned_runs_checked"] = 1
    report = run(values)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "validated_run_coverage_mismatch" and e["report"] == "output_integrity" for e in report["errors"])


def test_safety_flag_must_be_explicitly_false():
    values = fixtures()
    values[4]["execution_connected"] = True
    report = run(values)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "safety_flag_not_explicitly_false" and e["report"] == "metric_semantics" for e in report["errors"])


def test_warning_is_preserved():
    values = fixtures()
    values[3]["status"] = "PASS_WITH_WARNINGS"
    report = run(values)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert "row_window_lineage_passed_with_warnings" in report["warnings"]


def test_wrong_registry_identity_fails():
    values = fixtures()
    values[2]["registry_id"] = "different-registry"
    report = run(values)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "registry_id_mismatch" and e["report"] == "output_integrity" for e in report["errors"])


def test_deployment_authorization_true_fails():
    values = fixtures()
    values[1]["deployment_authorized"] = True
    report = run(values)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "deployment_must_remain_unauthorized" and e["report"] == "registry" for e in report["errors"])

@pytest.mark.parametrize("raw", [b'{"metric":NaN}', b'{"metric":Infinity}', b'{"metric":-Infinity}', b'{"value":1,"value":2}'])
def test_certification_json_readers_reject_nonstandard_json(tmp_path, raw):
    from app.research_evaluation_integrity_certification import _read_object
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        _read_object(path)


def test_missing_required_errors_array_fails_closed():
    values = fixtures()
    values[1].pop("errors", None)
    report = run(values)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "upstream_errors_schema_invalid" and e["report"] == "registry" for e in report["errors"])


def test_non_list_warnings_fails_closed():
    values = fixtures()
    values[4]["warnings"] = "not-a-list"
    report = run(values)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "upstream_warnings_schema_invalid" and e["report"] == "metric_semantics" for e in report["errors"])


def test_plan_may_omit_errors_array_but_cannot_supply_malformed_one():
    values = fixtures()
    report = run(values)
    assert report["status"] == "PASS"
    values = fixtures()
    values[0]["errors"] = "not-a-list"
    report = run(values)
    assert report["status"] == "FAIL"
    assert any(e["code"] == "upstream_errors_schema_invalid" and e["report"] == "plan" for e in report["errors"])

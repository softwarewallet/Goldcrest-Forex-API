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
        "expected_run_ids": ["r1", "r2"],
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
    values[3]["warnings"] = ["fixture warning"]
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

@pytest.mark.parametrize(
    ("report_index", "field", "expected_report"),
    [
        (0, "errors", "plan"),
        (3, "warnings", "row_window_lineage"),
        (4, "warnings", "metric_semantics"),
    ],
)
def test_explicit_null_diagnostics_fail_closed(report_index, field, expected_report):
    values = fixtures()
    values[report_index][field] = None

    report = run(values)

    assert report["status"] == "FAIL"
    expected_code = "upstream_errors_schema_invalid" if field == "errors" else "upstream_warnings_schema_invalid"
    assert any(
        error["code"] == expected_code and error["report"] == expected_report
        for error in report["errors"]
    )



def test_plan_run_count_must_match_run_matrix():
    values = list(fixtures())
    values[0]["expected_runs"] = 3

    report = run(values)

    assert report["status"] == "FAIL"
    assert any(error["code"] == "planned_run_count_matrix_mismatch" for error in report["errors"])


def test_plan_expected_run_ids_must_match_matrix_order_and_identity():
    values = list(fixtures())
    values[0]["expected_run_ids"] = ["r1", "other"]

    report = run(values)

    assert report["status"] == "FAIL"
    assert any(error["code"] == "expected_run_ids_matrix_mismatch" for error in report["errors"])


def test_plan_duplicate_run_ids_fail_even_when_reported_counts_match():
    values = list(fixtures())
    values[0]["run_matrix"] = [{"run_id": "r1"}, {"run_id": "r1"}]
    values[0]["expected_run_ids"] = ["r1", "r1"]

    report = run(values)

    assert report["status"] == "FAIL"
    assert any(error["code"] == "planned_run_matrix_duplicate_run_id" for error in report["errors"])
    assert any(error["code"] == "expected_run_ids_invalid" for error in report["errors"])


@pytest.mark.parametrize(
    ("report_index", "field", "value", "expected_code"),
    [
        (0, "plan_id", "", "plan_id_invalid"),
        (0, "plan_id", " plan-1 ", "plan_id_invalid"),
        (0, "plan_sha256", "A" * 64, "plan_sha256_invalid"),
        (0, "plan_sha256", "not-a-digest", "plan_sha256_invalid"),
        (1, "registry_id", "   ", "registry_id_invalid"),
        (1, "registry_id", " registry-1 ", "registry_id_invalid"),
    ],
)
def test_noncanonical_certification_identity_fails_closed(report_index, field, value, expected_code):
    values = fixtures()
    values[report_index][field] = value

    report = run(values)

    assert report["status"] == "FAIL"
    assert any(error["code"] == expected_code for error in report["errors"])


@pytest.mark.parametrize(
    ("matrix", "expected_ids", "expected_code"),
    [
        ([{"run_id": " r1 "}, {"run_id": "r2"}], [" r1 ", "r2"], "planned_run_matrix_entry_invalid"),
        ([{"run_id": "r1"}, {"run_id": "r2"}], ["r1", " r2 "], "expected_run_ids_invalid"),
    ],
)
def test_whitespace_padded_run_ids_fail_closed(matrix, expected_ids, expected_code):
    values = fixtures()
    values[0]["run_matrix"] = matrix
    values[0]["expected_run_ids"] = expected_ids

    report = run(values)

    assert report["status"] == "FAIL"
    assert any(error["code"] == expected_code for error in report["errors"])


@pytest.mark.parametrize(
    ("status", "warnings"),
    [("PASS", ["warning not reflected in status"]), ("PASS_WITH_WARNINGS", [])],
)
def test_upstream_status_must_match_warning_collection(status, warnings):
    values = fixtures()
    values[3]["status"] = status
    values[3]["warnings"] = warnings

    report = run(values)

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "upstream_status_warnings_mismatch"
        and error["report"] == "row_window_lineage"
        for error in report["errors"]
    )


@pytest.mark.parametrize("bad_status", [["PASS"], {"value": "PASS"}])
def test_unhashable_upstream_status_fails_closed(bad_status):
    values = fixtures()
    values[1]["status"] = bad_status

    report = run(values)

    assert report["status"] == "FAIL"
    assert any(
        error["code"] == "input_report_not_passing" and error["report"] == "registry"
        for error in report["errors"]
    )

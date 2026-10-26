"""Offline certification gate combining evaluation output, row-window, and metric-semantic audits.

This is a report-only meta-audit. It validates report identity, statuses, counts, and safety flags;
it does not rerun upstream audits or imply that a strategy is profitable or deployable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads

PASS_STATUSES = {"PASS", "PASS_WITH_WARNINGS"}
REQUIRED_FALSE_FLAGS = (
    "actionable",
    "inferential",
    "statistical_significance_claimed",
    "automatic_model_selection",
    "model_training_performed",
    "execution_connected",
    "automatic_repair_performed",
)


def certify_evaluation_integrity(
    plan: dict[str, Any],
    registry: dict[str, Any],
    output_integrity: dict[str, Any],
    row_window_lineage: dict[str, Any],
    metric_semantics: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "plan": ("research_experiment_plan", {"PASS"}),
        "registry": ("research_experiment_registry", PASS_STATUSES),
        "output_integrity": ("research_evaluation_output_integrity", PASS_STATUSES),
        "row_window_lineage": ("research_evaluation_window_lineage", PASS_STATUSES),
        "metric_semantics": ("research_metric_semantic_recalculation", PASS_STATUSES),
    }
    inputs = {
        "plan": plan,
        "registry": registry,
        "output_integrity": output_integrity,
        "row_window_lineage": row_window_lineage,
        "metric_semantics": metric_semantics,
    }
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    for name, payload in inputs.items():
        audit_name, allowed = expected[name]
        if not isinstance(payload, dict):
            errors.append({"code": "input_report_invalid", "report": name})
            continue
        if payload.get("audit_name") != audit_name:
            errors.append({"code": "input_audit_name_mismatch", "report": name, "expected": audit_name, "actual": payload.get("audit_name")})
        if payload.get("status") not in allowed:
            errors.append({"code": "input_report_not_passing", "report": name, "status": payload.get("status")})
        for flag in REQUIRED_FALSE_FLAGS:
            if payload.get(flag) is not False:
                errors.append({"code": "safety_flag_not_explicitly_false", "report": name, "flag": flag})
        if payload.get("deployment_authorized") is not False:
            errors.append({"code": "deployment_must_remain_unauthorized", "report": name})
        if payload.get("status") == "PASS_WITH_WARNINGS":
            warnings.append(f"{name}_passed_with_warnings")
        upstream_errors = payload.get("errors")
        if name != "plan" and not isinstance(upstream_errors, list):
            errors.append({"code": "upstream_errors_schema_invalid", "report": name})
        elif "errors" in payload and not isinstance(upstream_errors, list):
            errors.append({"code": "upstream_errors_schema_invalid", "report": name})
        elif upstream_errors:
            errors.append({"code": "upstream_report_contains_errors", "report": name})
        upstream_warnings = payload.get("warnings")
        if "warnings" in payload and (
            not isinstance(upstream_warnings, list)
            or any(not isinstance(item, str) or not item.strip() for item in upstream_warnings)
        ):
            errors.append({"code": "upstream_warnings_schema_invalid", "report": name})

    if any(not isinstance(payload, dict) for payload in inputs.values()):
        return _result(plan, registry, output_integrity, row_window_lineage, metric_semantics, errors, warnings)

    plan_id = plan.get("plan_id")
    plan_sha = plan.get("plan_sha256")
    registry_id = registry.get("registry_id")
    for name, payload in (
        ("output_integrity", output_integrity),
        ("row_window_lineage", row_window_lineage),
        ("metric_semantics", metric_semantics),
    ):
        if payload.get("plan_id") != plan_id:
            errors.append({"code": "plan_id_mismatch", "report": name})
        if payload.get("plan_sha256") != plan_sha:
            errors.append({"code": "plan_sha256_mismatch", "report": name})
        if payload.get("registry_id") != registry_id:
            errors.append({"code": "registry_id_mismatch", "report": name})

    run_matrix = plan.get("run_matrix")
    matrix_run_ids: list[str] = []
    matrix_valid = isinstance(run_matrix, list) and bool(run_matrix)
    if not matrix_valid:
        errors.append({"code": "planned_run_matrix_invalid"})
    else:
        seen_run_ids: set[str] = set()
        for index, entry in enumerate(run_matrix):
            run_id = entry.get("run_id") if isinstance(entry, dict) else None
            if not isinstance(run_id, str) or not run_id.strip():
                errors.append({"code": "planned_run_matrix_entry_invalid", "index": index})
                matrix_valid = False
                continue
            if run_id in seen_run_ids:
                errors.append({"code": "planned_run_matrix_duplicate_run_id", "run_id": run_id})
                matrix_valid = False
            seen_run_ids.add(run_id)
            matrix_run_ids.append(run_id)

    expected_run_ids = plan.get("expected_run_ids")
    expected_ids_valid = (
        isinstance(expected_run_ids, list)
        and all(isinstance(run_id, str) and bool(run_id.strip()) for run_id in expected_run_ids)
        and len(expected_run_ids) == len(set(expected_run_ids))
    )
    if not expected_ids_valid:
        errors.append({"code": "expected_run_ids_invalid"})
    elif matrix_valid and expected_run_ids != matrix_run_ids:
        errors.append({"code": "expected_run_ids_matrix_mismatch"})

    planned = plan.get("expected_runs")
    if not isinstance(planned, int) or isinstance(planned, bool) or planned < 1:
        planned = len(run_matrix) if isinstance(run_matrix, list) else None
        errors.append({"code": "planned_run_count_invalid"})
    elif isinstance(run_matrix, list) and planned != len(run_matrix):
        errors.append({
            "code": "planned_run_count_matrix_mismatch",
            "expected": planned,
            "actual": len(run_matrix),
        })
    if isinstance(planned, int):
        for name, payload, field in (
            ("output_integrity", output_integrity, "planned_runs_checked"),
            ("row_window_lineage", row_window_lineage, "runs_checked"),
            ("metric_semantics", metric_semantics, "runs_checked"),
        ):
            value = payload.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value != planned:
                errors.append({"code": "validated_run_coverage_mismatch", "report": name, "field": field, "expected": planned, "actual": value})

    missing = output_integrity.get("missing_planned_runs")
    if missing != []:
        errors.append({"code": "output_integrity_missing_planned_runs"})
    for name, payload in (("row_window_lineage", row_window_lineage), ("metric_semantics", metric_semantics)):
        if payload.get("warnings"):
            warnings.append(f"{name}_contains_warnings")

    return _result(plan, registry, output_integrity, row_window_lineage, metric_semantics, errors, warnings)


def _result(plan, registry, output_integrity, row_window_lineage, metric_semantics, errors, warnings):
    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_evaluation_integrity_certification",
        "status": status,
        "readiness": "NOT_CERTIFIED" if errors else ("EVALUATION_INTEGRITY_CERTIFIED_WITH_WARNINGS" if warnings else "EVALUATION_INTEGRITY_CERTIFIED"),
        "plan_id": plan.get("plan_id") if isinstance(plan, dict) else None,
        "plan_sha256": plan.get("plan_sha256") if isinstance(plan, dict) else None,
        "registry_id": registry.get("registry_id") if isinstance(registry, dict) else None,
        "planned_runs": plan.get("expected_runs") if isinstance(plan, dict) else None,
        "output_integrity_status": output_integrity.get("status") if isinstance(output_integrity, dict) else None,
        "row_window_lineage_status": row_window_lineage.get("status") if isinstance(row_window_lineage, dict) else None,
        "metric_semantics_status": metric_semantics.get("status") if isinstance(metric_semantics, dict) else None,
        "evaluation_rows_checked": row_window_lineage.get("evaluation_rows_checked") if isinstance(row_window_lineage, dict) else None,
        "metrics_recalculated": metric_semantics.get("metrics_recalculated") if isinstance(metric_semantics, dict) else None,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
        "execution_connected": False,
        "automatic_repair_performed": False,
        "deployment_authorized": False,
        "scope_limitations": [
            "Combines supplied audit reports; it does not rerun the underlying audits.",
            "Certification means metadata, evaluation-row window lineage, output integrity, and explicitly contracted simple metrics passed their checks.",
            "Does not prove that evaluation code generated the metrics artifact, validate complex trading PnL/drawdown semantics, or establish profitability.",
            "Offline report-only; does not execute experiments, train/select models, connect to brokers, or authorize deployment.",
        ],
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--registry-report", required=True, type=Path)
    parser.add_argument("--output-integrity-report", required=True, type=Path)
    parser.add_argument("--row-window-lineage-report", required=True, type=Path)
    parser.add_argument("--metric-semantics-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = certify_evaluation_integrity(
            _read_object(args.plan),
            _read_object(args.registry_report),
            _read_object(args.output_integrity_report),
            _read_object(args.row_window_lineage_report),
            _read_object(args.metric_semantics_report),
        )
        if args.output.exists():
            raise ValueError(f"output_already_exists:{args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "readiness": report["readiness"], "planned_runs": report["planned_runs"], "metrics_recalculated": report["metrics_recalculated"], "error_count": len(report["errors"]), "report_path": str(args.output.resolve()), "deployment_authorized": False}, indent=2))
        return 0 if report["status"] in PASS_STATUSES else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

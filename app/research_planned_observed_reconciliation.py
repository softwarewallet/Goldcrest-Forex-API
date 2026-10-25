"""Reconcile a deterministic experiment plan against an audited experiment registry.

This offline audit checks planned-versus-observed run IDs and run dimensions. It does not
launch or repair experiments, select models, infer significance, or connect to trading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PASS_STATUSES = {"PASS", "PASS_WITH_WARNINGS"}
FALSE_FLAGS = {
    "actionable": False,
    "inferential": False,
    "statistical_significance_claimed": False,
    "automatic_model_selection": False,
    "model_training_performed": False,
    "execution_connected": False,
    "automatic_repair_performed": False,
}


def reconcile_planned_observed_runs(
    plan: dict[str, Any], registry_report: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("audit_name") != "research_experiment_plan":
        raise ValueError("plan_report_invalid")
    if plan.get("status") != "PASS":
        raise ValueError("plan_report_not_passing")
    if not isinstance(registry_report, dict) or registry_report.get("audit_name") != "research_experiment_registry":
        raise ValueError("registry_report_invalid")
    if registry_report.get("status") not in PASS_STATUSES:
        raise ValueError("registry_report_not_passing")

    matrix = plan.get("run_matrix")
    expected_count = plan.get("expected_runs")
    if (
        not isinstance(matrix, list)
        or not matrix
        or not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count != len(matrix)
    ):
        raise ValueError("plan_run_matrix_invalid")
    expected_by_id: dict[str, dict[str, Any]] = {}
    for item in matrix:
        if not isinstance(item, dict) or not isinstance(item.get("run_id"), str) or not item["run_id"]:
            raise ValueError("plan_run_identity_invalid")
        if item["run_id"] in expected_by_id:
            raise ValueError("plan_run_id_duplicate")
        expected_by_id[item["run_id"]] = item

    observed = registry_report.get("run_results")
    if not isinstance(observed, list):
        raise ValueError("registry_run_results_invalid")
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_observed: set[str] = set()
    observed_by_id: dict[str, dict[str, Any]] = {}
    invalid_observed = 0

    for index, item in enumerate(observed):
        if not isinstance(item, dict):
            errors.append({"code": "observed_run_record_invalid", "index": index})
            invalid_observed += 1
            continue
        run_id = item.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            errors.append({"code": "observed_run_id_missing", "index": index})
            invalid_observed += 1
            continue
        if run_id in seen_observed:
            errors.append({"code": "observed_run_id_duplicate", "run_id": run_id})
            invalid_observed += 1
            continue
        seen_observed.add(run_id)
        observed_by_id[run_id] = item
        if item.get("status") != "PASS":
            errors.append({"code": "observed_run_not_validated", "run_id": run_id})
            invalid_observed += 1

    expected_ids = set(expected_by_id)
    observed_ids = set(observed_by_id)
    missing_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)
    for run_id in missing_ids:
        errors.append({"code": "planned_run_missing", "run_id": run_id})
    for run_id in unexpected_ids:
        warnings.append(f"unexpected_observed_run:{run_id}")

    matched = 0
    dimension_checks = 0
    for run_id in sorted(expected_ids & observed_ids):
        expected = expected_by_id[run_id]
        actual = observed_by_id[run_id]
        checks = {
            "seed": (expected.get("seed"), actual.get("seed")),
            "fold_id": (str(expected.get("fold_id")), str(actual.get("fold_id"))),
            "window_id": (str(expected.get("window_id")), str(actual.get("window_id"))),
            "evaluation_window": (expected.get("evaluation_window"), actual.get("evaluation_window")),
            "context": (expected.get("context"), actual.get("context")),
            "config_sha256": (expected.get("config_sha256"), actual.get("config_sha256")),
            "dataset_sha256": (expected.get("dataset_sha256"), actual.get("dataset_sha256")),
            "code_commit_sha": (expected.get("code_commit_sha"), actual.get("code_commit_sha")),
        }
        mismatches = [field for field, (left, right) in checks.items() if left != right]
        dimension_checks += len(checks)
        if mismatches:
            for field in mismatches:
                errors.append({
                    "code": "planned_observed_dimension_mismatch",
                    "run_id": run_id,
                    "field": field,
                    "planned": checks[field][0],
                    "observed": checks[field][1],
                })
        else:
            matched += 1

    registry_supplied = registry_report.get("runs_supplied")
    if isinstance(registry_supplied, int) and registry_supplied != len(observed):
        errors.append({
            "code": "registry_count_inconsistent",
            "runs_supplied": registry_supplied,
            "run_results_count": len(observed),
        })
    registry_validated = registry_report.get("runs_validated")
    if isinstance(registry_validated, int) and registry_validated != sum(
        1 for item in observed if isinstance(item, dict) and item.get("status") == "PASS"
    ):
        errors.append({"code": "registry_validated_count_inconsistent"})

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_planned_observed_reconciliation",
        "plan_id": plan.get("plan_id"),
        "plan_sha256": plan.get("plan_sha256"),
        "registry_id": registry_report.get("registry_id"),
        "status": status,
        "readiness": "NOT_RECONCILED" if errors else (
            "RECONCILED_WITH_WARNINGS" if warnings else "PLANNED_OBSERVED_RUNS_RECONCILED"
        ),
        "planned_runs": len(expected_by_id),
        "observed_runs": len(observed),
        "matched_runs": matched,
        "missing_run_ids": missing_ids,
        "unexpected_run_ids": unexpected_ids,
        "invalid_observed_run_records": invalid_observed,
        "dimension_fields_checked_per_matched_run": 8,
        "dimension_comparisons_performed": dimension_checks,
        "errors": errors,
        "warnings": warnings,
        **FALSE_FLAGS,
        "jobs_launched": False,
        "automatic_repair_performed": False,
        "scope_limitations": [
            "Reconciles supplied plan and registry audit JSON only; it does not rerun either audit.",
            "Unexpected runs are warnings because exploratory runs may be legitimate; missing planned runs fail the audit.",
            "Successful reconciliation proves metadata and declared coverage consistency only, not strategy quality or profitability.",
            "This report never authorizes deployment or trading execution.",
        ],
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="Planner output JSON")
    parser.add_argument("--registry-report", required=True, type=Path, help="Registry audit output JSON")
    parser.add_argument("--output", required=True, type=Path, help="New reconciliation report path")
    args = parser.parse_args()
    try:
        report = reconcile_planned_observed_runs(
            _read_object(args.plan), _read_object(args.registry_report)
        )
        if args.output.exists():
            raise ValueError(f"output_already_exists:{args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": report["status"],
            "readiness": report["readiness"],
            "planned_runs": report["planned_runs"],
            "observed_runs": report["observed_runs"],
            "matched_runs": report["matched_runs"],
            "missing_runs": len(report["missing_run_ids"]),
            "unexpected_runs": len(report["unexpected_run_ids"]),
            "report_path": str(args.output.resolve()),
            "execution_connected": False,
        }, indent=2))
        return 0 if report["status"] in PASS_STATUSES else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

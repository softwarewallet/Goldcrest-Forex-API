"""Final offline reproducibility readiness gate for plan, registry, reconciliation, and evidence.

This is a meta-audit over supplied JSON reports. It never executes experiments or trading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects instead of silently accepting the last duplicate."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    """Reject NaN and Infinity tokens, which are not valid RFC 8259 JSON."""
    raise ValueError(f"nonstandard_json_constant:{value}")


def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonstandard_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


PASS_STATUSES = {"PASS", "PASS_WITH_WARNINGS"}
REQUIRED_SAFETY_FALSE = (
    "actionable",
    "inferential",
    "execution_connected",
    "model_training_performed",
    "automatic_model_selection",
    "automatic_repair_performed",
)


def audit_reproducibility_readiness(
    plan: dict[str, Any],
    registry: dict[str, Any],
    reconciliation: dict[str, Any],
    evidence_pack: dict[str, Any],
) -> dict[str, Any]:
    inputs = {
        "plan": (plan, "research_experiment_plan", {"PASS"}),
        "registry": (registry, "research_experiment_registry", PASS_STATUSES),
        "reconciliation": (reconciliation, "research_planned_observed_reconciliation", PASS_STATUSES),
        "evidence_pack": (evidence_pack, "research_evidence_pack", PASS_STATUSES),
    }
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    for name, (payload, expected_audit, allowed_statuses) in inputs.items():
        if not isinstance(payload, dict):
            errors.append({"code": "input_report_invalid", "report": name})
            continue
        if payload.get("audit_name") != expected_audit:
            errors.append({
                "code": "input_audit_name_mismatch",
                "report": name,
                "expected": expected_audit,
                "actual": payload.get("audit_name"),
            })
        if payload.get("status") not in allowed_statuses:
            errors.append({
                "code": "input_report_not_passing",
                "report": name,
                "status": payload.get("status"),
            })
        for flag in REQUIRED_SAFETY_FALSE:
            if payload.get(flag) is not False:
                errors.append({
                    "code": "safety_flag_not_explicitly_false",
                    "report": name,
                    "flag": flag,
                })

    if errors and any(not isinstance(payload, dict) for payload, _, _ in inputs.values()):
        return _result(plan, registry, reconciliation, evidence_pack, errors, warnings)

    plan_sha = plan.get("plan_sha256")
    if reconciliation.get("plan_sha256") != plan_sha:
        errors.append({"code": "reconciliation_plan_fingerprint_mismatch"})
    if reconciliation.get("plan_id") != plan.get("plan_id"):
        errors.append({"code": "reconciliation_plan_id_mismatch"})
    if reconciliation.get("registry_id") != registry.get("registry_id"):
        errors.append({"code": "reconciliation_registry_id_mismatch"})

    planned = plan.get("expected_runs")
    observed = registry.get("runs_supplied")
    validated = registry.get("runs_validated")
    reconciled_planned = reconciliation.get("planned_runs")
    reconciled_observed = reconciliation.get("observed_runs")
    matched = reconciliation.get("matched_runs")
    if not isinstance(planned, int) or isinstance(planned, bool) or planned < 1:
        errors.append({"code": "planned_run_count_invalid"})
    if not isinstance(observed, int) or isinstance(observed, bool) or observed < 0:
        errors.append({"code": "registry_observed_run_count_invalid"})
    if not isinstance(validated, int) or isinstance(validated, bool) or validated < 0:
        errors.append({"code": "registry_validated_run_count_invalid"})
    if isinstance(planned, int) and reconciled_planned != planned:
        errors.append({"code": "planned_run_count_mismatch"})
    if isinstance(observed, int) and reconciled_observed != observed:
        errors.append({"code": "observed_run_count_mismatch"})
    if isinstance(planned, int) and isinstance(matched, int) and matched != planned:
        errors.append({"code": "not_all_planned_runs_matched"})
    if isinstance(observed, int) and isinstance(validated, int) and validated != observed:
        errors.append({"code": "not_all_observed_runs_validated"})

    plan_dataset = plan.get("dataset_sha256")
    plan_config = plan.get("config_sha256")
    plan_commit = plan.get("code_commit_sha")
    run_results = registry.get("run_results")
    if not isinstance(run_results, list):
        errors.append({"code": "registry_run_results_invalid"})
        run_results = []
    for index, item in enumerate(run_results):
        if not isinstance(item, dict):
            errors.append({"code": "registry_run_result_invalid", "index": index})
            continue
        if item.get("status") != "PASS":
            errors.append({"code": "registry_contains_failed_run", "run_id": item.get("run_id")})
        for field, expected in (
            ("dataset_sha256", plan_dataset),
            ("config_sha256", plan_config),
            ("code_commit_sha", plan_commit),
        ):
            actual = item.get(field)
            if actual is not None and actual != expected:
                errors.append({
                    "code": "run_provenance_mismatch",
                    "run_id": item.get("run_id"),
                    "field": field,
                    "expected": expected,
                    "actual": actual,
                })
            elif actual is None:
                errors.append({
                    "code": "run_provenance_missing",
                    "run_id": item.get("run_id"),
                    "field": field,
                })

    if reconciliation.get("missing_run_ids") != []:
        errors.append({"code": "reconciliation_has_missing_runs"})
    if reconciliation.get("errors"):
        errors.append({"code": "reconciliation_contains_errors"})
    if evidence_pack.get("errors"):
        errors.append({"code": "evidence_pack_contains_errors"})
    if evidence_pack.get("dataset_fingerprints_reconciled"):
        fingerprints = {str(value).lower() for value in evidence_pack["dataset_fingerprints_reconciled"]}
        if plan_dataset not in fingerprints:
            errors.append({"code": "evidence_pack_dataset_differs_from_plan"})
    else:
        warnings.append("evidence_pack_has_no_reconciled_dataset_fingerprint")

    for name, payload in (("registry", registry), ("reconciliation", reconciliation), ("evidence_pack", evidence_pack)):
        if payload.get("status") == "PASS_WITH_WARNINGS":
            warnings.append(f"{name}_passed_with_warnings")

    return _result(plan, registry, reconciliation, evidence_pack, errors, warnings)


def _result(plan, registry, reconciliation, evidence_pack, errors, warnings):
    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_reproducibility_readiness",
        "status": status,
        "readiness": "NOT_READY" if errors else (
            "READY_WITH_WARNINGS" if warnings else "REPRODUCIBILITY_EVIDENCE_RECONCILED"
        ),
        "plan_id": plan.get("plan_id") if isinstance(plan, dict) else None,
        "plan_sha256": plan.get("plan_sha256") if isinstance(plan, dict) else None,
        "registry_id": registry.get("registry_id") if isinstance(registry, dict) else None,
        "planned_runs": plan.get("expected_runs") if isinstance(plan, dict) else None,
        "observed_runs": registry.get("runs_supplied") if isinstance(registry, dict) else None,
        "validated_runs": registry.get("runs_validated") if isinstance(registry, dict) else None,
        "matched_runs": reconciliation.get("matched_runs") if isinstance(reconciliation, dict) else None,
        "evidence_pack_id": evidence_pack.get("pack_id") if isinstance(evidence_pack, dict) else None,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "execution_connected": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
        "automatic_repair_performed": False,
        "deployment_authorized": False,
        "scope_limitations": [
            "Validates supplied report consistency; does not rerun underlying research or audits.",
            "Passing establishes reproducibility metadata and evidence-chain consistency only.",
            "No conclusion about predictive performance, profitability, or deployment suitability is implied.",
            "This gate never authorizes broker connectivity or trading execution.",
        ],
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path.read_bytes(), source=str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--registry-report", required=True, type=Path)
    parser.add_argument("--reconciliation-report", required=True, type=Path)
    parser.add_argument("--evidence-pack-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = audit_reproducibility_readiness(
            _read_object(args.plan),
            _read_object(args.registry_report),
            _read_object(args.reconciliation_report),
            _read_object(args.evidence_pack_report),
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
            "matched_runs": report["matched_runs"],
            "error_count": len(report["errors"]),
            "report_path": str(args.output.resolve()),
            "deployment_authorized": False,
        }, indent=2))
        return 0 if report["status"] in PASS_STATUSES else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

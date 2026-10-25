"""Final offline gate linking reproducibility readiness to evaluation evidence-bundle verification.

This report-only meta-audit does not rerun upstream audits, execute experiments, or authorize
training, model selection, broker connectivity, deployment, or trading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def audit_final_certification_chain(
    readiness: dict[str, Any],
    bundle: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    inputs = {
        "readiness": (readiness, "research_reproducibility_readiness", PASS_STATUSES),
        "bundle": (bundle, "research_evaluation_certification_evidence_bundle", {"PASS"}),
        "verification": (
            verification,
            "research_evaluation_certification_evidence_bundle_verification",
            {"PASS"},
        ),
    }
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    for name, (payload, audit_name, statuses) in inputs.items():
        if not isinstance(payload, dict):
            errors.append({"code": "input_report_invalid", "report": name})
            continue
        if payload.get("audit_name") != audit_name:
            errors.append({
                "code": "input_audit_name_mismatch",
                "report": name,
                "expected": audit_name,
                "actual": payload.get("audit_name"),
            })
        if payload.get("status") not in statuses:
            errors.append({
                "code": "input_report_not_passing",
                "report": name,
                "status": payload.get("status"),
            })
        for flag in REQUIRED_FALSE_FLAGS:
            if payload.get(flag) is not False:
                errors.append({
                    "code": "safety_flag_not_explicitly_false",
                    "report": name,
                    "flag": flag,
                })
        if payload.get("deployment_authorized") is not False:
            errors.append({
                "code": "deployment_must_remain_unauthorized",
                "report": name,
            })
        if payload.get("errors"):
            errors.append({"code": "input_report_contains_errors", "report": name})
        if payload.get("status") == "PASS_WITH_WARNINGS":
            warnings.append(f"{name}_passed_with_warnings")

    if all(isinstance(payload, dict) for payload, _, _ in inputs.values()):
        if readiness.get("plan_id") != bundle.get("plan_id"):
            errors.append({"code": "plan_id_mismatch_between_readiness_and_bundle"})
        if readiness.get("plan_sha256") != bundle.get("plan_sha256"):
            errors.append({"code": "plan_sha256_mismatch_between_readiness_and_bundle"})
        if readiness.get("registry_id") != bundle.get("registry_id"):
            errors.append({"code": "registry_id_mismatch_between_readiness_and_bundle"})
        bundle_sha = bundle.get("bundle_sha256")
        if not isinstance(bundle_sha, str) or len(bundle_sha) != 64:
            errors.append({"code": "bundle_digest_invalid"})
        if verification.get("bundle_sha256") != bundle_sha:
            errors.append({"code": "verification_bundle_digest_mismatch"})
        declared = bundle.get("report_count")
        verified_declared = verification.get("reports_declared")
        verified_count = verification.get("reports_verified")
        if not isinstance(declared, int) or isinstance(declared, bool) or declared < 1:
            errors.append({"code": "bundle_report_count_invalid"})
        if verified_declared != declared:
            errors.append({"code": "verified_report_count_mismatch"})
        if verified_count != declared:
            errors.append({"code": "not_all_bundle_reports_verified"})
        if verification.get("readiness") != "EVIDENCE_BUNDLE_CONTENT_VERIFIED":
            errors.append({"code": "evidence_bundle_content_not_verified"})
        if verification.get("errors") != []:
            errors.append({"code": "bundle_verification_contains_errors"})
        if readiness.get("status") == "PASS_WITH_WARNINGS":
            warnings.append("reproducibility_readiness_has_warnings")

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_final_certification_chain",
        "status": status,
        "readiness": (
            "NOT_READY" if errors else
            "FINAL_CERTIFICATION_CHAIN_READY_WITH_WARNINGS" if warnings else
            "FINAL_CERTIFICATION_CHAIN_VERIFIED"
        ),
        "plan_id": readiness.get("plan_id") if isinstance(readiness, dict) else None,
        "plan_sha256": readiness.get("plan_sha256") if isinstance(readiness, dict) else None,
        "registry_id": readiness.get("registry_id") if isinstance(readiness, dict) else None,
        "bundle_sha256": bundle.get("bundle_sha256") if isinstance(bundle, dict) else None,
        "reports_declared": bundle.get("report_count") if isinstance(bundle, dict) else None,
        "reports_verified": verification.get("reports_verified") if isinstance(verification, dict) else None,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "automatic_model_selection": False,
        "model_training_performed": False,
        "execution_connected": False,
        "automatic_repair_performed": False,
        "deployment_authorized": False,
        "scope_limitations": [
            "Links supplied reproducibility-readiness and content-addressed evidence-bundle reports; it does not rerun upstream audits.",
            "Passing proves only the reported identities, statuses, digest linkage, and declared verification coverage are consistent.",
            "It does not establish strategy profitability, predictive validity, statistical significance, or deployment suitability.",
            "Offline report-only; no experiments, model training, broker connectivity, deployment, or trades are initiated.",
        ],
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-report", required=True, type=Path)
    parser.add_argument("--evidence-bundle", required=True, type=Path)
    parser.add_argument("--bundle-verification-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = audit_final_certification_chain(
            _read_object(args.readiness_report),
            _read_object(args.evidence_bundle),
            _read_object(args.bundle_verification_report),
        )
        if args.output.exists():
            raise ValueError(f"output_already_exists:{args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": report["status"],
            "readiness": report["readiness"],
            "reports_declared": report["reports_declared"],
            "reports_verified": report["reports_verified"],
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

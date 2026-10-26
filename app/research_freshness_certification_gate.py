"""Gate final research certification on current source/config/data provenance.

This gate consumes existing reports only. It does not rerun audits or regenerate artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads

def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


PASS_STATUSES = {"PASS", "PASS_WITH_WARNINGS"}
FALSE_FLAGS = (
    "actionable",
    "inferential",
    "statistical_significance_claimed",
    "automatic_model_selection",
    "model_training_performed",
    "execution_connected",
    "automatic_repair_performed",
    "deployment_authorized",
)


def gate_freshness_certification(
    freshness: dict[str, Any],
    certification: dict[str, Any],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    expected = (
        ("freshness", freshness, "research_artifact_freshness_audit", {"PASS"}),
        ("certification", certification, "research_final_certification_chain", PASS_STATUSES),
    )
    for label, payload, audit_name, allowed_statuses in expected:
        if not isinstance(payload, dict):
            errors.append({"code": "input_report_invalid", "report": label})
            continue
        if payload.get("audit_name") != audit_name:
            errors.append({
                "code": "input_audit_name_mismatch",
                "report": label,
                "expected": audit_name,
                "actual": payload.get("audit_name"),
            })
        if payload.get("status") not in allowed_statuses:
            errors.append({
                "code": "input_report_not_passing",
                "report": label,
                "status": payload.get("status"),
            })
        for flag in FALSE_FLAGS:
            if payload.get(flag) is not False:
                errors.append({"code": "safety_flag_not_explicitly_false", "report": label, "flag": flag})
        if payload.get("errors"):
            errors.append({"code": "input_report_contains_errors", "report": label})
        if payload.get("status") == "PASS_WITH_WARNINGS":
            warnings.append(f"{label}_passed_with_warnings")

    if isinstance(freshness, dict):
        if freshness.get("regeneration_required_count") != 0:
            errors.append({"code": "research_artifacts_require_regeneration"})
        if freshness.get("current_count") != freshness.get("artifact_count"):
            errors.append({"code": "freshness_artifact_counts_inconsistent"})
        artifacts = freshness.get("artifacts")
        if not isinstance(artifacts, list):
            errors.append({"code": "freshness_artifact_inventory_invalid"})
        elif any(not isinstance(item, dict) or item.get("classification") != "CURRENT" or item.get("needs_regeneration") is not False for item in artifacts):
            errors.append({"code": "freshness_inventory_contains_noncurrent_artifact"})
    if isinstance(certification, dict) and certification.get("status") == "PASS_WITH_WARNINGS":
        warnings.append("final_certification_chain_has_warnings")

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_freshness_certification_gate",
        "status": status,
        "readiness": (
            "NOT_READY" if errors else
            "CERTIFIED_WITH_WARNINGS" if warnings else
            "CERTIFIED_AND_CURRENT"
        ),
        "freshness_status": freshness.get("status") if isinstance(freshness, dict) else None,
        "certification_status": certification.get("status") if isinstance(certification, dict) else None,
        "artifact_count": freshness.get("artifact_count") if isinstance(freshness, dict) else None,
        "current_count": freshness.get("current_count") if isinstance(freshness, dict) else None,
        "regeneration_required_count": freshness.get("regeneration_required_count") if isinstance(freshness, dict) else None,
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
            "A passing gate means the supplied final-certification and freshness reports agree and declare all inventoried artifacts current.",
            "It does not independently recompute upstream evidence bundle hashes or establish that the manifest is complete.",
            "The source revision, config fingerprint, and dataset fingerprint are only as trustworthy as the caller-supplied inputs to the freshness audit.",
            "No artifacts are regenerated and no experiments, model training, broker connectivity, deployment, or trades are initiated.",
            "Certification does not establish profitability, predictive validity, statistical significance, or deployment suitability.",
        ],
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path.read_bytes(), source=str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freshness-report", required=True, type=Path)
    parser.add_argument("--certification-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = gate_freshness_certification(
            _read_object(args.freshness_report),
            _read_object(args.certification_report),
        )
        if args.output.exists():
            raise ValueError(f"output_already_exists:{args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "readiness": report["readiness"],
            "artifact_count": report["artifact_count"],
            "regeneration_required_count": report["regeneration_required_count"],
            "error_count": len(report["errors"]),
            "report_path": str(args.output.resolve()),
            "deployment_authorized": False,
        }, indent=2))
        return 0 if report["status"] in PASS_STATUSES else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}", "deployment_authorized": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

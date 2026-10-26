"""Final offline gate linking reproducibility readiness to evaluation evidence-bundle verification.

This report-only meta-audit does not rerun upstream audits, execute experiments, or authorize
training, model selection, broker connectivity, deployment, or trading.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _valid_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


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
        if "errors" in payload:
            input_errors = payload.get("errors")
            if not isinstance(input_errors, list):
                errors.append({"code": "input_errors_schema_invalid", "report": name})
            elif input_errors:
                errors.append({"code": "input_report_contains_errors", "report": name})
        if "warnings" in payload:
            input_warnings = payload.get("warnings")
            if not isinstance(input_warnings, list) or any(
                not isinstance(item, str) or not item.strip() for item in input_warnings
            ):
                errors.append({"code": "input_warnings_schema_invalid", "report": name})
        if payload.get("status") == "PASS_WITH_WARNINGS":
            warnings.append(f"{name}_passed_with_warnings")

    if all(isinstance(payload, dict) for payload, _, _ in inputs.values()):
        for name, payload in (("readiness", readiness), ("bundle", bundle)):
            if not _valid_nonempty_string(payload.get("plan_id")):
                errors.append({"code": "plan_id_invalid", "report": name})
            if not _is_sha256_hex(payload.get("plan_sha256")):
                errors.append({"code": "plan_digest_invalid", "report": name})
            if not _valid_nonempty_string(payload.get("registry_id")):
                errors.append({"code": "registry_id_invalid", "report": name})
        if readiness.get("plan_id") != bundle.get("plan_id"):
            errors.append({"code": "plan_id_mismatch_between_readiness_and_bundle"})
        if readiness.get("plan_sha256") != bundle.get("plan_sha256"):
            errors.append({"code": "plan_sha256_mismatch_between_readiness_and_bundle"})
        if readiness.get("registry_id") != bundle.get("registry_id"):
            errors.append({"code": "registry_id_mismatch_between_readiness_and_bundle"})
        bundle_sha = bundle.get("bundle_sha256")
        if not _is_sha256_hex(bundle_sha):
            errors.append({"code": "bundle_digest_invalid"})
        if verification.get("bundle_sha256") != bundle_sha:
            errors.append({"code": "verification_bundle_digest_mismatch"})
        declared = bundle.get("report_count")
        verified_declared = verification.get("reports_declared")
        verified_count = verification.get("reports_verified")
        if not isinstance(declared, int) or isinstance(declared, bool) or declared < 1:
            errors.append({"code": "bundle_report_count_invalid"})
        if (
            not isinstance(verified_declared, int)
            or isinstance(verified_declared, bool)
            or verified_declared < 1
        ):
            errors.append({"code": "verified_declared_report_count_invalid"})
        elif isinstance(declared, int) and not isinstance(declared, bool) and verified_declared != declared:
            errors.append({"code": "verified_report_count_mismatch"})
        if (
            not isinstance(verified_count, int)
            or isinstance(verified_count, bool)
            or verified_count < 0
        ):
            errors.append({"code": "verified_report_count_invalid"})
        elif isinstance(declared, int) and not isinstance(declared, bool) and verified_count != declared:
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
    payload = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def _write_report_no_clobber(
    output_path: str | Path,
    payload: dict[str, Any],
    input_paths: list[str | Path],
) -> Path:
    """Publish a certification report without replacing inputs or concurrent outputs."""
    output = Path(output_path).expanduser().resolve()
    inputs = {Path(path).expanduser().resolve() for path in input_paths}
    if output in inputs:
        raise ValueError("output_must_not_alias_input")
    if output.exists():
        raise ValueError(f"output_already_exists:{output}")

    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    published_identity: tuple[int, int] | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.link(temporary, output)
        stat = output.stat()
        published_identity = (stat.st_dev, stat.st_ino)
        temporary.unlink()
        temporary = None
        return output
    except Exception:
        if published_identity is not None:
            try:
                current = output.stat()
                if (current.st_dev, current.st_ino) == published_identity:
                    output.unlink()
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-report", required=True, type=Path)
    parser.add_argument("--evidence-bundle", required=True, type=Path)
    parser.add_argument("--bundle-verification-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    input_paths = [args.readiness_report, args.evidence_bundle, args.bundle_verification_report]
    try:
        report = audit_final_certification_chain(
            _read_object(args.readiness_report),
            _read_object(args.evidence_bundle),
            _read_object(args.bundle_verification_report),
        )
        output = _write_report_no_clobber(args.output, report, input_paths)
        print(json.dumps({
            "status": report["status"],
            "readiness": report["readiness"],
            "reports_declared": report["reports_declared"],
            "reports_verified": report["reports_verified"],
            "error_count": len(report["errors"]),
            "report_path": str(output),
            "deployment_authorized": False,
        }, indent=2))
        return 0 if report["status"] in PASS_STATUSES else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(json.dumps({
            "status": "FAIL",
            "error": f"{type(exc).__name__}:{exc}",
            "deployment_authorized": False,
        }, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

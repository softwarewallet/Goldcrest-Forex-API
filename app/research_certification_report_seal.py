"""Create and verify a detached integrity seal for unified research certification reports.

This is a content-integrity mechanism, not a digital signature or proof that the report's claims are true.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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


def canonical_report_bytes(report: dict[str, Any]) -> bytes:
    if not isinstance(report, dict):
        raise ValueError("certification_report_must_be_object")
    if report.get("audit_name") != "research_unified_provenance_certification":
        raise ValueError("certification_report_audit_name_invalid")
    if report.get("status") not in {"PASS", "PASS_WITH_WARNINGS", "FAIL"}:
        raise ValueError("certification_report_status_invalid")
    errors = report.get("errors")
    if not isinstance(errors, list):
        raise ValueError("certification_report_errors_must_be_list")
    if report.get("status") in {"PASS", "PASS_WITH_WARNINGS"} and errors:
        raise ValueError("passing_certification_report_must_not_contain_errors")
    for flag in FALSE_FLAGS:
        if report.get(flag) is not False:
            raise ValueError(f"certification_report_safety_flag_not_explicitly_false:{flag}")
    revision = report.get("source_revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("certification_report_source_revision_invalid")
    return (json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def create_report_seal(report: dict[str, Any]) -> dict[str, Any]:
    raw = canonical_report_bytes(report)
    return {
        "schema_version": 1,
        "audit_name": "research_unified_certification_report_detached_seal",
        "canonicalization": "json-sort-keys-compact-utf8-trailing-newline-v1",
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "report_bytes": len(raw),
        "report_status": report["status"],
        "source_revision": report["source_revision"],
        "error_count": len(report.get("errors", [])) if isinstance(report.get("errors", []), list) else None,
        **FALSE_FLAGS,
        "scope_limitations": [
            "The seal detects report edits only when retained separately or its digest is anchored in a separately controlled location.",
            "The seal is not a digital signature and does not authenticate the report producer.",
            "A valid seal proves canonical byte consistency only; it does not prove the report's claims or source evidence are truthful.",
        ],
    }


def verify_report_seal(report: dict[str, Any], seal: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    if not isinstance(seal, dict) or seal.get("audit_name") != "research_unified_certification_report_detached_seal":
        errors.append({"code": "REPORT_SEAL_SCHEMA_INVALID"})
    else:
        try:
            expected = create_report_seal(report)
        except (TypeError, ValueError) as exc:
            expected = None
            errors.append({"code": "CERTIFICATION_REPORT_INVALID", "detail": str(exc)})
        if expected is not None:
            for key in ("schema_version", "canonicalization", "report_sha256", "report_bytes",
                        "report_status", "source_revision", "error_count"):
                if seal.get(key) != expected.get(key):
                    errors.append({"code": "REPORT_SEAL_FIELD_MISMATCH", "field": key})
            for flag in FALSE_FLAGS:
                if seal.get(flag) is not False:
                    errors.append({"code": "SAFETY_FLAG_NOT_EXPLICITLY_FALSE", "field": flag})
    return {
        "schema_version": 1,
        "audit_name": "research_unified_certification_report_seal_verification",
        "status": "FAIL" if errors else "PASS",
        "report_sha256": seal.get("report_sha256") if isinstance(seal, dict) else None,
        "errors": errors,
        **FALSE_FLAGS,
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("create", "verify"), required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seal", type=Path, help="Required only in verify mode")
    parser.add_argument("--output", type=Path, help="Required for create mode")
    args = parser.parse_args()
    try:
        report = _read_object(args.report)
        if args.mode == "create":
            if args.output is None:
                raise ValueError("output_required_for_create_mode")
            seal = create_report_seal(report)
            if args.output.exists():
                raise ValueError(f"output_already_exists:{args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps({"status": "PASS", "report_sha256": seal["report_sha256"],
                              "seal_path": str(args.output.resolve()), "deployment_authorized": False}, indent=2))
            return 0
        if args.seal is None:
            raise ValueError("seal_required_for_verify_mode")
        verification = verify_report_seal(report, _read_object(args.seal))
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 0 if verification["status"] == "PASS" else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}",
                          "deployment_authorized": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

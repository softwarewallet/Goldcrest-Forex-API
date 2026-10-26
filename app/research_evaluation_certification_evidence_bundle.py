"""Build and verify a content-addressed evidence bundle for offline evaluation certification.

The bundle binds exact report bytes and selected report identities with SHA-256. It does not
rerun audits, infer trading semantics, or authorize model selection/deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads
from app.research_atomic_output import write_text_no_clobber

REPORT_SPECS = {
    "plan": ("research_experiment_plan", {"PASS"}),
    "registry": ("research_experiment_registry", {"PASS", "PASS_WITH_WARNINGS"}),
    "output_integrity": ("research_evaluation_output_integrity", {"PASS", "PASS_WITH_WARNINGS"}),
    "row_window_lineage": ("research_evaluation_window_lineage", {"PASS", "PASS_WITH_WARNINGS"}),
    "metric_semantics": ("research_metric_semantic_recalculation", {"PASS", "PASS_WITH_WARNINGS"}),
    "certification": ("research_evaluation_integrity_certification", {"PASS", "PASS_WITH_WARNINGS"}),
}
FALSE_FLAGS = {
    "actionable": False,
    "inferential": False,
    "statistical_significance_claimed": False,
    "automatic_model_selection": False,
    "model_training_performed": False,
    "execution_connected": False,
    "automatic_repair_performed": False,
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _valid_identity_string(value: object) -> bool:
    return isinstance(value, str) and IDENTITY_RE.fullmatch(value) is not None


def _valid_bundle_filename(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and value not in {".", ".."}
        and "/" not in value
        and chr(92) not in value
        and Path(value).name == value
    )


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _read_report(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise ValueError(f"report_symlink_not_allowed:{path}")
    if not path.is_file():
        raise ValueError(f"report_must_be_regular_file:{path}")
    raw = path.read_bytes()
    try:
        payload = strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"report_json_invalid:{path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"report_root_must_be_object:{path}")
    return payload, raw


def _identity(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_name": payload.get("audit_name"),
        "status": payload.get("status"),
        "plan_id": payload.get("plan_id"),
        "plan_sha256": payload.get("plan_sha256"),
        "registry_id": payload.get("registry_id"),
        "deployment_authorized": payload.get("deployment_authorized"),
    }


def build_evidence_bundle(report_paths: dict[str, Path]) -> dict[str, Any]:
    if set(report_paths) != set(REPORT_SPECS):
        raise ValueError("report_roles_must_match_required_roles")
    entries: dict[str, Any] = {}
    shared_plan_id = shared_plan_sha = shared_registry_id = None
    for role in REPORT_SPECS:
        path = report_paths[role]
        if not _valid_bundle_filename(path.name):
            raise ValueError(f"report_filename_must_be_canonical_basename:{role}")
        payload, raw = _read_report(path)
        expected_name, allowed_statuses = REPORT_SPECS[role]
        if payload.get("audit_name") != expected_name:
            raise ValueError(f"audit_name_mismatch:{role}")
        report_status = payload.get("status")
        if not isinstance(report_status, str) or report_status not in allowed_statuses:
            raise ValueError(f"report_not_passing:{role}")
        report_errors = payload.get("errors")
        if role != "plan" and not isinstance(report_errors, list):
            raise ValueError(f"report_errors_schema_invalid:{role}")
        if "errors" in payload and not isinstance(report_errors, list):
            raise ValueError(f"report_errors_schema_invalid:{role}")
        if report_errors:
            raise ValueError(f"report_contains_errors:{role}")
        report_warnings = payload.get("warnings")
        if "warnings" in payload and (
            not isinstance(report_warnings, list)
            or any(not isinstance(item, str) or not item.strip() for item in report_warnings)
        ):
            raise ValueError(f"report_warnings_schema_invalid:{role}")
        if (
            (payload.get("status") == "PASS" and report_warnings)
            or (payload.get("status") == "PASS_WITH_WARNINGS" and not report_warnings)
        ):
            raise ValueError(f"report_status_warning_mismatch:{role}")
        if role != "plan" and payload.get("deployment_authorized") is not False:
            raise ValueError(f"deployment_authorization_not_false:{role}")
        for flag, expected in FALSE_FLAGS.items():
            if payload.get(flag) is not expected:
                raise ValueError(f"safety_flag_not_false:{role}:{flag}")
        plan_id = payload.get("plan_id")
        plan_sha = payload.get("plan_sha256")
        registry_id = payload.get("registry_id")
        if role == "plan":
            if not _valid_identity_string(plan_id):
                raise ValueError("plan_id_invalid:plan")
            if not _valid_sha256(plan_sha):
                raise ValueError("plan_sha256_invalid:plan")
            shared_plan_id, shared_plan_sha = plan_id, plan_sha
        else:
            if plan_id != shared_plan_id and shared_plan_id is not None:
                raise ValueError(f"plan_id_mismatch:{role}")
            if plan_sha != shared_plan_sha and shared_plan_sha is not None:
                raise ValueError(f"plan_sha256_mismatch:{role}")
            if role != "registry" and registry_id != shared_registry_id and shared_registry_id is not None:
                raise ValueError(f"registry_id_mismatch:{role}")
        if role == "registry":
            if not _valid_identity_string(registry_id):
                raise ValueError("registry_id_invalid:registry")
            shared_registry_id = registry_id
        entries[role] = {
            "path": path.name,
            "sha256": _sha(raw),
            "size_bytes": len(raw),
            "identity": _identity(payload),
        }
    bundle_body = {
        "schema_version": 1,
        "audit_name": "research_evaluation_certification_evidence_bundle",
        "status": "PASS",
        "plan_id": shared_plan_id,
        "plan_sha256": shared_plan_sha,
        "registry_id": shared_registry_id,
        "report_count": len(entries),
        "reports": entries,
        "verification_contract": "Exact report bytes must match each declared SHA-256; audit identities/statuses must match the bundle inventory.",
        **FALSE_FLAGS,
        "deployment_authorized": False,
        "scope_limitations": [
            "The bundle records exact report-byte digests and selected identity metadata; it does not rerun or independently validate upstream calculations.",
            "A bundle can be regenerated if all inputs are replaced; retain the bundle and its digest in a separate trusted location for tamper evidence.",
            "Passing establishes content consistency only, not profitability, strategy validity, or deployment suitability.",
        ],
    }
    bundle_body["bundle_sha256"] = _sha(_canonical(bundle_body))
    return bundle_body


def verify_evidence_bundle(bundle: dict[str, Any], *, evidence_root: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    if not isinstance(bundle, dict) or bundle.get("audit_name") != "research_evaluation_certification_evidence_bundle":
        raise ValueError("evidence_bundle_invalid")
    expected_digest = bundle.get("bundle_sha256")
    body = dict(bundle)
    body.pop("bundle_sha256", None)
    if not isinstance(expected_digest, str) or _sha(_canonical(body)) != expected_digest:
        errors.append({"code": "bundle_sha256_mismatch"})
    if bundle.get("status") != "PASS":
        errors.append({"code": "bundle_status_not_passing"})
    if not _valid_identity_string(bundle.get("plan_id")):
        errors.append({"code": "bundle_plan_id_invalid"})
    if not _valid_sha256(bundle.get("plan_sha256")):
        errors.append({"code": "bundle_plan_sha256_invalid"})
    if not _valid_identity_string(bundle.get("registry_id")):
        errors.append({"code": "bundle_registry_id_invalid"})
    for flag, expected in FALSE_FLAGS.items():
        if bundle.get(flag) is not expected:
            errors.append({"code": "bundle_safety_flag_not_false", "flag": flag})
    if bundle.get("deployment_authorized") is not False:
        errors.append({"code": "bundle_deployment_authorization_not_false"})
    reports = bundle.get("reports")
    if not isinstance(reports, dict) or set(reports) != set(REPORT_SPECS):
        raise ValueError("bundle_report_inventory_invalid")
    declared_report_count = bundle.get("report_count")
    if not isinstance(declared_report_count, int) or isinstance(declared_report_count, bool) or declared_report_count < 1:
        errors.append({"code": "bundle_report_count_invalid", "declared": declared_report_count})
    elif declared_report_count != len(reports):
        errors.append({
            "code": "bundle_report_count_mismatch",
            "declared": declared_report_count,
            "actual": len(reports),
        })
    root = evidence_root.resolve()
    verified = 0
    for role, (audit_name, allowed_statuses) in REPORT_SPECS.items():
        entry = reports.get(role)
        if not isinstance(entry, dict):
            errors.append({"code": "bundle_report_entry_invalid", "role": role})
            continue
        name = entry.get("path")
        if not _valid_bundle_filename(name):
            errors.append({"code": "bundle_report_path_invalid", "role": role})
            continue
        candidate = root / name
        if candidate.is_symlink():
            errors.append({"code": "bundle_report_symlink_not_allowed", "role": role})
            continue
        path = candidate.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append({"code": "bundle_report_path_escapes_root", "role": role})
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            errors.append({"code": "bundle_report_unreadable", "role": role})
            continue
        if _sha(raw) != entry.get("sha256"):
            errors.append({"code": "bundle_report_sha256_mismatch", "role": role})
            continue
        if len(raw) != entry.get("size_bytes"):
            errors.append({"code": "bundle_report_size_mismatch", "role": role})
            continue
        payload = _read_report(path)[0]
        report_status = payload.get("status")
        if payload.get("audit_name") != audit_name or not isinstance(report_status, str) or report_status not in allowed_statuses:
            errors.append({"code": "bundle_report_identity_or_status_mismatch", "role": role})
            continue
        report_errors = payload.get("errors")
        if role != "plan" and not isinstance(report_errors, list):
            errors.append({"code": "bundle_report_errors_schema_invalid", "role": role})
            continue
        if "errors" in payload and not isinstance(report_errors, list):
            errors.append({"code": "bundle_report_errors_schema_invalid", "role": role})
            continue
        if report_errors:
            errors.append({"code": "bundle_report_contains_errors", "role": role})
            continue
        report_warnings = payload.get("warnings")
        if "warnings" in payload and (
            not isinstance(report_warnings, list)
            or any(not isinstance(item, str) or not item.strip() for item in report_warnings)
        ):
            errors.append({"code": "bundle_report_warnings_schema_invalid", "role": role})
            continue
        if (
            (payload.get("status") == "PASS" and report_warnings)
            or (payload.get("status") == "PASS_WITH_WARNINGS" and not report_warnings)
        ):
            errors.append({"code": "bundle_report_status_warning_mismatch", "role": role})
            continue
        invalid_flag = next((flag for flag, expected in FALSE_FLAGS.items() if payload.get(flag) is not expected), None)
        if invalid_flag is not None:
            errors.append({"code": "bundle_report_safety_flag_not_false", "role": role, "flag": invalid_flag})
            continue
        if role != "plan" and payload.get("deployment_authorized") is not False:
            errors.append({"code": "bundle_report_deployment_authorized", "role": role})
            continue
        if _identity(payload) != entry.get("identity"):
            errors.append({"code": "bundle_report_identity_metadata_mismatch", "role": role})
            continue
        if payload.get("plan_id") != bundle.get("plan_id") or payload.get("plan_sha256") != bundle.get("plan_sha256"):
            errors.append({"code": "bundle_report_plan_binding_mismatch", "role": role})
            continue
        if role != "plan" and payload.get("registry_id") != bundle.get("registry_id"):
            errors.append({"code": "bundle_report_registry_binding_mismatch", "role": role})
            continue
        verified += 1
    status = "FAIL" if errors else "PASS"
    return {
        "schema_version": 1,
        "audit_name": "research_evaluation_certification_evidence_bundle_verification",
        "status": status,
        "readiness": "BUNDLE_INVALID" if errors else "EVIDENCE_BUNDLE_CONTENT_VERIFIED",
        "bundle_sha256": expected_digest,
        "reports_declared": len(reports),
        "reports_verified": verified,
        "errors": errors,
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "automatic_model_selection": False,
        "model_training_performed": False,
        "execution_connected": False,
        "automatic_repair_performed": False,
        "deployment_authorized": False,
        "scope_limitations": [
            "Verification compares report bytes and identity metadata to the supplied bundle.",
            "Store the bundle digest separately in a trusted location if it must detect coordinated replacement of both reports and bundle.",
            "No upstream audit is rerun and no trading or deployment action is performed.",
        ],
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    for role in REPORT_SPECS:
        build.add_argument(f"--{role.replace('_', '-')}", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--bundle", required=True, type=Path)
    verify.add_argument("--evidence-root", required=True, type=Path)
    verify.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "build":
            paths = {role: getattr(args, role) for role in REPORT_SPECS}
            report = build_evidence_bundle(paths)
        else:
            report = verify_evidence_bundle(_read_object(args.bundle), evidence_root=args.evidence_root)
        write_text_no_clobber(
            args.output,
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        print(json.dumps({"status": report["status"], "readiness": report.get("readiness", "BUNDLE_BUILT"), "bundle_sha256": report.get("bundle_sha256"), "reports_verified": report.get("reports_verified"), "error_count": len(report.get("errors", [])), "report_path": str(args.output.resolve()), "deployment_authorized": False}, indent=2))
        return 0 if report["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

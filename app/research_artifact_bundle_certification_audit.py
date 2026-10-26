"""Certify provenance and safety invariants across a bundle of research audit reports.

The manifest lists reports to inspect and expected audit types/digests. This is a
structural bundle audit, not a proof of strategy validity or absence of all leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.research_atomic_output import write_text_no_clobber

from app.research_strict_json import strict_json_loads

SAFE_FALSE_FLAGS = (
    "actionable",
    "automatic_model_selection",
    "inferential",
    "statistical_significance_claimed",
    "execution_connected",
    "model_training_performed",
    "automatic_repair_performed",
    "deployment_authorized",
)
KNOWN_STATUSES = {"PASS", "PASS_WITH_WARNINGS", "NOT_COMPARABLE", "FAIL"}


def _is_canonical_relative_posix_path(value: str) -> bool:
    """Accept one unambiguous, confined POSIX spelling for manifest artifact paths."""
    candidate = Path(value)
    return (
        bool(value)
        and value == value.strip()
        and chr(92) not in value
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and "." not in candidate.parts
        and candidate.as_posix() == value
    )


def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def audit_research_artifact_bundle(
    manifest: dict[str, Any],
    *,
    manifest_sha256: str,
    base_dir: Path,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest_must_be_object")
    if not _valid_sha256(manifest_sha256):
        raise ValueError("manifest_sha256_invalid")

    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("artifacts_must_be_nonempty_list")
    required = manifest.get("required_audits", [])
    if not isinstance(required, list) or any(not isinstance(item, str) or not item for item in required):
        raise ValueError("required_audits_invalid")
    expected_dataset = manifest.get("expected_dataset_sha256")
    if expected_dataset is not None and not _valid_sha256(expected_dataset):
        raise ValueError("expected_dataset_sha256_invalid")

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_types: set[str] = set()
    seen_paths: set[str] = set()

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append({"code": "artifact_entry_invalid", "index": index})
            continue
        audit_type = entry.get("audit_name")
        relative_path = entry.get("path")
        entry_errors: list[str] = []
        if not isinstance(audit_type, str) or not audit_type:
            entry_errors.append("audit_name_invalid")
            audit_type = f"unknown_{index}"
        if not isinstance(relative_path, str) or not relative_path:
            entry_errors.append("artifact_path_invalid")
            relative_path = ""
        if relative_path:
            candidate = Path(relative_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                entry_errors.append("artifact_path_must_be_relative_and_confined")
            elif not _is_canonical_relative_posix_path(relative_path):
                entry_errors.append("artifact_path_must_be_canonical_relative_posix")
            resolved = (base_dir / candidate).resolve()
            try:
                resolved.relative_to(base_dir.resolve())
            except ValueError:
                entry_errors.append("artifact_path_escapes_manifest_directory")
        else:
            resolved = base_dir / "__invalid_path__"
        if relative_path in seen_paths:
            entry_errors.append("duplicate_artifact_path")
        if relative_path:
            seen_paths.add(relative_path)
        if audit_type in seen_types:
            entry_errors.append("duplicate_audit_type")
        seen_types.add(audit_type)

        payload: dict[str, Any] | None = None
        digest = None
        if not any(code in entry_errors for code in (
            "artifact_path_invalid",
            "artifact_path_must_be_relative_and_confined",
            "artifact_path_must_be_canonical_relative_posix",
            "artifact_path_escapes_manifest_directory",
        )):
            try:
                raw = resolved.read_bytes()
                digest = _sha256(raw)
                decoded = _load_json_object(raw, source=str(resolved))
                if not isinstance(decoded, dict):
                    entry_errors.append("artifact_root_not_object")
                else:
                    payload = decoded
            except OSError:
                entry_errors.append("artifact_unreadable")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                entry_errors.append("artifact_json_invalid")

        if payload is not None:
            if payload.get("audit_name") != audit_type:
                entry_errors.append("audit_name_mismatch")
            schema_version = payload.get("schema_version")
            if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != 1:
                entry_errors.append("schema_version_invalid")
            status = payload.get("status")
            if not isinstance(status, str) or status not in KNOWN_STATUSES:
                entry_errors.append("status_unknown_or_missing")
            if status == "FAIL":
                entry_errors.append("upstream_audit_failed")
            upstream_errors = payload.get("errors")
            if not isinstance(upstream_errors, list):
                entry_errors.append("upstream_errors_field_missing_or_invalid")
            elif upstream_errors:
                entry_errors.append("upstream_report_contains_errors")
            upstream_warnings = payload.get("warnings")
            if (
                not isinstance(upstream_warnings, list)
                or any(not isinstance(item, str) or not item.strip() for item in upstream_warnings)
            ):
                entry_errors.append("upstream_warnings_field_invalid")
            if status == "PASS" and isinstance(upstream_warnings, list) and upstream_warnings:
                entry_errors.append("status_warnings_mismatch")
            elif status == "PASS_WITH_WARNINGS" and (
                not isinstance(upstream_warnings, list) or not upstream_warnings
            ):
                entry_errors.append("status_warnings_mismatch")
            for flag in SAFE_FALSE_FLAGS:
                if payload.get(flag) is not False:
                    entry_errors.append(f"safety_flag_not_false:{flag}")
            expected_report_digest = entry.get("expected_report_sha256")
            if expected_report_digest is not None:
                if not _valid_sha256(expected_report_digest):
                    entry_errors.append("expected_report_sha256_invalid")
                elif digest != expected_report_digest.lower():
                    entry_errors.append("report_sha256_mismatch")
            entry_expected_dataset = entry.get("expected_dataset_sha256", expected_dataset)
            if entry_expected_dataset is not None:
                if not _valid_sha256(entry_expected_dataset):
                    entry_errors.append("entry_expected_dataset_sha256_invalid")
                elif payload.get("dataset_sha256") != entry_expected_dataset.lower():
                    entry_errors.append("dataset_sha256_mismatch_or_missing")
            if status == "NOT_COMPARABLE":
                warnings.append(f"artifact_not_comparable:{audit_type}")
            if status == "PASS_WITH_WARNINGS":
                warnings.append(f"artifact_passed_with_warnings:{audit_type}")
            if payload.get("actionable") is not False:
                warnings.append(f"artifact_missing_explicit_non_actionable_flag:{audit_type}")
        for code in entry_errors:
            errors.append({"code": code, "audit_name": audit_type, "path": relative_path})
        results.append({
            "audit_name": audit_type,
            "path": relative_path,
            "report_sha256": digest,
            "upstream_status": payload.get("status") if payload else None,
            "passed": not entry_errors,
            "errors": entry_errors,
        })

    missing = sorted(set(required) - seen_types)
    for audit_type in missing:
        errors.append({"code": "required_audit_missing", "audit_name": audit_type})

    return {
        "schema_version": 1,
        "audit_name": "research_artifact_bundle_certification",
        "manifest_sha256": manifest_sha256.lower(),
        "artifacts_checked": len(results),
        "required_audits": required,
        "missing_required_audits": missing,
        "artifact_results": results,
        "errors": errors,
        "warnings": warnings,
        "status": "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS"),
        "actionable": False,
        "inferential": False,
        "model_training_performed": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "automatic_repair_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON bundle manifest")
    parser.add_argument("--output", type=Path, help="New report path; existing files are never overwritten")
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
        manifest = _load_json_object(raw, source=str(args.input))
        report = audit_research_artifact_bundle(
            manifest,
            manifest_sha256=_sha256(raw),
            base_dir=args.input.resolve().parent,
        )
        output = args.output or args.input.with_name(args.input.stem + "_certification.json")
        write_text_no_clobber(
            output,
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        print(json.dumps({
            "status": report["status"],
            "report_path": str(output.resolve()),
            "artifacts_checked": report["artifacts_checked"],
            "error_count": len(report["errors"]),
            "warning_count": len(report["warnings"]),
            "execution_connected": False,
        }, indent=2))
        return 0 if report["status"] != "FAIL" else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

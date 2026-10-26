"""Offline final evidence-pack audit for experiment registry and validation outputs.

This is an integrity/coverage gate over already-produced JSON reports. It does not run
experiments or upstream audits, train/select models, infer significance, or authorize trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads

def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


PASS_STATUSES = {"PASS", "PASS_WITH_WARNINGS"}
REQUIRED_ROLES = {"experiment_registry", "validation_workflow"}
FALSE_FLAGS = (
    "actionable",
    "inferential",
    "execution_connected",
    "model_training_performed",
    "automatic_model_selection",
    "automatic_repair_performed",
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdefABCDEF" for char in value
    )


def audit_research_evidence_pack(
    manifest: dict[str, Any], *, manifest_sha256: str, base_dir: Path
) -> dict[str, Any]:
    """Validate report integrity, required evidence roles, coverage, and safety flags."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest_must_be_object")
    if manifest.get("schema_version") != 1:
        raise ValueError("schema_version_unsupported")
    if not _valid_sha256(manifest_sha256):
        raise ValueError("manifest_sha256_invalid")
    pack_id = manifest.get("pack_id")
    if not isinstance(pack_id, str) or not pack_id.strip():
        raise ValueError("pack_id_invalid")
    entries = manifest.get("reports")
    if not isinstance(entries, list) or not entries:
        raise ValueError("reports_must_be_nonempty_list")
    required_roles = manifest.get("required_roles", sorted(REQUIRED_ROLES))
    if (
        not isinstance(required_roles, list)
        or not required_roles
        or any(not isinstance(role, str) or not role.strip() for role in required_roles)
        or len(set(required_roles)) != len(required_roles)
    ):
        raise ValueError("required_roles_invalid")
    if not REQUIRED_ROLES.issubset(set(required_roles)):
        raise ValueError("required_roles_must_include_core_roles")

    root = base_dir.resolve()
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    results: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    seen_run_ids: set[str] = set()
    dataset_fingerprints: set[str] = set()

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append({"code": "report_entry_invalid", "index": index})
            continue
        role = entry.get("role")
        relpath = entry.get("path")
        item_errors: list[str] = []
        if not isinstance(role, str) or not role.strip():
            role = f"invalid_role_{index}"
            item_errors.append("report_role_invalid")
        if role in seen_roles:
            item_errors.append("report_role_duplicate")
        seen_roles.add(role)
        if not isinstance(relpath, str) or not relpath:
            relpath = ""
            item_errors.append("report_path_invalid")
        if relpath in seen_paths:
            item_errors.append("report_path_duplicate")
        if relpath:
            seen_paths.add(relpath)

        payload: dict[str, Any] | None = None
        actual_hash: str | None = None
        if relpath:
            relative = Path(relpath)
            if relative.is_absolute() or ".." in relative.parts:
                item_errors.append("report_path_must_be_relative_and_confined")
            else:
                resolved = (root / relative).resolve()
                try:
                    resolved.relative_to(root)
                except ValueError:
                    item_errors.append("report_path_escapes_pack_root")
                else:
                    try:
                        raw = resolved.read_bytes()
                        actual_hash = _sha256(raw)
                        expected = entry.get("expected_sha256")
                        if not _valid_sha256(expected):
                            item_errors.append("expected_sha256_invalid")
                        elif actual_hash != expected.lower():
                            item_errors.append("report_sha256_mismatch")
                        decoded = _load_json_object(raw, source=str(resolved))
                        if not isinstance(decoded, dict):
                            item_errors.append("report_root_must_be_object")
                        else:
                            payload = decoded
                    except OSError:
                        item_errors.append("report_unreadable")
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                        item_errors.append("report_json_invalid")

        if payload is not None:
            if payload.get("status") not in PASS_STATUSES:
                item_errors.append("upstream_report_not_passing")
            for flag in FALSE_FLAGS:
                if payload.get(flag) is not False:
                    item_errors.append(f"safety_flag_not_explicitly_false:{flag}")
            shared_fingerprints = payload.get("dataset_fingerprints")
            if isinstance(shared_fingerprints, dict):
                for digest in shared_fingerprints:
                    if not _valid_sha256(digest):
                        item_errors.append("dataset_fingerprint_invalid")
                    else:
                        dataset_fingerprints.add(digest.lower())

            if role == "experiment_registry":
                supplied, validated = payload.get("runs_supplied"), payload.get("runs_validated")
                if (
                    not isinstance(supplied, int) or isinstance(supplied, bool)
                    or not isinstance(validated, int) or isinstance(validated, bool)
                    or supplied < 1 or validated != supplied
                ):
                    item_errors.append("registry_run_coverage_incomplete")
                run_results = payload.get("run_results")
                if not isinstance(run_results, list) or len(run_results) != validated:
                    item_errors.append("registry_run_results_incomplete")
                else:
                    for run in run_results:
                        run_id = run.get("run_id") if isinstance(run, dict) else None
                        if not isinstance(run_id, str) or not run_id:
                            item_errors.append("registry_run_id_missing")
                        elif run_id in seen_run_ids:
                            item_errors.append("registry_run_id_duplicate")
                        else:
                            seen_run_ids.add(run_id)
                        if not isinstance(run, dict) or run.get("status") != "PASS":
                            item_errors.append("registry_contains_unvalidated_run")
            elif role == "validation_workflow":
                checked = payload.get("artifacts_checked")
                missing = payload.get("missing_required_audits")
                fingerprints = payload.get("dataset_fingerprints")
                if not isinstance(checked, int) or isinstance(checked, bool) or checked < 1:
                    item_errors.append("validation_artifact_coverage_missing")
                if missing != []:
                    item_errors.append("validation_required_audit_coverage_incomplete")
                if not isinstance(fingerprints, dict) or not fingerprints:
                    item_errors.append("validation_dataset_fingerprint_missing")
                elif any(not _valid_sha256(digest) for digest in fingerprints):
                    item_errors.append("validation_dataset_fingerprint_invalid")
            elif role == "experiment_comparison":
                if payload.get("selection_performed") is not False:
                    item_errors.append("comparison_selection_flag_not_false")
                compared = payload.get("runs_compared")
                if not isinstance(compared, int) or isinstance(compared, bool) or compared < 2:
                    item_errors.append("comparison_run_coverage_insufficient")

        for code in item_errors:
            errors.append({"code": code, "role": role, "path": relpath})
        results.append({
            "role": role,
            "path": relpath,
            "report_sha256": actual_hash,
            "upstream_status": payload.get("status") if payload else None,
            "passed": not item_errors,
            "errors": item_errors,
        })

    for role in sorted(set(required_roles) - seen_roles):
        errors.append({"code": "required_report_role_missing", "role": role})
    if len(dataset_fingerprints) > 1:
        errors.append({
            "code": "dataset_fingerprint_mismatch",
            "dataset_fingerprints": sorted(dataset_fingerprints),
        })
    if not dataset_fingerprints:
        warnings.append("no_dataset_fingerprint_reconciled")

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_evidence_pack",
        "pack_id": pack_id,
        "manifest_sha256": manifest_sha256.lower(),
        "status": status,
        "readiness": "NOT_READY" if errors else (
            "READY_WITH_WARNINGS" if warnings else "EVIDENCE_INTEGRITY_VALIDATED"
        ),
        "reports_supplied": len(entries),
        "reports_validated": sum(1 for item in results if item["passed"]),
        "report_results": results,
        "required_roles": required_roles,
        "dataset_fingerprints_reconciled": sorted(dataset_fingerprints),
        "run_ids_reconciled": sorted(seen_run_ids),
        "errors": errors,
        "warnings": warnings,
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "execution_connected": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
        "automatic_repair_performed": False,
        "scope_limitations": [
            "Checks supplied report files and declared evidence only; it does not rerun upstream audits.",
            "A passing pack does not establish predictive validity, profitability, or absence of all leakage.",
            "This report never authorizes deployment or trading execution.",
        ],
    }


def _write_json_no_clobber(output_path: Path, report: dict[str, Any]) -> Path:
    """Publish one JSON receipt atomically without replacing a concurrent destination."""
    destination = output_path.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"output_already_exists:{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        temporary = None
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON evidence-pack manifest")
    parser.add_argument("--output", type=Path, help="New report path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        input_path = args.input.expanduser().resolve()
        raw = input_path.read_bytes()
        manifest = _load_json_object(raw, source=str(args.input))
        report = audit_research_evidence_pack(
            manifest, manifest_sha256=_sha256(raw), base_dir=input_path.parent
        )
        output = (args.output or input_path.with_name(input_path.stem + "_evidence_pack_audit.json")).expanduser().resolve()
        if output == input_path:
            raise ValueError("output_must_not_alias_input")
        published = _write_json_no_clobber(output, report)
        print(json.dumps({
            "status": report["status"],
            "readiness": report["readiness"],
            "report_path": str(published),
            "reports_validated": report["reports_validated"],
            "error_count": len(report["errors"]),
            "execution_connected": False,
        }, indent=2))
        return 0 if report["status"] in PASS_STATUSES else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

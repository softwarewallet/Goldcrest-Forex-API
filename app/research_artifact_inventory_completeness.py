"""Audit declared provenance artifacts against an independently curated inventory contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from app.research_strict_json import strict_json_loads
from app.research_atomic_output import write_text_no_clobber

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


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}_path_required")
    if value != value.strip():
        raise ValueError(f"{label}_path_must_be_canonical_unpadded")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." == value or "\\" in value:
        raise ValueError(f"{label}_path_must_be_safe_relative_posix_path")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"{label}_path_must_be_normalized:{value}")
    return normalized


def _index(entries: Any, label: str) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, Any]]]:
    if not isinstance(entries, list):
        raise ValueError(f"{label}_artifacts_must_be_list")
    if not entries:
        raise ValueError(f"{label}_artifacts_must_not_be_empty")
    by_id: dict[str, str] = {}
    paths: dict[str, str] = {}
    normalized: list[dict[str, str]] = []
    errors: list[dict[str, Any]] = []
    for position, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"{label}_artifact_entry_must_be_object:{position}")
        artifact_id = item.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ValueError(f"{label}_artifact_id_required:{position}")
        if artifact_id != artifact_id.strip():
            raise ValueError(f"{label}_artifact_id_must_be_canonical_unpadded:{position}")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", artifact_id) is None:
            raise ValueError(f"{label}_artifact_id_must_be_canonical_token:{position}")
        path = _safe_relative_path(item.get("path"), label)
        if artifact_id in by_id:
            errors.append({"code": "DUPLICATE_ARTIFACT_ID", "inventory": label, "artifact_id": artifact_id})
        if path in paths:
            errors.append({"code": "DUPLICATE_ARTIFACT_PATH", "inventory": label, "path": path,
                           "artifact_ids": sorted({paths[path], artifact_id})})
        by_id[artifact_id] = path
        paths[path] = artifact_id
        normalized.append({"artifact_id": artifact_id, "path": path})
    return by_id, normalized, errors


def audit_artifact_inventory_completeness(
    manifest: dict[str, Any], inventory_spec: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("audit_name") != "research_artifact_provenance_manifest":
        raise ValueError("provenance_manifest_invalid")
    if not isinstance(inventory_spec, dict) or inventory_spec.get("audit_name") != "research_artifact_inventory_specification":
        raise ValueError("inventory_specification_invalid")
    schema_version = inventory_spec.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("inventory_specification_schema_version_unsupported")
    manifest_entries = manifest.get("artifacts")
    spec_entries = inventory_spec.get("required_artifacts")
    manifest_index, manifest_normalized, manifest_errors = _index(manifest_entries, "manifest")
    spec_index, spec_normalized, spec_errors = _index(spec_entries, "inventory_spec")
    errors = [*manifest_errors, *spec_errors]
    declared_artifact_count = manifest.get("artifact_count")
    if (
        not isinstance(declared_artifact_count, int)
        or isinstance(declared_artifact_count, bool)
        or declared_artifact_count != len(manifest_normalized)
    ):
        errors.append({"code": "MANIFEST_ARTIFACT_COUNT_MISMATCH",
                       "declared": declared_artifact_count, "actual": len(manifest_normalized)})
    missing = [
        {"artifact_id": artifact_id, "path": path}
        for artifact_id, path in sorted(spec_index.items())
        if manifest_index.get(artifact_id) != path
    ]
    unexpected = [
        {"artifact_id": artifact_id, "path": path}
        for artifact_id, path in sorted(manifest_index.items())
        if spec_index.get(artifact_id) != path
    ]
    if missing:
        errors.append({"code": "REQUIRED_ARTIFACTS_MISSING_OR_MISLOCATED", "count": len(missing)})
    if unexpected:
        errors.append({"code": "UNEXPECTED_ARTIFACTS_PRESENT", "count": len(unexpected)})
    canonical_spec = (json.dumps(inventory_spec, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    canonical_manifest = (json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    status = "FAIL" if errors else "PASS"
    return {
        "schema_version": 1,
        "audit_name": "research_artifact_inventory_completeness_audit",
        "status": status,
        "readiness": "INVENTORY_INCOMPLETE" if errors else "INVENTORY_COMPLETE",
        "manifest_artifact_count": len(manifest_normalized),
        "manifest_sha256": hashlib.sha256(canonical_manifest).hexdigest(),
        "expected_artifact_count": len(spec_normalized),
        "missing_or_mislocated_artifacts": missing,
        "unexpected_or_mislocated_artifacts": unexpected,
        "inventory_spec_sha256": hashlib.sha256(canonical_spec).hexdigest(),
        "inventory_spec_artifact_count": len(spec_normalized),
        "errors": errors,
        **FALSE_FLAGS,
        "scope_limitations": [
            "Completeness is only relative to the independently curated inventory specification supplied to this audit.",
            "A generated specification copied from the same manifest cannot independently prove that omitted artifacts were identified.",
            "The inventory specification digest is reported for provenance; this audit does not authenticate its author or independently seal the specification.",
            "No files are created, modified, regenerated, or executed by this audit.",
        ],
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance-manifest", required=True, type=Path)
    parser.add_argument("--inventory-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = audit_artifact_inventory_completeness(
            _read_object(args.provenance_manifest), _read_object(args.inventory_spec)
        )
        write_text_no_clobber(
            args.output,
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        print(json.dumps({
            "status": report["status"],
            "manifest_artifact_count": report["manifest_artifact_count"],
            "expected_artifact_count": report["expected_artifact_count"],
            "missing_count": len(report["missing_or_mislocated_artifacts"]),
            "unexpected_count": len(report["unexpected_or_mislocated_artifacts"]),
            "report_path": str(args.output.resolve()),
            "deployment_authorized": False,
        }, indent=2))
        return 0 if report["status"] == "PASS" else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}",
                          "deployment_authorized": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

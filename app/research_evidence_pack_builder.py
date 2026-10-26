"""Assemble a hash-addressed offline research evidence-pack manifest and audit it."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from app.research_evidence_pack_audit import audit_research_evidence_pack
from app.research_strict_json import strict_json_loads

_CORE_ROLES = {"experiment_registry", "validation_workflow"}


def _read_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report_root_must_be_object:{path}")
    return value, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json_no_clobber(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
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
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse_report_spec(spec: str) -> tuple[str, Path]:
    role, separator, raw_path = spec.partition("=")
    if not separator or not role.strip() or not raw_path.strip():
        raise ValueError("report_spec_must_be_ROLE_EQUALS_PATH")
    return role.strip(), Path(raw_path.strip()).expanduser().resolve()


def assemble_evidence_pack(
    *,
    pack_id: str,
    report_specs: list[str],
    required_roles: list[str] | None,
    output_path: str | Path,
    audit_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a confined manifest and immediately audit the referenced evidence."""
    if not isinstance(pack_id, str) or not pack_id.strip():
        raise ValueError("pack_id_missing")
    if not report_specs:
        raise ValueError("reports_required")
    manifest_path = Path(output_path).expanduser().resolve()
    audit_path = (
        Path(audit_output_path).expanduser().resolve()
        if audit_output_path is not None
        else manifest_path.with_name(f"{manifest_path.stem}_audit.json")
    )
    if manifest_path == audit_path:
        raise ValueError("manifest_and_audit_output_must_differ")

    parsed = [_parse_report_spec(spec) for spec in report_specs]
    roles = [role for role, _ in parsed]
    paths = [path for _, path in parsed]
    if len(set(roles)) != len(roles):
        raise ValueError("report_roles_must_be_unique")
    if len(set(paths)) != len(paths):
        raise ValueError("report_paths_must_be_unique")

    root = manifest_path.parent.resolve()
    entries: list[dict[str, str]] = []
    for role, path in parsed:
        if not path.is_file():
            raise ValueError(f"report_not_found:{path}")
        try:
            relative = path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"report_path_outside_pack_directory:{path}") from exc
        _payload, raw = _read_object(path)
        entries.append({
            "role": role,
            "path": relative.as_posix(),
            "expected_sha256": _sha256(raw),
        })

    selected_required = list(required_roles or sorted(_CORE_ROLES))
    if len(set(selected_required)) != len(selected_required):
        raise ValueError("required_roles_must_be_unique")
    if not _CORE_ROLES.issubset(set(selected_required)):
        raise ValueError("required_roles_must_include_core_roles")
    missing = sorted(set(selected_required) - set(roles))
    if missing:
        raise ValueError(f"required_roles_not_supplied:{','.join(missing)}")

    input_paths = set(paths)
    if manifest_path in input_paths or audit_path in input_paths:
        raise ValueError("output_must_not_alias_input_report")
    if manifest_path.exists() or audit_path.exists():
        existing = manifest_path if manifest_path.exists() else audit_path
        raise ValueError(f"output_already_exists:{existing}")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "pack_id": pack_id.strip(),
        "required_roles": selected_required,
        "reports": entries,
    }
    manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    audit = audit_research_evidence_pack(
        manifest,
        manifest_sha256=_sha256(manifest_raw),
        base_dir=root,
    )
    _write_json_no_clobber(manifest_path, manifest)
    _write_json_no_clobber(audit_path, audit)
    return {
        "status": audit["status"],
        "readiness": audit["readiness"],
        "pack_id": pack_id.strip(),
        "manifest_path": str(manifest_path),
        "audit_path": str(audit_path),
        "reports_supplied": audit["reports_supplied"],
        "reports_validated": audit["reports_validated"],
        "error_count": len(audit["errors"]),
        "warning_count": len(audit["warnings"]),
        "execution_connected": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-id", required=True)
    parser.add_argument("--report", action="append", required=True,
                        help="Report role and path as ROLE=PATH; repeat for each report")
    parser.add_argument("--required-role", action="append", default=None,
                        help="Additional required role; core roles are always required")
    parser.add_argument("--output", required=True, type=Path, help="New evidence-pack manifest path")
    parser.add_argument("--audit-output", type=Path, help="New audit receipt path (default: <manifest-stem>_audit.json)")
    args = parser.parse_args(argv)
    try:
        result = assemble_evidence_pack(
            pack_id=args.pack_id,
            report_specs=args.report,
            required_roles=args.required_role,
            output_path=args.output,
            audit_output_path=args.audit_output,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

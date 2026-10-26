"""Create and verify a detached integrity seal for artifact inventory specifications."""
from __future__ import annotations

import argparse
import hashlib
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


def canonical_inventory_spec_bytes(spec: dict[str, Any]) -> bytes:
    if not isinstance(spec, dict):
        raise ValueError("inventory_spec_must_be_object")
    if spec.get("audit_name") != "research_artifact_inventory_specification":
        raise ValueError("inventory_specification_invalid")
    if spec.get("schema_version") != 1:
        raise ValueError("inventory_specification_schema_version_unsupported")
    entries = spec.get("required_artifacts")
    if not isinstance(entries, list):
        raise ValueError("inventory_spec_required_artifacts_must_be_list")
    return (json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def create_inventory_spec_seal(spec: dict[str, Any]) -> dict[str, Any]:
    raw = canonical_inventory_spec_bytes(spec)
    entries = spec["required_artifacts"]
    return {
        "schema_version": 1,
        "audit_name": "research_artifact_inventory_specification_detached_seal",
        "canonicalization": "json-sort-keys-compact-utf8-trailing-newline-v1",
        "inventory_spec_sha256": hashlib.sha256(raw).hexdigest(),
        "inventory_spec_bytes": len(raw),
        "inventory_spec_schema_version": spec["schema_version"],
        "required_artifact_count": len(entries),
        **FALSE_FLAGS,
        "scope_limitations": [
            "The seal detects specification edits only when retained separately or its digest is anchored in a separately controlled location.",
            "The seal is not a digital signature and does not authenticate the specification author.",
            "A sealed inventory can still be incomplete or incorrect if its contents were not independently curated.",
        ],
    }


def verify_inventory_spec_seal(spec: dict[str, Any], seal: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    if not isinstance(seal, dict) or seal.get("audit_name") != "research_artifact_inventory_specification_detached_seal":
        errors.append({"code": "INVENTORY_SEAL_SCHEMA_INVALID"})
    else:
        try:
            expected = create_inventory_spec_seal(spec)
        except (TypeError, ValueError) as exc:
            expected = None
            errors.append({"code": "INVENTORY_SPEC_INVALID", "detail": str(exc)})
        if expected is not None:
            for key in ("schema_version", "canonicalization", "inventory_spec_sha256",
                        "inventory_spec_bytes", "inventory_spec_schema_version", "required_artifact_count"):
                if seal.get(key) != expected.get(key):
                    errors.append({"code": "INVENTORY_SEAL_FIELD_MISMATCH", "field": key})
            for flag in FALSE_FLAGS:
                if seal.get(flag) is not False:
                    errors.append({"code": "SAFETY_FLAG_NOT_EXPLICITLY_FALSE", "field": flag})
    return {
        "schema_version": 1,
        "audit_name": "research_artifact_inventory_specification_seal_verification",
        "status": "FAIL" if errors else "PASS",
        "inventory_spec_sha256": seal.get("inventory_spec_sha256") if isinstance(seal, dict) else None,
        "errors": errors,
        **FALSE_FLAGS,
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path.read_bytes(), source=str(path))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("create", "verify"), required=True)
    parser.add_argument("--inventory-spec", required=True, type=Path)
    parser.add_argument("--seal", type=Path, help="Required only in verify mode")
    parser.add_argument("--output", type=Path, help="Required for create mode")
    args = parser.parse_args()
    try:
        spec = _read_object(args.inventory_spec)
        if args.mode == "create":
            if args.output is None:
                raise ValueError("output_required_for_create_mode")
            seal = create_inventory_spec_seal(spec)
            if args.output.exists():
                raise ValueError(f"output_already_exists:{args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps({"status": "PASS", "inventory_spec_sha256": seal["inventory_spec_sha256"],
                              "seal_path": str(args.output.resolve()), "deployment_authorized": False}, indent=2))
            return 0
        if args.seal is None:
            raise ValueError("seal_required_for_verify_mode")
        report = verify_inventory_spec_seal(spec, _read_object(args.seal))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "PASS" else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}",
                          "deployment_authorized": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

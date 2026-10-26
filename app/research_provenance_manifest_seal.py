"""Create and verify a detached SHA-256 seal for research provenance manifests.

The seal detects accidental or malicious manifest edits only when the seal is retained in a
separately controlled location. It is not a digital signature and does not establish signer identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads

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


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    if not isinstance(manifest, dict):
        raise ValueError("manifest_must_be_json_object")
    return (json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def create_manifest_seal(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest_must_be_json_object")
    if manifest.get("audit_name") != "research_artifact_provenance_manifest":
        raise ValueError("manifest_audit_name_invalid")
    for flag in FALSE_FLAGS:
        if manifest.get(flag) is not False:
            raise ValueError(f"manifest_safety_flag_not_explicitly_false:{flag}")
    raw = canonical_manifest_bytes(manifest)
    revision = manifest.get("source_revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdefABCDEF" for c in revision):
        raise ValueError("manifest_source_revision_must_be_full_git_sha")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("manifest_artifacts_must_be_list")
    if manifest.get("artifact_count") != len(artifacts):
        raise ValueError("manifest_artifact_count_mismatch")
    return {
        "schema_version": 1,
        "audit_name": "research_provenance_manifest_detached_seal",
        "canonicalization": "json-sort-keys-compact-utf8-trailing-newline-v1",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_bytes": len(raw),
        "source_revision": revision.lower(),
        "artifact_count": len(artifacts),
        "manifest_audit_name": manifest.get("audit_name"),
        **FALSE_FLAGS,
        "scope_limitations": [
            "A matching seal proves byte-level agreement with the sealed canonical manifest, not that the manifest inventory is complete or truthful.",
            "The seal is not a digital signature and does not identify or authenticate its creator.",
            "The seal must be retained separately or its digest anchored in a separately controlled location.",
        ],
    }


def verify_manifest_seal(manifest: dict[str, Any], seal: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    if not isinstance(seal, dict) or seal.get("audit_name") != "research_provenance_manifest_detached_seal":
        errors.append({"code": "SEAL_SCHEMA_INVALID"})
    else:
        try:
            expected = create_manifest_seal(manifest)
        except (TypeError, ValueError) as exc:
            expected = None
            errors.append({"code": "MANIFEST_INVALID", "detail": str(exc)})
        if expected is not None:
            for key in ("schema_version", "canonicalization", "manifest_sha256", "manifest_bytes",
                        "source_revision", "artifact_count", "manifest_audit_name"):
                if seal.get(key) != expected.get(key):
                    errors.append({"code": "SEAL_FIELD_MISMATCH", "field": key})
            for flag in FALSE_FLAGS:
                if seal.get(flag) is not False:
                    errors.append({"code": "SAFETY_FLAG_NOT_EXPLICITLY_FALSE", "field": flag})
    return {
        "schema_version": 1,
        "audit_name": "research_provenance_manifest_seal_verification",
        "status": "FAIL" if errors else "PASS",
        "manifest_sha256": seal.get("manifest_sha256") if isinstance(seal, dict) else None,
        "errors": errors,
        **FALSE_FLAGS,
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("create", "verify"), required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seal", type=Path, help="Required only in verify mode")
    parser.add_argument("--output", type=Path, help="Required for create mode; verification prints its report")
    args = parser.parse_args()
    try:
        manifest = _read_object(args.manifest)
        if args.mode == "create":
            if args.output is None:
                raise ValueError("output_required_for_create_mode")
            seal = create_manifest_seal(manifest)
            if args.output.exists():
                raise ValueError(f"output_already_exists:{args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps({"status": "PASS", "manifest_sha256": seal["manifest_sha256"],
                              "seal_path": str(args.output.resolve()), "deployment_authorized": False}, indent=2))
            return 0
        if args.seal is None:
            raise ValueError("seal_required_for_verify_mode")
        report = verify_manifest_seal(manifest, _read_object(args.seal))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "PASS" else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}",
                          "deployment_authorized": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

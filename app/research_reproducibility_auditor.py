"""Cross-artifact reproducibility audit for completed offline research runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads
from app.research_experiment_fingerprint import canonical_json, verify_experiment_fingerprint


def _read_object(path: Path) -> dict[str, Any]:
    """Read a strict JSON object; malformed or ambiguous inputs fail closed."""
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("json_root_must_be_object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_artifact(value: object, manifest_path: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        relative_to_manifest = manifest_path.parent / candidate
        if relative_to_manifest.is_file():
            return relative_to_manifest.resolve()
    return candidate.resolve()


def _check_file_hash(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    path_field: str,
    hash_field: str,
    check_name: str,
    checks: dict[str, str],
    errors: list[str],
) -> Path | None:
    path = _resolve_artifact(manifest.get(path_field), manifest_path)
    if path is None or not path.is_file():
        checks[check_name] = "FAIL"
        errors.append(f"artifact_missing:{path_field}")
        return None
    try:
        actual = _sha256(path)
    except OSError:
        checks[check_name] = "FAIL"
        errors.append(f"artifact_unreadable:{path_field}")
        return None
    if manifest.get(hash_field) != actual:
        checks[check_name] = "FAIL"
        errors.append(f"artifact_hash_mismatch:{hash_field}")
    else:
        checks[check_name] = "PASS"
    return path


def audit_experiment_reproducibility(
    fingerprint_path: str | Path,
    pipeline_manifest_path: str | Path,
) -> dict[str, Any]:
    """Check fingerprint self-integrity, pipeline linkage, and referenced artifact bytes."""
    fp_path = Path(fingerprint_path).expanduser().resolve()
    pipe_path = Path(pipeline_manifest_path).expanduser().resolve()
    errors: list[str] = []
    checks: dict[str, str] = {}
    fingerprint: dict[str, Any] = {}
    manifest: dict[str, Any] = {}

    try:
        fingerprint = _read_object(fp_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"fingerprint_unreadable:{type(exc).__name__}")
    try:
        manifest = _read_object(pipe_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"pipeline_manifest_unreadable:{type(exc).__name__}")

    if fingerprint:
        verification = verify_experiment_fingerprint(fingerprint)
        checks["fingerprint_self_hash"] = verification["status"]
        errors.extend(verification["errors"])
    if manifest:
        if manifest.get("actionable") is not False:
            errors.append("pipeline_actionable_must_be_false")
        if manifest.get("execution_connected") is not False:
            errors.append("pipeline_execution_connected_must_be_false")
        claimed_manifest_hash = manifest.get("pipeline_manifest_sha256")
        unhashed_manifest = dict(manifest)
        unhashed_manifest.pop("pipeline_manifest_sha256", None)
        try:
            actual_manifest_hash = hashlib.sha256(canonical_json(unhashed_manifest)).hexdigest()
        except ValueError:
            actual_manifest_hash = None
            errors.append("pipeline_manifest_not_canonicalizable")
        checks["pipeline_manifest_self_hash"] = (
            "PASS" if claimed_manifest_hash == actual_manifest_hash else "FAIL"
        )
        if claimed_manifest_hash != actual_manifest_hash:
            errors.append("pipeline_manifest_sha256_mismatch")

    if fingerprint and manifest:
        if manifest.get("experiment_fingerprint_sha256") != fingerprint.get("experiment_fingerprint_sha256"):
            checks["pipeline_fingerprint_link"] = "FAIL"
            errors.append("pipeline_fingerprint_hash_mismatch")
        else:
            checks["pipeline_fingerprint_link"] = "PASS"

        actual_fingerprint_file_hash = _sha256(fp_path)
        if manifest.get("experiment_fingerprint_file_sha256") != actual_fingerprint_file_hash:
            checks["fingerprint_file_hash_link"] = "FAIL"
            errors.append("pipeline_fingerprint_file_hash_mismatch")
        else:
            checks["fingerprint_file_hash_link"] = "PASS"

        recorded_fingerprint_path = _resolve_artifact(manifest.get("experiment_fingerprint_path"), pipe_path)
        if recorded_fingerprint_path != fp_path:
            checks["fingerprint_path_link"] = "FAIL"
            errors.append("pipeline_fingerprint_path_mismatch")
        else:
            checks["fingerprint_path_link"] = "PASS"

        dataset_manifest_path = _check_file_hash(
            manifest,
            pipe_path,
            path_field="dataset_manifest_path",
            hash_field="dataset_manifest_sha256",
            check_name="dataset_manifest_file_hash",
            checks=checks,
            errors=errors,
        )
        if dataset_manifest_path is not None:
            try:
                dataset_manifest = _read_object(dataset_manifest_path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"dataset_manifest_unreadable:{type(exc).__name__}")
            else:
                if dataset_manifest.get("output_dataset_sha256") != fingerprint.get("dataset_sha256"):
                    errors.append("fingerprint_dataset_hash_mismatch")
                if dataset_manifest.get("feature_implementation_sha256") != fingerprint.get("feature_implementation_sha256"):
                    errors.append("fingerprint_feature_implementation_hash_mismatch")
                if dataset_manifest.get("feature_schema_sha256") != fingerprint.get("feature_schema_sha256"):
                    errors.append("fingerprint_feature_schema_hash_mismatch")

        _check_file_hash(
            manifest,
            pipe_path,
            path_field="dataset_path",
            hash_field="dataset_sha256",
            check_name="dataset_file_hash",
            checks=checks,
            errors=errors,
        )
        _check_file_hash(
            manifest,
            pipe_path,
            path_field="raw_ohlc_path",
            hash_field="raw_ohlc_sha256",
            check_name="raw_ohlc_file_hash",
            checks=checks,
            errors=errors,
        )

    return {
        "schema_version": 1,
        "scope": "OFFLINE_RESEARCH_REPRODUCIBILITY_AUDIT",
        "status": "FAIL" if errors else "PASS",
        "errors": list(dict.fromkeys(errors)),
        "checks": checks,
        "fingerprint_sha256": fingerprint.get("experiment_fingerprint_sha256"),
        "pipeline_manifest_sha256": manifest.get("pipeline_manifest_sha256"),
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "Cross-artifact integrity only. A passing audit does not prove strategy profitability, "
            "absence of all research bias, or deployment readiness."
        ),
    }


def write_reproducibility_audit(
    fingerprint_path: str | Path,
    pipeline_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write an offline reproducibility audit receipt without overwriting prior evidence."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing_to_overwrite_audit_receipt:{output}")
    result = audit_experiment_reproducibility(fingerprint_path, pipeline_manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

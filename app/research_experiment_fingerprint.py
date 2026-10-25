"""Canonical reproducibility fingerprints for offline research experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


_SCOPE = "OFFLINE_RESEARCH_REPRODUCIBILITY"


def canonical_json(payload: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("fingerprint_payload_not_canonicalizable") from exc


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_experiment_fingerprint(
    *,
    configuration: dict[str, Any],
    raw_ohlc_sha256: str,
    dataset_sha256: str,
    dataset_manifest_path: str | Path,
    dataset_manifest: dict[str, Any],
    implementation_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Fingerprint source data, prepared dataset, declared configuration, and code files."""
    if not isinstance(configuration, dict):
        raise ValueError("configuration_must_be_object")
    if not _is_sha256(raw_ohlc_sha256):
        raise ValueError("raw_ohlc_sha256_invalid")
    if not _is_sha256(dataset_sha256):
        raise ValueError("dataset_sha256_invalid")
    if not isinstance(dataset_manifest, dict):
        raise ValueError("dataset_manifest_must_be_object")
    if dataset_manifest.get("actionable") is not False:
        raise ValueError("dataset_manifest_must_be_explicitly_non_actionable")
    if dataset_manifest.get("output_dataset_sha256") != dataset_sha256:
        raise ValueError("dataset_manifest_dataset_hash_mismatch")

    for field in ("feature_implementation_sha256", "feature_schema_sha256"):
        if not _is_sha256(dataset_manifest.get(field)):
            raise ValueError(f"dataset_manifest_{field}_invalid")

    manifest_path = Path(dataset_manifest_path)
    manifest_bytes_sha256 = sha256_file(manifest_path)
    manifest_without_hash = dict(dataset_manifest)
    declared_manifest_hash = manifest_without_hash.pop("manifest_sha256", None)
    actual_manifest_hash = hashlib.sha256(canonical_json(manifest_without_hash)).hexdigest()
    if not _is_sha256(declared_manifest_hash) or declared_manifest_hash != actual_manifest_hash:
        raise ValueError("dataset_manifest_self_hash_mismatch")

    if not implementation_paths:
        raise ValueError("implementation_paths_must_not_be_empty")
    implementation_hashes: dict[str, str] = {}
    for name, path in sorted(implementation_paths.items()):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("implementation_path_name_invalid")
        implementation_hashes[name] = sha256_file(path)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "scope": _SCOPE,
        "configuration": configuration,
        "raw_ohlc_sha256": raw_ohlc_sha256,
        "dataset_sha256": dataset_sha256,
        "dataset_manifest_file_sha256": manifest_bytes_sha256,
        "dataset_manifest_content_sha256": actual_manifest_hash,
        "feature_implementation_sha256": dataset_manifest["feature_implementation_sha256"],
        "feature_schema_sha256": dataset_manifest["feature_schema_sha256"],
        "feature_backend": dataset_manifest.get("feature_backend"),
        "implementation_file_sha256": implementation_hashes,
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
    }
    payload["experiment_fingerprint_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def verify_experiment_fingerprint(receipt: dict[str, Any]) -> dict[str, Any]:
    """Verify the canonical self-hash, required digests, and research-only safety flags."""
    errors: list[str] = []
    if not isinstance(receipt, dict):
        receipt = {}
        errors.append("receipt_must_be_object")

    if receipt.get("schema_version") != 1:
        errors.append("schema_version_invalid")
    if receipt.get("scope") != _SCOPE:
        errors.append("scope_invalid")
    if not isinstance(receipt.get("configuration"), dict):
        errors.append("configuration_must_be_object")

    for field in (
        "raw_ohlc_sha256",
        "dataset_sha256",
        "dataset_manifest_file_sha256",
        "dataset_manifest_content_sha256",
        "feature_implementation_sha256",
        "feature_schema_sha256",
        "experiment_fingerprint_sha256",
    ):
        if not _is_sha256(receipt.get(field)):
            errors.append(f"{field}_invalid")

    implementation_hashes = receipt.get("implementation_file_sha256")
    if (
        not isinstance(implementation_hashes, dict)
        or not implementation_hashes
        or any(
            not isinstance(name, str) or not name.strip() or not _is_sha256(digest)
            for name, digest in implementation_hashes.items()
        )
    ):
        errors.append("implementation_file_sha256_invalid")

    for flag in ("actionable", "execution_connected", "automatic_model_selection"):
        if receipt.get(flag) is not False:
            errors.append(f"{flag}_must_be_false")

    claimed = receipt.get("experiment_fingerprint_sha256")
    unhashed = dict(receipt)
    unhashed.pop("experiment_fingerprint_sha256", None)
    try:
        actual = hashlib.sha256(canonical_json(unhashed)).hexdigest()
    except ValueError:
        actual = None
        errors.append("receipt_not_canonicalizable")
    if claimed != actual:
        errors.append("experiment_fingerprint_sha256_mismatch")

    return {
        "schema_version": 1,
        "scope": _SCOPE,
        "status": "FAIL" if errors else "PASS",
        "errors": list(dict.fromkeys(errors)),
        "experiment_fingerprint_sha256": claimed if isinstance(claimed, str) else None,
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
    }

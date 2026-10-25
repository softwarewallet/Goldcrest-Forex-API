"""Verify that a prepared research dataset is bound to its feature schema and source code.

This audit is offline and read-only. It validates artifact identity and declared feature-source
fingerprints; it does not by itself prove that arbitrary feature code is causally correct.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


_EXPECTED_SCOPE = "OFFLINE_RESEARCH_DATASET_PREPARATION"
_EXPECTED_IMPLEMENTATION_FILE = "app/causal_ta_features.py"
_METADATA_FIELDS = {
    "symbol", "timeframe", "timestamp", "open", "high", "low", "close",
    "label", "label_available_at", "label_elapsed_hours", "label_eligible",
}


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def audit_feature_source_provenance(
    dataset_csv: str | Path,
    manifest_path: str | Path,
    *,
    feature_source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate dataset bytes, manifest self-hash, feature schema, and source-code hash."""
    dataset_path = Path(dataset_csv).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    source_path = (
        Path(feature_source_path).expanduser().resolve()
        if feature_source_path is not None
        else Path(__file__).with_name("causal_ta_features.py").resolve()
    )
    errors: list[str] = []
    manifest: dict[str, Any] = {}

    try:
        loaded = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("manifest_root_must_be_object")
        manifest = loaded
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "schema_version": 1,
            "audit_name": "research_feature_source_provenance_audit",
            "status": "FAIL",
            "errors": [f"manifest_unreadable:{type(exc).__name__}"],
            "warnings": [],
            "dataset_path": str(dataset_path),
            "manifest_path": str(manifest_file),
            "feature_source_path": str(source_path),
            "actionable": False,
            "execution_connected": False,
            "model_training_performed": False,
            "automatic_model_selection": False,
        }

    if manifest.get("scope") != _EXPECTED_SCOPE:
        errors.append("manifest_scope_mismatch")
    if manifest.get("actionable") is not False:
        errors.append("manifest_must_be_explicitly_non_actionable")

    declared_manifest_hash = manifest.get("manifest_sha256")
    unhashed_manifest = dict(manifest)
    unhashed_manifest.pop("manifest_sha256", None)
    actual_manifest_hash = hashlib.sha256(_canonical_json(unhashed_manifest)).hexdigest()
    if not _is_sha256(declared_manifest_hash) or declared_manifest_hash != actual_manifest_hash:
        errors.append("manifest_sha256_mismatch")

    feature_names = manifest.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or any(not isinstance(name, str) or not name.strip() for name in feature_names)
        or len(feature_names) != len(set(feature_names))
    ):
        errors.append("feature_names_invalid")
        feature_names = []
    expected_schema_hash = hashlib.sha256(
        _canonical_json({"feature_names": feature_names})
    ).hexdigest()
    declared_schema_hash = manifest.get("feature_schema_sha256")
    if not _is_sha256(declared_schema_hash) or declared_schema_hash != expected_schema_hash:
        errors.append("feature_schema_sha256_mismatch")

    if manifest.get("feature_implementation_file") != _EXPECTED_IMPLEMENTATION_FILE:
        errors.append("feature_implementation_file_mismatch")
    declared_implementation_hash = manifest.get("feature_implementation_sha256")
    if not _is_sha256(declared_implementation_hash):
        errors.append("feature_implementation_sha256_invalid")
    try:
        actual_implementation_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError:
        actual_implementation_hash = None
        errors.append("feature_implementation_source_unreadable")
    if actual_implementation_hash is not None and declared_implementation_hash != actual_implementation_hash:
        errors.append("feature_implementation_sha256_mismatch")

    try:
        dataset_bytes = dataset_path.read_bytes()
        actual_dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
        if manifest.get("output_dataset_sha256") != actual_dataset_hash:
            errors.append("output_dataset_sha256_mismatch")
        with dataset_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = list(reader.fieldnames or [])
            row_count = sum(1 for _ in reader)
        actual_feature_columns = [column for column in columns if column not in _METADATA_FIELDS]
        if actual_feature_columns != feature_names:
            errors.append("dataset_feature_columns_mismatch")
        declared_rows = manifest.get("output_rows")
        if isinstance(declared_rows, bool) or not isinstance(declared_rows, int) or declared_rows != row_count:
            errors.append("dataset_output_row_count_mismatch")
    except (OSError, UnicodeDecodeError, csv.Error):
        actual_dataset_hash = None
        row_count = None
        errors.append("dataset_unreadable")

    status = "FAIL" if errors else "PASS"
    return {
        "schema_version": 1,
        "audit_name": "research_feature_source_provenance_audit",
        "status": status,
        "dataset_path": str(dataset_path),
        "dataset_sha256": actual_dataset_hash,
        "manifest_path": str(manifest_file),
        "manifest_sha256_declared": declared_manifest_hash,
        "manifest_sha256_actual": actual_manifest_hash,
        "feature_source_path": str(source_path),
        "feature_implementation_sha256_declared": declared_implementation_hash,
        "feature_implementation_sha256_actual": actual_implementation_hash,
        "feature_schema_sha256_declared": declared_schema_hash,
        "feature_count": len(feature_names),
        "dataset_rows": row_count,
        "errors": errors,
        "warnings": [
            "source fingerprint equality does not prove feature implementation causality; use point-in-time invariance audits"
        ],
        "actionable": False,
        "execution_connected": False,
        "model_training_performed": False,
        "automatic_model_selection": False,
    }

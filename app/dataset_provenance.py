"""Explicit provenance contracts for offline market-data snapshots.

This module validates operator-supplied metadata; it does not fetch data or claim
that a source, license, or snapshot is authentic. The content hash must be computed
from the exact immutable source artifact before parsing/transformation.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _utc(value: datetime, *, field: str) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping using stable canonical JSON."""
    try:
        payload = json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"provenance contains non-canonical JSON values: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DatasetProvenance:
    source_name: str
    source_version: str
    snapshot_sha256: str
    retrieved_at: str
    timestamp_timezone: str
    duplicate_policy: str
    adjustment_policy: str
    license_or_terms_reference: str
    instruments: tuple[str, ...]
    timeframe: str
    row_count: int
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["instruments"] = list(self.instruments)
        return result


def build_dataset_provenance(
    *,
    source_name: str,
    source_version: str,
    snapshot_sha256: str,
    retrieved_at: datetime,
    timestamp_timezone: str,
    duplicate_policy: str,
    adjustment_policy: str,
    license_or_terms_reference: str,
    instruments: list[str] | tuple[str, ...],
    timeframe: str,
    row_count: int,
) -> DatasetProvenance:
    """Build a provenance record only when essential data lineage is explicit."""
    required = {
        "source_name": source_name,
        "source_version": source_version,
        "timestamp_timezone": timestamp_timezone,
        "duplicate_policy": duplicate_policy,
        "adjustment_policy": adjustment_policy,
        "license_or_terms_reference": license_or_terms_reference,
        "timeframe": timeframe,
    }
    for field, value in required.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be explicitly provided")
    if not isinstance(snapshot_sha256, str) or not _SHA256.fullmatch(snapshot_sha256):
        raise ValueError("snapshot_sha256 must be a lowercase 64-character SHA-256 digest")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
        raise ValueError("row_count must be a positive integer")
    normalized_instruments = tuple(sorted(set(instruments)))
    if not normalized_instruments or any(not isinstance(item, str) or not item.strip() for item in normalized_instruments):
        raise ValueError("instruments must contain non-empty explicit identifiers")
    return DatasetProvenance(
        source_name=source_name.strip(),
        source_version=source_version.strip(),
        snapshot_sha256=snapshot_sha256,
        retrieved_at=_utc(retrieved_at, field="retrieved_at"),
        timestamp_timezone=timestamp_timezone.strip(),
        duplicate_policy=duplicate_policy.strip(),
        adjustment_policy=adjustment_policy.strip(),
        license_or_terms_reference=license_or_terms_reference.strip(),
        instruments=normalized_instruments,
        timeframe=timeframe.strip(),
        row_count=row_count,
    )


def verify_dataset_provenance(
    provenance: DatasetProvenance | Mapping[str, Any],
    *,
    actual_snapshot_sha256: str,
    actual_row_count: int,
    actual_instruments: list[str] | tuple[str, ...],
    actual_timeframe: str,
) -> dict[str, Any]:
    """Compare caller-computed snapshot and parsed dataset facts to declared metadata.

    The function verifies consistency only. Callers must hash the original snapshot
    bytes themselves and derive actual facts from the exact parsed dataset.
    """
    if isinstance(provenance, DatasetProvenance):
        declared = provenance.to_dict()
    else:
        declared = dict(provenance)
    if declared.get("schema_version") != 1:
        raise ValueError("unsupported dataset provenance schema_version")
    declared_hash = declared.get("snapshot_sha256")
    if not isinstance(declared_hash, str) or not _SHA256.fullmatch(declared_hash):
        raise ValueError("declared snapshot_sha256 is invalid")
    if actual_snapshot_sha256 != declared_hash:
        raise ValueError("dataset snapshot SHA-256 does not match provenance")
    if isinstance(actual_row_count, bool) or actual_row_count != declared.get("row_count"):
        raise ValueError("parsed dataset row count does not match provenance")
    declared_instruments = declared.get("instruments")
    if not isinstance(declared_instruments, (list, tuple)) or sorted(set(actual_instruments)) != sorted(set(declared_instruments)):
        raise ValueError("parsed dataset instruments do not match provenance")
    if actual_timeframe != declared.get("timeframe"):
        raise ValueError("parsed dataset timeframe does not match provenance")
    digest = canonical_sha256(declared)
    return {
        "verified": True,
        "provenance_sha256": digest,
        "snapshot_sha256": declared_hash,
        "row_count": actual_row_count,
        "instruments": sorted(set(actual_instruments)),
        "timeframe": actual_timeframe,
        "source_authenticated": False,
        "actionable": False,
    }

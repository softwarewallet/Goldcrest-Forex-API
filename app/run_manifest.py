"""Reproducible manifests and a local registry for offline research runs.

This module is intentionally independent of prediction serving and trading. It never
loads credentials, calls a model, or places orders.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.evaluation import ForexBar

MANIFEST_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _bar_payload(bar: ForexBar) -> dict[str, Any]:
    return {
        "symbol": bar.symbol,
        "timeframe": bar.timeframe,
        "timestamp": _utc_iso(bar.timestamp),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "bid": bar.bid,
        "ask": bar.ask,
    }


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    code_commit: str
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_fingerprint: str
    config_fingerprint: str
    row_count: int
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        payload["timeframes"] = list(self.timeframes)
        return payload


def build_run_manifest(
    bars: Sequence[ForexBar],
    *,
    code_commit: str,
    model_id: str,
    model_revision: str,
    dataset_id: str,
    evaluation_config: Mapping[str, Any],
) -> RunManifest:
    """Build stable content fingerprints; created_at is metadata, not part of run_id."""
    if not bars:
        raise ValueError("bars must not be empty")
    for name, value in (
        ("code_commit", code_commit), ("model_id", model_id),
        ("model_revision", model_revision), ("dataset_id", dataset_id),
    ):
        if not value or not value.strip():
            raise ValueError(f"{name} must be explicitly provided")
    bar_payload = [_bar_payload(bar) for bar in bars]
    data_fingerprint = _sha256(_canonical_json(bar_payload))
    config_fingerprint = _sha256(_canonical_json(dict(evaluation_config)))
    identity = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "code_commit": code_commit,
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_id": dataset_id,
        "dataset_fingerprint": data_fingerprint,
        "config_fingerprint": config_fingerprint,
    }
    run_id = _sha256(_canonical_json(identity))
    return RunManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        code_commit=code_commit,
        model_id=model_id,
        model_revision=model_revision,
        dataset_id=dataset_id,
        dataset_fingerprint=data_fingerprint,
        config_fingerprint=config_fingerprint,
        row_count=len(bars),
        symbols=tuple(sorted({bar.symbol for bar in bars})),
        timeframes=tuple(sorted({bar.timeframe for bar in bars})),
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


class ExperimentRegistry:
    """Small SQLite registry for immutable run manifests and later outcome summaries."""

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('REGISTERED', 'COMPLETED', 'FAILED')),
                    outcome_summary_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection

    def register(self, manifest: RunManifest) -> bool:
        """Register a run once; return False for an identical already-registered run."""
        payload = _canonical_json(manifest.to_dict())
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT manifest_json FROM experiment_runs WHERE run_id = ?",
                (manifest.run_id,),
            ).fetchone()
            if existing:
                existing_manifest = json.loads(existing["manifest_json"])
                incoming = json.loads(payload)
                # created_at is descriptive metadata; run identity is content-addressed.
                existing_manifest.pop("created_at", None)
                incoming.pop("created_at", None)
                if _canonical_json(existing_manifest) != _canonical_json(incoming):
                    raise ValueError("run_id collision with different manifest content")
                return False
            connection.execute(
                """INSERT INTO experiment_runs
                   (run_id, manifest_json, status, created_at, updated_at)
                   VALUES (?, ?, 'REGISTERED', ?, ?)""",
                (manifest.run_id, payload, now, now),
            )
        return True

    def record_result(self, run_id: str, *, status: str,
                      outcome_summary: Mapping[str, Any] | None = None) -> None:
        if status not in {"COMPLETED", "FAILED"}:
            raise ValueError("status must be COMPLETED or FAILED")
        summary = None if outcome_summary is None else _canonical_json(dict(outcome_summary))
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE experiment_runs
                   SET status = ?, outcome_summary_json = ?, updated_at = ?
                   WHERE run_id = ?""",
                (status, summary, now, run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError("unknown run_id")

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "manifest": json.loads(row["manifest_json"]),
            "status": row["status"],
            "outcome_summary": (None if row["outcome_summary_json"] is None
                                else json.loads(row["outcome_summary_json"])),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

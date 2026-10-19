"""SQLite ledger for identity-keyed out-of-sample research predictions.

This module is offline-only. It does not serve predictions, fetch market data, or
authorize trading. Each row represents one model's prediction for one fold/partition
observation; overlapping test windows are rejected by comparison unless identities are
made unique through the explicitly recorded fold index.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from app.probability_diagnostics import compare_probability_models


def _utc_iso(value: datetime, name: str) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    symbol: str
    timeframe: str
    observation_timestamp: datetime
    fold_index: int
    partition: str
    model_name: str
    model_version: str
    probability: float
    label: int | None = None
    label_available_at: datetime | None = None
    actionable: bool = False

    def validate(self) -> None:
        for name in ("run_id", "symbol", "timeframe", "model_name", "model_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        _utc_iso(self.observation_timestamp, "observation_timestamp")
        if isinstance(self.fold_index, bool) or not isinstance(self.fold_index, int) or self.fold_index < 0:
            raise ValueError("fold_index must be a non-negative integer")
        if self.partition not in {"validation", "test"}:
            raise ValueError("partition must be validation or test")
        if isinstance(self.probability, bool) or not isinstance(self.probability, (int, float)):
            raise ValueError("probability must be numeric")
        if not math.isfinite(float(self.probability)) or not 0 <= self.probability <= 1:
            raise ValueError("probability must be finite and in [0, 1]")
        if self.actionable is not False:
            raise ValueError("research ledger records must remain actionable=False")
        if (self.label is None) != (self.label_available_at is None):
            raise ValueError("label and label_available_at must be supplied together")
        if self.label is not None:
            if isinstance(self.label, bool) or not isinstance(self.label, int) or self.label not in (0, 1):
                raise ValueError("label must be integer 0 or 1")
            available = _utc_iso(self.label_available_at, "label_available_at")
            observed = _utc_iso(self.observation_timestamp, "observation_timestamp")
            if available < observed:
                raise ValueError("label cannot be available before the observation")


@dataclass(frozen=True)
class LedgerComparison:
    sample_count: int
    candidate_name: str
    baseline_name: str
    comparison: object
    actionable: bool = False


class PredictionLedger:
    """Durable SQLite ledger with idempotent writes and conflict detection."""

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS prediction_ledger_runs (
                    run_id TEXT PRIMARY KEY,
                    manifest_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
                    report_sha256 TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL
                )
            """)
            # Older versions stored ledger lifecycle rows in experiment_runs, which
            # is also the manifest registry's table name but has an incompatible schema.
            # Migrate only when the legacy table is positively identified by its columns.
            legacy = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='experiment_runs'"
            ).fetchone()
            if legacy is not None:
                legacy_columns = {row["name"] for row in connection.execute("PRAGMA table_info(experiment_runs)")}
                required = {"run_id", "manifest_sha256", "status", "report_sha256", "error", "updated_at"}
                if required.issubset(legacy_columns):
                    connection.execute("""
                        INSERT OR IGNORE INTO prediction_ledger_runs
                            (run_id, manifest_sha256, status, report_sha256, error, updated_at)
                        SELECT run_id, manifest_sha256, status, report_sha256, error, updated_at
                        FROM experiment_runs
                    """)
                    # Preserve the old table under a legacy name so ExperimentRegistry
                    # can create its own, schema-compatible experiment_runs table later.
                    backup_name = "legacy_prediction_ledger_runs_v1"
                    suffix = 1
                    while connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (backup_name,)
                    ).fetchone() is not None:
                        suffix += 1
                        backup_name = f"legacy_prediction_ledger_runs_v{suffix}"
                    connection.execute(f'ALTER TABLE experiment_runs RENAME TO "{backup_name}"')
            connection.execute("""
                CREATE TABLE IF NOT EXISTS prediction_ledger_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    report_sha256 TEXT,
                    error TEXT,
                    UNIQUE(run_id, attempt_number)
                )
            """)
            # Backfill one historical attempt for existing lifecycle rows. This preserves
            # compatibility with databases created before attempt history existed.
            connection.execute("""
                INSERT INTO prediction_ledger_attempts
                    (run_id, attempt_number, status, started_at, finished_at, report_sha256, error)
                SELECT r.run_id, 1, r.status, r.updated_at,
                       CASE WHEN r.status='running' THEN NULL ELSE r.updated_at END,
                       r.report_sha256, r.error
                FROM prediction_ledger_runs AS r
                WHERE NOT EXISTS (
                    SELECT 1 FROM prediction_ledger_attempts AS a WHERE a.run_id=r.run_id
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS oos_predictions (
                    run_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    observation_timestamp TEXT NOT NULL,
                    fold_index INTEGER NOT NULL,
                    partition TEXT NOT NULL CHECK(partition IN ('validation','test')),
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    probability REAL NOT NULL CHECK(probability >= 0 AND probability <= 1),
                    label INTEGER CHECK(label IN (0,1)),
                    label_available_at TEXT,
                    actionable INTEGER NOT NULL CHECK(actionable = 0),
                    PRIMARY KEY (run_id, symbol, timeframe, observation_timestamp,
                                 fold_index, partition, model_name, model_version),
                    CHECK ((label IS NULL AND label_available_at IS NULL) OR
                           (label IS NOT NULL AND label_available_at IS NOT NULL))
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _payload(record: PredictionRecord) -> tuple:
        record.validate()
        return (
            record.run_id, record.symbol, record.timeframe,
            _utc_iso(record.observation_timestamp, "observation_timestamp"),
            record.fold_index, record.partition, record.model_name, record.model_version,
            float(record.probability), record.label,
            None if record.label_available_at is None else _utc_iso(record.label_available_at, "label_available_at"),
            0,
        )

    def start_run(self, run_id: str, manifest: dict) -> None:
        """Start an attempt without rewriting completed results or hiding prior attempts.

        Failed attempts may be retried with the identical run contract. A completed run
        is immutable under its run_id; use a new run_id for a deliberate rerun.
        Concurrent attempts for the same run_id are rejected.
        """
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be non-empty")
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        manifest_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT manifest_sha256, status FROM prediction_ledger_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if existing is not None and existing["manifest_sha256"] != manifest_sha:
                raise ValueError("run_id already registered with a different manifest")
            if existing is not None and existing["status"] == "completed":
                raise ValueError("run_id already completed; completed results are immutable")
            if existing is not None and existing["status"] == "running":
                raise ValueError("run_id already has a running attempt")
            connection.execute(
                """INSERT INTO prediction_ledger_runs (run_id, manifest_sha256, status, report_sha256, error, updated_at)
                   VALUES (?, ?, 'running', NULL, NULL, ?)
                   ON CONFLICT(run_id) DO UPDATE SET status='running', report_sha256=NULL, error=NULL, updated_at=excluded.updated_at""",
                (run_id, manifest_sha, now),
            )
            attempt_number = connection.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) + 1 FROM prediction_ledger_attempts WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
            connection.execute(
                """INSERT INTO prediction_ledger_attempts
                   (run_id, attempt_number, status, started_at, finished_at, report_sha256, error)
                   VALUES (?, ?, 'running', ?, NULL, NULL, NULL)""",
                (run_id, attempt_number, now),
            )

    def get_run_attempts(self, run_id: str) -> list[dict]:
        """Return append-only attempt history in execution order."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT attempt_id, run_id, attempt_number, status, started_at,
                          finished_at, report_sha256, error
                   FROM prediction_ledger_attempts WHERE run_id=? ORDER BY attempt_number""",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def finish_run(self, run_id: str, report_sha256: str) -> None:
        """Mark a run complete only after its report has been built and optionally written."""
        if not isinstance(report_sha256, str) or len(report_sha256) != 64:
            raise ValueError("report_sha256 must be a SHA-256 hex digest")
        try:
            bytes.fromhex(report_sha256)
        except ValueError as exc:
            raise ValueError("report_sha256 must be a SHA-256 hex digest") from exc
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE prediction_ledger_runs SET status='completed', report_sha256=?, error=NULL, updated_at=?
                   WHERE run_id=? AND status='running'""",
                (report_sha256, now, run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("run must be registered as running before completion")
            attempt = connection.execute(
                """UPDATE prediction_ledger_attempts
                   SET status='completed', finished_at=?, report_sha256=?, error=NULL
                   WHERE run_id=? AND status='running'""",
                (now, report_sha256, run_id),
            )
            if attempt.rowcount != 1:
                raise ValueError("exactly one running attempt is required for completion")

    def fail_run(self, run_id: str, error: str) -> None:
        """Record a bounded failure diagnostic; partial prediction rows remain auditable."""
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        message = str(error).replace("\\x00", "")[:2000]
        with self._connect() as connection:
            connection.execute(
                """UPDATE prediction_ledger_runs SET status='failed', error=?, updated_at=?
                   WHERE run_id=? AND status='running'""",
                (message, now, run_id),
            )
            connection.execute(
                """UPDATE prediction_ledger_attempts
                   SET status='failed', finished_at=?, error=?
                   WHERE run_id=? AND status='running'""",
                (now, message, run_id),
            )

    def mark_interrupted_runs(self, *, older_than: datetime) -> list[str]:
        """Mark attempts left running before a cutoff as interrupted after a process crash.

        This is an explicit recovery operation: callers choose a conservative age
        threshold. It never changes completed/failed runs and returns affected IDs.
        """
        cutoff = _utc_iso(older_than, "older_than")
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        interrupted: list[str] = []
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT run_id FROM prediction_ledger_runs
                   WHERE status='running' AND updated_at < ? ORDER BY run_id""",
                (cutoff,),
            ).fetchall()
            for row in rows:
                run_id = row["run_id"]
                cursor = connection.execute(
                    """UPDATE prediction_ledger_runs
                       SET status='failed', error=?, updated_at=?
                       WHERE run_id=? AND status='running' AND updated_at < ?""",
                    ("Interrupted attempt detected during recovery; safe retry requires the identical run contract.",
                     now, run_id, cutoff),
                )
                if cursor.rowcount != 1:
                    continue
                connection.execute(
                    """UPDATE prediction_ledger_attempts
                       SET status='failed', finished_at=?,
                           error=?
                       WHERE run_id=? AND status='running'""",
                    (now, "Interrupted attempt detected during recovery.", run_id),
                )
                interrupted.append(run_id)
        return interrupted

    def get_run_status(self, run_id: str) -> dict | None:
        """Return the durable lifecycle state for an experiment run, if registered."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT run_id, manifest_sha256, status, report_sha256, error, updated_at FROM prediction_ledger_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def verify_completed_report(self, run_id: str, report: dict) -> bool:
        """Verify a report's canonical payload hash against a completed ledger run.

        This proves consistency between the supplied report and stored fingerprint only;
        it does not validate the correctness of source data or experiment assumptions.
        """
        if not isinstance(report, dict):
            raise ValueError("report must be a mapping")
        supplied = report.get("report_sha256")
        if not isinstance(supplied, str) or len(supplied) != 64:
            return False
        try:
            bytes.fromhex(supplied)
        except ValueError:
            return False
        payload = dict(report)
        payload.pop("report_sha256", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False)
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if actual != supplied:
            return False
        status = self.get_run_status(run_id)
        return bool(status and status["status"] == "completed"
                    and status["report_sha256"] == supplied
                    and report.get("run_id") == run_id)

    def record(self, record: PredictionRecord) -> bool:
        """Insert a record; identical retries return False, conflicting retries fail."""
        values = self._payload(record)
        columns = ("run_id","symbol","timeframe","observation_timestamp","fold_index",
                   "partition","model_name","model_version","probability","label",
                   "label_available_at","actionable")
        with self._connect() as connection:
            key = values[:8]
            existing = connection.execute(
                """SELECT * FROM oos_predictions WHERE run_id=? AND symbol=? AND timeframe=?
                   AND observation_timestamp=? AND fold_index=? AND partition=?
                   AND model_name=? AND model_version=?""", key
            ).fetchone()
            if existing is not None:
                current = tuple(existing[column] for column in columns)
                if current != values:
                    raise ValueError("prediction identity already exists with conflicting content")
                return False
            connection.execute(
                f"INSERT INTO oos_predictions ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                values,
            )
        return True

    def record_many(self, records: Iterable[PredictionRecord]) -> int:
        """Atomically record a batch; any conflict rolls the whole batch back."""
        inserted = 0
        with self._connect() as connection:
            for record in records:
                values = self._payload(record)
                columns = ("run_id","symbol","timeframe","observation_timestamp","fold_index",
                           "partition","model_name","model_version","probability","label",
                           "label_available_at","actionable")
                key = values[:8]
                existing = connection.execute(
                    """SELECT * FROM oos_predictions WHERE run_id=? AND symbol=? AND timeframe=?
                       AND observation_timestamp=? AND fold_index=? AND partition=?
                       AND model_name=? AND model_version=?""", key
                ).fetchone()
                if existing is not None:
                    if tuple(existing[c] for c in columns) != values:
                        raise ValueError("prediction identity already exists with conflicting content")
                    continue
                connection.execute(
                    f"INSERT INTO oos_predictions ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                    values,
                )
                inserted += 1
        return inserted

    def compare(
        self, *, run_id: str, symbol: str, timeframe: str, fold_index: int,
        partition: str, candidate_name: str, candidate_version: str,
        baseline_name: str, baseline_version: str, block_length: int,
        bootstrap_resamples: int = 5000, confidence_level: float = 0.95,
        random_seed: int = 20261018,
    ) -> LedgerComparison:
        if partition not in {"validation", "test"}:
            raise ValueError("partition must be validation or test")
        if candidate_name == baseline_name and candidate_version == baseline_version:
            raise ValueError("candidate and baseline must differ")
        query = """SELECT observation_timestamp, model_name, model_version, probability,
                          label, label_available_at
                   FROM oos_predictions WHERE run_id=? AND symbol=? AND timeframe=?
                     AND fold_index=? AND partition=?
                     AND ((model_name=? AND model_version=?) OR
                          (model_name=? AND model_version=?))
                   ORDER BY observation_timestamp"""
        args = (run_id, symbol, timeframe, fold_index, partition, candidate_name,
                candidate_version, baseline_name, baseline_version)
        with self._connect() as connection:
            rows = connection.execute(query, args).fetchall()
        grouped = {(r["model_name"], r["model_version"]): {} for r in rows}
        for row in rows:
            identity = row["observation_timestamp"]
            model_key = (row["model_name"], row["model_version"])
            if identity in grouped[model_key]:
                raise ValueError("duplicate observation identity in model results")
            if row["label"] is None or row["label_available_at"] is None:
                continue
            grouped[model_key][identity] = (row["label"], row["probability"], row["label_available_at"])
        candidate_key = (candidate_name, candidate_version)
        baseline_key = (baseline_name, baseline_version)
        candidate = grouped.get(candidate_key, {})
        baseline = grouped.get(baseline_key, {})
        if not candidate or set(candidate) != set(baseline):
            raise ValueError("candidate and baseline must have identical matured observation identities")
        for identity in candidate:
            c, b = candidate[identity], baseline[identity]
            if c[0] != b[0] or c[2] != b[2]:
                raise ValueError("paired observations disagree on label or label maturity")
        labels = [candidate[k][0] for k in sorted(candidate)]
        candidate_probabilities = [candidate[k][1] for k in sorted(candidate)]
        baseline_probabilities = [baseline[k][1] for k in sorted(candidate)]
        result = compare_probability_models(
            labels, candidate_probabilities, baseline_probabilities,
            candidate_name=candidate_name, baseline_name=baseline_name,
            block_length=block_length, bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level, random_seed=random_seed,
        )
        return LedgerComparison(len(labels), candidate_name, baseline_name, result, False)

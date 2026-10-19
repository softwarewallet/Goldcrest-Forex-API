"""Cross-check persisted offline experiment reports against the SQLite prediction ledger.

This is an audit/read-back utility only. It never fetches market data, fits models,
changes run state, serves predictions, or authorizes trading.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.prediction_ledger import PredictionLedger


@dataclass(frozen=True)
class ResearchArtifactAudit:
    run_id: str
    report_sha256: str
    expected_prediction_rows: int
    actual_prediction_rows: int
    actionable_rows: int
    ledger_status: str
    verified: bool = True
    actionable: bool = False


def verify_research_artifacts(
    report_path: str | Path,
    *,
    ledger: PredictionLedger,
) -> ResearchArtifactAudit:
    """Verify report hash, completed-run binding, and persisted prediction row counts.

    The expected row count is derived from each model/fold's validation and test
    sample counts. Database checks are read-only. Any missing, changed, incomplete,
    or internally inconsistent artifact fails closed with ValueError.
    """
    path = Path(report_path)
    try:
        report: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"report artifact cannot be read as JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("report artifact must contain a JSON object")
    run_id = report.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("report must contain a non-empty run_id")
    if report.get("interpretation", {}).get("actionable") is not False:
        raise ValueError("report must explicitly declare actionable=false")
    if not ledger.verify_completed_report(run_id, report):
        raise ValueError("report hash or completed-run ledger binding is invalid")

    fold_results = report.get("fold_results")
    if not isinstance(fold_results, dict) or not fold_results:
        raise ValueError("report fold_results must be a non-empty object")
    expected_rows = 0
    for model_name, folds in fold_results.items():
        if not isinstance(model_name, str) or not model_name.strip() or not isinstance(folds, list) or not folds:
            raise ValueError("each report model must contain a non-empty fold list")
        seen: set[int] = set()
        for fold in folds:
            if not isinstance(fold, dict):
                raise ValueError("fold result must be an object")
            index = fold.get("fold_index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen:
                raise ValueError("fold indices must be unique non-negative integers per model")
            seen.add(index)
            if fold.get("actionable", False) is not False:
                raise ValueError("all fold results must remain actionable=false")
            for partition in ("validation", "test"):
                metrics = fold.get(partition)
                if not isinstance(metrics, dict):
                    raise ValueError(f"fold {partition} metrics must be an object")
                count = metrics.get("sample_count")
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise ValueError(f"fold {partition} sample_count must be a positive integer")
                expected_rows += count

    try:
        with sqlite3.connect(f"file:{Path(ledger.database_path).resolve()}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT COUNT(*) AS total, COALESCE(SUM(actionable), 0) AS actionable "
                "FROM oos_predictions WHERE run_id = ?", (run_id,)
            ).fetchone()
    except sqlite3.Error as exc:
        raise ValueError(f"prediction ledger cannot be audited: {exc}") from exc
    actual_rows = int(rows["total"])
    actionable_rows = int(rows["actionable"])
    if actual_rows != expected_rows:
        raise ValueError(
            f"prediction ledger row count mismatch: report expects {expected_rows}, ledger has {actual_rows}"
        )
    if actionable_rows != 0:
        raise ValueError("ledger contains actionable research predictions")
    status = ledger.get_run_status(run_id)
    if not status or status["status"] != "completed":
        raise ValueError("experiment run is not completed")
    return ResearchArtifactAudit(
        run_id=run_id,
        report_sha256=report["report_sha256"],
        expected_prediction_rows=expected_rows,
        actual_prediction_rows=actual_rows,
        actionable_rows=actionable_rows,
        ledger_status=status["status"],
    )

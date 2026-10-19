"""Cross-check persisted offline experiment reports against the SQLite prediction ledger.

This is a read-only audit utility. It never fetches market data, fits models, changes
run state, serves predictions, or authorizes trading.
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
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
    """Verify report hash, completed-run binding, and per-model/fold/partition counts.

    Reconciliation is performed at model name, fitted model version, fold index, and
    validation/test partition granularity. This catches misplaced or missing rows that
    a single aggregate row-count comparison would miss. It is a count-level reconciliation;
    it does not claim source-data authenticity or prove model validity.
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
    interpretation = report.get("interpretation")
    if not isinstance(interpretation, dict) or interpretation.get("actionable") is not False:
        raise ValueError("report must explicitly declare actionable=false")
    if not ledger.verify_completed_report(run_id, report):
        raise ValueError("report hash or completed-run ledger binding is invalid")

    fold_results = report.get("fold_results")
    if not isinstance(fold_results, dict) or not fold_results:
        raise ValueError("report fold_results must be a non-empty object")
    expected: dict[tuple[str, str, int, str], int] = {}
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
            version = fold.get("model_version")
            if not isinstance(version, str) or not version.strip():
                raise ValueError(f"fold {index} for {model_name!r} must declare model_version")
            for partition in ("validation", "test"):
                metrics = fold.get(partition)
                if not isinstance(metrics, dict):
                    raise ValueError(f"fold {partition} metrics must be an object")
                count = metrics.get("sample_count")
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise ValueError(f"fold {partition} sample_count must be a positive integer")
                expected[(model_name, version, index, partition)] = count

    try:
        with sqlite3.connect(f"file:{Path(ledger.database_path).resolve()}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """SELECT model_name, model_version, fold_index, partition,
                          COUNT(*) AS row_count, COALESCE(SUM(actionable), 0) AS actionable_count
                   FROM oos_predictions WHERE run_id = ?
                   GROUP BY model_name, model_version, fold_index, partition""",
                (run_id,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise ValueError(f"prediction ledger cannot be audited: {exc}") from exc

    actual: dict[tuple[str, str, int, str], int] = {}
    actionable_rows = 0
    for row in rows:
        key = (row["model_name"], row["model_version"], int(row["fold_index"]), row["partition"])
        actual[key] = int(row["row_count"])
        actionable_rows += int(row["actionable_count"])
    expected_rows = sum(expected.values())
    actual_rows = sum(actual.values())
    if actual != expected:
        missing = {str(key): count for key, count in expected.items() if actual.get(key) != count}
        unexpected = {str(key): count for key, count in actual.items() if expected.get(key) != count}
        raise ValueError(
            "prediction ledger group mismatch; "
            f"missing_or_wrong_counts={missing}; unexpected_groups={unexpected}"
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

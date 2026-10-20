"""Command-line entry points for offline GoldcrestAI research operations.

These commands inspect persisted research artifacts only. They never fetch market
data, train models, serve predictions, or authorize trading.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sqlite3
import sys
from pathlib import Path

from app.prediction_ledger import PredictionLedger
from app.research_artifact_audit import verify_research_artifacts


def _emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goldcrest-research",
        description="Offline research artifact inspection; no trading actions are available.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser(
        "audit",
        help="verify a completed report against its SQLite prediction ledger",
    )
    audit.add_argument("--report", required=True, type=Path, help="path to fingerprinted JSON report")
    audit.add_argument("--ledger", required=True, type=Path, help="path to the research SQLite ledger")

    status = subparsers.add_parser(
        "status",
        help="inspect durable run state and attempt history without changing the ledger",
    )
    status.add_argument("--run-id", required=True, help="registered experiment run ID")
    status.add_argument("--ledger", required=True, type=Path, help="path to the research SQLite ledger")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        if not args.report.is_file():
            print(f"ERROR: report file does not exist: {args.report}", file=sys.stderr)
            return 2
        if not args.ledger.is_file():
            print(f"ERROR: ledger file does not exist: {args.ledger}", file=sys.stderr)
            return 2
        try:
            ledger = PredictionLedger(args.ledger)
            result = verify_research_artifacts(args.report, ledger=ledger)
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            _emit({"verified": False, "actionable": False, "error": str(exc)})
            return 1
        _emit(asdict(result))
        return 0 if result.verified is True and result.actionable is False else 1

    if args.command == "status":
        if not args.ledger.is_file():
            print(f"ERROR: ledger file does not exist: {args.ledger}", file=sys.stderr)
            return 2
        try:
            ledger = PredictionLedger(args.ledger)
            run = ledger.get_run_status(args.run_id)
            if run is None:
                _emit({"found": False, "run_id": args.run_id, "attempts": []})
                return 2
            attempts = ledger.get_run_attempts(args.run_id)
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            _emit({"found": False, "run_id": args.run_id, "error": str(exc)})
            return 1
        _emit({
            "found": True,
            "run": run,
            "attempts": attempts,
            "actionable": False,
            "interpretation": "Read-only lifecycle inspection; no run state was changed.",
        })
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

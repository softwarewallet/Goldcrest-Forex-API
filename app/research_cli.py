"""Command-line entry points for offline GoldcrestAI research operations.

This CLI audits persisted research artifacts only. It never fetches market data,
trains models, serves predictions, or authorizes trading.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

from app.prediction_ledger import PredictionLedger
from app.research_artifact_audit import verify_research_artifacts


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
    audit.add_argument("--ledger", required=True, type=Path, help="path to the experiment SQLite ledger")
    audit.add_argument(
        "--json",
        action="store_true",
        help="emit the audit result as machine-readable JSON (the default is also JSON)",
    )
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
        except (OSError, ValueError, RuntimeError) as exc:
            print(json.dumps({"verified": False, "actionable": False, "error": str(exc)},
                             sort_keys=True, ensure_ascii=False), file=sys.stderr)
            return 1
        print(json.dumps(asdict(result), sort_keys=True, ensure_ascii=False, indent=2))
        return 0 if result.verified is True and result.actionable is False else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

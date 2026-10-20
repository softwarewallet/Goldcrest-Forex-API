"""Audit historical OHLC quality and candle cadence before model research.

Cadence findings are descriptive, not proof of missing market data: session breaks,
weekends, holidays, provider conventions, and daylight-saving changes can create
legitimate gaps. This module never sorts, fills, drops, or synthesizes candles.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Sequence

REQUIRED_COLUMNS = {"symbol", "timeframe", "timestamp", "open", "high", "low", "close"}
TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


def _parse_timestamp(value: str, line: int) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"line {line}: timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"line {line}: timestamp must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def audit_historical_ohlc(input_csv: str | Path) -> dict[str, object]:
    """Return a reproducible quality/cadence audit without modifying source data."""
    source = Path(input_csv)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no data rows")

    parsed: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    for line, row in enumerate(rows, start=2):
        try:
            symbol = row["symbol"].strip().upper()
            timeframe = row["timeframe"].strip().upper()
            timestamp = _parse_timestamp(row["timestamp"], line)
            prices = {name: float(row[name]) for name in ("open", "high", "low", "close")}
            if not symbol or timeframe not in TIMEFRAME_MINUTES:
                raise ValueError("symbol must be non-empty and timeframe supported")
            if any(not math.isfinite(value) or value <= 0 for value in prices.values()):
                raise ValueError("OHLC values must be finite and positive")
            if prices["high"] < max(prices["open"], prices["close"], prices["low"]):
                raise ValueError("high is below another OHLC value")
            if prices["low"] > min(prices["open"], prices["close"], prices["high"]):
                raise ValueError("low is above another OHLC value")
            parsed.append({"symbol": symbol, "timeframe": timeframe, "timestamp": timestamp, **prices})
        except (KeyError, TypeError, ValueError) as exc:
            invalid_rows.append({"line": line, "reason": str(exc)})

    symbols = sorted({str(row["symbol"]) for row in parsed})
    timeframes = sorted({str(row["timeframe"]) for row in parsed})
    if len(symbols) != 1 or len(timeframes) != 1:
        raise ValueError("quality audit requires one symbol/timeframe per CSV")
    times = [row["timestamp"] for row in parsed]
    duplicates = len(times) - len(set(times))
    non_monotonic = sum(1 for previous, current in zip(times, times[1:]) if current < previous)
    expected_minutes = TIMEFRAME_MINUTES[timeframes[0]]
    deltas = [
        (current - previous).total_seconds() / 60.0
        for previous, current in zip(times, times[1:])
    ]
    positive_deltas = [delta for delta in deltas if delta > 0]
    long_intervals = [delta for delta in positive_deltas if delta > expected_minutes]
    irregular_intervals = [delta for delta in positive_deltas if delta != expected_minutes]
    estimated_missing = sum(
        max(0, int(delta // expected_minutes) - 1)
        for delta in long_intervals
        if math.isclose(delta / expected_minutes, round(delta / expected_minutes), abs_tol=1e-9)
    )
    one_bar_returns = []
    for previous, current in zip(parsed, parsed[1:]):
        if previous["timestamp"] < current["timestamp"] and previous["close"] > 0:
            one_bar_returns.append(float(current["close"]) / float(previous["close"]) - 1.0)
    abs_returns = sorted(abs(value) for value in one_bar_returns)
    report = {
        "audit_version": 1,
        "input_path": str(source),
        "input_sha256": digest,
        "row_count": len(rows),
        "parsed_row_count": len(parsed),
        "invalid_row_count": len(invalid_rows),
        "invalid_rows_sample": invalid_rows[:20],
        "symbol": symbols[0],
        "timeframe": timeframes[0],
        "expected_interval_minutes": expected_minutes,
        "duplicate_timestamp_count": duplicates,
        "non_monotonic_interval_count": non_monotonic,
        "observed_interval_count": len(deltas),
        "exact_expected_interval_count": sum(delta == expected_minutes for delta in deltas),
        "long_interval_count": len(long_intervals),
        "short_or_irregular_interval_count": sum(delta < expected_minutes or delta != expected_minutes for delta in deltas),
        "median_observed_interval_minutes": median(positive_deltas) if positive_deltas else None,
        "maximum_observed_interval_minutes": max(positive_deltas) if positive_deltas else None,
        "estimated_missing_bars_on_exact_interval_multiples": estimated_missing,
        "absolute_close_return_p95": abs_returns[min(len(abs_returns) - 1, math.ceil(len(abs_returns) * 0.95) - 1)] if abs_returns else None,
        "warnings": [
            "Cadence gaps are candidates for review, not confirmed missing candles; market sessions, weekends, holidays, DST and provider conventions may explain them.",
            "Estimated missing bars are counted only for exact multiples of the nominal timeframe and are not a data-repair instruction.",
            "No records were sorted, dropped, imputed, filled, or synthesized.",
        ],
        "structurally_valid": not invalid_rows and duplicates == 0 and non_monotonic == 0,
        "actionable": False,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit historical OHLC integrity and candle cadence; never repairs data.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--report", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_historical_ohlc(args.input_csv)
    destination = Path(args.report)
    if Path(args.input_csv).resolve() == destination.resolve():
        raise ValueError("input and report paths must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["structurally_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

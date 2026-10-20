"""Prepare causal, label-free features from a real historical OHLC CSV.

This utility validates source bars and computes features using only the current and
past rows. It deliberately does not create labels, fetch market data, or evaluate a
model; label horizon and neutral-outcome policy must be specified separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Sequence

from app.evaluation import ForexBar, validate_bars

REQUIRED_COLUMNS = {"symbol", "timeframe", "timestamp", "open", "high", "low", "close"}
FEATURE_COLUMNS = (
    "return_1", "return_3", "return_5", "range_pct", "body_pct",
    "close_location", "volatility_14",
)


def _timestamp(value: str, line: int) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"line {line}: timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"line {line}: timestamp must include a timezone offset")
    return parsed


def prepare_features(input_csv: str | Path, output_csv: str | Path) -> dict[str, object]:
    """Validate a single instrument/timeframe and write causal features after warm-up."""
    source = Path(input_csv)
    destination = Path(output_csv)
    if source.resolve() == destination.resolve():
        raise ValueError("input and output paths must differ")
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) < 16:
        raise ValueError("at least 16 OHLC rows are required for the 14-return volatility warm-up")

    bars: list[ForexBar] = []
    for line, row in enumerate(rows, start=2):
        try:
            bars.append(ForexBar(
                symbol=row["symbol"].strip().upper(),
                timeframe=row["timeframe"].strip().upper(),
                timestamp=_timestamp(row["timestamp"], line),
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid source row at line {line}: {exc}") from exc

    quality = validate_bars(bars)
    if quality.invalid_rows or quality.duplicate_timestamps or quality.non_monotonic_timestamps:
        raise ValueError(
            "source OHLC validation failed: "
            f"invalid_rows={quality.invalid_rows}, duplicates={quality.duplicate_timestamps}, "
            f"non_monotonic={quality.non_monotonic_timestamps}"
        )
    if len(quality.symbols) != 1 or len(quality.timeframes) != 1:
        raise ValueError("one feature-preparation run must contain exactly one symbol and timeframe")
    if [b.timestamp.astimezone().timestamp() for b in bars] != sorted(
        b.timestamp.astimezone().timestamp() for b in bars
    ):
        raise ValueError("source rows must already be chronological")

    closes = [b.close for b in bars]
    returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    output_rows: list[dict[str, str | float]] = []
    for i, bar in enumerate(bars):
        # 14 completed one-bar returns are required; no future rows are consulted.
        if i < 15:
            continue
        one_bar_returns = returns[i - 14:i]
        candle_range = bar.high - bar.low
        features = {
            "return_1": returns[i - 1],
            "return_3": closes[i] / closes[i - 3] - 1.0,
            "return_5": closes[i] / closes[i - 5] - 1.0,
            "range_pct": candle_range / bar.close,
            "body_pct": (bar.close - bar.open) / bar.open,
            "close_location": (bar.close - bar.low) / candle_range if candle_range else 0.5,
            "volatility_14": statistics.pstdev(one_bar_returns),
        }
        if not all(math.isfinite(value) for value in features.values()):
            raise ValueError(f"non-finite feature calculated at row {i + 2}")
        output_rows.append({
            "symbol": bar.symbol,
            "timeframe": bar.timeframe,
            "timestamp": bar.timestamp.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            **features,
        })

    if not output_rows:
        raise ValueError("no rows remained after feature warm-up")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["symbol", "timeframe", "timestamp", "open", "high", "low", "close", *FEATURE_COLUMNS]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "input_path": str(source),
        "output_path": str(destination),
        "input_sha256": digest,
        "input_rows": len(bars),
        "output_rows": len(output_rows),
        "warmup_rows_excluded": len(bars) - len(output_rows),
        "symbol": bars[0].symbol,
        "timeframe": bars[0].timeframe,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_policy": "current-and-past OHLC only; no future data used",
        "label_policy": "not generated; label horizon and neutral/tie policy remain explicit user inputs",
        "actionable": False,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate raw historical OHLC CSV and create causal features; does not create labels."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report", required=True, help="JSON audit report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = prepare_features(args.input_csv, args.output_csv)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

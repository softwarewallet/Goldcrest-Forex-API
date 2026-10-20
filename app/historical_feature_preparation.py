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
WARMUP_ROWS = 34
FEATURE_COLUMNS = (
    "return_1", "return_3", "return_5", "return_10", "range_pct", "body_pct",
    "close_location", "volatility_14", "ema_gap_12_26", "rsi_14", "atr_14_pct",
    "macd_hist_pct",
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



def _ema(values: list[float], period: int) -> list[float]:
    """Recursive EMA seeded with the first observation; causal by construction."""
    alpha = 2.0 / (period + 1.0)
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * float(value) + (1.0 - alpha) * result[-1])
    return result


def _wilder_average(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return result
    result[period] = sum(values[1:period + 1]) / period
    for index in range(period + 1, len(values)):
        prior = result[index - 1]
        assert prior is not None
        result[index] = (prior * (period - 1) + values[index]) / period
    return result


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
    if len(rows) <= WARMUP_ROWS:
        raise ValueError(f"at least {WARMUP_ROWS + 1} OHLC rows are required for the documented indicator warm-up")

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

    closes = [bar.close for bar in bars]
    returns = [0.0] + [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    gains = [0.0] + [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [0.0] + [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    true_ranges = [bars[0].high - bars[0].low] + [
        max(bars[i].high - bars[i].low, abs(bars[i].high - closes[i - 1]), abs(bars[i].low - closes[i - 1]))
        for i in range(1, len(bars))
    ]
    avg_gain = _wilder_average(gains, 14)
    avg_loss = _wilder_average(losses, 14)
    atr = _wilder_average(true_ranges, 14)
    ema12, ema26 = _ema(closes, 12), _ema(closes, 26)
    macd_line = [fast - slow for fast, slow in zip(ema12, ema26)]
    macd_signal = _ema(macd_line, 9)
    output_rows: list[dict[str, str | float]] = []
    for i, bar in enumerate(bars):
        if i < WARMUP_ROWS:
            continue
        candle_range = bar.high - bar.low
        up, down = avg_gain[i], avg_loss[i]
        atr_value = atr[i]
        assert up is not None and down is not None and atr_value is not None
        if down == 0:
            rsi = 50.0 if up == 0 else 100.0
        else:
            rsi = 100.0 - 100.0 / (1.0 + up / down)
        features = {
            "return_1": returns[i],
            "return_3": closes[i] / closes[i - 3] - 1.0,
            "return_5": closes[i] / closes[i - 5] - 1.0,
            "return_10": closes[i] / closes[i - 10] - 1.0,
            "range_pct": candle_range / bar.close,
            "body_pct": (bar.close - bar.open) / bar.open,
            "close_location": (bar.close - bar.low) / candle_range if candle_range else 0.5,
            "volatility_14": statistics.pstdev(returns[i - 13:i + 1]),
            "ema_gap_12_26": (ema12[i] - ema26[i]) / bar.close,
            "rsi_14": rsi,
            "atr_14_pct": atr_value / bar.close,
            "macd_hist_pct": (macd_line[i] - macd_signal[i]) / bar.close,
        }
        if not all(math.isfinite(value) for value in features.values()):
            raise ValueError(f"non-finite feature calculated at row {i + 2}")
        if not 0 <= features["rsi_14"] <= 100:
            raise ValueError(f"RSI out of bounds at row {i + 2}")
        output_rows.append({
            "symbol": bar.symbol, "timeframe": bar.timeframe,
            "timestamp": bar.timestamp.isoformat(),
            "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close,
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
        "warmup_policy": f"exclude first {WARMUP_ROWS} rows for indicator initialization",
        "symbol": bars[0].symbol,
        "timeframe": bars[0].timeframe,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_definitions": {
            "return_1/3/5/10": "close-to-close returns over the named number of bars",
            "range_pct": "(high-low)/close",
            "body_pct": "(close-open)/open",
            "close_location": "(close-low)/(high-low), or 0.5 for zero-range candles",
            "volatility_14": "population standard deviation of the latest 14 close-to-close returns",
            "ema_gap_12_26": "(recursive EMA12 - EMA26)/close; each EMA seeded with first close",
            "rsi_14": "Wilder-smoothed RSI from close changes; initial averages use 14 changes",
            "atr_14_pct": "Wilder-smoothed true range over 14 bars divided by close",
            "macd_hist_pct": "(EMA12-EMA26 minus EMA9 signal EMA)/close",
        },
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

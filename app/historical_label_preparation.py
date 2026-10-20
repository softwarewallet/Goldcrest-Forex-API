"""Add explicit forward-return binary labels to causal historical feature rows.

This is an offline data-preparation tool. It does not fetch data, train models, or trade.
The target compares the current close with the close exactly N bars later. The caller
must explicitly choose the horizon, neutral band, and neutral-label policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Sequence

REQUIRED_COLUMNS = {"symbol", "timeframe", "timestamp", "open", "high", "low", "close"}


def _parse_timestamp(value: str, line: int) -> datetime:
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


def label_historical_features(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    horizon_bars: int,
    neutral_band_bps: float,
    neutral_policy: str,
) -> dict[str, object]:
    """Create matured labels from future close returns without altering feature values."""
    source, destination = Path(input_csv), Path(output_csv)
    if source.resolve() == destination.resolve():
        raise ValueError("input and output paths must differ")
    if isinstance(horizon_bars, bool) or horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    if not math.isfinite(neutral_band_bps) or neutral_band_bps < 0:
        raise ValueError("neutral_band_bps must be finite and >= 0")
    if neutral_policy not in {"up", "down", "error"}:
        raise ValueError("neutral_policy must be one of: up, down, error")

    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - set(fields)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        if "label" in fields or "label_available_at" in fields:
            raise ValueError("input already contains labels; refusing to overwrite existing target fields")
        rows = list(reader)
    if len(rows) <= horizon_bars:
        raise ValueError("row count must exceed horizon_bars")

    symbol_timeframe: tuple[str, str] | None = None
    timestamps: list[datetime] = []
    closes: list[float] = []
    for line, row in enumerate(rows, start=2):
        try:
            symbol = row["symbol"].strip().upper()
            timeframe = row["timeframe"].strip().upper()
            current = (symbol, timeframe)
            if symbol_timeframe is None:
                symbol_timeframe = current
            elif current != symbol_timeframe:
                raise ValueError("one labeling run must contain exactly one symbol/timeframe")
            timestamp = _parse_timestamp(row["timestamp"], line)
            close = float(row["close"])
            if not math.isfinite(close) or close <= 0:
                raise ValueError("close must be finite and positive")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid row at line {line}: {exc}") from exc
        timestamps.append(timestamp)
        closes.append(close)

    utc_timestamps = [ts.timestamp() for ts in timestamps]
    if utc_timestamps != sorted(utc_timestamps) or len(set(utc_timestamps)) != len(utc_timestamps):
        raise ValueError("timestamps must be strictly chronological and unique")

    output_fields = [*fields, "label", "label_available_at", "forward_return_bps"]
    output_rows: list[dict[str, str]] = []
    neutral_count = up_count = down_count = 0
    for index in range(len(rows) - horizon_bars):
        future_index = index + horizon_bars
        forward_bps = (closes[future_index] / closes[index] - 1.0) * 10_000.0
        if not math.isfinite(forward_bps):
            raise ValueError(f"non-finite forward return at input row {index + 2}")
        if abs(forward_bps) <= neutral_band_bps:
            neutral_count += 1
            if neutral_policy == "error":
                raise ValueError(
                    f"neutral outcome at input row {index + 2}: "
                    f"|forward_return_bps| <= {neutral_band_bps}"
                )
            label = 1 if neutral_policy == "up" else 0
        else:
            label = 1 if forward_bps > 0 else 0
        if label == 1:
            up_count += 1
        else:
            down_count += 1
        output_row = {key: value for key, value in rows[index].items()}
        output_row["label"] = str(label)
        output_row["label_available_at"] = timestamps[future_index].isoformat()
        output_row["forward_return_bps"] = f"{forward_bps:.10f}"
        output_rows.append(output_row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    target_rule = (
        f"label=1 when close[t+{horizon_bars}] / close[t] - 1 is positive "
        f"and above neutral band {neutral_band_bps} bps; label=0 when negative "
        "and below the negative band; outcomes within or equal to the band use "
        f"neutral_policy={neutral_policy}"
    )
    return {
        "input_path": str(source),
        "output_path": str(destination),
        "input_sha256": digest,
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "tail_rows_without_matured_outcome": horizon_bars,
        "symbol": symbol_timeframe[0] if symbol_timeframe else None,
        "timeframe": symbol_timeframe[1] if symbol_timeframe else None,
        "horizon_bars": horizon_bars,
        "neutral_band_bps": neutral_band_bps,
        "neutral_policy": neutral_policy,
        "neutral_outcomes": neutral_count,
        "label_1_count": up_count,
        "label_0_count": down_count,
        "class_1_means": "UP",
        "target_rule": target_rule,
        "label_availability_rule": "timestamp of close[t+horizon_bars]; final horizon rows have no matured target and are excluded",
        "actionable": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create explicit binary forward-return labels for historical features; offline only."
    )
    parser.add_argument("--input-csv", required=True, help="Causal-feature CSV with OHLC columns and no existing labels")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--horizon-bars", required=True, type=int)
    parser.add_argument("--neutral-band-bps", required=True, type=float)
    parser.add_argument("--neutral-policy", required=True, choices=("up", "down", "error"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = label_historical_features(
        args.input_csv, args.output_csv, horizon_bars=args.horizon_bars,
        neutral_band_bps=args.neutral_band_bps, neutral_policy=args.neutral_policy,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

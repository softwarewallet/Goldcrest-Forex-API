"""Build a point-in-time feature/label CSV for existing offline research workflows.

Input is an immutable, already-downloaded OHLC CSV. This module never makes network
requests, authenticates to cTrader, or connects to a broker. Labels are explicit
forward-close direction labels and include the timestamp at which each label matures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from app.causal_ta_features import build_causal_ohlc_features
from app.evaluation import ForexBar

_REQUIRED_OHLC = {"timestamp", "open", "high", "low", "close"}
_LABEL_MEANINGS = {"UP", "DOWN"}


def _parse_timestamp(value: str, *, field: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def prepare_research_dataset(
    input_csv: str | Path,
    output_csv: str | Path,
    manifest_path: str | Path,
    *,
    symbol: str,
    timeframe: str,
    horizon_bars: int,
    class_1_means: str,
    data_source: str,
    source_version: str,
    retrieved_at_utc: str,
    terms_reference: str,
) -> dict[str, Any]:
    """Create a causal, explicitly labelled dataset accepted by the walk-forward CLI."""
    if not symbol.strip() or symbol.strip() != symbol.strip().upper():
        raise ValueError("symbol must be non-empty and uppercase")
    symbol = symbol.strip().upper()
    timeframe = timeframe.strip().upper()
    if timeframe not in {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}:
        raise ValueError("unsupported timeframe")
    if isinstance(horizon_bars, bool) or not isinstance(horizon_bars, int) or horizon_bars < 1:
        raise ValueError("horizon_bars must be a positive integer")
    class_1_means = class_1_means.strip().upper()
    if class_1_means not in _LABEL_MEANINGS:
        raise ValueError("class_1_means must be UP or DOWN")
    for name, value in (
        ("data_source", data_source), ("source_version", source_version),
        ("terms_reference", terms_reference),
    ):
        if not value.strip():
            raise ValueError(f"{name} must be explicitly provided")
    retrieved_at = _parse_timestamp(retrieved_at_utc, field="retrieved_at_utc")
    source_path = Path(input_csv)
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = _REQUIRED_OHLC - columns
        if missing:
            raise ValueError(f"input CSV is missing columns: {sorted(missing)}")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError("input CSV contains no OHLC rows")

    bars: list[ForexBar] = []
    for line, row in enumerate(source_rows, start=2):
        try:
            ts = _parse_timestamp(row["timestamp"], field=f"line {line} timestamp")
            values = {name: float(row[name]) for name in ("open", "high", "low", "close")}
            if any(not math.isfinite(value) for value in values.values()):
                raise ValueError("OHLC values must be finite")
            if (
                any(value <= 0 for value in values.values())
                or values["low"] > min(values["open"], values["close"])
                or values["high"] < max(values["open"], values["close"])
                or values["high"] < values["low"]
            ):
                raise ValueError("invalid OHLC relationships")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid input at line {line}: {exc}") from exc

        if "symbol" in row and row["symbol"].strip().upper() != symbol:
            raise ValueError(f"line {line} symbol does not match requested symbol {symbol}")
        if "timeframe" in row and row["timeframe"].strip().upper() != timeframe:
            raise ValueError(f"line {line} timeframe does not match requested timeframe")
        bars.append(ForexBar(
            symbol=symbol, timeframe=timeframe, timestamp=ts,
            open=values["open"], high=values["high"], low=values["low"], close=values["close"],
        ))

    timestamps = [bar.timestamp for bar in bars]
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        raise ValueError("input timestamps must already be strictly chronological and unique")

    feature_batch = build_causal_ohlc_features(bars)
    output_path = Path(output_csv)
    manifest_output = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    feature_names = list(feature_batch.feature_names)
    output_fields = [
        "symbol", "timeframe", "timestamp", "open", "high", "low", "close",
        "label", "label_available_at", *feature_names,
    ]
    emitted: list[dict[str, Any]] = []
    skipped_warmup = 0
    for index in range(0, len(bars) - horizon_bars):
        feature_row = feature_batch.rows[index]
        if any(feature_row[name] is None for name in feature_names):
            skipped_warmup += 1
            continue
        future_close = bars[index + horizon_bars].close
        current_close = bars[index].close
        is_up = future_close > current_close
        is_down = future_close < current_close
        label = int(is_up if class_1_means == "UP" else is_down)
        row: dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": timestamps[index].isoformat().replace("+00:00", "Z"),
            "open": bars[index].open,
            "high": bars[index].high,
            "low": bars[index].low,
            "close": bars[index].close,
            "label": label,
            "label_available_at": timestamps[index + horizon_bars].isoformat().replace("+00:00", "Z"),
        }
        row.update({name: feature_row[name] for name in feature_names})
        emitted.append(row)

    if not emitted:
        raise ValueError("not enough rows to produce mature labels after feature warm-up")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(emitted)
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "scope": "OFFLINE_RESEARCH_DATASET_PREPARATION",
        "actionable": False,
        "symbol": symbol,
        "timeframe": timeframe,
        "data_source": data_source.strip(),
        "source_version": source_version.strip(),
        "retrieved_at_utc": retrieved_at.isoformat().replace("+00:00", "Z"),
        "terms_reference": terms_reference.strip(),
        "input_snapshot_sha256": source_sha256,
        "output_dataset_sha256": output_sha256,
        "input_rows": len(bars),
        "output_rows": len(emitted),
        "feature_names": feature_names,
        "feature_backend": feature_batch.indicator_backend,
        "feature_timestamp_rule": "Each feature at t uses only OHLC observations timestamped <= t.",
        "label_definition": f"Binary forward-close direction over {horizon_bars} bars; class 1 means {class_1_means}; equality maps to class 0.",
        "label_horizon_bars": horizon_bars,
        "label_availability_rule": "label_available_at is the timestamp of the future close used to define the label.",
        "warmup_rows_skipped": skipped_warmup,
        "trailing_unlabelled_rows_excluded": min(horizon_bars, len(bars)),
        "automatic_sorting": False,
        "automatic_imputation": False,
        "automatic_model_selection": False,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare an offline Goldcrest research dataset from a canonical OHLC CSV.")
    parser.add_argument("--input-csv", required=True, help="Canonical CSV with timestamp,open,high,low,close; UTC or offset timestamps")
    parser.add_argument("--output-csv", required=True, help="Dataset CSV consumed by app.historical_tabular_research")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--symbol", required=True, help="Uppercase instrument identifier")
    parser.add_argument("--timeframe", required=True, choices=("M1", "M5", "M15", "M30", "H1", "H4", "D1"))
    parser.add_argument("--horizon-bars", type=int, required=True)
    parser.add_argument("--class-1-means", choices=("UP", "DOWN"), required=True)
    parser.add_argument("--data-source", required=True, help="For example cTrader Open API or Yahoo Finance via yfinance")
    parser.add_argument("--source-version", required=True, help="Provider/API or export version")
    parser.add_argument("--retrieved-at-utc", required=True, help="ISO-8601 UTC retrieval time ending in Z")
    parser.add_argument("--terms-reference", required=True, help="Actual applicable provider terms/license reference; do not guess")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = prepare_research_dataset(
        args.input_csv, args.output_csv, args.manifest,
        symbol=args.symbol, timeframe=args.timeframe,
        horizon_bars=args.horizon_bars, class_1_means=args.class_1_means,
        data_source=args.data_source, source_version=args.source_version,
        retrieved_at_utc=args.retrieved_at_utc, terms_reference=args.terms_reference,
    )
    print(json.dumps({
        "output_dataset_sha256": manifest["output_dataset_sha256"],
        "input_rows": manifest["input_rows"],
        "output_rows": manifest["output_rows"],
        "feature_count": len(manifest["feature_names"]),
        "feature_backend": manifest["feature_backend"],
        "actionable": False,
        "manifest_path": args.manifest,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

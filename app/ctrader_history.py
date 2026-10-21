"""Normalize cTrader Open API trendbar responses into canonical research OHLC CSV.

This adapter is deliberately transport-free: the caller supplies trendbars returned by
an already-authenticated cTrader Open API client. It does not open sockets, authenticate,
request data, or access order-related API methods.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _field(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def _timestamp(record: Any) -> datetime:
    # cTrader ProtoOATrendbar uses UTC minutes since Unix epoch.
    minute_value = _field(record, "utcTimestampInMinutes")
    if minute_value is not None:
        return datetime.fromtimestamp(float(minute_value) * 60.0, tz=timezone.utc)
    value = _field(record, "timestamp_utc")
    if value is None:
        value = _field(record, "timestamp")
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("cTrader timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Generic SDK adapters may provide epoch seconds under timestamp.
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("cTrader timestamps must be timezone-aware")
        return parsed.astimezone(timezone.utc)
    raise ValueError("trendbar must provide utcTimestampInMinutes or an aware timestamp")


def normalize_ctrader_trendbars(
    trendbars: Iterable[Any],
    *,
    symbol: str,
    timeframe: str,
    output_csv: str | Path,
    price_scale: float = 100000.0,
) -> dict[str, Any]:
    """Convert cTrader trendbar protobuf objects/mappings to canonical UTC OHLC CSV.

    For ProtoOATrendbar, low is the base price and deltaOpen/deltaHigh/deltaClose
    are offsets from that base. price_scale is explicit because instruments/SDK wrappers
    may expose different representations; verify it against the symbol's price precision.
    """
    instrument = symbol.strip().upper()
    period = timeframe.strip().upper()
    if not instrument:
        raise ValueError("symbol must not be empty")
    if period not in {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}:
        raise ValueError("unsupported timeframe")
    if isinstance(price_scale, bool) or not math.isfinite(price_scale) or price_scale <= 0:
        raise ValueError("price_scale must be finite and positive")

    normalized: list[dict[str, Any]] = []
    for index, bar in enumerate(trendbars):
        try:
            ts = _timestamp(bar)
            raw_low = _field(bar, "low")
            if raw_low is None:
                # Generic normalized records may already have absolute OHLC.
                values = {name: float(_field(bar, name)) for name in ("open", "high", "low", "close")}
            else:
                low_base = float(raw_low) / price_scale
                open_delta = float(_field(bar, "deltaOpen")) / price_scale
                high_delta = float(_field(bar, "deltaHigh")) / price_scale
                close_delta = float(_field(bar, "deltaClose")) / price_scale
                values = {
                    "open": low_base + open_delta,
                    "high": low_base + high_delta,
                    "low": low_base,
                    "close": low_base + close_delta,
                }
            if any(not math.isfinite(value) or value <= 0 for value in values.values()):
                raise ValueError("prices must be finite and positive")
            if values["low"] > min(values["open"], values["close"]) or values["high"] < max(values["open"], values["close"]) or values["high"] < values["low"]:
                raise ValueError("invalid OHLC relationships")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid cTrader trendbar at index {index}: {exc}") from exc
        normalized.append({
            "symbol": instrument,
            "timeframe": period,
            "timestamp": ts.isoformat().replace("+00:00", "Z"),
            **values,
        })

    if not normalized:
        raise ValueError("cTrader trendbar response is empty")
    timestamps = [row["timestamp"] for row in normalized]
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        raise ValueError("cTrader trendbars must already be strictly chronological and unique")

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["symbol", "timeframe", "timestamp", "open", "high", "low", "close"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(normalized)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "scope": "CTRADER_OPEN_API_HISTORICAL_OHLC_NORMALIZATION",
        "actionable": False,
        "data_source": "cTrader Open API",
        "symbol": instrument,
        "timeframe": period,
        "rows": len(normalized),
        "first_timestamp_utc": timestamps[0],
        "last_timestamp_utc": timestamps[-1],
        "price_scale": price_scale,
        "price_encoding": "ProtoOATrendbar low + deltaOpen/deltaHigh/deltaClose; verify scale against instrument precision",
        "snapshot_sha256": digest,
        "transport_or_auth_performed": False,
        "order_api_used": False,
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest

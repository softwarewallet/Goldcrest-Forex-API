"""Explicit, provenance-preserving Yahoo Finance history download for offline research.

This adapter only downloads and exports OHLCV. It never creates labels, produces trading
signals, or connects to a broker. Network access occurs only when the caller explicitly
invokes download_yfinance_ohlc().
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_ALLOWED_INTERVALS = {"1d", "5d", "1wk", "1mo", "3mo", "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


@dataclass(frozen=True)
class YFinanceSnapshot:
    symbol: str
    provider_symbol: str
    interval: str
    rows: int
    first_timestamp_utc: str
    last_timestamp_utc: str
    snapshot_sha256: str
    data_source: str = "Yahoo Finance via yfinance"
    auto_adjust: bool = False
    actionable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "provider_symbol": self.provider_symbol,
            "interval": self.interval,
            "rows": self.rows,
            "first_timestamp_utc": self.first_timestamp_utc,
            "last_timestamp_utc": self.last_timestamp_utc,
            "snapshot_sha256": self.snapshot_sha256,
            "data_source": self.data_source,
            "auto_adjust": self.auto_adjust,
            "actionable": False,
        }


def normalize_yfinance_symbol(symbol: str) -> str:
    """Map common slash-separated FX pairs to Yahoo's =X notation."""
    value = symbol.strip().upper().replace(" ", "")
    if len(value) == 7 and value[3] == "/" and value[:3].isalpha() and value[4:].isalpha():
        return value.replace("/", "") + "=X"
    if not value:
        raise ValueError("symbol must not be empty")
    return value


def download_yfinance_ohlc(
    symbol: str,
    *,
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    output_csv: str | Path | None = None,
    output_manifest: str | Path | None = None,
) -> tuple[pd.DataFrame, YFinanceSnapshot]:
    """Download explicit historical OHLCV and optionally persist a hash-bound snapshot.

    Supply either period or start/end. Yahoo Finance availability, licensing, adjustments,
    and historical intraday limits must be checked for the instrument and intended use.
    """
    if interval not in _ALLOWED_INTERVALS:
        raise ValueError(f"unsupported interval {interval!r}")
    if period and (start or end):
        raise ValueError("use either period or start/end, not both")
    if not period and not start:
        raise ValueError("provide period or start (and optional end)")
    provider_symbol = normalize_yfinance_symbol(symbol)
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required; install the project requirements") from exc

    options: dict[str, Any] = {
        "tickers": provider_symbol,
        "interval": interval,
        "auto_adjust": False,
        "actions": False,
        "progress": False,
        "threads": False,
        "group_by": "column",
    }
    if period:
        options["period"] = period
    else:
        options["start"] = start
        if end:
            options["end"] = end
    raw = yf.download(**options)
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo Finance returned no rows for {provider_symbol} at interval {interval}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(column[0]) for column in raw.columns]
    raw.columns = [str(column).strip().lower().replace(" ", "_") for column in raw.columns]
    required = ("open", "high", "low", "close")
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"Yahoo Finance response missing OHLC columns: {missing}")
    frame = raw.copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame.index.name = "timestamp"
    if frame.index.duplicated().any():
        raise ValueError("Yahoo Finance returned duplicate timestamps; snapshot rejected")
    frame = frame.sort_index()
    if frame.empty:
        raise ValueError("no timestamp rows remain after validation")
    for name in required:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if "volume" not in frame:
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    if frame[list(required)].isna().any().any():
        raise ValueError("OHLC contains missing or non-numeric values")
    if not np.isfinite(frame[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("OHLC contains non-finite values")
    if ((frame["low"] > frame[["open", "close"]].min(axis=1)) |
        (frame["high"] < frame[["open", "close"]].max(axis=1)) |
        (frame["high"] < frame["low"]) |
        (frame["close"] <= 0)).any():
        raise ValueError("Yahoo Finance returned invalid OHLC relationships")

    export = frame.reset_index()
    csv_bytes = export.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S%z").encode("utf-8")
    digest = hashlib.sha256(csv_bytes).hexdigest()
    snapshot = YFinanceSnapshot(
        symbol=symbol.strip().upper(),
        provider_symbol=provider_symbol,
        interval=interval,
        rows=len(export),
        first_timestamp_utc=export["timestamp"].iloc[0].isoformat().replace("+00:00", "Z"),
        last_timestamp_utc=export["timestamp"].iloc[-1].isoformat().replace("+00:00", "Z"),
        snapshot_sha256=digest,
    )
    if output_csv is not None:
        destination = Path(output_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(csv_bytes)
    if output_manifest is not None:
        destination = Path(output_manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "
", encoding="utf-8")
    return export, snapshot

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from app.yfinance_history import download_yfinance_ohlc, normalize_yfinance_symbol


def test_fx_symbol_maps_to_yahoo_notation():
    assert normalize_yfinance_symbol("EUR/USD") == "EURUSD=X"
    assert normalize_yfinance_symbol(" gbpusd=x ") == "GBPUSD=X"


def test_empty_symbol_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_yfinance_symbol("   ")


def test_download_is_explicit_and_writes_hash_bound_snapshot(monkeypatch, tmp_path):
    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
        name="Date",
    )
    source = pd.DataFrame(
        {
            "Open": [1.10, 1.11],
            "High": [1.12, 1.13],
            "Low": [1.09, 1.10],
            "Close": [1.11, 1.12],
            "Volume": [10, 20],
        },
        index=index,
    )
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(download=lambda **kwargs: (calls.append(kwargs) or source.copy())),
    )
    csv_path = tmp_path / "history.csv"
    manifest_path = tmp_path / "history.manifest.json"

    frame, snapshot = download_yfinance_ohlc(
        "EUR/USD",
        interval="1d",
        period="1mo",
        output_csv=csv_path,
        output_manifest=manifest_path,
    )

    assert len(calls) == 1
    assert calls[0]["tickers"] == "EURUSD=X"
    assert calls[0]["auto_adjust"] is False
    assert calls[0]["actions"] is False
    assert frame["timestamp"].dt.tz is not None
    assert snapshot.rows == 2
    assert snapshot.actionable is False
    assert len(snapshot.snapshot_sha256) == 64
    assert csv_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["snapshot_sha256"] == snapshot.snapshot_sha256
    assert manifest["actionable"] is False


def test_download_rejects_invalid_ohlc_without_persisting(monkeypatch, tmp_path):
    source = pd.DataFrame(
        {"Open": [1.10], "High": [1.05], "Low": [1.00], "Close": [1.02]},
        index=pd.DatetimeIndex(["2026-01-01T00:00:00Z"]),
    )
    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(download=lambda **kwargs: source.copy()),
    )
    destination = tmp_path / "must-not-exist.csv"
    with pytest.raises(ValueError, match="invalid OHLC relationships"):
        download_yfinance_ohlc(
            "EUR/USD", start="2026-01-01", output_csv=destination
        )
    assert not destination.exists()

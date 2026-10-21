from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.ctrader_history import normalize_ctrader_trendbars
from app.research_dataset_builder import prepare_research_dataset


def test_ctrader_trendbars_normalize_to_canonical_utc_ohlc(tmp_path):
    output = tmp_path / "ctrader.csv"
    manifest = normalize_ctrader_trendbars(
        [
            {"utcTimestampInMinutes": 29_000_000, "low": 110000, "deltaOpen": 200, "deltaHigh": 800, "deltaClose": 500},
            {"utcTimestampInMinutes": 29_000_015, "low": 110100, "deltaOpen": 100, "deltaHigh": 600, "deltaClose": 400},
        ],
        symbol="eurusd",
        timeframe="M15",
        output_csv=output,
        price_scale=100000,
    )
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["symbol"] == "EURUSD"
    assert rows[0]["timeframe"] == "M15"
    assert rows[0]["open"] == "1.102"
    assert rows[0]["high"] == "1.108"
    assert rows[0]["low"] == "1.1"
    assert rows[0]["close"] == "1.105"
    assert manifest["data_source"] == "cTrader Open API"
    assert manifest["actionable"] is False
    assert manifest["transport_or_auth_performed"] is False
    assert manifest["snapshot_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_ctrader_rejects_unsorted_or_duplicate_bars(tmp_path):
    with pytest.raises(ValueError, match="strictly chronological"):
        normalize_ctrader_trendbars(
            [
                {"timestamp": "2026-01-01T00:15:00Z", "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15},
                {"timestamp": "2026-01-01T00:00:00Z", "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15},
            ],
            symbol="EURUSD", timeframe="M15", output_csv=tmp_path / "bad.csv",
        )


def test_research_dataset_builder_emits_matured_labels_and_hashes(tmp_path):
    source = tmp_path / "ohlc.csv"
    output = tmp_path / "dataset.csv"
    manifest_path = tmp_path / "manifest.json"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "open", "high", "low", "close"])
        writer.writeheader()
        for i in range(70):
            close = 1.1 + i * 0.0002 + (i % 4) * 0.00003
            writer.writerow({
                "timestamp": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i)).isoformat().replace("+00:00", "Z"),
                "open": close - 0.00005,
                "high": close + 0.0001,
                "low": close - 0.0001,
                "close": close,
            })

    manifest = prepare_research_dataset(
        source, output, manifest_path,
        symbol="EURUSD", timeframe="M15", horizon_bars=3, class_1_means="UP",
        data_source="cTrader Open API", source_version="0.9.2",
        retrieved_at_utc="2026-10-21T12:00:00Z",
        terms_reference="operator-supplied cTrader Open API terms reference",
    )
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {"label", "label_available_at", "rsi_14", "atr_14_pct"} <= set(rows[0])
    assert rows[0]["label_available_at"] > rows[0]["timestamp"]
    assert manifest["actionable"] is False
    assert manifest["label_horizon_bars"] == 3
    assert manifest["output_dataset_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["manifest_sha256"] == manifest["manifest_sha256"]
    assert persisted["output_rows"] == len(rows)


def test_research_dataset_builder_rejects_unsorted_rows_and_bad_timestamp(tmp_path):
    source = tmp_path / "unsorted.csv"
    source.write_text(
        "timestamp,open,high,low,close\n"
        "2026-01-01T00:15:00Z,1.1,1.2,1.0,1.15\n"
        "2026-01-01T00:00:00Z,1.1,1.2,1.0,1.15\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly chronological"):
        prepare_research_dataset(
            source, tmp_path / "out.csv", tmp_path / "manifest.json",
            symbol="EURUSD", timeframe="M15", horizon_bars=1, class_1_means="UP",
            data_source="test", source_version="test", retrieved_at_utc="2026-10-21T12:00:00Z",
            terms_reference="test terms",
        )

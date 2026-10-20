from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pytest

from app.historical_feature_preparation import FEATURE_COLUMNS, prepare_features


def write_ohlc(path, count=24):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "symbol", "timeframe", "timestamp", "open", "high", "low", "close",
        ])
        writer.writeheader()
        for i in range(count):
            close = 1.1 + i * 0.0002 + (i % 3) * 0.00001
            writer.writerow({
                "symbol": "EUR/USD",
                "timeframe": "M15",
                "timestamp": (start + timedelta(minutes=15 * i)).isoformat().replace("+00:00", "Z"),
                "open": close - 0.00005,
                "high": close + 0.0001,
                "low": close - 0.0001,
                "close": close,
            })


def test_prepare_features_is_causal_and_records_provenance(tmp_path):
    source, output = tmp_path / "raw.csv", tmp_path / "features.csv"
    write_ohlc(source, 24)

    report = prepare_features(source, output)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert report["input_rows"] == 24
    assert report["output_rows"] == 9
    assert report["warmup_rows_excluded"] == 15
    assert len(report["input_sha256"]) == 64
    assert report["actionable"] is False
    assert set(FEATURE_COLUMNS).issubset(rows[0])
    assert "label" not in rows[0]
    assert all(float(row[name]) == float(row[name]) for row in rows for name in FEATURE_COLUMNS)


def test_prepare_features_rejects_nonchronological_input(tmp_path):
    source, output = tmp_path / "unordered.csv", tmp_path / "features.csv"
    write_ohlc(source, 20)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[3], rows[4] = rows[4], rows[3]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="OHLC validation failed|chronological"):
        prepare_features(source, output)


def test_prepare_features_rejects_naive_timestamps(tmp_path):
    source, output = tmp_path / "naive.csv", tmp_path / "features.csv"
    write_ohlc(source, 20)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["timestamp"] = "2025-01-01T00:00:00"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="timezone offset"):
        prepare_features(source, output)


def test_prepare_features_rejects_same_input_and_output_path(tmp_path):
    source = tmp_path / "raw.csv"
    write_ohlc(source, 20)
    with pytest.raises(ValueError, match="paths must differ"):
        prepare_features(source, source)

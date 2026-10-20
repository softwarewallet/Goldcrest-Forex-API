from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pytest

from app.historical_data_quality import audit_historical_ohlc


def write_bars(path, timestamps=None):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = timestamps or [start + timedelta(minutes=15 * i) for i in range(5)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "symbol", "timeframe", "timestamp", "open", "high", "low", "close",
        ])
        writer.writeheader()
        for i, timestamp in enumerate(timestamps):
            close = 1.1 + i * 0.0001
            writer.writerow({
                "symbol": "EUR/USD", "timeframe": "M15",
                "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                "open": close - 0.00002, "high": close + 0.00005,
                "low": close - 0.00005, "close": close,
            })


def test_quality_audit_reports_cadence_and_source_fingerprint(tmp_path):
    source = tmp_path / "raw.csv"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    write_bars(source, [
        start,
        start + timedelta(minutes=15),
        start + timedelta(minutes=30),
        start + timedelta(minutes=60),
        start + timedelta(minutes=75),
    ])

    report = audit_historical_ohlc(source)

    assert report["structurally_valid"] is True
    assert report["row_count"] == 5
    assert report["expected_interval_minutes"] == 15
    assert report["long_interval_count"] == 1
    assert report["estimated_missing_bars_on_exact_interval_multiples"] == 1
    assert len(report["input_sha256"]) == 64
    assert report["actionable"] is False
    assert "not confirmed missing candles" in report["warnings"][0]


def test_quality_audit_flags_duplicates_and_non_monotonic_rows(tmp_path):
    source = tmp_path / "raw.csv"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    write_bars(source, [
        start,
        start + timedelta(minutes=15),
        start + timedelta(minutes=15),
        start + timedelta(minutes=10),
    ])

    report = audit_historical_ohlc(source)

    assert report["duplicate_timestamp_count"] == 1
    assert report["non_monotonic_interval_count"] == 1
    assert report["structurally_valid"] is False


def test_quality_audit_rejects_naive_timestamp_and_invalid_ohlc_schema(tmp_path):
    source = tmp_path / "raw.csv"
    write_bars(source)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["timestamp"] = "2025-01-01T00:00:00"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="timezone offset"):
        audit_historical_ohlc(source)


def test_quality_audit_rejects_mixed_timeframes(tmp_path):
    source = tmp_path / "raw.csv"
    write_bars(source)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["timeframe"] = "H1"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="one symbol/timeframe"):
        audit_historical_ohlc(source)

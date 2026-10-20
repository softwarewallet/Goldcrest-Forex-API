from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pytest

from app.historical_tabular_research import load_historical_dataset


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = [
            "symbol", "timeframe", "timestamp", "open", "high", "low", "close",
            "label", "label_available_at", "rsi", "return_5",
        ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields if key in row} for row in rows)


def sample_rows(count=8):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(count):
        timestamp = start + timedelta(minutes=5 * index)
        close = 1.1 + index * 0.0001
        rows.append({
            "symbol": "EUR/USD",
            "timeframe": "M5",
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "open": str(close),
            "high": str(close + 0.0002),
            "low": str(close - 0.0002),
            "close": str(close),
            "label": str(index % 2),
            "label_available_at": (timestamp + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
            "rsi": str(40 + index),
            "return_5": str(index / 1000),
        })
    return rows


def test_load_historical_dataset_preserves_order_and_explicit_labels(tmp_path):
    path = tmp_path / "historical.csv"
    write_csv(path, sample_rows())

    bars, examples, digest = load_historical_dataset(path, ("rsi", "return_5"))

    assert len(bars) == 8
    assert len(examples) == 8
    assert examples[0].label == 0
    assert examples[1].label == 1
    assert examples[0].features == {"rsi": 40.0, "return_5": 0.0}
    assert examples[0].label_available_at > examples[0].timestamp
    assert len(digest) == 64


def test_loader_rejects_missing_feature_column(tmp_path):
    path = tmp_path / "missing.csv"
    rows = sample_rows()
    write_csv(path, rows, fields=[
        "symbol", "timeframe", "timestamp", "open", "high", "low", "close",
        "label", "label_available_at", "rsi",
    ])

    with pytest.raises(ValueError, match="missing required columns"):
        load_historical_dataset(path, ("rsi", "return_5"))


def test_loader_rejects_nonchronological_rows_instead_of_sorting(tmp_path):
    path = tmp_path / "unordered.csv"
    rows = sample_rows()
    rows[2], rows[3] = rows[3], rows[2]
    write_csv(path, rows)

    with pytest.raises(ValueError, match="OHLC/order checks|chronological"):
        load_historical_dataset(path, ("rsi", "return_5"))


def test_loader_rejects_label_availability_before_feature_timestamp(tmp_path):
    path = tmp_path / "immature.csv"
    rows = sample_rows()
    rows[0]["label_available_at"] = rows[0]["timestamp"]
    # Equality is valid: the outcome is explicitly known at the feature timestamp.
    write_csv(path, rows)
    assert len(load_historical_dataset(path, ("rsi", "return_5"))[1]) == 8

    rows[0]["label_available_at"] = "2025-12-31T23:59:00Z"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="label_available_at cannot precede"):
        load_historical_dataset(path, ("rsi", "return_5"))


def test_loader_rejects_nonbinary_labels_and_nonfinite_features(tmp_path):
    path = tmp_path / "bad-label.csv"
    rows = sample_rows()
    rows[0]["label"] = "2"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="label must be exactly 0 or 1"):
        load_historical_dataset(path, ("rsi", "return_5"))

    rows = sample_rows()
    rows[0]["rsi"] = "nan"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="must be finite"):
        load_historical_dataset(path, ("rsi", "return_5"))

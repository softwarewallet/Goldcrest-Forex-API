from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from app.historical_label_preparation import label_historical_features


def write_features(path, closes):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "return_1"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, close in enumerate(closes):
            writer.writerow({
                "symbol": "EUR/USD",
                "timeframe": "M15",
                "timestamp": (start + timedelta(minutes=15 * i)).isoformat().replace("+00:00", "Z"),
                "open": close,
                "high": close + 0.01,
                "low": close - 0.01,
                "close": close,
                "return_1": 0.0,
            })


def test_labels_use_future_close_and_record_maturity(tmp_path):
    source, output = tmp_path / "features.csv", tmp_path / "labeled.csv"
    write_features(source, [1.0, 1.1, 1.2, 1.0, 0.9])

    report = label_historical_features(
        source, output, horizon_bars=2, neutral_band_bps=0.0, neutral_policy="error"
    )

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert [row["label"] for row in rows] == ["1", "0", "0"]
    assert rows[0]["label_available_at"] == "2025-01-01T00:30:00+00:00"
    assert rows[0]["return_1"] == "0.0"
    assert report["tail_rows_without_matured_outcome"] == 2
    assert report["class_1_means"] == "UP"
    assert len(report["input_sha256"]) == 64
    assert report["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report["actionable"] is False


def test_neutral_policy_is_explicit_and_reported(tmp_path):
    source, output = tmp_path / "features.csv", tmp_path / "labeled.csv"
    write_features(source, [1.0, 1.0, 1.1, 1.1])

    with pytest.raises(ValueError, match="neutral outcome"):
        label_historical_features(
            source, output, horizon_bars=1, neutral_band_bps=0.0, neutral_policy="error"
        )

    report = label_historical_features(
        source, output, horizon_bars=1, neutral_band_bps=0.0, neutral_policy="up"
    )
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["label"] == "1"
    assert rows[2]["label"] == "1"
    assert report["neutral_outcomes"] == 2
    assert report["neutral_policy"] == "up"


def test_rejects_existing_labels_and_input_output_collision(tmp_path):
    source, output = tmp_path / "features.csv", tmp_path / "labeled.csv"
    write_features(source, [1.0, 1.1, 1.2])
    with pytest.raises(ValueError, match="paths must differ"):
        label_historical_features(
            source, source, horizon_bars=1, neutral_band_bps=0.0, neutral_policy="error"
        )

    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0]) + ["label"]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "label": "1"})
    with pytest.raises(ValueError, match="already contains labels"):
        label_historical_features(
            source, output, horizon_bars=1, neutral_band_bps=0.0, neutral_policy="error"
        )


def test_rejects_nonchronological_timestamps(tmp_path):
    source, output = tmp_path / "unordered.csv", tmp_path / "labeled.csv"
    write_features(source, [1.0, 1.1, 1.2])
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1], rows[2] = rows[2], rows[1]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="strictly chronological"):
        label_historical_features(
            source, output, horizon_bars=1, neutral_band_bps=0.0, neutral_policy="error"
        )

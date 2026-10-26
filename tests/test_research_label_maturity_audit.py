from __future__ import annotations

import csv
import json
from pathlib import Path

from app.research_label_maturity_audit import audit_label_maturity


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _fixtures(tmp_path: Path):
    raw = _write_csv(
        tmp_path / "raw.csv",
        ["symbol", "timeframe", "timestamp", "open", "high", "low", "close"],
        [
            ["EURUSD", "M15", timestamp, close, close + 0.01, close - 0.01, close]
            for timestamp, close in [
                ("2026-01-01T00:00:00Z", 1.00),
                ("2026-01-01T00:15:00Z", 1.10),
                ("2026-01-01T00:30:00Z", 1.20),
                ("2026-01-01T00:45:00Z", 1.00),
                ("2026-01-01T01:00:00Z", 0.90),
            ]
        ],
    )
    dataset = _write_csv(
        tmp_path / "dataset.csv",
        ["symbol", "timeframe", "timestamp", "close", "label", "label_available_at"],
        [
            ["EURUSD", "M15", "2026-01-01T00:00:00Z", 1.00, 1, "2026-01-01T00:30:00Z"],
            ["EURUSD", "M15", "2026-01-01T00:15:00Z", 1.10, 0, "2026-01-01T00:45:00Z"],
        ],
    )
    return raw, dataset


def test_label_maturity_audit_accepts_correct_forward_labels(tmp_path):
    raw, dataset = _fixtures(tmp_path)

    report = audit_label_maturity(
        raw, dataset, symbol="EURUSD", timeframe="M15",
        horizon_bars=2, class_1_means="UP",
    )

    assert report["status"] == "PASS"
    assert report["checked_label_rows"] == 2
    assert report["errors"] == []
    assert report["actionable"] is False


def test_label_maturity_audit_rejects_maturity_timestamp_mismatch(tmp_path):
    raw, dataset = _fixtures(tmp_path)
    content = dataset.read_text(encoding="utf-8").replace(
        "2026-01-01T00:30:00Z", "2026-01-01T00:45:00Z", 1
    )
    dataset.write_text(content, encoding="utf-8")

    report = audit_label_maturity(
        raw, dataset, symbol="EURUSD", timeframe="M15",
        horizon_bars=2, class_1_means="UP",
    )

    assert report["status"] == "FAIL"
    assert "label_maturity_timestamp_mismatch_line_2" in report["errors"]


def test_label_maturity_audit_rejects_label_value_mismatch(tmp_path):
    raw, dataset = _fixtures(tmp_path)
    content = dataset.read_text(encoding="utf-8").replace(
        "2026-01-01T00:00:00Z,1.0,1,", "2026-01-01T00:00:00Z,1.0,0,", 1
    )
    dataset.write_text(content, encoding="utf-8")

    report = audit_label_maturity(
        raw, dataset, symbol="EURUSD", timeframe="M15",
        horizon_bars=2, class_1_means="UP",
    )

    assert report["status"] == "FAIL"
    assert "dataset_label_value_mismatch_line_2" in report["errors"]


def test_label_maturity_audit_checks_eligibility_mask_against_elapsed_time(tmp_path):
    raw, dataset = _fixtures(tmp_path)
    dataset = _write_csv(
        tmp_path / "dataset-capped.csv",
        ["symbol", "timeframe", "timestamp", "close", "label", "label_available_at", "label_elapsed_hours", "label_eligible"],
        [
            ["EURUSD", "M15", "2026-01-01T00:00:00Z", 1.00, 1, "2026-01-01T00:30:00Z", 0.5, 1],
            ["EURUSD", "M15", "2026-01-01T00:15:00Z", 1.10, 0, "2026-01-01T00:45:00Z", 0.5, 1],
        ],
    )

    report = audit_label_maturity(
        raw, dataset, symbol="EURUSD", timeframe="M15",
        horizon_bars=2, class_1_means="UP", max_label_elapsed_hours=0.25,
    )

    assert report["status"] == "FAIL"
    assert report["eligible_label_rows"] == 0
    assert report["ineligible_label_rows"] == 2
    assert "dataset_label_eligibility_mask_mismatch_line_2" in report["errors"]
    assert "dataset_label_eligibility_mask_mismatch_line_3" in report["errors"]

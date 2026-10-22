from __future__ import annotations

import csv
import hashlib
import json

from app.research_data_quality import audit_ohlc_csv, audit_ohlc_csv_file


def write_csv(path, rows, *, columns=("symbol", "timeframe", "timestamp", "open", "high", "low", "close")):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def row(ts, *, symbol="EURUSD", timeframe="M15", open_="1.0", high="1.2", low="0.9", close="1.1"):
    return {"symbol": symbol, "timeframe": timeframe, "timestamp": ts,
            "open": open_, "high": high, "low": low, "close": close}


def test_quality_audit_passes_valid_ordered_csv(tmp_path):
    source = tmp_path / "bars.csv"
    write_csv(source, [
        row("2026-01-01T00:00:00Z"),
        row("2026-01-01T00:15:00Z"),
        row("2026-01-01T00:30:00+00:00"),
    ])
    result = audit_ohlc_csv(source, expected_symbol="EURUSD", expected_timeframe="M15")
    assert result["status"] == "PASS"
    assert result["valid_row_count"] == 3
    assert result["duplicate_timestamps"] == 0
    assert result["observed_gap_count"] == 0
    assert result["actionable"] is False


def test_quality_audit_fails_duplicates_ordering_and_bad_ohlc(tmp_path):
    source = tmp_path / "bad.csv"
    write_csv(source, [
        row("2026-01-01T00:15:00Z"),
        row("2026-01-01T00:15:00Z"),
        row("2026-01-01T00:00:00Z"),
        row("2026-01-01T00:30:00Z", low="1.15"),
    ])
    result = audit_ohlc_csv(source)
    assert result["status"] == "FAIL"
    assert result["duplicate_timestamps"] == 1
    assert result["non_monotonic_timestamps"] == 1
    assert result["invalid_ohlc_rows"] == 1
    assert "invalid_ohlc_rows_present" in result["errors"]


def test_quality_audit_flags_identity_mismatches_and_reports_gap(tmp_path):
    source = tmp_path / "gap.csv"
    write_csv(source, [
        row("2026-01-01T00:00:00Z", symbol="GBPUSD"),
        row("2026-01-01T01:00:00Z", symbol="GBPUSD"),
    ])
    result = audit_ohlc_csv(source, expected_symbol="EURUSD", expected_timeframe="M15")
    assert result["status"] == "FAIL"
    assert result["symbol_mismatches"] == 2
    assert result["observed_gap_count"] == 1
    assert "symbol_mismatches_present" in result["errors"]
    assert "observed_timestamp_gaps_require_session_calendar_review" in result["warnings"]


def test_quality_audit_missing_columns_fails_closed(tmp_path):
    source = tmp_path / "missing.csv"
    source.write_text("timestamp,open,close\n2026-01-01T00:00:00Z,1,1.1\n", encoding="utf-8")
    result = audit_ohlc_csv(source)
    assert result["status"] == "FAIL"
    assert result["errors"] == ["missing_required_columns:high,low"]


def test_quality_receipt_binds_exact_source_bytes(tmp_path):
    source = tmp_path / "bars.csv"
    receipt = tmp_path / "audit" / "quality.json"
    write_csv(source, [row("2026-01-01T00:00:00Z")])
    result = audit_ohlc_csv_file(source, receipt, expected_symbol="EURUSD", expected_timeframe="M15")
    stored = json.loads(receipt.read_text(encoding="utf-8"))
    assert stored["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["receipt_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert stored["status"] == "PASS"

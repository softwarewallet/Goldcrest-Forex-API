from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.research_data_audit import audit_raw_ohlc_file


def _write_snapshot(tmp_path: Path, timestamps: list[datetime]) -> tuple[Path, Path]:
    csv_path = tmp_path / "EURUSD_M15.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["symbol", "timeframe", "timestamp", "open", "high", "low", "close"],
            lineterminator="\n",
        )
        writer.writeheader()
        for ts in timestamps:
            writer.writerow({
                "symbol": "EURUSD",
                "timeframe": "M15",
                "timestamp": ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "open": "1.10000",
                "high": "1.10100",
                "low": "1.09900",
                "close": "1.10050",
            })
    manifest_path = Path(str(csv_path) + ".manifest.json")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    first = timestamps[0].astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    last = timestamps[-1].astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest = {
        "schema_version": 1,
        "scope": "CTRADER_OPEN_API_HISTORICAL_OHLC_NORMALIZATION",
        "actionable": False,
        "symbol": "EURUSD",
        "timeframe": "M15",
        "rows": len(timestamps),
        "first_timestamp_utc": first,
        "last_timestamp_utc": last,
        "snapshot_sha256": digest,
        "host": "demo",
        "pagination": {
            "complete": True,
            "window_count": 1,
            "pages_requested": 1,
            "duplicate_bars_removed": 0,
            "has_more_flag_available": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return csv_path, manifest_path


def test_raw_ohlc_audit_passes_matching_manifest_and_valid_bars(tmp_path):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    path, manifest = _write_snapshot(tmp_path, [start, start + timedelta(minutes=15), start + timedelta(minutes=30)])

    result = audit_raw_ohlc_file(path, manifest_path=manifest, expected_symbol="EURUSD", expected_timeframe="M15")

    assert result["status"] == "PASS"
    assert result["row_count"] == 3
    assert result["symbol"] == "EURUSD"
    assert result["timeframe"] == "M15"
    assert result["invalid_ohlc_rows"] == 0
    assert result["nonfinite_ohlc_rows"] == 0
    assert result["unaligned_timestamp_rows"] == 0
    assert result["pagination"]["complete"] is True
    assert result["actionable"] is False
    assert result["execution_connected"] is False


def test_raw_ohlc_audit_rejects_changed_snapshot_hash(tmp_path):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    path, manifest = _write_snapshot(tmp_path, [start, start + timedelta(minutes=15)])
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = audit_raw_ohlc_file(path, manifest_path=manifest)

    assert result["status"] == "FAIL"
    assert "snapshot_sha256_mismatch" in result["errors"]


def test_raw_ohlc_audit_rejects_non_chronological_duplicate_and_invalid_ohlc(tmp_path):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    path, manifest = _write_snapshot(tmp_path, [start, start])
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    rows[1]["low"] = "1.102"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    result = audit_raw_ohlc_file(path, manifest_path=manifest)

    assert result["status"] == "FAIL"
    assert any(item.startswith("timestamps_not_strictly_increasing_line_") for item in result["errors"])
    assert "invalid_ohlc_rows:1" in result["errors"]
    assert "snapshot_sha256_mismatch" in result["errors"]


def test_raw_ohlc_audit_warns_on_long_market_gaps_without_calling_them_corruption(tmp_path):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    path, manifest = _write_snapshot(tmp_path, [start, start + timedelta(minutes=15), start + timedelta(hours=9)])

    result = audit_raw_ohlc_file(path, manifest_path=manifest)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["large_gap_count_over_8_hours"] == 1
    assert "large_time_gaps_detected_review_market_closures_and_data_coverage" in result["warnings"]

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.research_calendar_gap_adjudication import adjudicate_market_calendar_gaps


def fixture(tmp_path: Path, *, intervals=None, coverage_start=None, coverage_end=None):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=9)
    csv_path = tmp_path / "EURUSD_M15.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "timeframe", "timestamp", "open", "high", "low", "close"])
        writer.writeheader()
        for stamp in (start, end):
            writer.writerow({"symbol": "EURUSD", "timeframe": "M15", "timestamp": stamp.isoformat().replace("+00:00", "Z"),
                             "open": "1.1", "high": "1.2", "low": "1.0", "close": "1.15"})
    snapshot_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "snapshot.manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1, "scope": "CTRADER_OPEN_API_HISTORICAL_OHLC_NORMALIZATION",
        "actionable": False, "symbol": "EURUSD", "timeframe": "M15", "rows": 2,
        "first_timestamp_utc": start.isoformat().replace("+00:00", "Z"),
        "last_timestamp_utc": end.isoformat().replace("+00:00", "Z"),
        "snapshot_sha256": snapshot_hash, "host": "demo", "provider": "ctrader-open-api",
    }), encoding="utf-8")
    source_path = tmp_path / "official-calendar.txt"
    source_path.write_text("Archived official calendar evidence fixture.\n", encoding="utf-8")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    calendar_path = tmp_path / "calendar-evidence.json"
    calendar_path.write_text(json.dumps({
        "schema_version": 1, "evidence_type": "market_calendar_gap_adjudication",
        "provider": "ctrader-open-api", "symbol": "EURUSD", "timeframe": "M15",
        "authority": "Example instrument venue calendar", "source_reference": "https://example.org/calendar.pdf",
        "retrieved_at_utc": "2026-01-05T12:00:00Z", "source_evidence_sha256": source_hash,
        "coverage_start_utc": (coverage_start or start - timedelta(days=1)).isoformat().replace("+00:00", "Z"),
        "coverage_end_utc": (coverage_end or end + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
        "closed_intervals": intervals if intervals is not None else [{
            "start_utc": (start + timedelta(minutes=15)).isoformat().replace("+00:00", "Z"),
            "end_utc": end.isoformat().replace("+00:00", "Z"),
            "reason": "Documented venue closure",
            "source_locator": "Page 2, dated session table",
        }],
    }), encoding="utf-8")
    return csv_path, manifest_path, calendar_path, source_path, start, end


def test_gap_adjudicated_only_when_hash_verified_interval_covers_missing_bars(tmp_path):
    csv_path, manifest, calendar, source, _, _ = fixture(tmp_path)
    report = adjudicate_market_calendar_gaps(csv_path, manifest, calendar, source)
    assert report["status"] == "PASS"
    assert report["adjudication_complete"] is True
    assert report["adjudicated_gap_count"] == 1
    assert report["gaps"][0]["adjudication"] == "ADJUDICATED_DOCUMENTED_CLOSURE"
    assert report["automatic_warning_suppression_performed"] is False
    assert report["automatic_repair_performed"] is False


def test_partial_calendar_overlap_does_not_adjudicate_gap(tmp_path):
    csv_path, manifest, calendar, source, start, end = fixture(
        tmp_path,
        intervals=[{
            "start_utc": (start + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
            "end_utc": (end - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "reason": "Partial closure only", "source_locator": "Page 2",
        }],
    )
    report = adjudicate_market_calendar_gaps(csv_path, manifest, calendar, source)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert report["adjudication_complete"] is False
    assert report["unresolved_gap_count"] == 1


def test_calendar_source_hash_mismatch_fails_closed(tmp_path):
    csv_path, manifest, calendar, source, _, _ = fixture(tmp_path)
    source.write_text("Changed after evidence was recorded.\n", encoding="utf-8")
    report = adjudicate_market_calendar_gaps(csv_path, manifest, calendar, source)
    assert report["status"] == "FAIL"
    assert "calendar_source_evidence_sha256_mismatch" in report["errors"]


def test_calendar_scope_mismatch_fails_closed(tmp_path):
    csv_path, manifest, calendar, source, _, _ = fixture(tmp_path)
    payload = json.loads(calendar.read_text(encoding="utf-8"))
    payload["symbol"] = "GBPUSD"
    calendar.write_text(json.dumps(payload), encoding="utf-8")
    report = adjudicate_market_calendar_gaps(csv_path, manifest, calendar, source)
    assert report["status"] == "FAIL"
    assert "calendar_evidence_symbol_mismatch" in report["errors"]


def test_unresolved_gap_does_not_silently_pass_without_closure_interval(tmp_path):
    csv_path, manifest, calendar, source, _, _ = fixture(tmp_path, intervals=[])
    report = adjudicate_market_calendar_gaps(csv_path, manifest, calendar, source)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert report["unresolved_gap_count"] == 1
    assert "unresolved_timestamp_gaps_remain" in report["warnings"]

from __future__ import annotations

import csv
import hashlib
import json

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.research_data_audit import (
    _gap_pattern_diagnostics,
    _timestamp_coverage_profile,
    _within_session_gap_diagnostics,
    audit_raw_ohlc_file,
    main as audit_main,
)


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


def test_raw_ohlc_audit_warns_on_non_weekend_long_gaps_without_calling_them_corruption(tmp_path):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    path, manifest = _write_snapshot(tmp_path, [start, start + timedelta(minutes=15), start + timedelta(hours=9)])

    result = audit_raw_ohlc_file(path, manifest_path=manifest)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["large_gap_count_over_8_hours"] == 1
    assert result["non_weekend_long_gap_count_over_8_hours"] == 1
    assert result["weekend_closure_like_gap_count_over_8_hours"] == 0
    assert result["large_gap_examples"][0]["gap_category"] == "non_weekend_long_gap"
    assert result["large_gap_examples"][0]["calendar_context"] == []
    assert result["holiday_context_gap_count_over_8_hours"] == 0
    assert "non_weekend_long_gaps_detected_review_data_coverage" in result["warnings"]






def test_raw_ohlc_audit_classifies_late_friday_to_sunday_gap_as_weekend_like(tmp_path):
    friday = datetime(2026, 1, 9, 21, 45, tzinfo=timezone.utc)
    sunday = datetime(2026, 1, 11, 21, 0, tzinfo=timezone.utc)
    path, manifest = _write_snapshot(tmp_path, [friday, sunday])

    result = audit_raw_ohlc_file(path, manifest_path=manifest)

    assert result["status"] == "PASS"
    assert result["warnings"] == []
    assert result["large_gap_count_over_8_hours"] == 1
    assert result["weekend_closure_like_gap_count_over_8_hours"] == 1
    assert result["non_weekend_long_gap_count_over_8_hours"] == 0
    assert result["large_gap_examples"][0]["gap_category"] == "weekend_closure_like"
    assert result["gap_classification"]["method"] == "conservative_utc_weekend_window_heuristic"


def test_raw_ohlc_audit_adds_christmas_context_without_suppressing_warning(tmp_path):
    before = datetime(2025, 12, 24, 21, 15, tzinfo=timezone.utc)
    after = datetime(2025, 12, 25, 22, 0, tzinfo=timezone.utc)
    path, manifest = _write_snapshot(tmp_path, [before, after])

    result = audit_raw_ohlc_file(path, manifest_path=manifest)

    gap = result["large_gap_examples"][0]
    assert result["status"] == "PASS_WITH_WARNINGS"
    assert gap["gap_category"] == "non_weekend_long_gap"
    assert gap["calendar_context"] == ["christmas_day"]
    assert result["holiday_context_gap_count_over_8_hours"] == 1
    assert result["holiday_context_gap_examples"][0]["calendar_context"] == ["christmas_day"]
    assert "non_weekend_long_gaps_detected_review_data_coverage" in result["warnings"]


def test_raw_ohlc_audit_adds_new_year_context_without_suppressing_warning(tmp_path):
    before = datetime(2025, 12, 31, 21, 15, tzinfo=timezone.utc)
    after = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)
    path, manifest = _write_snapshot(tmp_path, [before, after])

    result = audit_raw_ohlc_file(path, manifest_path=manifest)

    gap = result["large_gap_examples"][0]
    assert result["status"] == "PASS_WITH_WARNINGS"
    assert gap["gap_category"] == "non_weekend_long_gap"
    assert gap["calendar_context"] == ["new_years_day"]
    assert result["holiday_context_gap_count_over_8_hours"] == 1
    assert "non_weekend_long_gaps_detected_review_data_coverage" in result["warnings"]


def test_raw_ohlc_audit_can_audit_operator_csv_without_manifest(tmp_path):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    path, manifest = _write_snapshot(
        tmp_path, [start, start + timedelta(minutes=15), start + timedelta(minutes=30)]
    )
    manifest.unlink()

    result = audit_raw_ohlc_file(
        path, expected_symbol="EURUSD", expected_timeframe="M15", require_manifest=False
    )

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["errors"] == []
    assert "snapshot_manifest_missing_provenance_hash_unverified" in result["warnings"]
    assert result["actionable"] is False



def test_raw_ohlc_audit_warns_instead_of_rejecting_operator_manifest(tmp_path):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    path, manifest = _write_snapshot(
        tmp_path, [start, start + timedelta(minutes=15), start + timedelta(minutes=30)]
    )
    manifest.write_text(json.dumps({"scope": "OPERATOR_EXPORT_V1"}) + "\n", encoding="utf-8")

    result = audit_raw_ohlc_file(
        path, expected_symbol="EURUSD", expected_timeframe="M15", require_manifest=False
    )

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["errors"] == []
    assert "non_ctrader_manifest_metadata_unverified" in result["warnings"]


def test_raw_ohlc_audit_requires_manifest_by_default(tmp_path):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    path, manifest = _write_snapshot(tmp_path, [start, start + timedelta(minutes=15)])
    manifest.unlink()

    result = audit_raw_ohlc_file(path)

    assert result["status"] == "FAIL"
    assert "snapshot_manifest_not_found" in result["errors"]



def test_gap_pattern_diagnostics_summarize_duration_and_repeated_utc_boundaries():
    gaps = [
        {
            "from_utc": "2026-01-05T00:00:00Z",
            "to_utc": "2026-01-05T09:00:00Z",
            "gap_hours": 9.0,
            "gap_category": "non_weekend_long_gap",
        },
        {
            "from_utc": "2026-01-12T00:00:00Z",
            "to_utc": "2026-01-12T09:00:00Z",
            "gap_hours": 9.0,
            "gap_category": "non_weekend_long_gap",
        },
        {
            "from_utc": "2026-01-09T21:45:00Z",
            "to_utc": "2026-01-11T21:00:00Z",
            "gap_hours": 47.25,
            "gap_category": "weekend_closure_like",
        },
    ]

    result = _gap_pattern_diagnostics(gaps)

    assert result["gap_count"] == 3
    assert result["category_counts"] == {
        "non_weekend_long_gap": 2,
        "weekend_closure_like": 1,
    }
    assert result["duration_buckets"]["greater_than_8_to_12_hours"] == 2
    assert result["duration_buckets"]["greater_than_24_to_48_hours"] == 1
    assert sum(result["duration_buckets"].values()) == 3
    assert result["start_weekday_hour_utc_counts"][0] == {
        "weekday_utc": "MONDAY", "hour_utc": 0, "count": 2
    }
    assert result["repeated_boundary_patterns"] == [{
        "from_weekday_utc": "MONDAY",
        "from_hour_utc": 0,
        "to_weekday_utc": "MONDAY",
        "to_hour_utc": 9,
        "count": 2,
    }]
    assert result["unparsed_boundary_count"] == 0
    assert "do not establish an expected market closure" in result["interpretation"]



def test_within_session_gap_diagnostics_bucket_frame_multiples_and_ignore_over_8_hours():
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    timestamps = [
        start,
        start + timedelta(minutes=15),
        start + timedelta(minutes=45),
        start + timedelta(minutes=90),
        start + timedelta(minutes=180),
        start + timedelta(minutes=420),
        start + timedelta(minutes=1020),
    ]

    result = _within_session_gap_diagnostics(timestamps, frame_seconds=15 * 60)

    assert result["status"] == "AVAILABLE"
    assert result["expected_frame_seconds"] == 900
    assert result["gap_count"] == 4
    assert result["frame_multiple_buckets"] == {
        "greater_than_1_to_2_frames": 1,
        "greater_than_2_to_4_frames": 1,
        "greater_than_4_to_8_frames": 1,
        "greater_than_8_to_16_frames": 1,
        "greater_than_16_frames_to_8_hours": 0,
    }
    assert result["largest_gap_examples"][0]["interval_minutes"] == 240.0
    assert result["largest_gap_examples"][0]["elapsed_timeframe_equivalents"] == 16.0
    assert result["largest_gap_examples"][0]["aligned_to_timeframe_multiple"] is True
    assert "not confirmed missing candles" in result["interpretation"]


def test_within_session_gap_diagnostics_refuses_non_monotonic_timestamps():
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)

    result = _within_session_gap_diagnostics(
        [start, start + timedelta(minutes=15), start + timedelta(minutes=10)],
        frame_seconds=15 * 60,
    )

    assert result["status"] == "UNAVAILABLE_NON_MONOTONIC_TIMESTAMPS"
    assert result["gap_count"] is None
    assert result["largest_gap_examples"] == []


def test_raw_ohlc_audit_reports_sub_8_hour_spacing_without_reclassifying_it(tmp_path):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    path, manifest = _write_snapshot(
        tmp_path, [start, start + timedelta(minutes=15), start + timedelta(minutes=45)]
    )

    result = audit_raw_ohlc_file(path, manifest_path=manifest)

    diagnostics = result["gap_diagnostics_between_one_frame_and_8_hours"]
    assert result["status"] == "PASS"
    assert diagnostics["status"] == "AVAILABLE"
    assert diagnostics["gap_count"] == 1
    assert diagnostics["frame_multiple_buckets"]["greater_than_1_to_2_frames"] == 1
    assert diagnostics["largest_gap_examples"][0]["interval_minutes"] == 30.0
    assert result["large_gap_count_over_8_hours"] == 0
    assert result["actionable"] is False


def test_timestamp_coverage_profile_counts_observed_and_unobserved_utc_dates():
    monday = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    wednesday = datetime(2026, 1, 7, 12, 0, tzinfo=timezone.utc)
    timestamps = [
        monday,
        monday + timedelta(minutes=15),
        monday + timedelta(hours=23, minutes=45),
        wednesday,
        wednesday + timedelta(minutes=15),
    ]

    result = _timestamp_coverage_profile(timestamps)

    assert result["status"] == "AVAILABLE"
    assert result["timestamp_count"] == 5
    assert result["calendar_span_days_inclusive"] == 3
    assert result["observed_utc_dates_count"] == 2
    assert result["unobserved_utc_dates_within_span_count"] == 1
    assert result["unobserved_utc_date_examples"] == ["2026-01-06"]
    assert result["daily_row_count_summary"] == {
        "minimum": 2, "median": 2.5, "mean": 2.5, "maximum": 3,
    }
    assert result["daily_row_counts"][0] == {
        "date_utc": "2026-01-05",
        "row_count": 3,
        "first_timestamp_utc": "2026-01-05T00:00:00Z",
        "last_timestamp_utc": "2026-01-05T23:45:00Z",
    }
    assert result["hour_of_day_utc_counts"][0] == {"hour_utc": 0, "count": 2}
    assert result["hour_of_day_utc_counts"][12] == {"hour_utc": 12, "count": 2}
    assert "not automatically classified as missing-data incidents" in result["interpretation"]


def test_timestamp_coverage_profile_does_not_summarize_non_monotonic_data():
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)

    result = _timestamp_coverage_profile([
        start, start + timedelta(minutes=15), start + timedelta(minutes=10)
    ])

    assert result["status"] == "UNAVAILABLE_NON_MONOTONIC_TIMESTAMPS"
    assert result["timestamp_count"] == 3
    assert result["daily_row_counts"] == []
    assert result["unobserved_utc_dates_within_span_count"] == 0


def test_timestamp_coverage_profile_requires_valid_example_limit():
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="non-negative integer"):
        _timestamp_coverage_profile([start], unobserved_date_example_limit=-1)


def test_raw_ohlc_audit_includes_timestamp_coverage_profile(tmp_path):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    path, manifest = _write_snapshot(
        tmp_path, [start, start + timedelta(minutes=15), start + timedelta(days=2)]
    )

    result = audit_raw_ohlc_file(path, manifest_path=manifest)
    coverage = result["timestamp_coverage_profile"]

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert coverage["status"] == "AVAILABLE"
    assert coverage["observed_utc_dates_count"] == 2
    assert coverage["unobserved_utc_dates_within_span_count"] == 1
    assert coverage["unobserved_utc_date_examples"] == ["2026-01-06"]
    assert result["actionable"] is False



def _write_pipeline_manifest(tmp_path: Path, csv_path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "run_id": "test-run",
        "scope": "GOLDCREST_OFFLINE_RESEARCH_PIPELINE",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "raw_ohlc_path": str(csv_path.resolve()),
        "raw_ohlc_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "actionable": False,
        "execution_connected": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["pipeline_manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path = tmp_path / "test-run_pipeline_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return path


def test_audit_cli_resolves_exact_csv_from_verified_pipeline_manifest(tmp_path, capsys):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    csv_path, _ = _write_snapshot(
        tmp_path, [start, start + timedelta(minutes=15), start + timedelta(minutes=30)]
    )
    pipeline_manifest = _write_pipeline_manifest(tmp_path, csv_path)
    output = tmp_path / "audit.json"

    exit_code = audit_main([
        "--pipeline-manifest", str(pipeline_manifest),
        "--output", str(output),
        "--json",
    ])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["status"] == "PASS"
    assert report["csv_path"] == str(csv_path.resolve())
    assert report["symbol"] == "EURUSD"
    assert report["timeframe"] == "M15"
    assert report["pipeline_manifest_sha256_verified"] is True
    assert report["pipeline_manifest_raw_ohlc_sha256_verified"] is True
    assert output.is_file()


def test_audit_cli_rejects_csv_that_differs_from_pipeline_manifest_snapshot_hash(tmp_path, capsys):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    recorded_csv, _ = _write_snapshot(
        tmp_path, [start, start + timedelta(minutes=15), start + timedelta(minutes=30)]
    )
    pipeline_manifest = _write_pipeline_manifest(tmp_path, recorded_csv)

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_csv, _ = _write_snapshot(
        other_dir, [start + timedelta(days=1), start + timedelta(days=1, minutes=15)]
    )

    exit_code = audit_main([
        "--csv", str(other_csv),
        "--pipeline-manifest", str(pipeline_manifest),
        "--json",
    ])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert report["status"] == "FAIL"
    assert "pipeline_manifest_raw_ohlc_sha256_mismatch" in report["errors"]
    assert report["pipeline_manifest_raw_ohlc_sha256_verified"] is False


def test_audit_cli_rejects_tampered_pipeline_manifest_fingerprint(tmp_path, capsys):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    csv_path, _ = _write_snapshot(tmp_path, [start, start + timedelta(minutes=15)])
    pipeline_manifest = _write_pipeline_manifest(tmp_path, csv_path)
    payload = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
    payload["symbol"] = "GBPUSD"
    pipeline_manifest.write_text(json.dumps(payload) + "\\n", encoding="utf-8")

    exit_code = audit_main([
        "--pipeline-manifest", str(pipeline_manifest),
        "--json",
    ])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert report["status"] == "FAIL"
    assert "pipeline_manifest_sha256_mismatch" in report["errors"]

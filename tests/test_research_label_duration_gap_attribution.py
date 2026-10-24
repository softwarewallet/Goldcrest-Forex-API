from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone

from app.research_label_duration_gap_attribution import audit_label_duration_gap_attribution


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path, break_horizon=False):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = []
    for index in range(9):
        point = start + timedelta(minutes=15 * index)
        if index >= 3:
            point += timedelta(minutes=30)
        timestamps.append(point)

    raw_path = tmp_path / "raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "symbol", "timeframe", "timestamp", "open", "high", "low", "close"
        ], lineterminator="\n")
        writer.writeheader()
        for point in timestamps:
            writer.writerow({
                "symbol": "EURUSD", "timeframe": "M15",
                "timestamp": point.isoformat().replace("+00:00", "Z"),
                "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
            })

    dataset_path = tmp_path / "dataset.csv"
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "symbol", "timeframe", "timestamp", "label_available_at"
        ], lineterminator="\n")
        writer.writeheader()
        for index in range(5):
            target = index + 4
            if break_horizon and index == 0:
                target -= 1
            writer.writerow({
                "symbol": "EURUSD", "timeframe": "M15",
                "timestamp": timestamps[index].isoformat().replace("+00:00", "Z"),
                "label_available_at": timestamps[target].isoformat().replace("+00:00", "Z"),
            })

    manifest = tmp_path / "baseline_pipeline_manifest.json"
    _write_json(manifest, {
        "scope": "GOLDCREST_OFFLINE_RESEARCH_PIPELINE",
        "run_id": "fixture-baseline", "symbol": "EURUSD", "timeframe": "M15",
        "label_horizon_bars": 4, "label_eligibility_mode": "all_labels_included",
        "max_label_elapsed_hours": None,
        "raw_ohlc_path": str(raw_path), "raw_ohlc_sha256": _sha256(raw_path),
        "raw_ohlc_audit_status": "PASS_WITH_WARNINGS",
        "dataset_path": str(dataset_path), "dataset_sha256": _sha256(dataset_path),
        "actionable": False, "execution_connected": False,
        "automatic_winner_selection": False,
    })

    profile = tmp_path / "duration_profile.json"
    _write_json(profile, {
        "scope": "GOLDCREST_LABEL_ELAPSED_DURATION_CAP_BOUNDARY_PROFILE",
        "status": "PASS_WITH_WARNINGS",
        "baseline_manifest_path": str(manifest),
        "baseline_dataset_path": str(dataset_path),
        "baseline_dataset_sha256": _sha256(dataset_path),
        "baseline_raw_ohlc_sha256": _sha256(raw_path),
        "caps_hours": [1.0, 1.5, 8.0],
        "duration_statistics": {"observation_count": 5},
        "warnings": [], "inferential": False,
        "automatic_model_selection": False, "actionable": False,
        "execution_connected": False,
    })
    return manifest, profile


def test_gap_attribution_reconciles_nominal_duration_and_gap_excess(tmp_path):
    manifest, profile = _fixture(tmp_path)
    result = audit_label_duration_gap_attribution(manifest, profile)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["dataset_rows"] == result["processed_rows"] == 5
    assert result["row_index_horizon_mismatches"] == 0
    assert result["rows_with_any_interbar_gap"] == 3
    assert result["rows_with_elapsed_duration_above_nominal"] == 3
    assert result["rows_with_duration_gap_excess_reconciled"] == 5
    assert result["aggregate"]["nominal_bar_hours_total"] == 5.0
    assert result["aggregate"]["elapsed_hours_total"] == 6.5
    assert result["aggregate"]["duration_excess_hours_total"] == 1.5
    assert result["aggregate"]["observed_gap_excess_hours_total"] == 1.5
    assert [row["row_count"] for row in result["duration_bucket_attribution"]] == [2, 3, 0, 0]


def test_gap_attribution_fails_on_horizon_mismatch(tmp_path):
    manifest, profile = _fixture(tmp_path, break_horizon=True)
    result = audit_label_duration_gap_attribution(manifest, profile)

    assert result["status"] == "FAIL"
    assert result["row_index_horizon_mismatches"] == 1
    assert any("dataset_label_horizon_index_mismatch_line_2" in item for item in result["errors"])
    assert result["actionable"] is False
    assert result["execution_connected"] is False


def test_gap_attribution_rejects_changed_raw_file_hash(tmp_path):
    manifest, profile = _fixture(tmp_path)
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text(raw_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    try:
        audit_label_duration_gap_attribution(manifest, profile)
    except ValueError as exc:
        assert "baseline_raw_ohlc_file_hash_mismatch" in str(exc)
    else:
        raise AssertionError("Expected changed raw OHLC SHA-256 to be rejected")

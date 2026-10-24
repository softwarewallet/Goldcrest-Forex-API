from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone

from app.research_label_gap_event_profile import audit_label_gap_event_profile


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path, break_horizon=False):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = []
    for index in range(14):
        point = start + timedelta(minutes=15 * index)
        if index >= 4:
            point += timedelta(minutes=30)
        if index >= 10:
            point += timedelta(hours=48)
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
        for index in range(len(timestamps) - 4):
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

    attribution = tmp_path / "duration_gap_attribution.json"
    _write_json(attribution, {
        "scope": "GOLDCREST_LABEL_DURATION_GAP_ATTRIBUTION_AUDIT",
        "status": "PASS_WITH_WARNINGS",
        "baseline_manifest_path": str(manifest),
        "baseline_raw_ohlc_path": str(raw_path),
        "baseline_raw_ohlc_sha256": _sha256(raw_path),
        "baseline_dataset_path": str(dataset_path),
        "baseline_dataset_sha256": _sha256(dataset_path),
        "caps_hours": [1.0, 1.5, 8.0],
        "rows_with_any_interbar_gap": 8,
        "rows_with_elapsed_duration_above_nominal": 8,
        "rows_with_duration_gap_excess_reconciled": 10,
        "warnings": [], "inferential": False,
        "automatic_model_selection": False, "actionable": False,
        "execution_connected": False,
    })
    return manifest, attribution


def test_event_profile_maps_each_gap_to_its_affected_label_windows(tmp_path):
    manifest, attribution = _fixture(tmp_path)
    result = audit_label_gap_event_profile(manifest, attribution)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["dataset_rows"] == result["valid_label_rows"] == 10
    assert result["raw_non_nominal_gap_event_count"] == 2
    assert result["gap_event_counts_by_excess_class"] == {
        "excess_up_to_1h": 1,
        "excess_over_1_to_8h": 0,
        "excess_over_8h": 1,
    }
    assert result["unique_label_rows_crossing_any_gap"] == 8
    assert result["label_gap_association_count"] == 8
    assert result["label_rows_crossing_multiple_gap_events"] == 0
    assert result["event_count_by_affected_label_count"] == {"4": 2}
    assert result["labels_affected_by_gap_class"] == {
        "excess_up_to_1h": 4,
        "excess_over_1_to_8h": 0,
        "excess_over_8h": 4,
    }
    assert all(item["matches"] for item in result["duration_gap_attribution_cross_checks"])


def test_event_profile_fails_closed_when_label_horizon_does_not_match(tmp_path):
    manifest, attribution = _fixture(tmp_path, break_horizon=True)
    result = audit_label_gap_event_profile(manifest, attribution)

    assert result["status"] == "FAIL"
    assert result["row_index_horizon_mismatches"] == 1
    assert any("dataset_label_horizon_index_mismatch_line_2" in item for item in result["errors"])
    assert result["actionable"] is False
    assert result["execution_connected"] is False


def test_event_profile_rejects_changed_dataset_hash(tmp_path):
    manifest, attribution = _fixture(tmp_path)
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(dataset_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    try:
        audit_label_gap_event_profile(manifest, attribution)
    except ValueError as exc:
        assert "baseline_dataset_file_hash_mismatch" in str(exc)
    else:
        raise AssertionError("Expected changed dataset SHA-256 to be rejected")

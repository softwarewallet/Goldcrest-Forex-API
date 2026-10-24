from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone

from app.research_label_eligibility_duration_profile import audit_duration_cap_boundaries


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path, *, reported_eligible=(2, 4, 4)):
    raw_path = tmp_path / "raw.csv"
    raw_path.write_text(
        "timestamp,open,high,low,close\n"
        "2025-01-01T00:00:00Z,1,1,1,1\n",
        encoding="utf-8",
    )

    durations = [0.5, 1.0, 1.25, 1.5, 8.1, 49.5]
    dataset_path = tmp_path / "baseline-dataset.csv"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "label_available_at"])
        writer.writeheader()
        for index, hours in enumerate(durations):
            timestamp = start + timedelta(hours=index * 2)
            available = timestamp + timedelta(hours=hours)
            writer.writerow({
                "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                "label_available_at": available.isoformat().replace("+00:00", "Z"),
            })

    baseline_path = _write_json(tmp_path / "baseline_pipeline_manifest.json", {
        "scope": "GOLDCREST_OFFLINE_RESEARCH_PIPELINE",
        "run_id": "fixture-baseline",
        "label_eligibility_mode": "all_labels_included",
        "max_label_elapsed_hours": None,
        "raw_ohlc_path": str(raw_path),
        "raw_ohlc_sha256": _sha256(raw_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "actionable": False,
        "execution_connected": False,
        "automatic_winner_selection": False,
    })

    audit_path = tmp_path / "threshold_eligibility_audit.json"
    caps = [1.0, 1.5, 8.0]
    profiles = []
    for cap, eligible in zip(caps, reported_eligible):
        profiles.append({
            "cap_hours": cap,
            "dataset_rows": len(durations),
            "eligible_rows": eligible,
            "ineligible_rows": len(durations) - eligible,
        })
    _write_json(audit_path, {
        "scope": "GOLDCREST_LABEL_ELIGIBILITY_THRESHOLD_PROFILE_AUDIT",
        "status": "PASS",
        "raw_ohlc_sha256": _sha256(raw_path),
        "caps_audited_hours": caps,
        "cap_profiles": profiles,
        "warnings": [],
        "errors": [],
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
    })
    return baseline_path, audit_path


def test_duration_profile_reconciles_cap_counts_and_empty_middle_bucket(tmp_path):
    baseline, audit = _fixture(tmp_path)

    result = audit_duration_cap_boundaries(baseline, audit)

    assert result["status"] == "PASS"
    assert result["duration_statistics"]["observation_count"] == 6
    assert [bucket["row_count"] for bucket in result["duration_buckets"]] == [2, 2, 0, 2]
    assert [check["computed_eligible_rows"] for check in result["cap_count_cross_checks"]] == [2, 4, 4]
    assert all(check["counts_match_existing_audit"] for check in result["cap_count_cross_checks"])
    assert result["counts_reconcile_to_dataset"] is True


def test_duration_profile_fails_if_reported_threshold_count_does_not_match_dataset(tmp_path):
    baseline, audit = _fixture(tmp_path, reported_eligible=(3, 4, 4))

    result = audit_duration_cap_boundaries(baseline, audit)

    assert result["status"] == "FAIL"
    assert "cap_1h_eligible_count_mismatch:3!=2" in result["errors"]


def test_duration_profile_rejects_changed_dataset_hash(tmp_path):
    baseline, audit = _fixture(tmp_path)
    baseline_data = json.loads(baseline.read_text(encoding="utf-8"))
    dataset_path = tmp_path / "baseline-dataset.csv"
    dataset_path.write_text(dataset_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _write_json(baseline, baseline_data)

    try:
        audit_duration_cap_boundaries(baseline, audit)
    except ValueError as exc:
        assert "baseline_dataset_file_hash_mismatch" in str(exc)
    else:
        raise AssertionError("Expected changed dataset SHA-256 to be rejected")

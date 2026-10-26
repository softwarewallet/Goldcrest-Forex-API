from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone

from app.research_label_gap_policy_equivalence_audit import audit_gap_policy_equivalence


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path, bad_threshold_count=False):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = []
    for index in range(14):
        point = start + timedelta(minutes=15 * index)
        if index >= 4:
            point += timedelta(minutes=30)
        if index >= 10:
            point += timedelta(hours=48)
        timestamps.append(point)

    raw = tmp_path / "raw.csv"
    with raw.open("w", encoding="utf-8", newline="") as handle:
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

    dataset = tmp_path / "dataset.csv"
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "symbol", "timeframe", "timestamp", "label_available_at"
        ], lineterminator="\n")
        writer.writeheader()
        for index in range(10):
            writer.writerow({
                "symbol": "EURUSD", "timeframe": "M15",
                "timestamp": timestamps[index].isoformat().replace("+00:00", "Z"),
                "label_available_at": timestamps[index + 4].isoformat().replace("+00:00", "Z"),
            })

    manifest = _json(tmp_path / "baseline_pipeline_manifest.json", {
        "scope": "GOLDCREST_OFFLINE_RESEARCH_PIPELINE",
        "run_id": "fixture-baseline", "symbol": "EURUSD", "timeframe": "M15",
        "label_horizon_bars": 4, "label_eligibility_mode": "all_labels_included",
        "max_label_elapsed_hours": None,
        "raw_ohlc_path": str(raw), "raw_ohlc_sha256": _sha(raw),
        "raw_ohlc_audit_status": "PASS_WITH_WARNINGS",
        "dataset_path": str(dataset), "dataset_sha256": _sha(dataset),
        "actionable": False, "execution_connected": False,
        "automatic_winner_selection": False,
    })

    profile = _json(tmp_path / "duration_profile.json", {
        "scope": "GOLDCREST_LABEL_ELAPSED_DURATION_CAP_BOUNDARY_PROFILE",
        "status": "PASS_WITH_WARNINGS", "baseline_manifest_path": str(manifest),
        "baseline_dataset_path": str(dataset), "baseline_dataset_sha256": _sha(dataset),
        "baseline_raw_ohlc_sha256": _sha(raw), "caps_hours": [1.0, 1.5, 8.0],
        "duration_buckets": [
            {"interval_hours": "<= 1", "row_count": 2},
            {"interval_hours": "(1, 1.5]", "row_count": 4},
            {"interval_hours": "(1.5, 8]", "row_count": 0},
            {"interval_hours": "> 8", "row_count": 4},
        ],
        "warnings": [], "inferential": False, "automatic_model_selection": False,
        "actionable": False, "execution_connected": False,
    })

    attribution = _json(tmp_path / "duration_gap_attribution.json", {
        "scope": "GOLDCREST_LABEL_DURATION_GAP_ATTRIBUTION_AUDIT",
        "status": "PASS_WITH_WARNINGS", "baseline_manifest_path": str(manifest),
        "baseline_raw_ohlc_path": str(raw), "baseline_raw_ohlc_sha256": _sha(raw),
        "baseline_dataset_path": str(dataset), "baseline_dataset_sha256": _sha(dataset),
        "rows_with_any_interbar_gap": 8, "rows_with_duration_gap_excess_reconciled": 10,
        "warnings": [], "inferential": False, "automatic_model_selection": False,
        "actionable": False, "execution_connected": False,
    })

    events = [
        {
            "from_utc": timestamps[3].isoformat().replace("+00:00", "Z"),
            "to_utc": timestamps[4].isoformat().replace("+00:00", "Z"),
            "interval_hours": 0.75, "gap_excess_class": "excess_up_to_1h",
            "affected_label_rows": [2, 3, 4, 5],
        },
        {
            "from_utc": timestamps[9].isoformat().replace("+00:00", "Z"),
            "to_utc": timestamps[10].isoformat().replace("+00:00", "Z"),
            "interval_hours": 48.25, "gap_excess_class": "excess_over_8h",
            "affected_label_rows": [8, 9, 10, 11],
        },
    ]
    event_profile = _json(tmp_path / "gap_event_profile.json", {
        "scope": "GOLDCREST_LABEL_GAP_EVENT_PROFILE", "status": "PASS_WITH_WARNINGS",
        "baseline_manifest_path": str(manifest), "baseline_dataset_path": str(dataset),
        "baseline_dataset_sha256": _sha(dataset), "baseline_raw_ohlc_sha256": _sha(raw),
        "duration_gap_attribution_report_sha256": _sha(attribution),
        "raw_non_nominal_gap_event_count": 2, "unique_label_rows_crossing_any_gap": 8,
        "label_gap_association_count": 8,
        "duration_bucket_counts_recomputed": {"<= 1": 2, "(1, 1.5]": 4, "> 8": 4},
        "gap_events": events, "warnings": [], "inferential": False,
        "automatic_model_selection": False, "actionable": False, "execution_connected": False,
    })

    threshold = _json(tmp_path / "threshold_audit.json", {
        "scope": "GOLDCREST_LABEL_ELIGIBILITY_THRESHOLD_PROFILE_AUDIT",
        "status": "PASS_WITH_WARNINGS", "raw_ohlc_sha256": _sha(raw),
        "caps_audited_hours": [1.0, 1.5, 8.0],
        "cap_profiles": [
            {"cap_hours": 1.0, "dataset_rows": 10, "eligible_rows": 2 if not bad_threshold_count else 3, "ineligible_rows": 8},
            {"cap_hours": 1.5, "dataset_rows": 10, "eligible_rows": 6, "ineligible_rows": 4},
            {"cap_hours": 8.0, "dataset_rows": 10, "eligible_rows": 6, "ineligible_rows": 4},
        ],
        "warnings": [], "inferential": False, "automatic_model_selection": False,
        "actionable": False, "execution_connected": False,
    })
    return manifest, profile, attribution, event_profile, threshold, dataset


def _run(parts):
    manifest, profile, attribution, events, threshold, _ = parts
    return audit_gap_policy_equivalence(manifest, profile, attribution, events, threshold)


def test_gap_policies_match_elapsed_caps_by_exact_row_mask(tmp_path):
    parts = _fixture(tmp_path)
    result = _run(parts)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["dataset_rows"] == result["processed_rows"] == 10
    assert result["gap_event_recomputation"]["computed_event_count"] == 2
    assert result["gap_event_recomputation"]["event_cross_check_failures"] == 0
    policies = {item["policy"]: item for item in result["candidate_policies"]}
    assert policies["baseline_all_labels"]["eligible_rows"] == 10
    assert policies["elapsed_cap_1h"]["eligible_rows"] == 2
    assert policies["exclude_any_non_nominal_gap"]["eligible_rows"] == 2
    assert policies["elapsed_cap_1_5h"]["eligible_rows"] == 6
    assert policies["elapsed_cap_8h"]["eligible_rows"] == 6
    assert policies["exclude_labels_crossing_raw_gap_over_8h"]["eligible_rows"] == 6
    assert all(item["same_eligible_mask"] for item in result["eligibility_mask_equivalence_checks"])
    assert all(item["counts_match"] for item in result["threshold_audit_count_cross_checks"])


def test_gap_policy_audit_fails_if_threshold_audit_counts_disagree(tmp_path):
    result = _run(_fixture(tmp_path, bad_threshold_count=True))

    assert result["status"] == "FAIL"
    assert "threshold_audit_count_mismatch:1h" in result["errors"]


def test_gap_policy_audit_rejects_changed_dataset_bytes(tmp_path):
    parts = _fixture(tmp_path)
    dataset = parts[-1]
    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    try:
        _run(parts)
    except ValueError as exc:
        assert "baseline_dataset_file_hash_mismatch" in str(exc)
    else:
        raise AssertionError("Expected changed dataset SHA-256 to be rejected")



def test_gap_policy_audit_rejects_malformed_event_row_membership(tmp_path):
    parts = _fixture(tmp_path)
    event_profile = json.loads(parts[3].read_text(encoding="utf-8"))
    event_profile["gap_events"][0]["affected_label_rows"] = [2, "3", 4, 5]
    _json(parts[3], event_profile)

    result = _run(parts)

    assert result["status"] == "FAIL"
    assert any(error.startswith("gap_event_exposure_mismatch:") for error in result["errors"])


def test_gap_policy_audit_rejects_malformed_warning_schema(tmp_path):
    parts = _fixture(tmp_path)
    profile = json.loads(parts[1].read_text(encoding="utf-8"))
    profile["warnings"] = "not-a-list"
    _json(parts[1], profile)

    try:
        _run(parts)
    except ValueError as exc:
        assert "duration_profile_warnings_schema_invalid" in str(exc)
    else:
        raise AssertionError("Expected malformed warning schema to be rejected")


def test_gap_policy_audit_rejects_pass_status_with_errors(tmp_path):
    parts = _fixture(tmp_path)
    profile = json.loads(parts[1].read_text(encoding="utf-8"))
    profile["errors"] = ["unresolved_integrity_error"]
    _json(parts[1], profile)

    try:
        _run(parts)
    except ValueError as exc:
        assert "duration_profile_reports_errors_despite_pass_status" in str(exc)
    else:
        raise AssertionError("Expected contradictory report status to be rejected")


@pytest.mark.parametrize("status", [["PASS"], {"status": "PASS"}, None, 7])
def test_gap_policy_status_checker_rejects_malformed_status(status):
    with pytest.raises(ValueError, match="status_not_pass"):
        gap_audit._check_status({"status": status}, "fixture")

from __future__ import annotations

import json

from app.research_label_eligibility_threshold_report_audit import audit_threshold_report


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _threshold_fixture(tmp_path, profiles):
    raw_hash = "a" * 64
    seeds = [20261021, 20261022]
    caps = []
    for cap, eligible, ineligible, fold_rows in profiles:
        seed_results = []
        for seed in seeds:
            manifest_path = tmp_path / f"cap-{cap}-seed-{seed}-pipeline.json"
            _write_json(manifest_path, {
                "max_label_elapsed_hours": cap,
                "raw_ohlc_sha256": raw_hash,
                "label_eligibility_metadata": {
                    "mode": "elapsed_time_cap",
                    "max_label_elapsed_hours": cap,
                    "eligible_rows": eligible,
                    "ineligible_rows": ineligible,
                },
            })
            seed_results.append({
                "seed": seed,
                "status": "PASS",
                "capped_replay_manifest_path": str(manifest_path),
            })
        seed_sweep_path = tmp_path / f"cap-{cap}-seed-sweep.json"
        _write_json(seed_sweep_path, {
            "status": "PASS",
            "max_label_elapsed_hours": cap,
            "requested_seeds": seeds,
            "raw_ohlc_sha256": raw_hash,
            "seed_results": seed_results,
        })
        caps.append({
            "max_label_elapsed_hours": cap,
            "status": "PASS",
            "seed_sweep_report_path": str(seed_sweep_path),
            "fold_signature": [
                [0, "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z", fold_rows],
            ],
            "fold_window_signature": [
                [0, "2026-01-01T00:00:00Z", "2026-01-07T00:00:00Z"],
            ],
            "metrics": [{
                "model_type": "xgboost",
                "feature_variant": "all_features",
                "metrics": {"log_loss": {"mean_delta": 0.001}},
            }],
        })
    threshold_path = tmp_path / "eurusd_threshold_sweep_report.json"
    _write_json(threshold_path, {
        "schema_version": 1,
        "scope": "GOLDCREST_LABEL_ELIGIBILITY_THRESHOLD_SWEEP",
        "status": "PASS",
        "raw_ohlc_sha256": raw_hash,
        "seeds": seeds,
        "requested_caps_hours": [profile[0] for profile in profiles],
        "cap_results": caps,
        "warnings": [],
        "errors": [],
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
    })
    return threshold_path


def test_audit_identifies_caps_with_identical_eligibility_without_retraining(tmp_path):
    path = _threshold_fixture(tmp_path, [
        (1.5, 800, 200, 80),
        (8.0, 800, 200, 80),
    ])

    result = audit_threshold_report(path)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["cap_profiles"][0]["dataset_rows"] == 1000
    assert result["cap_profiles"][0]["eligible_fraction"] == 0.8
    pair = result["adjacent_cap_comparisons"][0]
    assert pair["same_dataset_eligibility_profile"] is True
    assert pair["eligible_rows_delta"] == 0
    assert pair["metric_summary_identical"] is True
    assert "caps_1.5h_and_8h_have_identical_eligibility_profiles" in result["warnings"]


def test_audit_reports_added_eligible_rows_between_caps(tmp_path):
    path = _threshold_fixture(tmp_path, [
        (1.5, 800, 200, 80),
        (8.0, 875, 125, 87),
    ])

    result = audit_threshold_report(path)

    # Identical metric summaries despite changed eligibility counts are flagged for review.
    assert result["status"] == "PASS_WITH_WARNINGS"
    pair = result["adjacent_cap_comparisons"][0]
    assert pair["same_dataset_eligibility_profile"] is False
    assert any("identical_aggregate_metric_summaries_despite_different_eligibility_profiles" in warning for warning in result["warnings"])
    assert pair["eligible_rows_delta"] == 75
    assert pair["ineligible_rows_delta"] == -75
    assert pair["eligible_test_rows_fold_deltas"][0]["eligible_test_rows_delta"] == 7


def test_audit_fails_when_eligibility_is_not_monotonic_by_cap(tmp_path):
    path = _threshold_fixture(tmp_path, [
        (1.5, 875, 125, 87),
        (8.0, 800, 200, 80),
    ])

    result = audit_threshold_report(path)

    assert result["status"] == "FAIL"
    assert any("eligibility_counts_not_monotonic" in error for error in result["errors"])


def test_audit_rejects_non_threshold_sweep_input(tmp_path):
    path = tmp_path / "not-a-threshold-report.json"
    _write_json(path, {"scope": "OTHER"})

    result_raised = False
    try:
        audit_threshold_report(path)
    except ValueError as exc:
        result_raised = "not_a_label_eligibility_threshold_sweep_report" in str(exc)
    assert result_raised



def test_threshold_audit_rejects_malformed_source_warning_schema(tmp_path):
    from pathlib import Path
    import pytest

    source = Path(_threshold_fixture(tmp_path, [
        (1.0, 10, 0, 5),
        (1.5, 10, 0, 5),
        (8.0, 10, 0, 5),
    ]))
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["warnings"] = "not-a-list"
    source.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = audit_threshold_report(source)

    assert result["status"] == "FAIL"
    assert "source_threshold_sweep_warnings_schema_invalid" in result["errors"]


def test_threshold_audit_rejects_seed_report_with_errors_despite_pass_status(tmp_path):
    from pathlib import Path
    import pytest

    source = Path(_threshold_fixture(tmp_path, [
        (1.0, 10, 0, 5),
        (1.5, 10, 0, 5),
        (8.0, 10, 0, 5),
    ]))
    payload = json.loads(source.read_text(encoding="utf-8"))
    child_path = Path(payload["cap_results"][0]["seed_sweep_report_path"])
    child = json.loads(child_path.read_text(encoding="utf-8"))
    child["errors"] = ["contradictory_error"]
    child_path.write_text(json.dumps(child, sort_keys=True) + "\n", encoding="utf-8")

    result = audit_threshold_report(source)

    assert result["status"] == "FAIL"
    assert any("reports_errors_despite_pass_status" in error for error in result["errors"])

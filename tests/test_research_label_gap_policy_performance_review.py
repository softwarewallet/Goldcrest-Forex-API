from __future__ import annotations

import hashlib
import json

import pytest

from app.research_label_gap_policy_performance_review import review_gap_policy_performance


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _metric_summary(mean_delta, minimum, maximum):
    return [{
        "model_type": "XGBoost",
        "feature_variant": "core",
        "metrics": {
            "log_loss": {
                "mean_delta": mean_delta,
                "median_delta": mean_delta,
                "minimum_delta": minimum,
                "maximum_delta": maximum,
                "favorable_seed_count": 0 if mean_delta > 0 else 1,
                "adverse_seed_count": 2 if mean_delta > 0 else 0,
                "neutral_seed_count": 0,
            }
        },
    }]


def _seed_report(path, cap, capped_values):
    results = []
    for seed, baseline, capped in zip((11, 22), (0.70, 0.71), capped_values):
        results.append({
            "seed": seed,
            "status": "PASS",
            "experiments": {
                "XGBoost/core": {
                    "model_type": "XGBoost",
                    "feature_variant": "core",
                    "metrics": {
                        "log_loss": {
                            "baseline": baseline,
                            "capped": capped,
                            "delta": capped - baseline,
                        }
                    },
                }
            },
        })
    _write(path, {
        "scope": "GOLDCREST_LABEL_ELIGIBILITY_MULTI_SEED_SENSITIVITY",
        "status": "PASS_WITH_WARNINGS",
        "max_label_elapsed_hours": cap,
        "requested_seeds": [11, 22],
        "successful_seed_count": 2,
        "failed_seed_count": 0,
        "raw_ohlc_sha256": "a" * 64,
        "seed_results": results,
        "warnings": [],
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
    })
    return path


def _fixture(tmp_path, bad_threshold_count=False):
    child_paths = {
        1.0: _seed_report(tmp_path / "seed_cap_1.json", 1.0, (0.72, 0.74)),
        1.5: _seed_report(tmp_path / "seed_cap_1p5.json", 1.5, (0.69, 0.71)),
        8.0: _seed_report(tmp_path / "seed_cap_8.json", 8.0, (0.69, 0.71)),
    }
    summary = {
        1.0: _metric_summary(0.025, 0.02, 0.03),
        1.5: _metric_summary(-0.005, -0.01, 0.0),
        8.0: _metric_summary(-0.005, -0.01, 0.0),
    }
    cap_results = []
    cap_profiles = []
    for cap in (1.0, 1.5, 8.0):
        eligible = 6 if cap == 1.0 else 8
        reported_eligible = eligible + int(bad_threshold_count and cap == 1.0)
        cap_results.append({
            "max_label_elapsed_hours": cap,
            "status": "PASS_WITH_WARNINGS",
            "seed_sweep_report_path": str(child_paths[cap]),
            "raw_ohlc_sha256": "a" * 64,
            "metrics": summary[cap],
            "inferential": False,
            "automatic_model_selection": False,
            "actionable": False,
            "execution_connected": False,
        })
        cap_profiles.append({
            "cap_hours": cap,
            "status": "PASS_WITH_WARNINGS",
            "dataset_rows": 10,
            "eligible_rows": reported_eligible,
            "ineligible_rows": 10 - reported_eligible,
            "metric_summary": summary[cap],
            "seed_count": 2,
        })

    sweep_path = _write(tmp_path / "threshold_sweep.json", {
        "scope": "GOLDCREST_LABEL_ELIGIBILITY_THRESHOLD_SWEEP",
        "status": "PASS_WITH_WARNINGS",
        "seeds": [11, 22],
        "raw_ohlc_sha256": "a" * 64,
        "same_raw_ohlc_snapshot_across_caps": True,
        "same_fold_windows_across_caps": True,
        "cap_results": cap_results,
        "warnings": [],
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
    })
    audit_path = _write(tmp_path / "threshold_audit.json", {
        "scope": "GOLDCREST_LABEL_ELIGIBILITY_THRESHOLD_PROFILE_AUDIT",
        "status": "PASS_WITH_WARNINGS",
        "raw_ohlc_sha256": "a" * 64,
        "threshold_sweep_report_path": str(sweep_path),
        "threshold_sweep_report_sha256": _sha(sweep_path),
        "requested_seeds": [11, 22],
        "caps_audited_hours": [1.0, 1.5, 8.0],
        "cap_profiles": cap_profiles,
        "warnings": [],
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
    })
    empty_hash = hashlib.sha256(b"\n").hexdigest()
    policy_path = _write(tmp_path / "gap_policy_audit.json", {
        "scope": "GOLDCREST_LABEL_GAP_POLICY_EQUIVALENCE_AUDIT",
        "status": "PASS_WITH_WARNINGS",
        "baseline_raw_ohlc_sha256": "a" * 64,
        "baseline_dataset_sha256": "b" * 64,
        "dataset_rows": 10,
        "input_reports": {
            "threshold_audit": {"path": str(audit_path), "sha256": _sha(audit_path)}
        },
        "candidate_policies": [
            {
                "policy": "baseline_all_labels", "dataset_rows": 10,
                "eligible_rows": 10, "ineligible_rows": 0, "eligible_fraction": 1.0,
                "eligible_row_numbers_sha256": "1" * 64, "ineligible_row_numbers_sha256": empty_hash,
            },
            {
                "policy": "elapsed_cap_1h", "dataset_rows": 10,
                "eligible_rows": 6, "ineligible_rows": 4, "eligible_fraction": 0.6,
                "eligible_row_numbers_sha256": "2" * 64, "ineligible_row_numbers_sha256": "3" * 64,
            },
            {
                "policy": "exclude_any_non_nominal_gap", "dataset_rows": 10,
                "eligible_rows": 6, "ineligible_rows": 4, "eligible_fraction": 0.6,
                "eligible_row_numbers_sha256": "2" * 64, "ineligible_row_numbers_sha256": "3" * 64,
            },
            {
                "policy": "elapsed_cap_1_5h", "dataset_rows": 10,
                "eligible_rows": 8, "ineligible_rows": 2, "eligible_fraction": 0.8,
                "eligible_row_numbers_sha256": "4" * 64, "ineligible_row_numbers_sha256": "5" * 64,
            },
            {
                "policy": "elapsed_cap_8h", "dataset_rows": 10,
                "eligible_rows": 8, "ineligible_rows": 2, "eligible_fraction": 0.8,
                "eligible_row_numbers_sha256": "4" * 64, "ineligible_row_numbers_sha256": "5" * 64,
            },
            {
                "policy": "exclude_labels_crossing_raw_gap_over_8h", "dataset_rows": 10,
                "eligible_rows": 8, "ineligible_rows": 2, "eligible_fraction": 0.8,
                "eligible_row_numbers_sha256": "4" * 64, "ineligible_row_numbers_sha256": "5" * 64,
            },
        ],
        "eligibility_mask_equivalence_checks": [
            {"left_policy": "elapsed_cap_1h", "right_policy": "exclude_any_non_nominal_gap",
             "same_eligible_mask": True, "eligible_row_symmetric_difference_count": 0},
            {"left_policy": "elapsed_cap_1_5h", "right_policy": "elapsed_cap_8h",
             "same_eligible_mask": True, "eligible_row_symmetric_difference_count": 0},
            {"left_policy": "elapsed_cap_1_5h", "right_policy": "exclude_labels_crossing_raw_gap_over_8h",
             "same_eligible_mask": True, "eligible_row_symmetric_difference_count": 0},
            {"left_policy": "elapsed_cap_8h", "right_policy": "exclude_labels_crossing_raw_gap_over_8h",
             "same_eligible_mask": True, "eligible_row_symmetric_difference_count": 0},
        ],
        "warnings": [], "errors": [],
        "inferential": False, "automatic_model_selection": False,
        "actionable": False, "execution_connected": False,
    })
    return policy_path, audit_path, sweep_path


def test_review_groups_existing_metrics_by_unique_masks_without_retraining(tmp_path):
    policy, audit, _ = _fixture(tmp_path)

    result = review_gap_policy_performance(policy, audit)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["dataset_rows"] == 10
    assert result["unique_mask_count_in_policy_audit"] == 3
    assert result["unique_capped_mask_count_in_existing_sweep"] == 2
    assert result["uncapped_baseline_reference_metrics_match_across_caps"] is True
    assert result["equivalent_cap_metric_checks"] == [{
        "cap_hours": [1.5, 8.0],
        "same_eligible_row_mask": True,
        "aggregate_metric_summaries_identical": True,
        "per_seed_results_identical": True,
    }]
    paired = result["paired_policy_metric_differences_1_5h_minus_1h"]
    assert len(paired) == 1
    assert paired[0]["mean_difference_of_paired_deltas"] == -0.03
    assert paired[0]["mean_capped_metric_difference"] == -0.03
    assert len(paired[0]["per_seed"]) == 2
    assert all(group["metrics_available"] for group in result["unique_capped_mask_groups"])


def test_review_fails_if_threshold_audit_counts_disagree_with_exact_mask_report(tmp_path):
    policy, audit, _ = _fixture(tmp_path, bad_threshold_count=True)

    with pytest.raises(ValueError, match="cap_1h_eligible_row_count_mismatch"):
        review_gap_policy_performance(policy, audit)


def test_review_rejects_a_changed_threshold_sweep_report(tmp_path):
    policy, audit, sweep = _fixture(tmp_path)
    sweep.write_text(sweep.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="threshold_sweep_report_hash_mismatch"):
        review_gap_policy_performance(policy, audit)

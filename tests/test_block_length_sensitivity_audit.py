import pytest
from app.block_length_sensitivity_audit import build_block_length_sensitivity_audit


def comparison():
    entries = [
        {"block_length": 2, "brier_difference_ci_low": -0.04, "brier_difference_ci_high": -0.01,
         "log_loss_difference_ci_low": -0.08, "log_loss_difference_ci_high": -0.02,
         "brier_direction": "CANDIDATE_LOWER", "log_loss_direction": "CANDIDATE_LOWER"},
        {"block_length": 4, "brier_difference_ci_low": -0.03, "brier_difference_ci_high": -0.005,
         "log_loss_difference_ci_low": -0.07, "log_loss_difference_ci_high": -0.01,
         "brier_direction": "CANDIDATE_LOWER", "log_loss_direction": "CANDIDATE_LOWER"},
        {"block_length": 8, "brier_difference_ci_low": -0.02, "brier_difference_ci_high": 0.01,
         "log_loss_difference_ci_low": -0.06, "log_loss_difference_ci_high": 0.02,
         "brier_direction": "INCLUDES_ZERO", "log_loss_direction": "INCLUDES_ZERO"},
    ]
    return {"candidate_name": "candidate", "baseline_name": "baseline", "fold_index": 0,
        "partition": "test", "sample_count": 50, "actionable": False,
        "comparison": {"block_length": 4, "brier_difference_ci_low": -0.03,
            "brier_difference_ci_high": -0.005, "log_loss_difference_ci_low": -0.07,
            "log_loss_difference_ci_high": -0.01, "block_length_sensitivity": entries}}


def test_audit_flags_interval_direction_changes():
    audit = build_block_length_sensitivity_audit([comparison()])
    assert audit["sensitive_count"] == 1
    assert audit["comparisons"][0]["status"] == "SENSITIVE_TO_BLOCK_LENGTH_CHOICE"
    assert audit["comparisons"][0]["tested_block_lengths"] == [2, 4, 8]


def test_audit_flags_stability_only_across_tested_lengths():
    item = comparison()
    for entry in item["comparison"]["block_length_sensitivity"]:
        entry["brier_difference_ci_high"] = -0.001
        entry["brier_direction"] = "CANDIDATE_LOWER"
        entry["log_loss_difference_ci_high"] = -0.001
        entry["log_loss_direction"] = "CANDIDATE_LOWER"
    item["comparison"]["brier_difference_ci_low"] = -0.03
    item["comparison"]["brier_difference_ci_high"] = -0.005
    item["comparison"]["log_loss_difference_ci_low"] = -0.07
    item["comparison"]["log_loss_difference_ci_high"] = -0.01
    assert build_block_length_sensitivity_audit([item])["direction_stable_count"] == 1


def test_audit_rejects_direction_mismatch():
    item = comparison()
    item["comparison"]["block_length_sensitivity"][0]["brier_direction"] = "CANDIDATE_HIGHER"
    with pytest.raises(ValueError, match="direction does not match"):
        build_block_length_sensitivity_audit([item])


def test_audit_rejects_primary_interval_mismatch():
    item = comparison()
    item["comparison"]["brier_difference_ci_low"] = -0.04
    with pytest.raises(ValueError, match="primary interval does not match"):
        build_block_length_sensitivity_audit([item])


def test_legacy_comparisons_remain_readable():
    audit = build_block_length_sensitivity_audit([{"candidate_name": "legacy", "comparison": {"block_length": 2}}])
    assert audit["comparison_count"] == 0
    assert audit["actionable"] is False

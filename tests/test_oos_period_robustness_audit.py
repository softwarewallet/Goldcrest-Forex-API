from copy import deepcopy

import pytest

from app.oos_period_robustness_audit import build_oos_period_robustness_audit


def fold(index, brier=0.2, log_loss=0.5, test_count=10):
    return {
        "fold_index": index,
        "validation": {"sample_count": test_count, "brier_score": 0.2, "log_loss": 0.5},
        "test": {
            "sample_count": test_count,
            "brier_score": brier,
            "log_loss": log_loss,
            "accuracy_at_fixed_half": 0.6,
            "expected_calibration_error": 0.1,
        },
        "actionable": False,
    }


def test_audit_reports_chronological_periods_without_model_selection():
    results = {
        "candidate": [fold(0, 0.1, 0.3), fold(1, 0.2, 0.5), fold(2, 0.3, 0.7), fold(3, 0.4, 0.9)],
        "training_prevalence": [fold(0), fold(1), fold(2), fold(3)],
    }
    audit = build_oos_period_robustness_audit(
        results, [{"candidate_name": "candidate", "baseline_name": "training_prevalence"}]
    )
    assert audit["status"] == "PERIOD_DIAGNOSTIC_AVAILABLE"
    assert audit["common_test_fold_indices"] == [0, 1, 2, 3]
    candidate = audit["candidates"][0]
    assert [period["period"] for period in candidate["periods"]] == ["EARLY", "MIDDLE", "LATE"]
    assert [period["fold_indices"] for period in candidate["periods"]] == [[0, 1], [2], [3]]
    assert candidate["test_brier_range"] == pytest.approx(0.3)
    assert candidate["test_expected_calibration_error_range"] == pytest.approx(0.0)
    assert candidate["expected_calibration_error_fold_count"] == 4
    assert candidate["periods"][0]["ece_fold_count"] == 2
    assert candidate["periods"][0]["median_test_expected_calibration_error"] == pytest.approx(0.1)
    assert candidate["periods"][0]["maximum_test_expected_calibration_error"] == pytest.approx(0.1)
    assert audit["model_selection_performed"] is False
    assert audit["test_metrics_used_for_selection"] is False
    assert audit["actionable"] is False


def test_audit_withholds_period_comparison_when_fewer_than_three_common_folds():
    results = {"candidate": [fold(0), fold(1)], "training_prevalence": [fold(0), fold(1)]}
    audit = build_oos_period_robustness_audit(results, [])
    assert audit["status"] == "INSUFFICIENT_COMMON_TEST_FOLDS"
    assert audit["candidates"] == []
    assert audit["period_group_count"] == 0


def test_audit_uses_only_common_complete_test_folds():
    results = {
        "a": [fold(0), fold(1), fold(2), fold(3)],
        "b": [fold(0), fold(1, brier=float("nan")), fold(2), fold(3)],
    }
    audit = build_oos_period_robustness_audit(results, [])
    assert audit["common_test_fold_indices"] == [0, 2, 3]


def test_audit_ignores_validation_metrics_and_is_deterministic():
    results = {"candidate": [fold(0), fold(1), fold(2)]}
    audit = build_oos_period_robustness_audit(results, [])
    changed = deepcopy(results)
    for row in changed["candidate"]:
        row["validation"]["brier_score"] = -999
        row["validation"]["log_loss"] = -999
    assert build_oos_period_robustness_audit(changed, []) == audit


def test_audit_verifies_timestamps_match_chronological_fold_order():
    results = {
        "candidate": [
            fold(0), fold(1), fold(2),
        ],
    }
    starts = [
        "2026-04-01T00:00:00Z",
        "2026-04-02T00:00:00Z",
        "2026-04-03T00:00:00Z",
    ]
    for row, start in zip(results["candidate"], starts):
        row["test_start_utc"] = start

    audit = build_oos_period_robustness_audit(results, [])

    assert audit["status"] == "PERIOD_DIAGNOSTIC_AVAILABLE"
    assert audit["chronology_integrity"]["status"] == "VERIFIED_TEST_START_ORDER"
    assert audit["chronology_integrity"]["observed_candidate_fold_timestamps"] == 3
    assert audit["chronology_integrity"]["fold_test_starts"][0]["test_start_utc"] == starts[0]


def test_audit_withholds_periods_when_timestamps_contradict_fold_order():
    results = {
        "candidate": [
            fold(0), fold(1), fold(2),
        ],
    }
    starts = [
        "2026-04-02T00:00:00Z",
        "2026-04-01T00:00:00Z",
        "2026-04-03T00:00:00Z",
    ]
    for row, start in zip(results["candidate"], starts):
        row["test_start_utc"] = start

    audit = build_oos_period_robustness_audit(results, [])

    assert audit["status"] == "CHRONOLOGY_INVALID"
    assert audit["chronology_integrity"]["status"] == "FOLD_ORDER_TIMESTAMP_MISMATCH"
    assert audit["candidates"] == []
    assert audit["period_group_count"] == 0


def test_audit_withholds_periods_when_candidates_disagree_on_fold_windows():
    results = {
        "candidate-a": [fold(0), fold(1), fold(2)],
        "candidate-b": [fold(0), fold(1), fold(2)],
    }
    starts = [
        "2026-04-01T00:00:00Z",
        "2026-04-02T00:00:00Z",
        "2026-04-03T00:00:00Z",
    ]
    for candidate_rows in results.values():
        for row, start in zip(candidate_rows, starts):
            row["test_start_utc"] = start
    results["candidate-b"][1]["test_start_utc"] = "2026-04-05T00:00:00Z"

    audit = build_oos_period_robustness_audit(results, [])

    assert audit["status"] == "CHRONOLOGY_INVALID"
    assert audit["chronology_integrity"]["status"] == "CROSS_CANDIDATE_WINDOW_MISMATCH"
    assert audit["candidates"] == []


def test_audit_rejects_out_of_domain_test_metrics():
    results = {
        "candidate": [
            fold(0, brier=0.2),
            fold(1, brier=1.2),
            fold(2, brier=0.3),
            fold(3, brier=0.4, log_loss=-0.2),
            fold(4, brier=0.5),
        ],
    }

    audit = build_oos_period_robustness_audit(results, [])

    assert audit["common_test_fold_indices"] == [0, 2, 4]
    assert audit["status"] == "PERIOD_DIAGNOSTIC_AVAILABLE"


def test_audit_rejects_invalid_minimum():
    with pytest.raises(ValueError, match="at least 3"):
        build_oos_period_robustness_audit({}, [], minimum_test_folds=2)

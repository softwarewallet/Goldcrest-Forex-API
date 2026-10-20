from copy import deepcopy

from app.model_selection_audit import build_model_selection_audit


def fold(index, validation_log_loss, validation_brier, test_log_loss=0.9, test_brier=0.4):
    return {
        "fold_index": index,
        "validation": {
            "sample_count": 20,
            "log_loss": validation_log_loss,
            "brier_score": validation_brier,
        },
        "test": {
            "sample_count": 20,
            "log_loss": test_log_loss,
            "brier_score": test_brier,
        },
        "actionable": False,
    }


def test_selection_uses_validation_only_and_deterministic_tie_breaks():
    results = {
        "model_a": [fold(0, 0.50, 0.20), fold(1, 0.55, 0.21)],
        "model_b": [fold(0, 0.45, 0.22), fold(1, 0.48, 0.23)],
        "training_prevalence": [fold(0, 0.60, 0.25), fold(1, 0.62, 0.26)],
    }
    comparisons = [{"candidate_name": name, "baseline_name": "training_prevalence", "partition": "validation"}
                   for name in ("model_a", "model_b")]
    audit = build_model_selection_audit(results, comparisons)
    assert audit["selected_candidate"] == "model_b"
    assert [row["candidate_name"] for row in audit["ranking"]] == ["model_b", "model_a"]
    assert audit["common_validation_fold_indices"] == [0, 1]
    assert audit["test_metrics_used_for_selection"] is False
    assert audit["actionable"] is False

    changed_test = deepcopy(results)
    for folds in changed_test.values():
        for item in folds:
            item["test"]["log_loss"] = -1000
            item["test"]["brier_score"] = -1000
    assert build_model_selection_audit(changed_test, comparisons) == audit


def test_selection_is_withheld_when_too_few_common_validation_folds():
    results = {
        "model_a": [fold(0, 0.5, 0.2)],
        "training_prevalence": [fold(0, 0.6, 0.3)],
    }
    audit = build_model_selection_audit(
        results, [{"candidate_name": "model_a", "baseline_name": "training_prevalence"}]
    )
    assert audit["selected_candidate"] is None
    assert audit["ranking"] == []
    assert audit["status"] == "INSUFFICIENT_COMMON_VALIDATION_FOLDS"


def test_selection_uses_only_complete_common_folds_for_all_candidates():
    results = {
        "model_a": [fold(0, 0.5, 0.2), fold(1, 0.4, 0.2), fold(2, 0.6, 0.3)],
        "model_b": [fold(0, 0.4, 0.2), fold(1, float("nan"), 0.1), fold(2, 0.5, 0.25)],
    }
    audit = build_model_selection_audit(results, [])
    assert audit["common_validation_fold_indices"] == [0, 2]
    assert audit["selected_candidate"] == "model_b"


def test_selection_rejects_invalid_minimum_fold_count():
    import pytest
    with pytest.raises(ValueError, match="at least 2"):
        build_model_selection_audit({}, [], minimum_validation_folds=1)

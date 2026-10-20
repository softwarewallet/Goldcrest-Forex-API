import pytest

from app.fold_stability import summarize_fold_stability


def _fold(index, brier, log_loss, count=10):
    return {
        "fold_index": index,
        "validation": {"sample_count": count, "brier_score": brier, "log_loss": log_loss},
        "test": {"sample_count": count, "brier_score": brier, "log_loss": log_loss},
        "actionable": False,
    }


def test_stability_reports_fold_direction_and_median_without_pooling():
    result = summarize_fold_stability(
        {
            "candidate": [_fold(0, 0.20, 0.50), _fold(1, 0.24, 0.65), _fold(2, 0.22, 0.60)],
            "baseline": [_fold(0, 0.25, 0.70), _fold(1, 0.23, 0.60), _fold(2, 0.25, 0.75)],
        },
        [{"candidate_name": "candidate", "baseline_name": "baseline", "partition": "test"}],
    )
    assert len(result) == 1
    row = result[0]
    assert row["fold_indices"] == [0, 1, 2]
    assert row["fold_count"] == 3
    assert row["actionable"] is False
    assert row["metrics"]["brier_score"]["candidate_wins"] == 2
    assert row["metrics"]["brier_score"]["candidate_losses"] == 1
    assert row["metrics"]["brier_score"]["median_candidate_minus_baseline"] == pytest.approx(-0.03)
    assert row["metrics"]["log_loss"]["candidate_wins"] == 2
    assert row["metrics"]["log_loss"]["candidate_losses"] == 1


def test_stability_uses_only_common_folds_and_preserves_partition():
    result = summarize_fold_stability(
        {"candidate": [_fold(0, 0.2, 0.4), _fold(1, 0.1, 0.3)],
         "baseline": [_fold(0, 0.3, 0.5)]},
        [{"candidate_name": "candidate", "baseline_name": "baseline", "partition": "validation"}],
    )
    assert result[0]["partition"] == "validation"
    assert result[0]["fold_indices"] == [0]
    assert result[0]["metrics"]["brier_score"]["candidate_wins"] == 1


def test_stability_fails_closed_when_common_fold_sample_counts_differ():
    result = None
    with pytest.raises(ValueError, match="sample counts differ"):
        summarize_fold_stability(
            {"candidate": [_fold(0, 0.2, 0.4, count=10)],
             "baseline": [_fold(0, 0.3, 0.5, count=11)]},
            [{"candidate_name": "candidate", "baseline_name": "baseline", "partition": "test"}],
        )


def test_stability_returns_empty_without_paired_model_definitions():
    assert summarize_fold_stability({"candidate": [_fold(0, 0.2, 0.4)]}, []) == []

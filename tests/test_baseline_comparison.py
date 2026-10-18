import pytest

from app.baseline_comparison import (
    compare_candidate_to_baseline,
    summarize_returns,
)


def test_summary_reports_sample_counts_and_zero_returns():
    result = summarize_returns([0.2, -0.1, 0.0, 0.1])
    assert result.count == 4
    assert result.total_net_price_units == pytest.approx(0.2)
    assert result.mean_net_price_units == pytest.approx(0.05)
    assert result.net_win_rate == pytest.approx(0.5)
    assert result.positive_count == 2
    assert result.negative_count == 1
    assert result.zero_count == 1


def test_candidate_baseline_comparison_is_deterministic_and_paired():
    candidate = [0.3, 0.2, -0.1, 0.2, 0.1]
    baseline = [0.1, 0.1, 0.0, 0.0, 0.0]
    first = compare_candidate_to_baseline(
        candidate, baseline, bootstrap_resamples=1000, random_seed=7
    )
    second = compare_candidate_to_baseline(
        candidate, baseline, bootstrap_resamples=1000, random_seed=7
    )
    assert first == second
    assert first.paired_difference_mean == pytest.approx(0.1)
    assert first.paired_difference_ci_low <= first.paired_difference_ci_high
    assert first.candidate.count == first.baseline.count == 5


def test_comparison_rejects_misaligned_or_invalid_samples():
    with pytest.raises(ValueError, match="equal length"):
        compare_candidate_to_baseline([0.1, 0.2], [0.1])
    with pytest.raises(ValueError, match="finite"):
        summarize_returns([0.1, float("nan")])
    with pytest.raises(ValueError, match=">= 100"):
        compare_candidate_to_baseline([0.1], [0.0], bootstrap_resamples=99)


def test_identical_series_has_zero_difference():
    result = compare_candidate_to_baseline(
        [0.1, -0.1, 0.0], [0.1, -0.1, 0.0],
        bootstrap_resamples=500,
    )
    assert result.paired_difference_mean == 0
    assert result.paired_difference_ci_low == 0
    assert result.paired_difference_ci_high == 0
    assert result.interpretation == "UNCERTAINTY_INTERVAL_INCLUDES_ZERO_NO_CLEAR_DIFFERENCE"

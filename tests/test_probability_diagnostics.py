import pytest

from app.probability_diagnostics import compare_probability_models, probability_diagnostics


def test_diagnostics_compute_scores_and_reliability_bins():
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.4, 0.6, 0.9]
    result = probability_diagnostics(labels, probabilities, bins=5)
    assert result.sample_count == 4
    assert result.positive_count == 2
    assert result.prevalence == 0.5
    assert result.brier_score == pytest.approx(0.085)
    assert result.accuracy_at_fixed_half == 1.0
    assert len(result.reliability_bins) == 5
    assert sum(item.count for item in result.reliability_bins) == 4
    assert 0 <= result.expected_calibration_error <= 1


def test_block_length_sensitivity_is_deterministic_and_includes_primary_length():
    labels = [0, 1] * 40
    candidate = [0.15 if y == 0 else 0.85 for y in labels]
    baseline = [0.4 if y == 0 else 0.6 for y in labels]
    first = compare_probability_models(labels, candidate, baseline, candidate_name="candidate",
        baseline_name="baseline", block_length=4, bootstrap_resamples=400, random_seed=19)
    second = compare_probability_models(labels, candidate, baseline, candidate_name="candidate",
        baseline_name="baseline", block_length=4, bootstrap_resamples=400, random_seed=19)
    assert first.block_length_sensitivity == second.block_length_sensitivity
    assert [item.block_length for item in first.block_length_sensitivity] == [2, 4, 8]
    primary = next(item for item in first.block_length_sensitivity if item.block_length == 4)
    assert primary.brier_difference_ci_low == first.brier_difference_ci_low
    assert primary.log_loss_difference_ci_high == first.log_loss_difference_ci_high


def test_probability_one_is_included_in_last_reliability_bin():
    result = probability_diagnostics([1, 0], [1.0, 0.0], bins=2)
    assert result.reliability_bins[-1].count == 1
    assert result.reliability_bins[0].count == 1


def test_paired_comparison_is_deterministic_and_reports_both_losses():
    labels = [0, 1] * 30
    candidate = [0.1 if y == 0 else 0.9 for y in labels]
    baseline = [0.4 if y == 0 else 0.6 for y in labels]
    first = compare_probability_models(
        labels, candidate, baseline, candidate_name="candidate",
        baseline_name="baseline", block_length=4, bootstrap_resamples=200, random_seed=17,
    )
    second = compare_probability_models(
        labels, candidate, baseline, candidate_name="candidate",
        baseline_name="baseline", block_length=4, bootstrap_resamples=200, random_seed=17,
    )
    assert first == second
    assert first.candidate_brier_score < first.baseline_brier_score
    assert first.candidate_log_loss < first.baseline_log_loss
    assert first.brier_difference_candidate_minus_baseline < 0
    assert first.log_loss_difference_candidate_minus_baseline < 0
    assert first.interpretation.startswith("CANDIDATE_HAS_LOWER_BRIER")


@pytest.mark.parametrize("labels, probabilities", [
    ([], []),
    ([0, 1], [0.2]),
    ([0, 2], [0.2, 0.8]),
    ([False, True], [0.2, 0.8]),
    ([0, 1], [0.2, float("nan")]),
    ([0, 1], [0.2, 1.1]),
])
def test_diagnostics_reject_invalid_inputs(labels, probabilities):
    with pytest.raises(ValueError):
        probability_diagnostics(labels, probabilities)


def test_comparison_rejects_unpaired_or_invalid_configuration():
    with pytest.raises(ValueError, match="aligned"):
        compare_probability_models([0, 1], [0.2], [0.3, 0.7],
                                   candidate_name="a", baseline_name="b", block_length=1)
    with pytest.raises(ValueError, match="block_length"):
        compare_probability_models([0, 1], [0.2, 0.8], [0.3, 0.7],
                                   candidate_name="a", baseline_name="b", block_length=3)
    with pytest.raises(ValueError, match="distinct"):
        compare_probability_models([0, 1], [0.2, 0.8], [0.3, 0.7],
                                   candidate_name="same", baseline_name="same", block_length=1)

import pytest

from app.block_bootstrap import compare_with_block_bootstrap


def test_block_bootstrap_is_reproducible_and_reports_configuration():
    candidate = [0.3, 0.2, -0.1, 0.2, 0.1, 0.0, 0.4, -0.2]
    baseline = [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, -0.1]
    first = compare_with_block_bootstrap(
        candidate, baseline, block_length=2, bootstrap_resamples=1000, random_seed=17
    )
    second = compare_with_block_bootstrap(
        candidate, baseline, block_length=2, bootstrap_resamples=1000, random_seed=17
    )
    assert first == second
    assert first.paired_difference_mean == pytest.approx(0.0875)
    assert first.block_length == 2
    assert first.bootstrap_resamples == 1000
    assert first.paired_difference_ci_low <= first.paired_difference_ci_high


def test_identical_series_has_zero_block_bootstrap_interval():
    values = [0.1, -0.1, 0.0, 0.2, -0.2]
    result = compare_with_block_bootstrap(values, values, block_length=2, bootstrap_resamples=300)
    assert result.paired_difference_mean == 0
    assert result.paired_difference_ci_low == 0
    assert result.paired_difference_ci_high == 0
    assert "INCLUDES_ZERO" in result.interpretation


@pytest.mark.parametrize("block_length", [0, -1, 1.5, True, 6])
def test_invalid_block_length_fails_closed(block_length):
    with pytest.raises(ValueError, match="block_length"):
        compare_with_block_bootstrap([0.1] * 5, [0.0] * 5, block_length=block_length)


def test_invalid_series_and_bootstrap_configuration_fail_closed():
    with pytest.raises(ValueError, match="equal length"):
        compare_with_block_bootstrap([0.1, 0.2], [0.0], block_length=1)
    with pytest.raises(ValueError, match="finite"):
        compare_with_block_bootstrap([0.1, float("inf")], [0.0, 0.0], block_length=1)
    with pytest.raises(ValueError, match=">= 100"):
        compare_with_block_bootstrap([0.1], [0.0], block_length=1, bootstrap_resamples=99)


def test_constant_difference_keeps_exact_interval():
    result = compare_with_block_bootstrap(
        [0.2, 0.3, 0.4], [0.1, 0.2, 0.3], block_length=2, bootstrap_resamples=300
    )
    assert result.paired_difference_ci_low == pytest.approx(0.1)
    assert result.paired_difference_ci_high == pytest.approx(0.1)

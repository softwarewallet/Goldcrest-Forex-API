from dataclasses import replace

import pytest

from app.calibration_uncertainty import moving_block_ece_uncertainty


def test_moving_block_ece_interval_is_deterministic_and_bounded():
    labels = [i % 2 for i in range(80)]
    probabilities = [0.35 if label else 0.65 for label in labels]
    first = moving_block_ece_uncertainty(
        labels, probabilities, block_length=5, bootstrap_resamples=200,
        seed=77,
    )
    second = moving_block_ece_uncertainty(
        labels, probabilities, block_length=5, bootstrap_resamples=200,
        seed=77,
    )
    assert first == second
    assert first.status == "DESCRIPTIVE_MOVING_BLOCK_PERCENTILE_INTERVAL"
    assert 0 <= first.interval_low <= first.interval_high <= 1
    assert first.sample_count == 80
    assert first.block_length == 5


def test_small_samples_withhold_calibration_interval():
    result = moving_block_ece_uncertainty(
        [0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6],
        block_length=2, bootstrap_resamples=100,
    )
    assert result.interval_low is None
    assert result.interval_high is None
    assert result.status == "WITHHELD_INSUFFICIENT_SAMPLE_COUNT"


@pytest.mark.parametrize("block_length", [0, -1, 31, 1.5, True])
def test_invalid_block_length_rejected(block_length):
    with pytest.raises(ValueError, match="block_length"):
        moving_block_ece_uncertainty(
            [0, 1] * 15, [0.2, 0.8] * 15,
            block_length=block_length, bootstrap_resamples=100,
        )


def test_invalid_probabilities_and_labels_rejected():
    with pytest.raises(ValueError, match="probabilities"):
        moving_block_ece_uncertainty([0, 1], [0.2, float("nan")],
                                     block_length=1, bootstrap_resamples=100)
    with pytest.raises(ValueError, match="labels"):
        moving_block_ece_uncertainty([0, 2], [0.2, 0.8],
                                     block_length=1, bootstrap_resamples=100)


def test_bootstrap_resamples_have_minimum():
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        moving_block_ece_uncertainty([0, 1] * 20, [0.2, 0.8] * 20,
                                     block_length=2, bootstrap_resamples=99)

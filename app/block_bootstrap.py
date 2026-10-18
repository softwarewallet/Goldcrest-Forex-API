"""Dependence-aware paired block bootstrap for offline Forex research.

Resamples contiguous blocks of paired candidate-minus-baseline differences, preserving
short-range serial dependence within each block. This is still an approximation and is
not suitable for every dependence structure; block length must be justified and tested
for sensitivity. No model, broker, or strategy-promotion behavior is included.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from app.baseline_comparison import ReturnSeriesSummary, summarize_returns


@dataclass(frozen=True)
class BlockBootstrapComparison:
    candidate: ReturnSeriesSummary
    baseline: ReturnSeriesSummary
    paired_difference_mean: float
    paired_difference_ci_low: float
    paired_difference_ci_high: float
    bootstrap_resamples: int
    confidence_level: float
    block_length: int
    random_seed: int
    interpretation: str


def compare_with_block_bootstrap(
    candidate_net_changes: Sequence[float],
    baseline_net_changes: Sequence[float],
    *,
    block_length: int,
    bootstrap_resamples: int = 5000,
    confidence_level: float = 0.95,
    random_seed: int = 20261018,
) -> BlockBootstrapComparison:
    """Compare aligned net series using a moving-block percentile bootstrap.

    Inputs must be chronological, paired one-for-one, and already net of equivalent
    documented costs. Non-circular contiguous blocks are sampled and concatenated until
    the original sample length is reached. This method does not correct selection bias,
    multiple testing, nonstationarity, or dependence beyond the chosen block structure.
    """
    for name, values in (("candidate_net_changes", candidate_net_changes),
                         ("baseline_net_changes", baseline_net_changes)):
        if not values:
            raise ValueError(f"{name} must not be empty")
        if any(not isfinite(value) for value in values):
            raise ValueError(f"{name} must contain only finite values")
    if len(candidate_net_changes) != len(baseline_net_changes):
        raise ValueError("candidate and baseline must be aligned and equal length")
    if not isinstance(block_length, int) or isinstance(block_length, bool) or block_length < 1:
        raise ValueError("block_length must be an integer >= 1")
    if block_length > len(candidate_net_changes):
        raise ValueError("block_length must not exceed sample length")
    if not isinstance(bootstrap_resamples, int) or isinstance(bootstrap_resamples, bool) or bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be an integer >= 100")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    differences = [c - b for c, b in zip(candidate_net_changes, baseline_net_changes)]
    observed_mean = sum(differences) / len(differences)
    n = len(differences)
    rng = random.Random(random_seed)
    means: list[float] = []
    for _ in range(bootstrap_resamples):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randrange(n - block_length + 1)
            sample.extend(differences[start:start + block_length])
        sample = sample[:n]
        means.append(sum(sample) / n)
    means.sort()
    alpha = 1.0 - confidence_level
    low_index = max(0, min(bootstrap_resamples - 1, int((alpha / 2) * bootstrap_resamples)))
    high_index = max(0, min(bootstrap_resamples - 1, int((1 - alpha / 2) * bootstrap_resamples) - 1))
    low, high = means[low_index], means[high_index]
    if low > 0:
        interpretation = "BLOCK_BOOTSTRAP_INTERVAL_ABOVE_ZERO_NOT_PROOF_OF_PROFITABILITY"
    elif high < 0:
        interpretation = "BLOCK_BOOTSTRAP_INTERVAL_BELOW_ZERO_CANDIDATE_UNDERPERFORMS_BASELINE"
    else:
        interpretation = "BLOCK_BOOTSTRAP_INTERVAL_INCLUDES_ZERO_NO_CLEAR_DIFFERENCE"
    return BlockBootstrapComparison(
        candidate=summarize_returns(candidate_net_changes),
        baseline=summarize_returns(baseline_net_changes),
        paired_difference_mean=observed_mean,
        paired_difference_ci_low=low,
        paired_difference_ci_high=high,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        block_length=block_length,
        random_seed=random_seed,
        interpretation=interpretation,
    )

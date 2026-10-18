"""Offline candidate-vs-baseline comparison with uncertainty reporting.

Inputs are aligned net price changes in raw price units, after applying the same
documented cost model. This module does not infer a strategy, fetch data, or promote it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True)
class ReturnSeriesSummary:
    count: int
    total_net_price_units: float
    mean_net_price_units: float
    net_win_rate: float
    positive_count: int
    negative_count: int
    zero_count: int


@dataclass(frozen=True)
class BaselineComparison:
    candidate: ReturnSeriesSummary
    baseline: ReturnSeriesSummary
    paired_difference_mean: float
    paired_difference_ci_low: float
    paired_difference_ci_high: float
    bootstrap_resamples: int
    confidence_level: float
    interpretation: str


def _validate_series(values: Sequence[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(not isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")


def summarize_returns(values: Sequence[float]) -> ReturnSeriesSummary:
    _validate_series(values, "returns")
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    zero = len(values) - positive - negative
    total = sum(values)
    return ReturnSeriesSummary(
        count=len(values),
        total_net_price_units=total,
        mean_net_price_units=total / len(values),
        net_win_rate=positive / len(values),
        positive_count=positive,
        negative_count=negative,
        zero_count=zero,
    )


def compare_candidate_to_baseline(
    candidate_net_changes: Sequence[float],
    baseline_net_changes: Sequence[float],
    *,
    bootstrap_resamples: int = 5000,
    confidence_level: float = 0.95,
    random_seed: int = 20261018,
) -> BaselineComparison:
    """Compare aligned paired outcomes and bootstrap their mean difference.

    Both series must represent the same observations in the same chronological order,
    and both must already include equivalent costs. The interval is a paired percentile
    bootstrap interval, not a guarantee of future performance. Overlapping trades or
    serial dependence require block/bootstrap or other dependence-aware inference.
    """
    _validate_series(candidate_net_changes, "candidate_net_changes")
    _validate_series(baseline_net_changes, "baseline_net_changes")
    if len(candidate_net_changes) != len(baseline_net_changes):
        raise ValueError("candidate and baseline must be aligned and equal length")
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be >= 100")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    differences = [candidate - baseline
                   for candidate, baseline in zip(candidate_net_changes, baseline_net_changes)]
    observed_mean = sum(differences) / len(differences)
    rng = random.Random(random_seed)
    n = len(differences)
    boot_means = []
    for _ in range(bootstrap_resamples):
        sample = [differences[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    alpha = 1.0 - confidence_level
    low_index = max(0, min(bootstrap_resamples - 1,
                           int((alpha / 2) * bootstrap_resamples)))
    high_index = max(0, min(bootstrap_resamples - 1,
                            int((1 - alpha / 2) * bootstrap_resamples) - 1))
    ci_low, ci_high = boot_means[low_index], boot_means[high_index]
    if ci_low > 0:
        interpretation = "BOOTSTRAP_INTERVAL_ABOVE_ZERO_NOT_PROOF_OF_PROFITABILITY"
    elif ci_high < 0:
        interpretation = "BOOTSTRAP_INTERVAL_BELOW_ZERO_CANDIDATE_UNDERPERFORMS_BASELINE"
    else:
        interpretation = "UNCERTAINTY_INTERVAL_INCLUDES_ZERO_NO_CLEAR_DIFFERENCE"
    return BaselineComparison(
        candidate=summarize_returns(candidate_net_changes),
        baseline=summarize_returns(baseline_net_changes),
        paired_difference_mean=observed_mean,
        paired_difference_ci_low=ci_low,
        paired_difference_ci_high=ci_high,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        interpretation=interpretation,
    )

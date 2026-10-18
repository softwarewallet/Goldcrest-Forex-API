"""Probability diagnostics and paired model comparison for offline research.

Metrics are computed only on explicitly aligned out-of-sample predictions. This module
does not fit models, calibrate them, choose thresholds, or authorize trading.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ReliabilityBin:
    lower_bound: float
    upper_bound: float
    count: int
    mean_predicted_probability: float | None
    observed_positive_rate: float | None


@dataclass(frozen=True)
class ProbabilityDiagnostics:
    sample_count: int
    positive_count: int
    prevalence: float
    brier_score: float
    log_loss: float
    accuracy_at_fixed_half: float
    mean_predicted_probability: float
    expected_calibration_error: float
    reliability_bins: tuple[ReliabilityBin, ...]
    interpretation: str = "OUT_OF_SAMPLE_DIAGNOSTICS_NOT_PROOF_OF_CALIBRATION_OR_PROFITABILITY"


@dataclass(frozen=True)
class PairedProbabilityComparison:
    sample_count: int
    candidate_name: str
    baseline_name: str
    candidate_brier_score: float
    baseline_brier_score: float
    brier_difference_candidate_minus_baseline: float
    brier_difference_ci_low: float
    brier_difference_ci_high: float
    candidate_log_loss: float
    baseline_log_loss: float
    log_loss_difference_candidate_minus_baseline: float
    log_loss_difference_ci_low: float
    log_loss_difference_ci_high: float
    block_length: int
    bootstrap_resamples: int
    confidence_level: float
    interpretation: str


def _validate(labels: Sequence[int], probabilities: Sequence[float], name: str) -> None:
    if not labels or len(labels) != len(probabilities):
        raise ValueError(f"{name} labels and probabilities must be non-empty and aligned")
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError("labels must be integer 0 or 1")
    for probability in probabilities:
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(f"{name} probabilities must be numeric")
        if not math.isfinite(float(probability)) or not 0 <= probability <= 1:
            raise ValueError(f"{name} probabilities must be finite and within [0, 1]")


def _losses(labels: Sequence[int], probabilities: Sequence[float]) -> tuple[list[float], list[float]]:
    eps = 1e-15
    brier = [(p - y) ** 2 for y, p in zip(labels, probabilities)]
    logloss = [
        -(y * math.log(max(eps, p)) + (1-y) * math.log(max(eps, 1-p)))
        for y, p in zip(labels, probabilities)
    ]
    return brier, logloss


def probability_diagnostics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> ProbabilityDiagnostics:
    """Return proper scoring rules and an equal-width reliability table.

    Reliability bins are left-closed/right-open except the final bin, which includes
    probability 1. Empty bins are retained with null statistics. ECE is descriptive
    and sensitive to bin count/sample size; do not treat it as a calibration guarantee.
    """
    _validate(labels, probabilities, "model")
    if not isinstance(bins, int) or isinstance(bins, bool) or not 2 <= bins <= 100:
        raise ValueError("bins must be an integer between 2 and 100")
    brier_values, logloss_values = _losses(labels, probabilities)
    count = len(labels)
    reliability: list[ReliabilityBin] = []
    weighted_gap = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            (y, p) for y, p in zip(labels, probabilities)
            if lower <= p < upper or (index == bins - 1 and p == 1)
        ]
        if not members:
            reliability.append(ReliabilityBin(lower, upper, 0, None, None))
            continue
        mean_p = sum(p for _, p in members) / len(members)
        observed = sum(y for y, _ in members) / len(members)
        weighted_gap += len(members) / count * abs(mean_p - observed)
        reliability.append(ReliabilityBin(lower, upper, len(members), mean_p, observed))
    return ProbabilityDiagnostics(
        sample_count=count,
        positive_count=sum(labels),
        prevalence=sum(labels) / count,
        brier_score=sum(brier_values) / count,
        log_loss=sum(logloss_values) / count,
        accuracy_at_fixed_half=sum((p >= 0.5) == bool(y) for y, p in zip(labels, probabilities)) / count,
        mean_predicted_probability=sum(probabilities) / count,
        expected_calibration_error=weighted_gap,
        reliability_bins=tuple(reliability),
    )


def compare_probability_models(
    labels: Sequence[int],
    candidate_probabilities: Sequence[float],
    baseline_probabilities: Sequence[float],
    *,
    candidate_name: str,
    baseline_name: str,
    block_length: int,
    bootstrap_resamples: int = 5000,
    confidence_level: float = 0.95,
    random_seed: int = 20261018,
) -> PairedProbabilityComparison:
    """Compare paired out-of-sample losses with a moving-block bootstrap.

    Negative candidate-minus-baseline differences favor the candidate. Inputs must be
    aligned by identical observations in chronological order. Moving blocks preserve
    local sequence dependence better than IID resampling, but do not fix nonstationarity,
    multiple testing, or poor block-length choices. Evaluate only once on a held-out
    test set after all selection decisions are frozen.
    """
    _validate(labels, candidate_probabilities, "candidate")
    _validate(labels, baseline_probabilities, "baseline")
    if len(candidate_probabilities) != len(baseline_probabilities):
        raise ValueError("candidate and baseline predictions must have equal length")
    if not candidate_name.strip() or not baseline_name.strip() or candidate_name == baseline_name:
        raise ValueError("candidate and baseline names must be distinct and non-empty")
    n = len(labels)
    if not isinstance(block_length, int) or isinstance(block_length, bool) or not 1 <= block_length <= n:
        raise ValueError("block_length must be an integer between 1 and sample count")
    if not isinstance(bootstrap_resamples, int) or isinstance(bootstrap_resamples, bool) or bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be an integer >= 100")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    candidate_brier, candidate_log = _losses(labels, candidate_probabilities)
    baseline_brier, baseline_log = _losses(labels, baseline_probabilities)
    brier_diff = [a-b for a, b in zip(candidate_brier, baseline_brier)]
    log_diff = [a-b for a, b in zip(candidate_log, baseline_log)]

    def interval(values: list[float], seed_offset: int) -> tuple[float, float]:
        rng = random.Random(random_seed + seed_offset)
        means: list[float] = []
        for _ in range(bootstrap_resamples):
            sample: list[float] = []
            while len(sample) < n:
                start = rng.randrange(n - block_length + 1)
                sample.extend(values[start:start + block_length])
            sample = sample[:n]
            means.append(sum(sample) / n)
        means.sort()
        alpha = 1 - confidence_level
        low = max(0, min(bootstrap_resamples - 1, int(alpha / 2 * bootstrap_resamples)))
        high = max(0, min(bootstrap_resamples - 1, int((1 - alpha / 2) * bootstrap_resamples) - 1))
        return means[low], means[high]

    brier_low, brier_high = interval(brier_diff, 0)
    log_low, log_high = interval(log_diff, 1)
    if brier_high < 0 and log_high < 0:
        interpretation = "CANDIDATE_HAS_LOWER_BRIER_AND_LOG_LOSS_IN_THIS_EVALUATION_NOT_PROOF_OF_GENERALIZATION"
    elif brier_low > 0 and log_low > 0:
        interpretation = "CANDIDATE_HAS_HIGHER_BRIER_AND_LOG_LOSS_IN_THIS_EVALUATION"
    else:
        interpretation = "NO_CLEAR_JOINT_LOSS_ADVANTAGE_OR_METRICS_DISAGREE"
    return PairedProbabilityComparison(
        sample_count=n,
        candidate_name=candidate_name,
        baseline_name=baseline_name,
        candidate_brier_score=sum(candidate_brier) / n,
        baseline_brier_score=sum(baseline_brier) / n,
        brier_difference_candidate_minus_baseline=sum(brier_diff) / n,
        brier_difference_ci_low=brier_low,
        brier_difference_ci_high=brier_high,
        candidate_log_loss=sum(candidate_log) / n,
        baseline_log_loss=sum(baseline_log) / n,
        log_loss_difference_candidate_minus_baseline=sum(log_diff) / n,
        log_loss_difference_ci_low=log_low,
        log_loss_difference_ci_high=log_high,
        block_length=block_length,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        interpretation=interpretation,
    )

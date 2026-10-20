"""Moving-block bootstrap uncertainty for binned calibration error.

This is a descriptive uncertainty diagnostic for chronological out-of-sample
probabilities. It does not calibrate probabilities or imply trading performance.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CalibrationUncertainty:
    interval_low: float | None
    interval_high: float | None
    sample_count: int
    bins: int
    block_length: int
    bootstrap_resamples: int
    confidence_level: float
    status: str


def _ece(labels: Sequence[int], probabilities: Sequence[float], bins: int) -> float:
    n = len(labels)
    weighted_gap = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            (label, probability)
            for label, probability in zip(labels, probabilities)
            if lower <= probability < upper or (index == bins - 1 and probability == 1.0)
        ]
        if members:
            observed = sum(label for label, _ in members) / len(members)
            predicted = sum(probability for _, probability in members) / len(members)
            weighted_gap += len(members) / n * abs(observed - predicted)
    return weighted_gap


def moving_block_ece_uncertainty(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    block_length: int,
    bootstrap_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 20261020,
    bins: int = 10,
    minimum_sample_count: int = 30,
) -> CalibrationUncertainty:
    """Estimate percentile interval for ECE by resampling contiguous index blocks.

    The input order must already be chronological. Small samples are explicitly
    withheld rather than assigned misleadingly precise intervals. Blocks wrap only
    through independently sampled starts; each sampled block itself remains contiguous.
    """
    if not labels or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be non-empty and aligned")
    if any(isinstance(y, bool) or not isinstance(y, int) or y not in (0, 1) for y in labels):
        raise ValueError("labels must be integer 0 or 1")
    if any(isinstance(p, bool) or not isinstance(p, (int, float))
           or not math.isfinite(float(p)) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("probabilities must be finite numeric values in [0, 1]")
    n = len(labels)
    if isinstance(block_length, bool) or not isinstance(block_length, int) or not 1 <= block_length <= n:
        raise ValueError("block_length must be an integer between 1 and sample_count")
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be an integer >= 100")
    if not 0 < confidence_level < 1 or isinstance(confidence_level, bool):
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    if isinstance(minimum_sample_count, bool) or not isinstance(minimum_sample_count, int) or minimum_sample_count < 2:
        raise ValueError("minimum_sample_count must be an integer >= 2")
    if n < minimum_sample_count:
        return CalibrationUncertainty(None, None, n, bins, block_length, bootstrap_resamples,
                                      confidence_level, "WITHHELD_INSUFFICIENT_SAMPLE_COUNT")

    rng = random.Random(seed)
    blocks_needed = math.ceil(n / block_length)
    values: list[float] = []
    for _ in range(bootstrap_resamples):
        sampled_indices: list[int] = []
        for _ in range(blocks_needed):
            start = rng.randrange(n)
            sampled_indices.extend((start + offset) % n for offset in range(block_length))
        sampled_indices = sampled_indices[:n]
        sample_labels = [labels[i] for i in sampled_indices]
        sample_probabilities = [probabilities[i] for i in sampled_indices]
        values.append(_ece(sample_labels, sample_probabilities, bins))
    values.sort()
    alpha = (1 - confidence_level) / 2
    low = values[min(len(values) - 1, max(0, math.floor(alpha * len(values))))]
    high = values[min(len(values) - 1, max(0, math.ceil((1 - alpha) * len(values)) - 1))]
    return CalibrationUncertainty(float(low), float(high), n, bins, block_length,
                                  bootstrap_resamples, confidence_level,
                                  "DESCRIPTIVE_MOVING_BLOCK_PERCENTILE_INTERVAL")

"""Deterministic, causal OHLC feature generation for offline research only.

Features are computed at finalized bar close and use only the current and earlier
bars. They are not evidence of predictive value. Callers must ensure bars are
point-in-time correct, sorted, and belong to a single instrument/timeframe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence


@dataclass(frozen=True)
class OHLCBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class FeatureRow:
    timestamp: datetime
    values: dict[str, float]


FEATURE_NAMES = (
    "return_1",
    "return_3",
    "return_5",
    "range_pct",
    "body_pct",
    "volatility_5",
    "volatility_10",
)


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0


def build_causal_ohlc_features(bars: Sequence[OHLCBar]) -> list[FeatureRow]:
    """Build deterministic bar-close features; omit rows lacking full lookback.

    Return and volatility features are simple close-to-close fractional returns.
    Volatility is population standard deviation over the trailing return window.
    Range/body are scaled by the previous close, so all denominators are known
    and positive. No forward fill, future access, imputation, or label creation.
    """
    if not bars:
        return []
    previous_timestamp: datetime | None = None
    closes: list[float] = []
    returns: list[float] = []
    for index, bar in enumerate(bars):
        if bar.timestamp.tzinfo is None or bar.timestamp.utcoffset() is None:
            raise ValueError("bar timestamps must be timezone-aware")
        if previous_timestamp is not None and bar.timestamp <= previous_timestamp:
            raise ValueError("bar timestamps must be strictly increasing")
        previous_timestamp = bar.timestamp
        values = (bar.open, bar.high, bar.low, bar.close)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("OHLC values must be finite")
        if not all(_finite_positive(v) for v in values):
            raise ValueError("OHLC values must be positive")
        if bar.high < max(bar.open, bar.close, bar.low):
            raise ValueError("bar high is inconsistent with OHLC values")
        if bar.low > min(bar.open, bar.close, bar.high):
            raise ValueError("bar low is inconsistent with OHLC values")
        closes.append(bar.close)
        if index:
            returns.append(bar.close / closes[index - 1] - 1.0)

    output: list[FeatureRow] = []
    # Five/ten-return rolling volatility requires 6/11 closes respectively.
    for i, bar in enumerate(bars):
        if i < 10:
            continue
        prior_close = closes[i - 1]
        trailing_returns = returns[i - 10:i]
        def stddev(window: list[float]) -> float:
            mean = sum(window) / len(window)
            return math.sqrt(sum((x - mean) ** 2 for x in window) / len(window))
        values = {
            "return_1": closes[i] / closes[i - 1] - 1.0,
            "return_3": closes[i] / closes[i - 3] - 1.0,
            "return_5": closes[i] / closes[i - 5] - 1.0,
            "range_pct": (bar.high - bar.low) / prior_close,
            "body_pct": (bar.close - bar.open) / prior_close,
            "volatility_5": stddev(trailing_returns[-5:]),
            "volatility_10": stddev(trailing_returns),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("generated features must be finite")
        output.append(FeatureRow(timestamp=bar.timestamp, values=values))
    return output

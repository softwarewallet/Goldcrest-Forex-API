"""Offline barrier-outcome and chronological walk-forward research tools.

No network, model, broker, or live execution dependencies. Not a signal generator.
"""
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Iterable, Literal

OutcomeLabel = Literal["TARGET_FIRST", "STOP_FIRST", "HORIZON_EXPIRED", "AMBIGUOUS", "INSUFFICIENT_DATA"]
Direction = Literal["LONG", "SHORT"]

@dataclass(frozen=True)
class PriceBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float

@dataclass(frozen=True)
class BarrierEvaluation:
    label: OutcomeLabel
    evaluated_bars: int
    observed_at: datetime | None
    reason: str

@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def _validate_bar(bar: PriceBar) -> None:
    if bar.timestamp.tzinfo is None or bar.timestamp.utcoffset() is None:
        raise ValueError("Bar timestamp must be timezone-aware")
    values = (bar.open, bar.high, bar.low, bar.close)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values) or any(not isfinite(v) or v <= 0 for v in values):
        raise ValueError("OHLC prices must be positive finite numbers")
    if bar.high < max(bar.open, bar.close, bar.low) or bar.low > min(bar.open, bar.close, bar.high):
        raise ValueError("Bar violates OHLC constraints")


def evaluate_barrier_outcome(*, prediction_at: datetime, direction: Direction, target_price: float, stop_price: float, horizon_bars: int, bars: Iterable[PriceBar]) -> BarrierEvaluation:
    """Evaluate strictly post-prediction bars; same-bar double hits are ambiguous."""
    if prediction_at.tzinfo is None or prediction_at.utcoffset() is None:
        raise ValueError("prediction_at must be timezone-aware")
    if direction not in ("LONG", "SHORT"):
        raise ValueError("direction must be LONG or SHORT")
    for value in (target_price, stop_price):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
            raise ValueError("Target and stop prices must be positive and finite")
    if isinstance(horizon_bars, bool) or not isinstance(horizon_bars, int) or horizon_bars < 1:
        raise ValueError("horizon_bars must be a positive integer")
    if (direction == "LONG" and stop_price >= target_price) or (direction == "SHORT" and target_price >= stop_price):
        raise ValueError("Target and stop ordering is inconsistent with direction")
    ordered = list(bars)
    previous = prediction_at
    for bar in ordered:
        _validate_bar(bar)
        if bar.timestamp <= previous:
            raise ValueError("Bars must be strictly ascending and after prediction_at")
        previous = bar.timestamp
    eligible = ordered[:horizon_bars]
    for index, bar in enumerate(eligible, 1):
        target_hit = bar.high >= target_price if direction == "LONG" else bar.low <= target_price
        stop_hit = bar.low <= stop_price if direction == "LONG" else bar.high >= stop_price
        if target_hit and stop_hit:
            return BarrierEvaluation("AMBIGUOUS", index, bar.timestamp, "Both barriers touched in one bar; intrabar order unknown")
        if target_hit:
            return BarrierEvaluation("TARGET_FIRST", index, bar.timestamp, "Target touched first")
        if stop_hit:
            return BarrierEvaluation("STOP_FIRST", index, bar.timestamp, "Stop touched first")
    if len(eligible) < horizon_bars:
        return BarrierEvaluation("INSUFFICIENT_DATA", len(eligible), eligible[-1].timestamp if eligible else None, "Full horizon unavailable")
    return BarrierEvaluation("HORIZON_EXPIRED", horizon_bars, eligible[-1].timestamp, "Neither barrier touched within horizon")


def make_walk_forward_folds(*, sample_count: int, min_train_size: int, test_size: int, step_size: int | None = None) -> list[WalkForwardFold]:
    """Create expanding-window folds; default test windows are sequential and disjoint."""
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer")
    if isinstance(min_train_size, bool) or not isinstance(min_train_size, int) or min_train_size < 1:
        raise ValueError("min_train_size must be a positive integer")
    if isinstance(test_size, bool) or not isinstance(test_size, int) or test_size < 1:
        raise ValueError("test_size must be a positive integer")
    step = test_size if step_size is None else step_size
    if isinstance(step, bool) or not isinstance(step, int) or step < test_size:
        raise ValueError("step_size must be >= test_size to prevent overlapping test windows")
    folds, train_end, number = [], min_train_size, 1
    while train_end + test_size <= sample_count:
        folds.append(WalkForwardFold(number, 0, train_end, train_end, train_end + test_size))
        number += 1
        train_end += step
    return folds

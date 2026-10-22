"""Deterministic, offline research evaluators.

This module has no model, network, database, or broker dependencies. It must not be
used as a live trading signal generator. Labels are research outcomes only.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal

OutcomeLabel = Literal[
    "TARGET_FIRST", "STOP_FIRST", "HORIZON_EXPIRED", "AMBIGUOUS", "INSUFFICIENT_DATA"
]
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
    if any(not isinstance(value, (int, float)) or value <= 0 for value in values):
        raise ValueError("OHLC prices must be positive finite numbers")
    if not all(value == value and abs(value) != float("inf") for value in values):
        raise ValueError("OHLC prices must be finite")
    if bar.high < max(bar.open, bar.close, bar.low):
        raise ValueError("Bar high violates OHLC constraints")
    if bar.low > min(bar.open, bar.close, bar.high):
        raise ValueError("Bar low violates OHLC constraints")


def evaluate_barrier_outcome(
    *,
    prediction_at: datetime,
    direction: Direction,
    target_price: float,
    stop_price: float,
    horizon_bars: int,
    bars: Iterable[PriceBar],
) -> BarrierEvaluation:
    """Evaluate a frozen target/stop against strictly post-prediction OHLC bars.

    If target and stop are both touched in the same candle, return AMBIGUOUS rather
    than guessing intrabar order. The caller is responsible for defining the
    target/stop prices using a documented entry-price and cost convention.
    """
    if prediction_at.tzinfo is None or prediction_at.utcoffset() is None:
        raise ValueError("prediction_at must be timezone-aware")
    if direction not in ("LONG", "SHORT"):
        raise ValueError("direction must be LONG or SHORT")
    if not (0 < target_price < float("inf")) or not (0 < stop_price < float("inf")):
        raise ValueError("Target and stop prices must be positive and finite")
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    if direction == "LONG" and not stop_price < target_price:
        raise ValueError("LONG requires stop_price < target_price")
    if direction == "SHORT" and not target_price < stop_price:
        raise ValueError("SHORT requires target_price < stop_price")

    ordered = list(bars)
    previous = prediction_at
    for bar in ordered:
        _validate_bar(bar)
        if bar.timestamp <= previous:
            raise ValueError("Bars must be strictly ascending and after prediction_at")
        previous = bar.timestamp

    eligible = ordered[:horizon_bars]
    for index, bar in enumerate(eligible, start=1):
        if direction == "LONG":
            target_hit = bar.high >= target_price
            stop_hit = bar.low <= stop_price
        else:
            target_hit = bar.low <= target_price
            stop_hit = bar.high >= stop_price

        if target_hit and stop_hit:
            return BarrierEvaluation(
                "AMBIGUOUS", index, bar.timestamp,
                "Target and stop touched in the same OHLC bar; intrabar order unavailable",
            )
        if target_hit:
            return BarrierEvaluation("TARGET_FIRST", index, bar.timestamp, "Target touched first")
        if stop_hit:
            return BarrierEvaluation("STOP_FIRST", index, bar.timestamp, "Stop touched first")

    if len(eligible) < horizon_bars:
        return BarrierEvaluation(
            "INSUFFICIENT_DATA", len(eligible),
            eligible[-1].timestamp if eligible else None,
            "Not enough post-prediction bars to evaluate the full horizon",
        )
    return BarrierEvaluation(
        "HORIZON_EXPIRED", horizon_bars, eligible[-1].timestamp,
        "Neither target nor stop touched within the configured horizon",
    )


def make_walk_forward_folds(
    *, sample_count: int, min_train_size: int, test_size: int, step_size: int | None = None
) -> list[WalkForwardFold]:
    """Build expanding-window folds using positional indices and no future training data."""
    if sample_count < 0:
        raise ValueError("sample_count must be >= 0")
    if min_train_size < 1 or test_size < 1:
        raise ValueError("min_train_size and test_size must be >= 1")
    step = test_size if step_size is None else step_size
    if step < 1:
        raise ValueError("step_size must be >= 1")

    folds = []
    train_end = min_train_size
    fold_number = 1
    while train_end + test_size <= sample_count:
        folds.append(WalkForwardFold(
            fold=fold_number,
            train_start=0,
            train_end=train_end,
            test_start=train_end,
            test_end=train_end + test_size,
        ))
        fold_number += 1
        train_end += step
    return folds

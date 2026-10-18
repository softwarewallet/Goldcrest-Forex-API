"""Chronological walk-forward fold generation for offline Forex research.

This module creates folds only. It does not fit models, score strategies, access brokers,
or promote candidates. Each fold is an expanding-window train/validation/test split with
horizon-based purging around boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from typing import Sequence

from app.evaluation import ForexBar, validate_bars


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train: tuple[ForexBar, ...]
    validation: tuple[ForexBar, ...]
    test: tuple[ForexBar, ...]
    purged_rows: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class WalkForwardPlan:
    symbol: str
    timeframe: str
    horizon_bars: int
    initial_train_bars: int
    validation_bars: int
    test_bars: int
    step_bars: int
    folds: tuple[WalkForwardFold, ...]
    unused_tail_bars: int


def generate_walk_forward_folds(
    bars: Sequence[ForexBar],
    *,
    initial_train_bars: int,
    validation_bars: int,
    test_bars: int,
    step_bars: int,
    horizon_bars: int,
) -> WalkForwardPlan:
    """Generate expanding train windows and forward-only validation/test windows.

    Partition sizes and step are pre-purge bar counts. A horizon-sized buffer is
    removed at each partition boundary. Bars must be chronological; nothing is sorted
    or silently cleaned.
    """
    sizes = {"initial_train_bars": initial_train_bars,
             "validation_bars": validation_bars, "test_bars": test_bars,
             "step_bars": step_bars, "horizon_bars": horizon_bars}
    for name, value in sizes.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    if not bars:
        raise ValueError("bars must not be empty")
    if len({(bar.symbol, bar.timeframe) for bar in bars}) != 1:
        raise ValueError("walk-forward evaluation requires one symbol/timeframe per plan")
    quality = validate_bars(bars)
    if quality.invalid_rows or quality.duplicate_timestamps or quality.non_monotonic_timestamps:
        raise ValueError("bars failed data-quality checks; fix or quarantine invalid rows first")
    first_symbol, first_timeframe = bars[0].symbol, bars[0].timeframe
    times = [bar.timestamp.astimezone(timezone.utc) for bar in bars]
    if times != sorted(times) or len(set(times)) != len(times):
        raise ValueError("bars must be strictly chronological with unique timestamps")

    folds: list[WalkForwardFold] = []
    n = len(bars)
    train_end = initial_train_bars
    while train_end + validation_bars + test_bars <= n:
        validation_end = train_end + validation_bars
        test_end = validation_end + test_bars
        purge = horizon_bars
        train = tuple(bars[:train_end - purge])
        validation_start = train_end + purge
        validation_stop = validation_end - purge
        validation = tuple(bars[validation_start:validation_stop])
        test_start = validation_end + purge
        test = tuple(bars[test_start:test_end])
        if not train or not validation or not test:
            raise ValueError("fold has an empty partition after purging; increase data or reduce horizon")
        folds.append(WalkForwardFold(
            fold_index=len(folds), train=train, validation=validation, test=test,
            purged_rows=(train_end - len(train)) + (validation_start - train_end)
                         + (validation_end - validation_start - len(validation))
                         + (test_start - validation_end),
            train_start=train[0].timestamp.isoformat(), train_end=train[-1].timestamp.isoformat(),
            validation_start=validation[0].timestamp.isoformat(),
            validation_end=validation[-1].timestamp.isoformat(),
            test_start=test[0].timestamp.isoformat(), test_end=test[-1].timestamp.isoformat(),
        ))
        train_end += step_bars
    if not folds:
        raise ValueError("not enough bars to create one complete purged walk-forward fold")
    last_test_timestamp = folds[-1].test[-1].timestamp
    last_test_end = next(i + 1 for i, bar in enumerate(bars) if bar.timestamp == last_test_timestamp)
    return WalkForwardPlan(
        symbol=first_symbol, timeframe=first_timeframe, horizon_bars=horizon_bars,
        initial_train_bars=initial_train_bars, validation_bars=validation_bars,
        test_bars=test_bars, step_bars=step_bars, folds=tuple(folds),
        unused_tail_bars=n - last_test_end,
    )

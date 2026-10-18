"""Offline Forex historical-evaluation primitives.

This module deliberately contains no broker connectivity, model invocation, or order
execution. It validates data and creates chronological splits only; it does not score
a strategy until explicit evaluation assumptions are supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ForexBar:
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    bid: float | None = None
    ask: float | None = None


@dataclass(frozen=True)
class DataQualityReport:
    row_count: int
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    duplicate_timestamps: int
    non_monotonic_timestamps: int
    invalid_rows: int
    missing_quote_rows: int


@dataclass(frozen=True)
class ChronologicalSplit:
    train: tuple[ForexBar, ...]
    validation: tuple[ForexBar, ...]
    test: tuple[ForexBar, ...]
    purged_rows: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Explicit inputs required before a scored evaluation can run.

    Costs are expressed in price units per unit traded, not pips or account currency.
    The caller must document conversion and instrument-specific semantics separately.
    """
    symbols: tuple[str, ...]
    timeframe: str
    horizon_bars: int
    data_source: str
    dataset_version: str
    entry_price_rule: str
    exit_price_rule: str
    spread_cost_price_units: float
    slippage_price_units: float
    commission_price_units: float
    target_rule: str
    stop_rule: str
    ambiguous_bar_rule: str
    primary_metric: str

    def validate(self) -> None:
        if not self.symbols or any(not s or s != s.upper() for s in self.symbols):
            raise ValueError("symbols must be a non-empty tuple of uppercase symbols")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be unique")
        if self.timeframe not in {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}:
            raise ValueError("unsupported timeframe")
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        required = (
            self.data_source, self.dataset_version, self.entry_price_rule,
            self.exit_price_rule, self.target_rule, self.stop_rule,
            self.ambiguous_bar_rule, self.primary_metric,
        )
        if any(not value.strip() for value in required):
            raise ValueError("all evaluation assumptions must be explicitly documented")
        costs = (self.spread_cost_price_units, self.slippage_price_units,
                 self.commission_price_units)
        if any(not isfinite(value) or value < 0 for value in costs):
            raise ValueError("cost assumptions must be finite and non-negative")


def validate_bars(bars: Iterable[ForexBar], *, require_quotes: bool = False) -> DataQualityReport:
    """Validate OHLC and ordering without silently sorting or dropping bad records."""
    rows = list(bars)
    invalid = 0
    missing_quotes = 0
    duplicates = 0
    non_monotonic = 0
    seen: set[tuple[str, str, datetime]] = set()
    previous: dict[tuple[str, str], datetime] = {}

    for bar in rows:
        key = (bar.symbol, bar.timeframe)
        timestamp = bar.timestamp
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            invalid += 1
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        prices = (bar.open, bar.high, bar.low, bar.close)
        if (not bar.symbol or bar.symbol != bar.symbol.upper()
                or bar.timeframe not in {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}
                or any(not isfinite(p) or p <= 0 for p in prices)
                or bar.high < max(bar.open, bar.close, bar.low)
                or bar.low > min(bar.open, bar.close, bar.high)):
            invalid += 1
        if (bar.bid is None or bar.ask is None):
            missing_quotes += 1
            if require_quotes:
                invalid += 1
        elif (not isfinite(bar.bid) or not isfinite(bar.ask)
              or bar.bid <= 0 or bar.ask < bar.bid):
            invalid += 1

        row_key = (bar.symbol, bar.timeframe, timestamp)
        if row_key in seen:
            duplicates += 1
        seen.add(row_key)
        last = previous.get(key)
        if last is not None and timestamp < last:
            non_monotonic += 1
        previous[key] = timestamp

    return DataQualityReport(
        row_count=len(rows),
        symbols=tuple(sorted({bar.symbol for bar in rows})),
        timeframes=tuple(sorted({bar.timeframe for bar in rows})),
        duplicate_timestamps=duplicates,
        non_monotonic_timestamps=non_monotonic,
        invalid_rows=invalid,
        missing_quote_rows=missing_quotes,
    )


def chronological_split(
    bars: Sequence[ForexBar],
    *,
    train_fraction: float,
    validation_fraction: float,
    horizon_bars: int,
) -> ChronologicalSplit:
    """Split chronologically and purge horizon_bars around split boundaries.

    Input must already be sorted by timestamp and contain one instrument/timeframe.
    Rows adjacent to validation/test boundaries are purged to reduce overlapping-label
    leakage. Fractions describe the unpurged dataset; small datasets fail closed.
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    if len(bars) < 3 * horizon_bars + 3:
        raise ValueError("not enough rows for requested horizon and split")
    if len({(bar.symbol, bar.timeframe) for bar in bars}) != 1:
        raise ValueError("split one symbol/timeframe at a time")
    timestamps = [bar.timestamp for bar in bars]
    if any(ts.tzinfo is None or ts.utcoffset() is None for ts in timestamps):
        raise ValueError("all timestamps must be timezone-aware")
    utc_times = [ts.astimezone(timezone.utc) for ts in timestamps]
    if utc_times != sorted(utc_times) or len(set(utc_times)) != len(utc_times):
        raise ValueError("bars must be sorted and timestamps unique")

    n = len(bars)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    purge = horizon_bars
    train = tuple(bars[:max(0, train_end - purge)])
    validation = tuple(bars[min(n, train_end + purge):max(train_end + purge, validation_end - purge)])
    test = tuple(bars[min(n, validation_end + purge):])
    if not train or not validation or not test:
        raise ValueError("split leaves an empty partition; provide more data or smaller horizon")
    return ChronologicalSplit(
        train=train,
        validation=validation,
        test=test,
        purged_rows=n - len(train) - len(validation) - len(test),
    )

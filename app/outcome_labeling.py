"Deterministic, offline Forex outcome labels for historical research.

Labels use the entry bar's close and inspect only subsequent bars. OHLC bars cannot
reveal intrabar order when target and stop are both touched; an explicit conservative
ambiguity policy is required. This module never calls a model or broker.
"
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Literal, Sequence

from app.evaluation import ForexBar

Side = Literal["LONG", "SHORT"]
AmbiguousPolicy = Literal["EXCLUDE", "STOP_FIRST"]
OutcomeLabel = Literal["TARGET_FIRST", "STOP_FIRST", "HORIZON_EXPIRED", "AMBIGUOUS"]


@dataclass(frozen=True)
class OutcomeLabelResult:
    symbol: str
    timeframe: str
    entry_timestamp: datetime
    observation_timestamp: datetime
    side: Side
    label: OutcomeLabel
    entry_price: float
    exit_price: float | None
    gross_price_change: float | None
    bars_observed: int
    reason: str


def label_outcome(
    bars: Sequence[ForexBar],
    *,
    entry_index: int,
    side: Side,
    target_distance: float,
    stop_distance: float,
    horizon_bars: int,
    ambiguous_policy: AmbiguousPolicy,
) -> OutcomeLabelResult:
    """Label a hypothetical trade using future OHLC bars only.

    Entry is at the close of bars[entry_index]. The following horizon_bars bars are
    inspected, up to available data. Distances are absolute price units, not pips,
    R multiples, or account currency. Bid/ask and transaction costs are not applied;
    a separate explicitly configured step must account for them.

    If target and stop are both touched in one OHLC bar, EXCLUDE returns an AMBIGUOUS
    label without an exit/return; STOP_FIRST resolves conservatively as a stop.
    """
    if side not in ("LONG", "SHORT"):
        raise ValueError("side must be LONG or SHORT")
    if not isfinite(target_distance) or target_distance <= 0:
        raise ValueError("target_distance must be finite and > 0")
    if not isfinite(stop_distance) or stop_distance <= 0:
        raise ValueError("stop_distance must be finite and > 0")
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    if ambiguous_policy not in ("EXCLUDE", "STOP_FIRST"):
        raise ValueError("ambiguous_policy must be EXCLUDE or STOP_FIRST")
    if entry_index < 0 or entry_index >= len(bars):
        raise ValueError("entry_index is outside the supplied bars")
    if len({(bar.symbol, bar.timeframe) for bar in bars}) != 1:
        raise ValueError("bars must contain one symbol/timeframe")
    times = [bar.timestamp for bar in bars]
    if any(ts.tzinfo is None or ts.utcoffset() is None for ts in times):
        raise ValueError("all timestamps must be timezone-aware")
    utc_times = [ts.astimezone(timezone.utc) for ts in times]
    if utc_times != sorted(utc_times) or len(set(utc_times)) != len(utc_times):
        raise ValueError("bars must be strictly chronological with unique timestamps")
    for bar in bars:
        prices = (bar.open, bar.high, bar.low, bar.close)
        if (any(not isfinite(p) or p <= 0 for p in prices)
                or bar.high < max(bar.open, bar.close, bar.low)
                or bar.low > min(bar.open, bar.close, bar.high)):
            raise ValueError("bars contain invalid OHLC values")

    entry = bars[entry_index]
    entry_price = entry.close
    target = entry_price + target_distance if side == "LONG" else entry_price - target_distance
    stop = entry_price - stop_distance if side == "LONG" else entry_price + stop_distance
    future = bars[entry_index + 1:entry_index + 1 + horizon_bars]

    def result(label: OutcomeLabel, bar: ForexBar, exit_price: float | None,
               count: int, reason: str) -> OutcomeLabelResult:
        change = None if exit_price is None else (
            exit_price - entry_price if side == "LONG" else entry_price - exit_price
        )
        return OutcomeLabelResult(
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            entry_timestamp=entry.timestamp,
            observation_timestamp=bar.timestamp,
            side=side,
            label=label,
            entry_price=entry_price,
            exit_price=exit_price,
            gross_price_change=change,
            bars_observed=count,
            reason=reason,
        )

    for offset, bar in enumerate(future, start=1):
        if side == "LONG":
            target_hit, stop_hit = bar.high >= target, bar.low <= stop
        else:
            target_hit, stop_hit = bar.low <= target, bar.high >= stop
        if target_hit and stop_hit:
            if ambiguous_policy == "EXCLUDE":
                return result("AMBIGUOUS", bar, None, offset,
                              "TARGET_AND_STOP_TOUCHED_WITHIN_SAME_OHLC_BAR")
            return result("STOP_FIRST", bar, stop, offset,
                          "AMBIGUOUS_BAR_RESOLVED_CONSERVATIVELY_AS_STOP")
        if target_hit:
            return result("TARGET_FIRST", bar, target, offset, "TARGET_TOUCHED")
        if stop_hit:
            return result("STOP_FIRST", bar, stop, offset, "STOP_TOUCHED")

    if future:
        last = future[-1]
        return result("HORIZON_EXPIRED", last, last.close, len(future),
                      "HORIZON_EXPIRED_AT_LAST_AVAILABLE_CLOSE")
    return result("HORIZON_EXPIRED", entry, entry_price, 0, "NO_FUTURE_BARS_AVAILABLE")

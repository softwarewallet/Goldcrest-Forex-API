"""Cost-aware scoring primitives for deterministic offline Forex outcomes.

Costs must be supplied in the same raw price units as gross_price_change and must
represent the complete round-trip cost per trade. This module does not infer pip value,
position sizing, account-currency P&L, or executable fills.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from app.outcome_labeling import OutcomeLabelResult


@dataclass(frozen=True)
class RoundTripCosts:
    spread_price_units: float
    slippage_price_units: float
    commission_price_units: float

    def validate(self) -> None:
        values = (self.spread_price_units, self.slippage_price_units,
                  self.commission_price_units)
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("round-trip costs must be finite and non-negative")

    @property
    def total_price_units(self) -> float:
        self.validate()
        return (self.spread_price_units + self.slippage_price_units
                + self.commission_price_units)


@dataclass(frozen=True)
class ScoredOutcome:
    symbol: str
    timeframe: str
    entry_timestamp: str
    side: str
    label: str
    gross_price_change: float
    total_cost_price_units: float
    net_price_change: float


@dataclass(frozen=True)
class CostAwareSummary:
    outcomes_received: int
    scored_outcomes: int
    ambiguous_excluded: int
    missing_gross_change_excluded: int
    gross_total_price_units: float
    costs_total_price_units: float
    net_total_price_units: float
    net_mean_per_scored_outcome: float | None
    net_win_rate: float | None
    positive_net_outcomes: int
    negative_net_outcomes: int
    zero_net_outcomes: int
    cost_assumption_price_units_per_trade: float


def score_outcomes(
    outcomes: Sequence[OutcomeLabelResult],
    *,
    costs: RoundTripCosts,
) -> tuple[tuple[ScoredOutcome, ...], CostAwareSummary]:
    """Subtract explicit round-trip costs and summarize outcomes without overclaiming.

    AMBIGUOUS results and any result lacking a gross price change are excluded from
    scored metrics but counted separately. Horizon-expired observations are scored
    using their recorded mark-to-close gross change. All metrics remain raw price
    units, not account-currency returns or portfolio performance.
    """
    total_cost = costs.total_price_units
    scored: list[ScoredOutcome] = []
    ambiguous = 0
    missing_change = 0

    for outcome in outcomes:
        if outcome.label == "AMBIGUOUS":
            ambiguous += 1
            continue
        gross = outcome.gross_price_change
        if gross is None or not isfinite(gross):
            missing_change += 1
            continue
        net = gross - total_cost
        scored.append(ScoredOutcome(
            symbol=outcome.symbol,
            timeframe=outcome.timeframe,
            entry_timestamp=outcome.entry_timestamp.isoformat(),
            side=outcome.side,
            label=outcome.label,
            gross_price_change=gross,
            total_cost_price_units=total_cost,
            net_price_change=net,
        ))

    gross_total = sum(item.gross_price_change for item in scored)
    net_total = sum(item.net_price_change for item in scored)
    wins = sum(item.net_price_change > 0 for item in scored)
    losses = sum(item.net_price_change < 0 for item in scored)
    zeros = len(scored) - wins - losses
    count = len(scored)
    summary = CostAwareSummary(
        outcomes_received=len(outcomes),
        scored_outcomes=count,
        ambiguous_excluded=ambiguous,
        missing_gross_change_excluded=missing_change,
        gross_total_price_units=gross_total,
        costs_total_price_units=total_cost * count,
        net_total_price_units=net_total,
        net_mean_per_scored_outcome=(net_total / count if count else None),
        net_win_rate=(wins / count if count else None),
        positive_net_outcomes=wins,
        negative_net_outcomes=losses,
        zero_net_outcomes=zeros,
        cost_assumption_price_units_per_trade=total_cost,
    )
    return tuple(scored), summary

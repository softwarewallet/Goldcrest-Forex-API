from datetime import datetime, timezone

import pytest

from app.outcome_labeling import OutcomeLabelResult
from app.cost_aware_evaluation import RoundTripCosts, score_outcomes


def outcome(label, gross):
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return OutcomeLabelResult(
        symbol="EUR/USD", timeframe="M5", entry_timestamp=now,
        observation_timestamp=now, side="LONG", label=label,
        entry_price=1.1, exit_price=None if gross is None else 1.1 + gross,
        gross_price_change=gross, bars_observed=1, reason="test",
    )


def test_costs_are_subtracted_from_each_scored_outcome():
    scored, summary = score_outcomes(
        [outcome("TARGET_FIRST", 0.002), outcome("STOP_FIRST", -0.001)],
        costs=RoundTripCosts(0.0001, 0.0001, 0.00005),
    )
    assert scored[0].net_price_change == pytest.approx(0.00175)
    assert scored[1].net_price_change == pytest.approx(-0.00125)
    assert summary.gross_total_price_units == pytest.approx(0.001)
    assert summary.costs_total_price_units == pytest.approx(0.0005)
    assert summary.net_total_price_units == pytest.approx(0.0005)
    assert summary.net_win_rate == pytest.approx(0.5)


def test_ambiguous_and_missing_returns_are_counted_but_not_scored():
    _, summary = score_outcomes(
        [outcome("AMBIGUOUS", None), outcome("TARGET_FIRST", None),
         outcome("HORIZON_EXPIRED", 0.0)],
        costs=RoundTripCosts(0, 0, 0),
    )
    assert summary.outcomes_received == 3
    assert summary.ambiguous_excluded == 1
    assert summary.missing_gross_change_excluded == 1
    assert summary.scored_outcomes == 1
    assert summary.zero_net_outcomes == 1
    assert summary.net_win_rate == 0


def test_empty_scored_set_has_no_fabricated_metrics():
    _, summary = score_outcomes(
        [outcome("AMBIGUOUS", None)],
        costs=RoundTripCosts(0, 0, 0),
    )
    assert summary.scored_outcomes == 0
    assert summary.net_mean_per_scored_outcome is None
    assert summary.net_win_rate is None


@pytest.mark.parametrize("costs", [
    RoundTripCosts(-0.1, 0, 0),
    RoundTripCosts(float("nan"), 0, 0),
    RoundTripCosts(0, float("inf"), 0),
])
def test_invalid_costs_fail_closed(costs):
    with pytest.raises(ValueError, match="finite and non-negative"):
        score_outcomes([outcome("TARGET_FIRST", 0.01)], costs=costs)

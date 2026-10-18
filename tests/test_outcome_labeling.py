from datetime import datetime, timedelta, timezone

import pytest

from app.evaluation import ForexBar
from app.outcome_labeling import label_outcome


def bars_from_ohlc(rows):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [
        ForexBar(
            symbol="EUR/USD", timeframe="M5",
            timestamp=start + timedelta(minutes=5 * i),
            open=o, high=h, low=l, close=c,
        )
        for i, (o, h, l, c) in enumerate(rows)
    ]


def test_long_target_first_uses_future_bar_only():
    bars = bars_from_ohlc([
        (1.1000, 1.1010, 1.0990, 1.1000),
        (1.1000, 1.1025, 1.0995, 1.1020),
    ])
    result = label_outcome(
        bars, entry_index=0, side="LONG", target_distance=0.002,
        stop_distance=0.001, horizon_bars=3, ambiguous_policy="EXCLUDE",
    )
    assert result.label == "TARGET_FIRST"
    assert result.exit_price == pytest.approx(1.102)
    assert result.bars_observed == 1


def test_short_stop_first():
    bars = bars_from_ohlc([
        (1.1000, 1.1010, 1.0990, 1.1000),
        (1.1000, 1.1020, 1.0995, 1.1015),
    ])
    result = label_outcome(
        bars, entry_index=0, side="SHORT", target_distance=0.002,
        stop_distance=0.001, horizon_bars=2, ambiguous_policy="EXCLUDE",
    )
    assert result.label == "STOP_FIRST"
    assert result.exit_price == pytest.approx(1.101)


def test_ambiguous_bar_is_excluded_without_fabricated_return():
    bars = bars_from_ohlc([
        (1.1000, 1.1010, 1.0990, 1.1000),
        (1.1000, 1.1030, 1.0970, 1.1000),
    ])
    result = label_outcome(
        bars, entry_index=0, side="LONG", target_distance=0.002,
        stop_distance=0.002, horizon_bars=1, ambiguous_policy="EXCLUDE",
    )
    assert result.label == "AMBIGUOUS"
    assert result.exit_price is None
    assert result.gross_price_change is None


def test_ambiguous_bar_can_only_be_resolved_conservatively():
    bars = bars_from_ohlc([
        (1.1000, 1.1010, 1.0990, 1.1000),
        (1.1000, 1.1030, 1.0970, 1.1000),
    ])
    result = label_outcome(
        bars, entry_index=0, side="LONG", target_distance=0.002,
        stop_distance=0.002, horizon_bars=1, ambiguous_policy="STOP_FIRST",
    )
    assert result.label == "STOP_FIRST"
    assert result.exit_price == pytest.approx(1.098)


def test_horizon_expired_marks_last_available_close():
    bars = bars_from_ohlc([
        (1.1000, 1.1010, 1.0990, 1.1000),
        (1.1000, 1.1005, 1.0995, 1.1002),
        (1.1002, 1.1006, 1.0998, 1.1004),
    ])
    result = label_outcome(
        bars, entry_index=0, side="LONG", target_distance=0.005,
        stop_distance=0.005, horizon_bars=5, ambiguous_policy="EXCLUDE",
    )
    assert result.label == "HORIZON_EXPIRED"
    assert result.exit_price == pytest.approx(1.1004)
    assert result.bars_observed == 2


@pytest.mark.parametrize("target,stop,horizon", [(0, 0.1, 1), (0.1, -1, 1), (0.1, 0.1, 0)])
def test_invalid_label_parameters_fail_closed(target, stop, horizon):
    bars = bars_from_ohlc([(1.1, 1.11, 1.09, 1.1), (1.1, 1.11, 1.09, 1.1)])
    with pytest.raises(ValueError):
        label_outcome(
            bars, entry_index=0, side="LONG", target_distance=target,
            stop_distance=stop, horizon_bars=horizon, ambiguous_policy="EXCLUDE",
        )

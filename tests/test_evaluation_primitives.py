from datetime import datetime, timedelta, timezone

import pytest

from app.evaluation import (
    EvaluationConfig,
    ForexBar,
    chronological_split,
    validate_bars,
)


def make_bars(count=100):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    bars = []
    for i in range(count):
        close = 1.1 + i * 0.0001
        bars.append(ForexBar(
            symbol="EUR/USD",
            timeframe="M5",
            timestamp=start + timedelta(minutes=5 * i),
            open=close - 0.00005,
            high=close + 0.0001,
            low=close - 0.0001,
            close=close,
            bid=close - 0.00001,
            ask=close + 0.00001,
        ))
    return bars


def test_validate_bars_accepts_valid_ordered_fx_bars():
    report = validate_bars(make_bars())
    assert report.row_count == 100
    assert report.symbols == ("EUR/USD",)
    assert report.timeframes == ("M5",)
    assert report.invalid_rows == 0
    assert report.duplicate_timestamps == 0
    assert report.non_monotonic_timestamps == 0
    assert report.missing_quote_rows == 0


def test_validate_bars_reports_duplicate_and_non_monotonic_timestamps():
    bars = make_bars(5)
    bars[3] = bars[1]
    report = validate_bars(bars)
    assert report.duplicate_timestamps == 1
    assert report.non_monotonic_timestamps == 1


def test_require_quotes_marks_missing_quotes_invalid():
    bars = make_bars(4)
    bars[2] = ForexBar(
        symbol=bars[2].symbol, timeframe=bars[2].timeframe,
        timestamp=bars[2].timestamp, open=bars[2].open, high=bars[2].high,
        low=bars[2].low, close=bars[2].close,
    )
    report = validate_bars(bars, require_quotes=True)
    assert report.missing_quote_rows == 1
    assert report.invalid_rows == 1


def test_chronological_split_is_ordered_and_purges_boundaries():
    bars = make_bars(100)
    split = chronological_split(
        bars, train_fraction=0.6, validation_fraction=0.2, horizon_bars=5
    )
    assert split.train[-1].timestamp < split.validation[0].timestamp
    assert split.validation[-1].timestamp < split.test[0].timestamp
    assert split.purged_rows >= 10
    assert len(split.train) + len(split.validation) + len(split.test) + split.purged_rows == len(bars)


def test_chronological_split_rejects_unsorted_input():
    bars = make_bars(40)
    bars[10], bars[11] = bars[11], bars[10]
    with pytest.raises(ValueError, match="sorted"):
        chronological_split(
            bars, train_fraction=0.6, validation_fraction=0.2, horizon_bars=2
        )


def test_evaluation_config_requires_explicit_assumptions():
    config = EvaluationConfig(
        symbols=("EUR/USD",),
        timeframe="M5",
        horizon_bars=12,
        data_source="synthetic-test-fixture",
        dataset_version="fixture-v1",
        entry_price_rule="buy at ask; sell at bid",
        exit_price_rule="buy exits at bid; sell exits at ask",
        spread_cost_price_units=0.0001,
        slippage_price_units=0.00005,
        commission_price_units=0.0,
        target_rule="externally specified target rule v1",
        stop_rule="externally specified stop rule v1",
        ambiguous_bar_rule="mark ambiguous; do not choose favorable touch",
        primary_metric="net expected R (not computed by this module)",
    )
    config.validate()


def test_evaluation_config_rejects_negative_costs():
    config = EvaluationConfig(
        symbols=("EUR/USD",),
        timeframe="M5",
        horizon_bars=12,
        data_source="fixture",
        dataset_version="v1",
        entry_price_rule="explicit rule",
        exit_price_rule="explicit rule",
        spread_cost_price_units=-0.0001,
        slippage_price_units=0.0,
        commission_price_units=0.0,
        target_rule="explicit rule",
        stop_rule="explicit rule",
        ambiguous_bar_rule="mark ambiguous",
        primary_metric="explicit metric",
    )
    with pytest.raises(ValueError, match="non-negative"):
        config.validate()

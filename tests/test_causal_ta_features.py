from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.causal_ta_features import FEATURE_FAMILIES, build_causal_ohlc_features, feature_ablation_sets
from app.evaluation import ForexBar


def bars(count=80, *, alter_after=None):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = []
    for i in range(count):
        close = 1.1 + i * 0.0001 + ((i % 5) - 2) * 0.00003
        if alter_after is not None and i >= alter_after:
            close += 0.05
        result.append(ForexBar(
            symbol="EUR/USD",
            timeframe="M15",
            timestamp=start + timedelta(minutes=15 * i),
            open=close - 0.00002,
            high=close + 0.0002,
            low=close - 0.0002,
            close=close,
        ))
    return result


def test_feature_batch_is_non_actionable_and_has_expected_schema():
    batch = build_causal_ohlc_features(bars())
    expected = {name for family in FEATURE_FAMILIES.values() for name in family}
    assert set(batch.feature_names) == expected
    assert len(batch.rows) == 80
    assert batch.actionable is False
    assert batch.indicator_backend in {"talib", "numpy_fallback"}
    assert batch.rows[0]["return_1"] is None
    assert batch.rows[-1]["return_1"] is not None
    assert all(value is None or value == value and abs(value) != float("inf")
               for row in batch.rows for value in row.values())


def test_future_bars_cannot_change_past_feature_values():
    original = build_causal_ohlc_features(bars())
    changed = build_causal_ohlc_features(bars(alter_after=50))
    for index in range(50):
        assert original.rows[index] == changed.rows[index]


def test_feature_builder_rejects_duplicate_or_unordered_timestamps():
    source = bars(5)
    source[3] = ForexBar(
        symbol=source[3].symbol,
        timeframe=source[3].timeframe,
        timestamp=source[2].timestamp,
        open=source[3].open,
        high=source[3].high,
        low=source[3].low,
        close=source[3].close,
    )
    with pytest.raises(ValueError, match="strictly chronological"):
        build_causal_ohlc_features(source)


def test_ablation_sets_are_deterministic_and_preserve_input_order():
    names = ("return_1", "rsi_14", "volatility_5", "body_pct")
    sets = feature_ablation_sets(names)
    assert sets["all_features"] == names
    assert sets["without_momentum"] == ("return_1", "volatility_5", "body_pct")
    assert sets["without_volatility"] == ("return_1", "rsi_14", "body_pct")
    assert "without_returns" in sets
    assert "without_candle_structure" in sets


def test_ablation_rejects_unknown_features():
    with pytest.raises(ValueError, match="unknown causal feature"):
        feature_ablation_sets(("return_1", "my_future_signal"))

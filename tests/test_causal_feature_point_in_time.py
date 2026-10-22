"""Empirical point-in-time invariance tests for causal OHLC feature engineering."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pytest

from app.causal_ta_features import build_causal_ohlc_features
from app.evaluation import ForexBar


def _bars(count: int = 96) -> list[ForexBar]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars: list[ForexBar] = []
    previous = 1.08
    for i in range(count):
        close = 1.08 + i * 0.00017 + math.sin(i / 3.7) * 0.0013 + math.cos(i / 8.1) * 0.0004
        open_ = previous
        high = max(open_, close) + 0.0002 + (i % 4) * 0.00001
        low = min(open_, close) - 0.00018 - (i % 3) * 0.00001
        bars.append(ForexBar(
            symbol="EURUSD",
            timeframe="H1",
            timestamp=start + timedelta(hours=i),
            open=open_,
            high=high,
            low=low,
            close=close,
        ))
        previous = close
    return bars


def _assert_feature_rows_equal(left: dict[str, float | None], right: dict[str, float | None]) -> None:
    assert left.keys() == right.keys()
    for name in left:
        a, b = left[name], right[name]
        if a is None or b is None:
            assert a is b, f"{name}: warm-up availability changed ({a!r} != {b!r})"
        else:
            assert a == pytest.approx(b, rel=1e-11, abs=1e-12), f"{name}: {a!r} != {b!r}"


def test_features_are_prefix_invariant_at_every_timestamp() -> None:
    """Appending later bars must not change any feature already computed at t."""
    bars = _bars()
    full = build_causal_ohlc_features(bars)
    assert full.actionable is False

    for cutoff in range(len(bars)):
        prefix = build_causal_ohlc_features(bars[: cutoff + 1])
        assert prefix.timestamps_utc[-1] == full.timestamps_utc[cutoff]
        _assert_feature_rows_equal(prefix.rows[-1], full.rows[cutoff])


def test_features_are_invariant_to_perturbing_all_future_ohlc() -> None:
    """Changing future market observations cannot affect features at an earlier cutoff."""
    bars = _bars()
    cutoff = 61
    baseline = build_causal_ohlc_features(bars)

    perturbed = list(bars)
    for i in range(cutoff + 1, len(perturbed)):
        bar = perturbed[i]
        # Preserve valid OHLC geometry and timestamps while materially changing future prices.
        new_open = bar.open * (1.0 + 0.035 + (i % 5) * 0.001)
        new_close = bar.close * (1.0 - 0.027 - (i % 3) * 0.001)
        perturbed[i] = ForexBar(
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            timestamp=bar.timestamp,
            open=new_open,
            high=max(new_open, new_close) * 1.002,
            low=min(new_open, new_close) * 0.998,
            close=new_close,
        )

    changed_future = build_causal_ohlc_features(perturbed)
    _assert_feature_rows_equal(baseline.rows[cutoff], changed_future.rows[cutoff])


def test_feature_rows_keep_fixed_schema_and_non_actionable_contract() -> None:
    batch = build_causal_ohlc_features(_bars(48))
    assert batch.feature_names
    assert len(set(batch.feature_names)) == len(batch.feature_names)
    assert all(tuple(row.keys()) == batch.feature_names for row in batch.rows)
    assert batch.actionable is False

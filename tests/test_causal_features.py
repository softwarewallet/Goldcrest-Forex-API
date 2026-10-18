from datetime import datetime, timedelta, timezone
import math

import pytest

from app.causal_features import OHLCBar, build_causal_ohlc_features


def bars(count=16):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = []
    for i in range(count):
        close = 100.0 + i * 0.7 + math.sin(i / 2)
        open_ = close - 0.2
        result.append(OHLCBar(start + timedelta(minutes=5 * i), open_, close + 0.4, open_ - 0.1, close))
    return result


def test_emits_only_after_full_ten_return_lookback_and_schema_is_stable():
    result = build_causal_ohlc_features(bars(16))
    assert len(result) == 6
    assert result[0].timestamp == bars(16)[10].timestamp
    assert tuple(result[0].values) == (
        "return_1", "return_3", "return_5", "range_pct", "body_pct",
        "volatility_5", "volatility_10",
    )
    assert all(math.isfinite(v) for row in result for v in row.values.values())


def test_future_bars_do_not_change_features_already_emitted():
    source = bars(18)
    prefix = build_causal_ohlc_features(source[:14])
    extended = build_causal_ohlc_features(source)
    assert [r.values for r in prefix] == [r.values for r in extended[:len(prefix)]]


def test_rejects_naive_or_non_increasing_timestamps():
    source = bars(12)
    source[2] = OHLCBar(datetime(2026, 1, 1), source[2].open, source[2].high, source[2].low, source[2].close)
    with pytest.raises(ValueError, match="timezone-aware"):
        build_causal_ohlc_features(source)
    source = bars(12)
    source[4] = OHLCBar(source[3].timestamp, source[4].open, source[4].high, source[4].low, source[4].close)
    with pytest.raises(ValueError, match="strictly increasing"):
        build_causal_ohlc_features(source)


def test_rejects_invalid_ohlc_and_nonfinite_prices():
    source = bars(12)
    bad = source.copy()
    bar = bad[3]
    bad[3] = OHLCBar(bar.timestamp, bar.open, bar.low - 1, bar.low, bar.close)
    with pytest.raises(ValueError, match="high is inconsistent"):
        build_causal_ohlc_features(bad)
    bad = source.copy()
    bar = bad[3]
    bad[3] = OHLCBar(bar.timestamp, float("nan"), bar.high, bar.low, bar.close)
    with pytest.raises(ValueError, match="finite"):
        build_causal_ohlc_features(bad)


def test_empty_and_short_inputs_return_no_rows():
    assert build_causal_ohlc_features([]) == []
    assert build_causal_ohlc_features(bars(10)) == []

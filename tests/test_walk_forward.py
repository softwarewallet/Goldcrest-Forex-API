from datetime import datetime, timedelta, timezone

import pytest

from app.evaluation import ForexBar
from app.walk_forward import generate_walk_forward_folds


def make_bars(count=120):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    result = []
    for i in range(count):
        close = 1.1 + i * 0.0001
        result.append(ForexBar(
            symbol="EUR/USD", timeframe="M5",
            timestamp=start + timedelta(minutes=5 * i),
            open=close - 0.00002, high=close + 0.0001,
            low=close - 0.0001, close=close,
            bid=close - 0.00001, ask=close + 0.00001,
        ))
    return result


def test_expanding_walk_forward_folds_are_chronological_and_purged():
    plan = generate_walk_forward_folds(
        make_bars(), initial_train_bars=50, validation_bars=20,
        test_bars=15, step_bars=15, horizon_bars=3,
    )
    assert len(plan.folds) >= 3
    for fold in plan.folds:
        assert fold.train[-1].timestamp < fold.validation[0].timestamp
        assert fold.validation[-1].timestamp < fold.test[0].timestamp
        assert len(fold.train) == (50 + fold.fold_index * 15) - 3
        assert len(fold.validation) == 20 - 6
        assert len(fold.test) == 15 - 3
    assert len(plan.folds[1].train) > len(plan.folds[0].train)


def test_walk_forward_rejects_mixed_instruments_and_bad_data():
    bars = make_bars(100)
    bars[5] = ForexBar("GBP/USD", "M5", bars[5].timestamp,
                       bars[5].open, bars[5].high, bars[5].low, bars[5].close)
    with pytest.raises(ValueError, match="one symbol/timeframe"):
        generate_walk_forward_folds(
            bars, initial_train_bars=40, validation_bars=20,
            test_bars=15, step_bars=10, horizon_bars=2,
        )


def test_walk_forward_fails_when_purge_empties_partition():
    with pytest.raises(ValueError, match="empty partition|not enough bars"):
        generate_walk_forward_folds(
            make_bars(40), initial_train_bars=10, validation_bars=4,
            test_bars=4, step_bars=2, horizon_bars=3,
        )


def test_walk_forward_rejects_unsorted_and_duplicate_timestamps():
    bars = make_bars(100)
    bars[10], bars[11] = bars[11], bars[10]
    with pytest.raises(ValueError, match="data-quality|chronological"):
        generate_walk_forward_folds(
            bars, initial_train_bars=40, validation_bars=20,
            test_bars=15, step_bars=10, horizon_bars=2,
        )


@pytest.mark.parametrize("name,value", [
    ("initial_train_bars", 0), ("validation_bars", 0),
    ("test_bars", -1), ("step_bars", 0), ("horizon_bars", 0),
])
def test_walk_forward_requires_positive_partition_sizes(name, value):
    kwargs = dict(initial_train_bars=40, validation_bars=20,
                  test_bars=15, step_bars=10, horizon_bars=2)
    kwargs[name] = value
    with pytest.raises(ValueError, match="integer >= 1"):
        generate_walk_forward_folds(make_bars(100), **kwargs)

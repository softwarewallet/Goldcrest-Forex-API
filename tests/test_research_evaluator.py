from datetime import datetime, timedelta, timezone

import pytest

from app.research_evaluator import PriceBar, evaluate_barrier_outcome, make_walk_forward_folds

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def bar(minutes, open_, high, low, close):
    return PriceBar(T0 + timedelta(minutes=minutes), open_, high, low, close)


def test_long_target_first_uses_only_post_prediction_bars():
    result = evaluate_barrier_outcome(
        prediction_at=T0, direction="LONG", target_price=105, stop_price=95,
        horizon_bars=3, bars=[bar(1, 100, 102, 99, 101), bar(2, 101, 106, 100, 105)],
    )
    assert result.label == "TARGET_FIRST"
    assert result.evaluated_bars == 2
    assert result.observed_at == T0 + timedelta(minutes=2)


def test_short_stop_first():
    result = evaluate_barrier_outcome(
        prediction_at=T0, direction="SHORT", target_price=95, stop_price=105,
        horizon_bars=2, bars=[bar(1, 100, 106, 98, 104), bar(2, 104, 104, 94, 96)],
    )
    assert result.label == "STOP_FIRST"


def test_both_barriers_touched_same_bar_is_ambiguous():
    result = evaluate_barrier_outcome(
        prediction_at=T0, direction="LONG", target_price=105, stop_price=95,
        horizon_bars=1, bars=[bar(1, 100, 106, 94, 101)],
    )
    assert result.label == "AMBIGUOUS"


def test_horizon_expired_and_insufficient_data_are_distinct():
    expired = evaluate_barrier_outcome(
        prediction_at=T0, direction="LONG", target_price=105, stop_price=95,
        horizon_bars=2, bars=[bar(1, 100, 102, 99, 101), bar(2, 101, 103, 100, 102)],
    )
    insufficient = evaluate_barrier_outcome(
        prediction_at=T0, direction="LONG", target_price=105, stop_price=95,
        horizon_bars=3, bars=[bar(1, 100, 102, 99, 101)],
    )
    assert expired.label == "HORIZON_EXPIRED"
    assert insufficient.label == "INSUFFICIENT_DATA"


def test_rejects_pre_prediction_or_unsorted_bars():
    with pytest.raises(ValueError, match="strictly ascending"):
        evaluate_barrier_outcome(
            prediction_at=T0, direction="LONG", target_price=105, stop_price=95,
            horizon_bars=2, bars=[bar(-1, 100, 101, 99, 100)],
        )


def test_rejects_invalid_ohlc():
    with pytest.raises(ValueError, match="OHLC"):
        evaluate_barrier_outcome(
            prediction_at=T0, direction="LONG", target_price=105, stop_price=95,
            horizon_bars=1, bars=[bar(1, 100, 99, 98, 101)],
        )


def test_walk_forward_folds_are_chronological_and_non_overlapping():
    folds = make_walk_forward_folds(sample_count=30, min_train_size=10, test_size=5)
    assert len(folds) == 4
    for fold in folds:
        assert fold.train_start == 0
        assert fold.train_end == fold.test_start
        assert fold.test_end - fold.test_start == 5
    assert all(left.test_end == right.test_start for left, right in zip(folds, folds[1:]))


def test_walk_forward_does_not_create_incomplete_final_test_fold():
    folds = make_walk_forward_folds(sample_count=22, min_train_size=10, test_size=5)
    assert [(f.test_start, f.test_end) for f in folds] == [(10, 15), (15, 20)]

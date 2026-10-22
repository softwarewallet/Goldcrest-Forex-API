from datetime import datetime, timedelta, timezone
import pytest
from app.research_evaluator import PriceBar, evaluate_barrier_outcome, make_walk_forward_folds

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
def bar(m, o, h, l, c): return PriceBar(T0 + timedelta(minutes=m), o, h, l, c)

def test_target_first_and_same_bar_ambiguity():
    result = evaluate_barrier_outcome(prediction_at=T0, direction="LONG", target_price=105, stop_price=95, horizon_bars=2, bars=[bar(1,100,102,99,101),bar(2,101,106,100,105)])
    assert result.label == "TARGET_FIRST"
    ambiguous = evaluate_barrier_outcome(prediction_at=T0, direction="LONG", target_price=105, stop_price=95, horizon_bars=1, bars=[bar(1,100,106,94,101)])
    assert ambiguous.label == "AMBIGUOUS"

def test_insufficient_data_and_horizon_expiry_differ():
    args = dict(prediction_at=T0, direction="LONG", target_price=105, stop_price=95)
    assert evaluate_barrier_outcome(**args, horizon_bars=2, bars=[bar(1,100,101,99,100)]).label == "INSUFFICIENT_DATA"
    assert evaluate_barrier_outcome(**args, horizon_bars=1, bars=[bar(1,100,101,99,100)]).label == "HORIZON_EXPIRED"

def test_walk_forward_test_windows_do_not_overlap():
    folds = make_walk_forward_folds(sample_count=30, min_train_size=10, test_size=5)
    assert [(f.test_start, f.test_end) for f in folds] == [(10,15),(15,20),(20,25),(25,30)]

def test_rejects_overlapping_test_windows():
    with pytest.raises(ValueError, match="step_size"):
        make_walk_forward_folds(sample_count=40, min_train_size=10, test_size=5, step_size=2)

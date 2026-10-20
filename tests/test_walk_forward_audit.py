from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.model_walk_forward import TimedBinaryExample
from app.walk_forward import WalkForwardFold, WalkForwardPlan
from app.walk_forward_audit import audit_walk_forward_plan


def make_plan(*, step_bars: int = 2) -> tuple[WalkForwardPlan, list[TimedBinaryExample]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = []
    examples = []
    for index in range(12):
        timestamp = start + timedelta(hours=index)
        bar = type("Bar", (), {
            "timestamp": timestamp, "symbol": "EUR/USD", "timeframe": "H1",
        })()
        bars.append(bar)
        examples.append(TimedBinaryExample(
            timestamp=timestamp,
            features={"x": float(index)},
            label=index % 2,
            label_available_at=timestamp + (timedelta(hours=2) if index in (2, 3) else timedelta(0)),
        ))
    fold1 = WalkForwardFold(
        fold_index=0, train=tuple(bars[0:4]), validation=tuple(bars[5:7]),
        test=tuple(bars[8:10]), purged_rows=2,
        train_start=bars[0].timestamp.isoformat(), train_end=bars[3].timestamp.isoformat(),
        validation_start=bars[5].timestamp.isoformat(), validation_end=bars[6].timestamp.isoformat(),
        test_start=bars[8].timestamp.isoformat(), test_end=bars[9].timestamp.isoformat(),
    )
    fold2 = WalkForwardFold(
        fold_index=1, train=tuple(bars[0:6]), validation=tuple(bars[7:9]),
        test=tuple(bars[9:11]), purged_rows=2,
        train_start=bars[0].timestamp.isoformat(), train_end=bars[5].timestamp.isoformat(),
        validation_start=bars[7].timestamp.isoformat(), validation_end=bars[8].timestamp.isoformat(),
        test_start=bars[9].timestamp.isoformat(), test_end=bars[10].timestamp.isoformat(),
    )
    return WalkForwardPlan(
        symbol="EUR/USD", timeframe="H1", horizon_bars=1,
        initial_train_bars=4, validation_bars=2, test_bars=2, step_bars=step_bars,
        folds=(fold1, fold2), unused_tail_bars=1,
    ), examples


def test_audit_reports_label_maturity_and_overlapping_test_timestamps():
    plan, examples = make_plan()
    audit = audit_walk_forward_plan(plan, examples)
    assert audit["actionable"] is False
    assert audit["fold_count"] == 2
    assert audit["training_rows_excluded_unmatured_across_folds"] == 1
    assert audit["test_windows_overlap"] is True
    assert audit["repeated_test_timestamp_observations"] == 1
    assert audit["unique_test_timestamps"] == 3
    assert audit["feature_causality_verified"] is False
    assert audit["folds"][0]["partitions_disjoint"] is True


def test_audit_rejects_partition_overlap():
    plan, examples = make_plan()
    fold = plan.folds[0]
    broken = WalkForwardFold(
        fold_index=0, train=fold.train, validation=fold.validation,
        test=(fold.validation[-1], *fold.test[1:]), purged_rows=fold.purged_rows,
        train_start=fold.train_start, train_end=fold.train_end,
        validation_start=fold.validation_start, validation_end=fold.validation_end,
        test_start=fold.test_start, test_end=fold.test_end,
    )
    bad_plan = WalkForwardPlan(
        symbol=plan.symbol, timeframe=plan.timeframe, horizon_bars=plan.horizon_bars,
        initial_train_bars=plan.initial_train_bars, validation_bars=plan.validation_bars,
        test_bars=plan.test_bars, step_bars=plan.step_bars, folds=(broken,),
        unused_tail_bars=plan.unused_tail_bars,
    )
    with pytest.raises(ValueError, match="share observation timestamps"):
        audit_walk_forward_plan(bad_plan, examples)


def test_audit_rejects_examples_missing_fold_timestamps():
    plan, examples = make_plan()
    with pytest.raises(ValueError, match="missing from examples"):
        audit_walk_forward_plan(plan, examples[:-1])

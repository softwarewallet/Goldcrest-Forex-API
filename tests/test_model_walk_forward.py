from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest

from app.evaluation import ForexBar
from app.model_walk_forward import TimedBinaryExample, evaluate_binary_walk_forward
from app.models.tabular import BinaryModelOutput
from app.walk_forward import generate_walk_forward_folds


def make_bars(count=120):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    result = []
    for i in range(count):
        close = 1.1 + (i % 7) * 0.0001
        result.append(ForexBar(
            symbol="EUR/USD", timeframe="M5",
            timestamp=start + timedelta(minutes=5*i), open=close,
            high=close + 0.0001, low=close - 0.0001, close=close,
        ))
    return result


class FakeModel:
    model_type = "fixture"
    model_version = "test-v1"

    def fit(self, feature_rows, labels, **kwargs):
        assert len(feature_rows) == len(labels)
        self.mean = sum(labels) / len(labels)
        return self

    def predict(self, feature_row):
        return BinaryModelOutput(self.model_type, self.model_version, float(self.mean))


def examples_for(bars, delay=2):
    return [
        TimedBinaryExample(
            timestamp=bar.timestamp,
            features={"close_delta": float(i % 3), "hour_bucket": float(i % 24)},
            label=i % 2,
            label_available_at=bars[min(i + delay, len(bars)-1)].timestamp,
        )
        for i, bar in enumerate(bars)
    ]


def test_walk_forward_fits_each_fold_and_reports_non_actionable_metrics():
    bars = make_bars()
    plan = generate_walk_forward_folds(
        bars, initial_train_bars=60, validation_bars=20,
        test_bars=20, step_bars=10, horizon_bars=2,
    )
    results = evaluate_binary_walk_forward(
        plan, examples_for(bars), feature_names=("close_delta", "hour_bucket"),
        model_factory=FakeModel,
    )
    assert len(results) == len(plan.folds)
    assert results[0].train_rows_excluded_unmatured > 0
    assert results[0].validation.sample_count > 0
    assert results[0].test.sample_count > 0
    assert results[0].actionable is False
    assert results[0].validation.brier_score >= 0
    assert results[0].validation.log_loss > 0


def test_rejects_unordered_or_mismatched_feature_rows():
    bars = make_bars()
    plan = generate_walk_forward_folds(
        bars, initial_train_bars=60, validation_bars=20,
        test_bars=20, step_bars=10, horizon_bars=2,
    )
    examples = examples_for(bars)
    examples[1], examples[2] = examples[2], examples[1]
    with pytest.raises(ValueError, match="chronological"):
        evaluate_binary_walk_forward(
            plan, examples, feature_names=("close_delta", "hour_bucket"),
            model_factory=FakeModel,
        )
    with pytest.raises(ValueError, match="schema"):
        evaluate_binary_walk_forward(
            plan, examples_for(bars), feature_names=("close_delta",),
            model_factory=FakeModel,
        )


def test_fails_closed_when_training_labels_have_one_class():
    bars = make_bars()
    plan = generate_walk_forward_folds(
        bars, initial_train_bars=60, validation_bars=20,
        test_bars=20, step_bars=10, horizon_bars=2,
    )
    rows = examples_for(bars)
    rows = [TimedBinaryExample(r.timestamp, r.features, 0, r.label_available_at) for r in rows]
    with pytest.raises(ValueError, match="both classes"):
        evaluate_binary_walk_forward(
            plan, rows, feature_names=("close_delta", "hour_bucket"),
            model_factory=FakeModel,
        )


def test_rejects_nonchronological_or_missing_fold_partitions():
    bars = make_bars()
    plan = generate_walk_forward_folds(
        bars, initial_train_bars=60, validation_bars=20,
        test_bars=20, step_bars=10, horizon_bars=2,
    )
    fold = plan.folds[0]
    reordered_fold = replace(fold, validation=fold.test, test=fold.validation)
    reordered_plan = replace(plan, folds=(reordered_fold,))
    with pytest.raises(ValueError, match="strictly chronological"):
        evaluate_binary_walk_forward(
            reordered_plan, examples_for(bars),
            feature_names=("close_delta", "hour_bucket"), model_factory=FakeModel,
        )

    missing_time = fold.validation[0].timestamp
    missing_examples = [row for row in examples_for(bars) if row.timestamp != missing_time]
    with pytest.raises(ValueError, match="timestamps missing from examples"):
        evaluate_binary_walk_forward(
            plan, missing_examples,
            feature_names=("close_delta", "hour_bucket"), model_factory=FakeModel,
        )


def test_rejects_float_labels_instead_of_silently_accepting_them():
    bars = make_bars()
    plan = generate_walk_forward_folds(
        bars, initial_train_bars=60, validation_bars=20,
        test_bars=20, step_bars=10, horizon_bars=2,
    )
    rows = examples_for(bars)
    rows[0] = TimedBinaryExample(rows[0].timestamp, rows[0].features, 1.0, rows[0].label_available_at)
    with pytest.raises(ValueError, match="integer 0 or 1"):
        evaluate_binary_walk_forward(
            plan, rows, feature_names=("close_delta", "hour_bucket"), model_factory=FakeModel,
        )

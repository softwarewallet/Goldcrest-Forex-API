from datetime import datetime, timedelta, timezone

import pytest

from app.evaluation import ForexBar
from app.model_comparison import TrainingPrevalenceBaseline, benchmark_binary_models
from app.model_walk_forward import TimedBinaryExample
from app.models.tabular import BinaryModelOutput
from app.walk_forward import generate_walk_forward_folds


def make_bars(count=160):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    result = []
    for i in range(count):
        close = 1.1 + (i % 7) * 0.0001
        result.append(ForexBar(
            symbol="EUR/USD", timeframe="M5",
            timestamp=start + timedelta(minutes=5 * i), open=close,
            high=close + 0.0001, low=close - 0.0001, close=close,
        ))
    return result


def examples_for(bars, delay=2):
    return [
        TimedBinaryExample(
            timestamp=bar.timestamp,
            features={"feature_a": float(i % 3), "feature_b": float(i % 5)},
            label=int(i % 2),
            label_available_at=bars[min(i + delay, len(bars) - 1)].timestamp,
        )
        for i, bar in enumerate(bars)
    ]


class ConstantCandidate:
    model_type = "constant_candidate"
    model_version = "fixture-v1"

    def fit(self, feature_rows, labels, **kwargs):
        self.p = 0.75
        return self

    def predict(self, feature_row):
        return BinaryModelOutput(self.model_type, self.model_version, self.p)


def plan_for(bars):
    return generate_walk_forward_folds(
        bars, initial_train_bars=80, validation_bars=25,
        test_bars=25, step_bars=23, horizon_bars=2,
    )


def test_prevalence_baseline_uses_only_supplied_training_labels():
    baseline = TrainingPrevalenceBaseline()
    baseline.fit([{"x": 1.0}] * 4, [1, 1, 0, 0])
    assert baseline.predict({"x": 999.0}).probability_class_1 == 0.5
    with pytest.raises(RuntimeError):
        TrainingPrevalenceBaseline().predict({"x": 1.0})


def test_benchmark_uses_same_folds_and_emits_candidate_minus_baseline_deltas():
    bars = make_bars()
    result = benchmark_binary_models(
        plan_for(bars), examples_for(bars),
        feature_names=("feature_a", "feature_b"),
        model_factories={"candidate": ConstantCandidate},
    )
    assert result.actionable is False
    assert set(result.fold_results) == {"candidate", "training_prevalence"}
    assert len(result.fold_results["candidate"]) == len(result.fold_results["training_prevalence"])
    assert len(result.differences_vs_baseline) == len(plan_for(bars).folds) * 2
    assert {row.partition for row in result.differences_vs_baseline} == {"validation", "test"}
    assert all(row.sample_count > 0 for row in result.differences_vs_baseline)
    assert all(fold.actionable is False for folds in result.fold_results.values() for fold in folds)


def test_rejects_missing_candidates_or_conflicting_baseline_name():
    bars = make_bars()
    plan = plan_for(bars)
    examples = examples_for(bars)
    with pytest.raises(ValueError, match="at least one"):
        benchmark_binary_models(plan, examples, feature_names=("feature_a",), model_factories={})
    with pytest.raises(ValueError, match="conflicts"):
        benchmark_binary_models(
            plan, examples, feature_names=("feature_a", "feature_b"),
            model_factories={"training_prevalence": ConstantCandidate},
        )

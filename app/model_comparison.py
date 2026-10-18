"""Fold-consistent offline benchmark for binary research models.

All candidates receive the same chronological fold plan, examples, feature schema,
and label-maturity rules. Results are descriptive research metrics only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from app.model_walk_forward import (
    BinaryMetrics,
    BinaryResearchModel,
    ModelFoldResult,
    TimedBinaryExample,
    evaluate_binary_walk_forward,
)
from app.models.tabular import BinaryModelOutput
from app.walk_forward import WalkForwardPlan


class TrainingPrevalenceBaseline:
    """Constant-probability baseline fitted exclusively on a fold's training labels."""

    model_type = "training_prevalence_baseline"

    def __init__(self, model_version: str = "prevalence-v1") -> None:
        if not model_version:
            raise ValueError("model_version must not be empty")
        self.model_version = model_version
        self._probability: float | None = None

    def fit(self, feature_rows, labels, **kwargs):
        if not labels or len(feature_rows) != len(labels):
            raise ValueError("baseline requires non-empty aligned training rows and labels")
        if any(isinstance(y, bool) or not isinstance(y, int) or y not in (0, 1) for y in labels):
            raise ValueError("baseline labels must be integer 0 or 1")
        # No validation/test data is passed to this fit call.
        self._probability = sum(labels) / len(labels)
        return self

    def predict(self, feature_row):
        if self._probability is None:
            raise RuntimeError("baseline has not been fitted")
        return BinaryModelOutput(self.model_type, self.model_version, self._probability)


@dataclass(frozen=True)
class FoldMetricDifference:
    fold_index: int
    candidate_name: str
    partition: str
    sample_count: int
    brier_difference_candidate_minus_baseline: float
    log_loss_difference_candidate_minus_baseline: float
    accuracy_difference_candidate_minus_baseline: float
    mean_probability_difference_candidate_minus_baseline: float


@dataclass(frozen=True)
class ModelBenchmark:
    fold_results: Mapping[str, tuple[ModelFoldResult, ...]]
    differences_vs_baseline: tuple[FoldMetricDifference, ...]
    baseline_name: str
    feature_names: tuple[str, ...]
    actionable: bool = False
    interpretation: str = (
        "RESEARCH_ONLY_FOLD_METRICS; NOT A PAIRED SIGNIFICANCE TEST OR PROFITABILITY CLAIM"
    )


def _difference(
    fold_index: int,
    candidate_name: str,
    partition: str,
    candidate: BinaryMetrics,
    baseline: BinaryMetrics,
) -> FoldMetricDifference:
    if candidate.sample_count != baseline.sample_count:
        raise ValueError("candidate and baseline must score identical observations per fold")
    return FoldMetricDifference(
        fold_index=fold_index,
        candidate_name=candidate_name,
        partition=partition,
        sample_count=candidate.sample_count,
        brier_difference_candidate_minus_baseline=candidate.brier_score - baseline.brier_score,
        log_loss_difference_candidate_minus_baseline=candidate.log_loss - baseline.log_loss,
        accuracy_difference_candidate_minus_baseline=(
            candidate.accuracy_at_fixed_half - baseline.accuracy_at_fixed_half
        ),
        mean_probability_difference_candidate_minus_baseline=(
            candidate.mean_predicted_probability - baseline.mean_predicted_probability
        ),
    )


def benchmark_binary_models(
    plan: WalkForwardPlan,
    examples: Sequence[TimedBinaryExample],
    *,
    feature_names: Sequence[str],
    model_factories: Mapping[str, Callable[[], BinaryResearchModel]],
    baseline_name: str = "training_prevalence",
) -> ModelBenchmark:
    """Evaluate candidates and a train-fold prevalence baseline on identical folds.

    Factories must return fresh, untrained models. The baseline prevalence is estimated
    only from matured labels available in each fold's training window. This function
    does not perform hyperparameter search, calibration, threshold optimization, or
    automatic model selection. Keep final-test metrics untouched until decisions freeze.
    """
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")
    if baseline_name in model_factories:
        raise ValueError("baseline_name conflicts with a candidate name")
    if not model_factories:
        raise ValueError("at least one candidate model factory is required")
    if any(not name.strip() for name in model_factories):
        raise ValueError("candidate names must be non-empty")

    factories: dict[str, Callable[[], BinaryResearchModel]] = dict(model_factories)
    factories[baseline_name] = TrainingPrevalenceBaseline
    results: dict[str, tuple[ModelFoldResult, ...]] = {}
    for name, factory in factories.items():
        model_results = evaluate_binary_walk_forward(
            plan, examples, feature_names=names, model_factory=factory
        )
        if len(model_results) != len(plan.folds):
            raise ValueError("candidate produced a different fold count")
        results[name] = model_results

    baseline = results[baseline_name]
    differences: list[FoldMetricDifference] = []
    for name, candidate_folds in results.items():
        if name == baseline_name:
            continue
        for candidate, base in zip(candidate_folds, baseline):
            if candidate.fold_index != base.fold_index:
                raise ValueError("candidate and baseline fold identities do not match")
            if (candidate.validation.sample_count != base.validation.sample_count
                    or candidate.test.sample_count != base.test.sample_count):
                raise ValueError("candidate and baseline observations are not aligned")
            differences.append(_difference(
                candidate.fold_index, name, "validation", candidate.validation, base.validation
            ))
            differences.append(_difference(
                candidate.fold_index, name, "test", candidate.test, base.test
            ))
    return ModelBenchmark(
        fold_results=results,
        differences_vs_baseline=tuple(differences),
        baseline_name=baseline_name,
        feature_names=names,
        actionable=False,
    )

"""Leakage-aware binary model walk-forward evaluation for offline research.

Features and labels must be prepared externally under a documented, point-in-time
feature specification. Labels carry the timestamp at which their outcome became known.
This module never invents features, fetches market data, chooses a trade threshold, or
connects research model outputs to serving/trading.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping, Protocol, Sequence

from app.walk_forward import WalkForwardPlan


class BinaryResearchModel(Protocol):
    model_type: str
    model_version: str

    def fit(self, feature_rows: Sequence[Mapping[str, float]], labels: Sequence[int], **kwargs: object) -> object: ...
    def predict(self, feature_row: Mapping[str, float]) -> object: ...


@dataclass(frozen=True)
class TimedBinaryExample:
    timestamp: datetime
    features: Mapping[str, float]
    label: int
    label_available_at: datetime


@dataclass(frozen=True)
class BinaryMetrics:
    sample_count: int
    positive_count: int
    negative_count: int
    prevalence: float
    brier_score: float
    log_loss: float
    accuracy_at_fixed_half: float
    mean_predicted_probability: float


@dataclass(frozen=True)
class ModelFoldResult:
    fold_index: int
    model_type: str
    model_version: str
    train_rows_available: int
    train_rows_excluded_unmatured: int
    validation: BinaryMetrics
    test: BinaryMetrics
    probability_semantics: str
    actionable: bool = False


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _validate_examples(examples: Sequence[TimedBinaryExample], feature_names: Sequence[str]) -> None:
    if not examples:
        raise ValueError("examples must not be empty")
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names) or any(not n for n in names):
        raise ValueError("feature_names must be non-empty and unique")
    previous: datetime | None = None
    for example in examples:
        timestamp = _utc(example.timestamp, "feature")
        available = _utc(example.label_available_at, "label availability")
        if previous is not None and timestamp <= previous:
            raise ValueError("examples must be strictly chronological with unique timestamps")
        previous = timestamp
        if available < timestamp:
            raise ValueError("label cannot be available before its feature timestamp")
        if example.label not in (0, 1) or isinstance(example.label, bool):
            raise ValueError("labels must be explicitly supplied as integer 0 or 1")
        if set(example.features) != set(names):
            raise ValueError("feature schema mismatch")
        for name in names:
            value = example.features[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"feature {name!r} must be finite numeric data")


def _metrics(labels: Sequence[int], probabilities: Sequence[float]) -> BinaryMetrics:
    if not labels or len(labels) != len(probabilities):
        raise ValueError("metrics require non-empty aligned labels and probabilities")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("predicted probabilities must be finite and within [0, 1]")
    eps = 1e-15
    brier = sum((p - y) ** 2 for y, p in zip(labels, probabilities)) / len(labels)
    logloss = -sum(y * math.log(max(eps, p)) + (1-y) * math.log(max(eps, 1-p))
                   for y, p in zip(labels, probabilities)) / len(labels)
    return BinaryMetrics(
        sample_count=len(labels),
        positive_count=sum(labels),
        negative_count=len(labels) - sum(labels),
        prevalence=sum(labels) / len(labels),
        brier_score=brier,
        log_loss=logloss,
        accuracy_at_fixed_half=sum((p >= 0.5) == bool(y) for y, p in zip(labels, probabilities)) / len(labels),
        mean_predicted_probability=sum(probabilities) / len(probabilities),
    )


def evaluate_binary_walk_forward(
    plan: WalkForwardPlan,
    examples: Sequence[TimedBinaryExample],
    *,
    feature_names: Sequence[str],
    model_factory: Callable[[], BinaryResearchModel],
) -> tuple[ModelFoldResult, ...]:
    """Fit a fresh model per fold and score validation/test without threshold tuning.

    Training examples are restricted to feature timestamps in the fold's train partition
    whose labels had matured by the final train timestamp. Validation/test labels are
    used only for reporting. Test results must remain untouched during model/feature
    selection. Every fold must have adequate examples of both classes in training.
    """
    _validate_examples(examples, feature_names)
    by_time = {_utc(row.timestamp, "feature"): row for row in examples}
    results: list[ModelFoldResult] = []
    for fold in plan.folds:
        train_times = {_utc(bar.timestamp, "fold") for bar in fold.train}
        validation_times = {_utc(bar.timestamp, "fold") for bar in fold.validation}
        test_times = {_utc(bar.timestamp, "fold") for bar in fold.test}
        train_end = max(train_times)
        train_all = [by_time[t] for t in sorted(train_times) if t in by_time]
        train = [row for row in train_all if _utc(row.label_available_at, "label availability") <= train_end]
        unmatured = len(train_all) - len(train)
        validation = [by_time[t] for t in sorted(validation_times) if t in by_time]
        test = [by_time[t] for t in sorted(test_times) if t in by_time]
        if not train or not validation or not test:
            raise ValueError("each fold requires feature/label records in train, validation, and test")
        if len({row.label for row in train}) != 2:
            raise ValueError("matured training labels must contain both classes")
        model = model_factory()
        model.fit([row.features for row in train], [row.label for row in train])
        def predict(rows: Sequence[TimedBinaryExample]) -> BinaryMetrics:
            probabilities = []
            for row in rows:
                output = model.predict(row.features)
                probability = getattr(output, "probability_class_1", None)
                if probability is None:
                    raise ValueError("model output lacks probability_class_1")
                probabilities.append(float(probability))
            return _metrics([row.label for row in rows], probabilities)
        results.append(ModelFoldResult(
            fold_index=fold.fold_index,
            model_type=str(getattr(model, "model_type", type(model).__name__)),
            model_version=str(getattr(model, "model_version", "unknown")),
            train_rows_available=len(train),
            train_rows_excluded_unmatured=unmatured,
            validation=predict(validation),
            test=predict(test),
            probability_semantics="uncalibrated research probability; metrics are not trading performance",
            actionable=False,
        ))
    if not results:
        raise ValueError("walk-forward plan contains no folds")
    return tuple(results)

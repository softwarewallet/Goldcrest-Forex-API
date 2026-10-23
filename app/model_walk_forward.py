"""Leakage-aware binary model walk-forward evaluation for offline research.

Features and labels must be prepared externally under a documented, point-in-time
feature specification. Labels carry the timestamp at which their outcome became known.
This module never invents features, fetches market data, chooses a trade threshold, or
connects research model outputs to serving/trading.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping, Protocol, Sequence

from app.calibration_uncertainty import CalibrationUncertainty, moving_block_ece_uncertainty
from app.probability_diagnostics import ReliabilityBin, probability_diagnostics
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
    label_eligible: bool = True


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
    expected_calibration_error: float = 0.0
    reliability_bins: tuple[ReliabilityBin, ...] = ()
    observation_identity_sha256: str | None = None
    calibration_uncertainty: CalibrationUncertainty | None = None


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
    train_rows_excluded_ineligible: int = 0
    validation_rows_excluded_ineligible: int = 0
    test_rows_excluded_ineligible: int = 0


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
        if isinstance(example.label, bool) or not isinstance(example.label, int) or example.label not in (0, 1):
            raise ValueError("labels must be explicitly supplied as integer 0 or 1")
        if not isinstance(example.label_eligible, bool):
            raise ValueError("label_eligible must be a boolean")
        if set(example.features) != set(names):
            raise ValueError("feature schema mismatch")
        for name in names:
            value = example.features[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"feature {name!r} must be finite numeric data")


def _metrics(
    labels: Sequence[int], probabilities: Sequence[float], *,
    observation_timestamps: Sequence[datetime] | None = None,
    calibration_block_length: int = 1,
    calibration_bootstrap_resamples: int = 2000,
    calibration_confidence_level: float = 0.95,
    calibration_random_seed: int = 20261020,
) -> BinaryMetrics:
    if not labels or len(labels) != len(probabilities):
        raise ValueError("metrics require non-empty aligned labels and probabilities")
    identity_sha256 = None
    if observation_timestamps is not None:
        if len(observation_timestamps) != len(labels):
            raise ValueError("observation timestamps must align with labels and probabilities")
        identities = [_utc(value, "observation identity").isoformat().replace("+00:00", "Z")
                      for value in observation_timestamps]
        if len(set(identities)) != len(identities):
            raise ValueError("observation timestamps must be unique within a scored partition")
        identity_sha256 = hashlib.sha256(json.dumps(
            sorted(identities), separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("predicted probabilities must be finite and within [0, 1]")
    eps = 1e-15
    brier = sum((p - y) ** 2 for y, p in zip(labels, probabilities)) / len(labels)
    logloss = -sum(y * math.log(max(eps, p)) + (1-y) * math.log(max(eps, 1-p))
                   for y, p in zip(labels, probabilities)) / len(labels)
    calibration = probability_diagnostics(labels, probabilities, bins=10)
    uncertainty = moving_block_ece_uncertainty(
        labels, probabilities, block_length=min(calibration_block_length, len(labels)),
        bootstrap_resamples=calibration_bootstrap_resamples,
        confidence_level=calibration_confidence_level,
        seed=calibration_random_seed,
        bins=10,
    )
    return BinaryMetrics(
        sample_count=len(labels),
        positive_count=sum(labels),
        negative_count=len(labels) - sum(labels),
        prevalence=sum(labels) / len(labels),
        brier_score=brier,
        log_loss=logloss,
        accuracy_at_fixed_half=sum((p >= 0.5) == bool(y) for y, p in zip(labels, probabilities)) / len(labels),
        mean_predicted_probability=sum(probabilities) / len(probabilities),
        expected_calibration_error=calibration.expected_calibration_error,
        reliability_bins=calibration.reliability_bins,
        observation_identity_sha256=identity_sha256,
        calibration_uncertainty=uncertainty,
    )


def evaluate_binary_walk_forward(
    plan: WalkForwardPlan,
    examples: Sequence[TimedBinaryExample],
    *,
    feature_names: Sequence[str],
    model_factory: Callable[[], BinaryResearchModel],
    calibration_block_length: int = 1,
    calibration_bootstrap_resamples: int = 2000,
    calibration_confidence_level: float = 0.95,
    calibration_random_seed: int = 20261020,
) -> tuple[ModelFoldResult, ...]:
    """Fit a fresh model per fold and score validation/test without threshold tuning.

    Training examples are restricted to feature timestamps in the fold's train partition
    whose labels had matured by the final train timestamp. Validation/test labels are
    used only for reporting. Test results must remain untouched during model/feature
    selection. Every fold must have adequate examples of both classes in training.
    """
    _validate_examples(examples, feature_names)
    if calibration_block_length < 1 or calibration_bootstrap_resamples < 100:
        raise ValueError("invalid calibration bootstrap configuration")
    if not 0 < calibration_confidence_level < 1:
        raise ValueError("calibration_confidence_level must be between 0 and 1")
    by_time = {_utc(row.timestamp, "feature"): row for row in examples}
    results: list[ModelFoldResult] = []
    if not plan.folds:
        raise ValueError("walk-forward plan contains no folds")
    seen_fold_indices: set[int] = set()
    for fold in plan.folds:
        if isinstance(fold.fold_index, bool) or not isinstance(fold.fold_index, int) or fold.fold_index < 0:
            raise ValueError("fold indices must be non-negative integers")
        if fold.fold_index in seen_fold_indices:
            raise ValueError("walk-forward plan contains duplicate fold indices")
        seen_fold_indices.add(fold.fold_index)
        train_times = {_utc(bar.timestamp, "fold") for bar in fold.train}
        validation_times = {_utc(bar.timestamp, "fold") for bar in fold.validation}
        test_times = {_utc(bar.timestamp, "fold") for bar in fold.test}
        if not train_times or not validation_times or not test_times:
            raise ValueError("each fold requires non-empty train, validation, and test partitions")
        if (train_times & validation_times) or (train_times & test_times) or (validation_times & test_times):
            raise ValueError("fold partitions must not share observation timestamps")
        if max(train_times) >= min(validation_times) or max(validation_times) >= min(test_times):
            raise ValueError("fold partitions must be strictly chronological: train before validation before test")
        missing_times = (train_times | validation_times | test_times) - by_time.keys()
        if missing_times:
            raise ValueError(f"fold references {len(missing_times)} timestamps missing from examples")
        train_end = max(train_times)
        train_all = [by_time[t] for t in sorted(train_times)]
        train_eligible = [row for row in train_all if row.label_eligible]
        train = [
            row for row in train_eligible
            if _utc(row.label_available_at, "label availability") <= train_end
        ]
        unmatured = len(train_eligible) - len(train)
        validation_all = [by_time[t] for t in sorted(validation_times)]
        validation = [row for row in validation_all if row.label_eligible]
        test_all = [by_time[t] for t in sorted(test_times)]
        test = [row for row in test_all if row.label_eligible]
        if not train or not validation or not test:
            raise ValueError("each fold requires eligible feature/label records in train, validation, and test")
        if len({row.label for row in train}) != 2:
            raise ValueError("matured training labels must contain both classes")
        model = model_factory()
        model.fit([row.features for row in train], [row.label for row in train])

        def predict(rows: Sequence[TimedBinaryExample], partition: str) -> BinaryMetrics:
            probabilities = []
            for row in rows:
                output = model.predict(row.features)
                probability = getattr(output, "probability_class_1", None)
                if probability is None:
                    raise ValueError("model output lacks probability_class_1")
                probabilities.append(float(probability))
            # Derive a stable but fold/partition-specific seed without depending on Python hash().
            seed_payload = f"{calibration_random_seed}:{fold.fold_index}:{partition}"
            seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:8], 16)
            return _metrics(
                [row.label for row in rows], probabilities,
                observation_timestamps=[row.timestamp for row in rows],
                calibration_block_length=calibration_block_length,
                calibration_bootstrap_resamples=calibration_bootstrap_resamples,
                calibration_confidence_level=calibration_confidence_level,
                calibration_random_seed=seed,
            )

        results.append(ModelFoldResult(
            fold_index=fold.fold_index,
            model_type=str(getattr(model, "model_type", type(model).__name__)),
            model_version=str(getattr(model, "model_version", "unknown")),
            train_rows_available=len(train),
            train_rows_excluded_unmatured=unmatured,
            validation=predict(validation, "validation"),
            test=predict(test, "test"),
            probability_semantics="uncalibrated research probability; metrics are not trading performance",
            actionable=False,
            train_rows_excluded_ineligible=len(train_all) - len(train_eligible),
            validation_rows_excluded_ineligible=len(validation_all) - len(validation),
            test_rows_excluded_ineligible=len(test_all) - len(test),
        ))
    if not results:
        raise ValueError("walk-forward plan contains no folds")
    return tuple(results)

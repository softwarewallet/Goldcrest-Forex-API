"""Optuna tuning on an explicitly supplied train/validation pair; research-only.

This utility deliberately has no test-partition argument. Callers must reserve final
test folds for evaluation only after tuning choices have been frozen.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

import optuna
from sklearn.metrics import log_loss

from app.model_walk_forward import TimedBinaryExample, _utc
from app.models.tabular import LightGBMResearchModel, XGBoostResearchModel
from app.resource_profile import get_ml_thread_count


@dataclass(frozen=True)
class OptunaTuningResult:
    model_type: str
    feature_names: tuple[str, ...]
    feature_schema_sha256: str
    study_name: str
    seed: int
    trial_count_requested: int
    trial_count_completed: int
    best_validation_log_loss: float
    best_params: Mapping[str, object]
    train_rows_available: int
    train_rows_excluded_unmatured: int
    validation_rows: int
    validation_start_utc: str
    validation_end_utc: str
    selection_metric: str = "validation_log_loss_minimize"
    test_partition_used: bool = False
    actionable: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        payload["best_params"] = dict(self.best_params)
        return payload


def _validate_partition(rows: Sequence[TimedBinaryExample], feature_names: tuple[str, ...], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} partition must not be empty")
    previous: datetime | None = None
    for row in rows:
        timestamp = _utc(row.timestamp, name)
        available = _utc(row.label_available_at, "label availability")
        if previous is not None and timestamp <= previous:
            raise ValueError(f"{name} timestamps must be strictly chronological and unique")
        previous = timestamp
        if available < timestamp:
            raise ValueError("label availability cannot precede feature timestamp")
        if set(row.features) != set(feature_names):
            raise ValueError("feature schema mismatch")
        for feature, value in row.features.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"feature {feature!r} must be numeric")
            import math
            if not math.isfinite(float(value)):
                raise ValueError(f"feature {feature!r} must be finite")
        if isinstance(row.label, bool) or row.label not in (0, 1):
            raise ValueError("labels must be explicitly supplied as 0 or 1")


def tune_on_validation(
    train_examples: Sequence[TimedBinaryExample],
    validation_examples: Sequence[TimedBinaryExample],
    *,
    feature_names: Sequence[str],
    model_type: str,
    n_trials: int = 25,
    seed: int = 20261021,
    study_name: str = "goldcrest-validation-tuning",
) -> OptunaTuningResult:
    """Tune XGBoost/LightGBM using only train and validation examples.

    Unmatured training labels are excluded at the train cutoff. Validation must occur
    strictly after the training cutoff. Validation log-loss ranks trials; the caller
    must not use this validation partition as a final test set.
    """
    names = tuple(feature_names)
    if not names or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("feature_names must contain non-empty names")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    if model_type not in {"xgboost", "lightgbm"}:
        raise ValueError("model_type must be 'xgboost' or 'lightgbm'")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or not 1 <= n_trials <= 100:
        raise ValueError("n_trials must be an integer between 1 and 100")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(study_name, str) or not study_name.strip():
        raise ValueError("study_name must not be empty")

    _validate_partition(train_examples, names, "train")
    _validate_partition(validation_examples, names, "validation")
    train_start = _utc(train_examples[0].timestamp, "train")
    train_end = _utc(train_examples[-1].timestamp, "train")
    validation_start = _utc(validation_examples[0].timestamp, "validation")
    validation_end = _utc(validation_examples[-1].timestamp, "validation")
    if train_start >= train_end:
        raise ValueError("training partition must contain at least two timestamps")
    if train_end >= validation_start:
        raise ValueError("validation must begin strictly after the training partition")

    matured_train = [
        row for row in train_examples
        if _utc(row.label_available_at, "label availability") <= train_end
    ]
    excluded = len(train_examples) - len(matured_train)
    if len(matured_train) < 4 or len({row.label for row in matured_train}) != 2:
        raise ValueError("matured training rows must contain at least four rows and both classes")
    if len(validation_examples) < 2 or len({row.label for row in validation_examples}) != 2:
        raise ValueError("validation rows must contain both classes and at least two rows")

    from app.models.tabular import feature_schema_fingerprint
    feature_sha = feature_schema_fingerprint(names)
    train_x = [row.features for row in matured_train]
    train_y = [row.label for row in matured_train]
    valid_x = [row.features for row in validation_examples]
    valid_y = [row.label for row in validation_examples]
    threads = get_ml_thread_count()

    def objective(trial: optuna.Trial) -> float:
        if model_type == "xgboost":
            params: dict[str, object] = {
                "n_estimators": trial.suggest_int("n_estimators", 40, 180, step=20),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "n_jobs": threads,
                "random_state": seed,
            }
            model = XGBoostResearchModel(names, model_version="optuna-research-v1")
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 40, 180, step=20),
                "num_leaves": trial.suggest_int("num_leaves", 7, 31),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "n_jobs": threads,
                "random_state": seed,
            }
            model = LightGBMResearchModel(names, model_version="optuna-research-v1")
        model.fit(train_x, train_y, params=params)
        probabilities = [model.predict(row).probability_class_1 for row in valid_x]
        score = float(log_loss(valid_y, probabilities, labels=[0, 1]))
        if not __import__("math").isfinite(score):
            raise ValueError("validation log-loss is non-finite")
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed or study.best_trial is None:
        raise RuntimeError("Optuna produced no completed trials")

    return OptunaTuningResult(
        model_type=model_type,
        feature_names=names,
        feature_schema_sha256=feature_sha,
        study_name=study_name,
        seed=seed,
        trial_count_requested=n_trials,
        trial_count_completed=len(completed),
        best_validation_log_loss=float(study.best_value),
        best_params=dict(study.best_params),
        train_rows_available=len(matured_train),
        train_rows_excluded_unmatured=excluded,
        validation_rows=len(validation_examples),
        validation_start_utc=validation_start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        validation_end_utc=validation_end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        actionable=False,
    )


def write_tuning_report(result: OptunaTuningResult, path: str) -> str:
    """Write deterministic, non-actionable JSON tuning provenance."""
    from pathlib import Path
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["report_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload["warning"] = "Validation-selected hyperparameters are research-only; do not interpret as test performance or trading profitability."
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return str(destination)

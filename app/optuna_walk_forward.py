"""End-to-end Optuna + purged walk-forward evaluation, strictly research-only.

Each fold tunes on its own chronological train/validation partitions, then refits the
selected configuration on labels matured before the test window and scores the untouched
test partition. No result from this module is actionable or connected to execution.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.evaluation import ForexBar
from app.model_walk_forward import TimedBinaryExample, _metrics, _utc
from app.models.tabular import LightGBMResearchModel, XGBoostResearchModel, feature_schema_fingerprint
from app.optuna_validation_tuning import tune_on_validation
from app.resource_profile import build_resource_profile
from app.walk_forward import WalkForwardPlan, generate_walk_forward_folds


@dataclass(frozen=True)
class TunedFoldResult:
    fold_index: int
    train_start_utc: str
    train_end_utc: str
    validation_start_utc: str
    validation_end_utc: str
    test_start_utc: str
    test_end_utc: str
    model_type: str
    best_params: Mapping[str, object]
    validation_log_loss: float
    trials_requested: int
    trials_completed: int
    train_rows_available_for_tuning: int
    train_rows_excluded_unmatured: int
    refit_rows_available: int
    refit_rows_excluded_unmatured: int
    test_metrics: Mapping[str, Any]
    test_predictions: tuple[dict[str, Any], ...]
    baseline_diagnostics: Mapping[str, Any]
    actionable: bool = False
    train_rows_excluded_ineligible_for_tuning: int = 0
    validation_rows_excluded_ineligible: int = 0
    refit_rows_excluded_ineligible: int = 0
    test_rows_available_before_eligibility: int = 0
    test_rows_excluded_ineligible: int = 0


@dataclass(frozen=True)
class OptunaWalkForwardReport:
    schema_version: int
    run_id: str
    symbol: str
    timeframe: str
    horizon_bars: int
    model_type: str
    feature_names: tuple[str, ...]
    feature_schema_sha256: str
    dataset_sha256: str
    n_trials_per_fold: int
    random_seed: int
    fold_count: int
    folds: tuple[TunedFoldResult, ...]
    resource_profile: Mapping[str, Any]
    test_metrics_used_for_tuning: bool = False
    actionable: bool = False
    interpretation: str = (
        "Research-only result. Hyperparameters were selected using each fold's validation "
        "partition; each reported test partition was excluded from that fold's tuning."
    )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["feature_names"] = list(self.feature_names)
        value["folds"] = [asdict(row) for row in self.folds]
        return value


def _iso(value: datetime) -> str:
    return _utc(value, "report").isoformat().replace("+00:00", "Z")


def run_optuna_walk_forward(
    bars: Sequence[ForexBar],
    examples: Sequence[TimedBinaryExample],
    *,
    feature_names: Sequence[str],
    run_id: str,
    dataset_sha256: str,
    model_type: str,
    initial_train_bars: int,
    validation_bars: int,
    test_bars: int,
    step_bars: int,
    horizon_bars: int,
    n_trials_per_fold: int = 20,
    random_seed: int = 20261021,
    bootstrap_resamples: int = 500,
    plan: WalkForwardPlan | None = None,
    class_1_means: str = "UP",
) -> OptunaWalkForwardReport:
    """Tune and test each fold independently; never tune on any fold's test rows."""
    names = tuple(feature_names)
    feature_schema_fingerprint(names)
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if len(dataset_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in dataset_sha256.lower()):
        raise ValueError("dataset_sha256 must be a 64-character hexadecimal digest")
    if model_type not in {"xgboost", "lightgbm"}:
        raise ValueError("model_type must be 'xgboost' or 'lightgbm'")
    if class_1_means not in {"UP", "DOWN"}:
        raise ValueError("class_1_means must be 'UP' or 'DOWN'")
    if isinstance(n_trials_per_fold, bool) or not isinstance(n_trials_per_fold, int) or not 1 <= n_trials_per_fold <= 100:
        raise ValueError("n_trials_per_fold must be between 1 and 100")
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be an integer >= 100")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")
    if len(bars) != len(examples) or not bars:
        raise ValueError("bars and examples must be non-empty and have matching row counts")
    bar_times = [_utc(bar.timestamp, "bar") for bar in bars]
    example_times = [_utc(row.timestamp, "example") for row in examples]
    if bar_times != example_times:
        raise ValueError("bars and examples must have exactly aligned timestamps and order")
    if len({(bar.symbol, bar.timeframe) for bar in bars}) != 1:
        raise ValueError("one run must contain exactly one symbol and timeframe")
    if any(set(row.features) != set(names) for row in examples):
        raise ValueError("feature schema mismatch")
    if any(row.label_available_at.tzinfo is None or row.label_available_at.utcoffset() is None for row in examples):
        raise ValueError("label availability timestamps must be timezone-aware")

    digest = hashlib.sha256()
    for row in examples:
        digest.update(_iso(row.timestamp).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.label).encode("ascii"))
        digest.update(b"\0")
        digest.update(_iso(row.label_available_at).encode("utf-8"))
        digest.update(b"\0")
        digest.update(b"eligible" if row.label_eligible else b"ineligible")
        for name in names:
            digest.update(b"\0")
            digest.update(name.encode("utf-8"))
            digest.update(b"=")
            digest.update(float(row.features[name]).hex().encode("ascii"))
    # This is an internal consistency fingerprint, not a replacement for the source CSV hash.
    examples_fingerprint = digest.hexdigest()
    if not examples_fingerprint:
        raise ValueError("failed to fingerprint research examples")

    walk_plan = plan or generate_walk_forward_folds(
        bars,
        initial_train_bars=initial_train_bars,
        validation_bars=validation_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        horizon_bars=horizon_bars,
    )
    if walk_plan.horizon_bars != horizon_bars:
        raise ValueError("provided plan horizon does not match requested horizon")
    by_time = {timestamp: row for timestamp, row in zip(example_times, examples)}
    bar_index_by_time = {timestamp: index for index, timestamp in enumerate(bar_times)}
    results: list[TunedFoldResult] = []

    for fold in walk_plan.folds:
        train_times = [_utc(bar.timestamp, "train fold") for bar in fold.train]
        validation_times = [_utc(bar.timestamp, "validation fold") for bar in fold.validation]
        test_times = [_utc(bar.timestamp, "test fold") for bar in fold.test]
        if not train_times or not validation_times or not test_times:
            raise ValueError("every fold must have non-empty train, validation, and test partitions")
        if max(train_times) >= min(validation_times) or max(validation_times) >= min(test_times):
            raise ValueError("fold partitions must be strictly chronological")
        try:
            train = [by_time[t] for t in train_times]
            validation = [by_time[t] for t in validation_times]
            test_all = [by_time[t] for t in test_times]
        except KeyError as exc:
            raise ValueError("fold references timestamps missing from examples") from exc

        test_start = min(test_times)
        eligible_validation = [row for row in validation if row.label_eligible]
        # Eligibility filtering is applied after the chronological fold plan is built,
        # preserving the exact same test windows and timestamp index for baseline/variant runs.
        immature_validation = [
            row for row in eligible_validation
            if _utc(row.label_available_at, "label availability") > test_start
        ]
        test = [row for row in test_all if row.label_eligible]
        if not test:
            raise ValueError(
                f"fold {fold.fold_index} has no eligible test rows after elapsed-time filtering"
            )
        if immature_validation:
            raise ValueError(
                f"fold {fold.fold_index} has validation labels not matured by test start; "
                "increase the purge gap or correct label_available_at"
            )

        tuning_seed = random_seed + fold.fold_index
        tuning = tune_on_validation(
            train,
            validation,
            feature_names=names,
            model_type=model_type,
            n_trials=n_trials_per_fold,
            seed=tuning_seed,
            study_name=f"{run_id}-{model_type}-fold-{fold.fold_index}",
        )

        refit_candidates_all = train + validation
        refit_candidates = [row for row in refit_candidates_all if row.label_eligible]
        refit_cutoff = test_start
        refit_rows = [
            row for row in refit_candidates
            if _utc(row.label_available_at, "label availability") <= refit_cutoff
        ]
        if len(refit_rows) < 4 or len({row.label for row in refit_rows}) != 2:
            raise ValueError(f"fold {fold.fold_index} has insufficient matured eligible refit labels")

        if model_type == "xgboost":
            model = XGBoostResearchModel(names, model_version="optuna-walk-forward-v1")
        else:
            model = LightGBMResearchModel(names, model_version="optuna-walk-forward-v1")
        fit_params = dict(tuning.best_params)
        fit_params["random_state"] = tuning_seed
        fit_params["n_jobs"] = build_resource_profile().ml_threads
        model.fit([row.features for row in refit_rows], [row.label for row in refit_rows], params=fit_params)

        probabilities = [model.predict(row.features).probability_class_1 for row in test]

        # Fit simple reference baselines only on matured train+validation labels.
        # No held-out test labels influence baseline prevalence, majority class, or direction skill.
        refit_labels = [row.label for row in refit_rows]
        refit_prevalence = sum(refit_labels) / len(refit_labels)
        refit_majority_class = int(sum(refit_labels) > (len(refit_labels) / 2.0))  # ties default to class 0
        direction_hits = 0
        direction_training_rows = 0
        class_1_is_up = class_1_means == "UP"
        for historical_row in refit_rows:
            historical_index = bar_index_by_time.get(_utc(historical_row.timestamp, "baseline history"))
            if historical_index is None or historical_index == 0:
                continue
            change = bars[historical_index].close - bars[historical_index - 1].close
            if not math.isfinite(change) or change == 0:
                continue
            predicts_up = change > 0
            direction_class = int(predicts_up) if class_1_is_up else int(not predicts_up)
            direction_training_rows += 1
            direction_hits += int(direction_class == historical_row.label)
        # Laplace smoothing prevents an unbounded 0/1 confidence estimate on small folds.
        lagged_direction_hit_rate = (
            (direction_hits + 1.0) / (direction_training_rows + 2.0)
        )

        test_predictions: list[dict[str, Any]] = []
        for row, model_probability in zip(test, probabilities):
            test_index = bar_index_by_time.get(_utc(row.timestamp, "baseline test"))
            lagged_probability = refit_prevalence
            lagged_direction_class: int | None = None
            if test_index is not None and test_index > 0:
                change = bars[test_index].close - bars[test_index - 1].close
                if math.isfinite(change) and change != 0:
                    predicts_up = change > 0
                    lagged_direction_class = (
                        int(predicts_up) if class_1_is_up else int(not predicts_up)
                    )
                    lagged_probability = (
                        lagged_direction_hit_rate
                        if lagged_direction_class == 1
                        else 1.0 - lagged_direction_hit_rate
                    )
            test_predictions.append({
                "timestamp_utc": _iso(row.timestamp),
                "label": int(row.label),
                "model_probability_class_1": float(model_probability),
                "constant_probability_class_1": 0.5,
                "matured_refit_prevalence_probability_class_1": float(refit_prevalence),
                "matured_refit_majority_class": refit_majority_class,
                "lagged_direction_probability_class_1": float(lagged_probability),
                "lagged_direction_prediction": int(lagged_probability >= 0.5),
            })
        baseline_diagnostics = {
            "class_1_means": class_1_means,
            "refit_rows_matured_before_test": len(refit_rows),
            "refit_positive_prevalence": refit_prevalence,
            "refit_majority_class": refit_majority_class,
            "lagged_direction_training_rows": direction_training_rows,
            "lagged_direction_hits": direction_hits,
            "lagged_direction_hit_rate_laplace_smoothed": lagged_direction_hit_rate,
            "baseline_labels_used_test_data": False,
        }

        seed_digest = hashlib.sha256(f"{random_seed}:{fold.fold_index}:test".encode()).hexdigest()
        metrics = _metrics(
            [row.label for row in test],
            probabilities,
            observation_timestamps=[row.timestamp for row in test],
            calibration_block_length=min(max(1, horizon_bars), len(test)),
            calibration_bootstrap_resamples=bootstrap_resamples,
            calibration_random_seed=int(seed_digest[:8], 16),
        )
        results.append(TunedFoldResult(
            fold_index=fold.fold_index,
            train_start_utc=_iso(min(train_times)),
            train_end_utc=_iso(max(train_times)),
            validation_start_utc=_iso(min(validation_times)),
            validation_end_utc=_iso(max(validation_times)),
            test_start_utc=_iso(min(test_times)),
            test_end_utc=_iso(max(test_times)),
            model_type=model_type,
            best_params=dict(tuning.best_params),
            validation_log_loss=tuning.best_validation_log_loss,
            trials_requested=tuning.trial_count_requested,
            trials_completed=tuning.trial_count_completed,
            train_rows_available_for_tuning=tuning.train_rows_available,
            train_rows_excluded_unmatured=tuning.train_rows_excluded_unmatured,
            refit_rows_available=len(refit_rows),
            refit_rows_excluded_unmatured=len(refit_candidates) - len(refit_rows),
            test_metrics=asdict(metrics),
            test_predictions=tuple(test_predictions),
            baseline_diagnostics=baseline_diagnostics,
            actionable=False,
            train_rows_excluded_ineligible_for_tuning=tuning.train_rows_excluded_ineligible,
            validation_rows_excluded_ineligible=tuning.validation_rows_excluded_ineligible,
            refit_rows_excluded_ineligible=len(refit_candidates_all) - len(refit_candidates),
            test_rows_available_before_eligibility=len(test_all),
            test_rows_excluded_ineligible=len(test_all) - len(test),
        ))

    if not results:
        raise ValueError("walk-forward plan contains no folds")
    return OptunaWalkForwardReport(
        schema_version=1,
        run_id=run_id,
        symbol=bars[0].symbol,
        timeframe=bars[0].timeframe,
        horizon_bars=horizon_bars,
        model_type=model_type,
        feature_names=names,
        feature_schema_sha256=feature_schema_fingerprint(names),
        dataset_sha256=dataset_sha256,
        n_trials_per_fold=n_trials_per_fold,
        random_seed=random_seed,
        fold_count=len(results),
        folds=tuple(results),
        resource_profile=build_resource_profile().to_dict(),
        actionable=False,
    )


def write_optuna_walk_forward_report(report: OptunaWalkForwardReport, path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["actionable"] = False
    payload["test_metrics_used_for_tuning"] = False
    payload["report_warning"] = (
        "Offline predictive research only. Metrics are not evidence of profitability; "
        "do not connect this report to broker execution."
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["report_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return str(destination)

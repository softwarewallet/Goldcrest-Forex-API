"""End-to-end Optuna + purged walk-forward evaluation, strictly research-only.

Each fold tunes on its own chronological train/validation partitions, then refits the
selected configuration on labels matured before the test window and scores the untouched
test partition. No result from this module is actionable or connected to execution.
"""
from __future__ import annotations

import hashlib
import json
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
    actionable: bool = False


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
            test = [by_time[t] for t in test_times]
        except KeyError as exc:
            raise ValueError("fold references timestamps missing from examples") from exc

        # All validation labels must have matured before the test begins; otherwise
        # this fold cannot faithfully simulate validation-time model selection.
        test_start = min(test_times)
        immature_validation = [
            row for row in validation if _utc(row.label_available_at, "label availability") > test_start
        ]
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

        refit_candidates = train + validation
        refit_cutoff = test_start
        refit_rows = [
            row for row in refit_candidates
            if _utc(row.label_available_at, "label availability") <= refit_cutoff
        ]
        if len(refit_rows) < 4 or len({row.label for row in refit_rows}) != 2:
            raise ValueError(f"fold {fold.fold_index} has insufficient matured refit labels")

        if model_type == "xgboost":
            model = XGBoostResearchModel(names, model_version="optuna-walk-forward-v1")
        else:
            model = LightGBMResearchModel(names, model_version="optuna-walk-forward-v1")
        fit_params = dict(tuning.best_params)
        fit_params["random_state"] = tuning_seed
        fit_params["n_jobs"] = build_resource_profile().ml_threads
        model.fit([row.features for row in refit_rows], [row.label for row in refit_rows], params=fit_params)

        probabilities = [model.predict(row.features).probability_class_1 for row in test]
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
            actionable=False,
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

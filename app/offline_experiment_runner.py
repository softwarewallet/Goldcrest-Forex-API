"""End-to-end offline evaluation runner; never serves or authorizes predictions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from app.experiment_report import build_experiment_report, write_experiment_report
from app.model_comparison import TrainingPrevalenceBaseline
from app.model_walk_forward import BinaryResearchModel, ModelFoldResult, TimedBinaryExample, _metrics, _utc, _validate_examples
from app.prediction_ledger import PredictionLedger, PredictionRecord
from app.walk_forward import WalkForwardPlan


@dataclass(frozen=True)
class OfflineExperimentResult:
    run_id: str
    report: dict
    prediction_rows_inserted: int
    ledger_path: str
    report_path: str | None
    actionable: bool = False


def run_offline_experiment(
    plan: WalkForwardPlan,
    examples: Sequence[TimedBinaryExample],
    *,
    feature_names: Sequence[str],
    manifest: Mapping[str, object],
    model_factories: Mapping[str, Callable[[], BinaryResearchModel]],
    ledger: PredictionLedger,
    evaluation_contract: Mapping[str, object],
    exclusions: Mapping[str, int],
    block_length: int,
    bootstrap_resamples: int = 5000,
    confidence_level: float = 0.95,
    random_seed: int = 20261018,
    report_path: str | Path | None = None,
) -> OfflineExperimentResult:
    """Train per fold, persist OOS predictions, compare a prevalence baseline, report."""
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("manifest must contain a non-empty run_id")
    if not evaluation_contract:
        raise ValueError("evaluation_contract must explicitly document run assumptions")
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be non-empty and unique")
    if not model_factories:
        raise ValueError("at least one candidate model factory is required")
    if "training_prevalence" in model_factories:
        raise ValueError("training_prevalence is a reserved baseline name")
    if block_length < 1 or bootstrap_resamples < 1 or not 0 < confidence_level < 1:
        raise ValueError("invalid block-bootstrap configuration")
    _validate_examples(examples, feature_names)
    if not plan.folds:
        raise ValueError("walk-forward plan must contain at least one fold")
    fold_indices = [fold.fold_index for fold in plan.folds]
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in fold_indices):
        raise ValueError("fold indices must be non-negative integers")
    if len(set(fold_indices)) != len(fold_indices):
        raise ValueError("walk-forward plan contains duplicate fold indices")
    by_time = {_utc(row.timestamp, "feature"): row for row in examples}
    factories = dict(model_factories)
    factories["training_prevalence"] = TrainingPrevalenceBaseline
    fold_results: dict[str, list[ModelFoldResult]] = {name: [] for name in factories}
    prediction_records: list[PredictionRecord] = []

    for fold in plan.folds:
        train_times = {_utc(bar.timestamp, "fold") for bar in fold.train}
        validation_times = {_utc(bar.timestamp, "fold") for bar in fold.validation}
        test_times = {_utc(bar.timestamp, "fold") for bar in fold.test}
        if not train_times or not validation_times or not test_times:
            raise ValueError("every fold must have non-empty train, validation, and test partitions")
        if (train_times & validation_times) or (train_times & test_times) or (validation_times & test_times):
            raise ValueError("fold partitions must not share observation timestamps")
        required_times = train_times | validation_times | test_times
        missing_times = required_times - by_time.keys()
        if missing_times:
            raise ValueError(f"fold references {len(missing_times)} timestamps missing from examples")
        train_end = max(train_times)
        train_all = [by_time[t] for t in sorted(train_times)]
        train = [row for row in train_all if _utc(row.label_available_at, "label availability") <= train_end]
        validation = [by_time[t] for t in sorted(validation_times)]
        test = [by_time[t] for t in sorted(test_times)]
        if not train or not validation or not test:
            raise ValueError("every fold needs labeled examples in train, validation, and test")
        if len({row.label for row in train}) != 2:
            raise ValueError("matured training labels must contain both classes")

        for name, factory in factories.items():
            model = factory()
            model.fit([row.features for row in train], [row.label for row in train])
            if getattr(model, "actionable", False) is not False:
                raise ValueError("research model must remain actionable=False")

            def score(rows: Sequence[TimedBinaryExample], partition: str):
                probabilities = []
                for row in rows:
                    output = model.predict(row.features)
                    if getattr(output, "actionable", False) is not False:
                        raise ValueError("model output must remain actionable=False")
                    probability = getattr(output, "probability_class_1", None)
                    if probability is None:
                        raise ValueError("model output lacks probability_class_1")
                    probability = float(probability)
                    probabilities.append(probability)
                    prediction_records.append(PredictionRecord(
                        run_id=run_id, symbol=plan.symbol, timeframe=plan.timeframe,
                        observation_timestamp=row.timestamp, fold_index=fold.fold_index,
                        partition=partition, model_name=name,
                        model_version=str(getattr(model, "model_version", "unknown")),
                        probability=probability, label=row.label,
                        label_available_at=row.label_available_at, actionable=False,
                    ))
                return _metrics([row.label for row in rows], probabilities)

            fold_results[name].append(ModelFoldResult(
                fold_index=fold.fold_index,
                model_type=str(getattr(model, "model_type", type(model).__name__)),
                model_version=str(getattr(model, "model_version", "unknown")),
                train_rows_available=len(train),
                train_rows_excluded_unmatured=len(train_all) - len(train),
                validation=score(validation, "validation"),
                test=score(test, "test"),
                probability_semantics="uncalibrated research probability; not trading performance",
                actionable=False,
            ))

    ledger.start_run(run_id, dict(manifest))
    try:
        inserted = ledger.record_many(prediction_records)
        comparisons = []
        versions = {name: str(getattr(factory(), "model_version", "candidate-v1"))
                    for name, factory in factories.items()}
        for candidate_name in factories:
            if candidate_name == "training_prevalence":
                continue
            for fold in plan.folds:
                for partition in ("validation", "test"):
                    comparisons.append(ledger.compare(
                        run_id=run_id, symbol=plan.symbol, timeframe=plan.timeframe,
                        fold_index=fold.fold_index, partition=partition,
                        candidate_name=candidate_name, candidate_version=versions[candidate_name],
                        baseline_name="training_prevalence", baseline_version=versions["training_prevalence"],
                        block_length=block_length, bootstrap_resamples=bootstrap_resamples,
                        confidence_level=confidence_level, random_seed=random_seed,
                    ))

        report = build_experiment_report(
            manifest=dict(manifest),
            fold_results={name: tuple(results) for name, results in fold_results.items()},
            paired_comparisons=comparisons,
            exclusions=exclusions,
            evaluation_contract=dict(evaluation_contract),
            warnings=[
                "Offline research only; all outputs remain actionable=false.",
                "Final-test results must remain untouched during model/feature selection.",
                "Block-bootstrap interpretation depends on the explicitly chosen block length.",
                "Fold windows may overlap; do not pool fold metrics or treat repeated rows as independent.",
            ],
        )
        saved_path = str(write_experiment_report(report, report_path)) if report_path is not None else None
        return OfflineExperimentResult(
            run_id=run_id, report=report, prediction_rows_inserted=inserted,
            ledger_path=ledger.database_path, report_path=saved_path, actionable=False,
        )
        ledger.finish_run(run_id, report["report_sha256"])
        return OfflineExperimentResult(
            run_id=run_id, report=report, prediction_rows_inserted=inserted,
            ledger_path=ledger.database_path, report_path=saved_path, actionable=False,
        )
    except Exception as exc:
        ledger.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise

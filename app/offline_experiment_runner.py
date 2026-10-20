"""End-to-end offline evaluation runner; never serves or authorizes predictions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from app.experiment_report import build_experiment_report, write_experiment_report
from app.model_comparison import TrainingPrevalenceBaseline
from app.model_walk_forward import BinaryResearchModel, ModelFoldResult, TimedBinaryExample, _metrics, _utc, _validate_examples
from app.prediction_ledger import PredictionLedger, PredictionRecord
from app.research_reproducibility import reproducibility_manifest, snapshot_model_configuration
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
    # Preflight factories once so the run contract captures exact model configuration.
    initial_models = {name: factory() for name, factory in factories.items()}
    if len({id(model) for model in initial_models.values()}) != len(initial_models):
        raise ValueError("each model factory must return a distinct fresh model instance")
    reproducibility = reproducibility_manifest(factories, initial_models)
    run_manifest = dict(manifest)
    run_manifest["reproducibility"] = reproducibility
    # Bind point-in-time regime checks to test boundaries from the actual evaluation plan.
    run_manifest["fold_temporal_boundaries"] = [
        {
            "fold_index": fold.fold_index,
            "test_start_utc": _utc(min(bar.timestamp for bar in fold.test), "test fold boundary").isoformat().replace("+00:00", "Z"),
            "test_end_utc": _utc(max(bar.timestamp for bar in fold.test), "test fold boundary").isoformat().replace("+00:00", "Z"),
        }
        for fold in plan.folds
    ]
    initial_model_configs = {name: snapshot_model_configuration(model) for name, model in initial_models.items()}
    fold_results: dict[str, list[ModelFoldResult]] = {name: [] for name in factories}
    prediction_records: list[PredictionRecord] = []
    model_versions: dict[str, str] = {}


    # Fingerprint the actual examples and exact fold membership, not only caller-supplied
    # manifest metadata. This makes same-run-ID retries fail closed when the supplied
    # rows/labels/features/fold boundaries have changed.
    canonical_examples = []
    for row in sorted(examples, key=lambda item: _utc(item.timestamp, "feature")):
        canonical_examples.append({
            "timestamp": _utc(row.timestamp, "feature").isoformat(),
            "features": {name: float(row.features[name]) for name in feature_names},
            "label": row.label,
            "label_available_at": _utc(row.label_available_at, "label availability").isoformat(),
        })
    canonical_folds = []
    for fold in plan.folds:
        canonical_folds.append({
            "fold_index": fold.fold_index,
            "train": sorted(_utc(bar.timestamp, "fold").isoformat() for bar in fold.train),
            "validation": sorted(_utc(bar.timestamp, "fold").isoformat() for bar in fold.validation),
            "test": sorted(_utc(bar.timestamp, "fold").isoformat() for bar in fold.test),
        })
    provenance_payload = {
        "examples": canonical_examples,
        "folds": canonical_folds,
        "feature_names": list(feature_names),
        "symbol": plan.symbol,
        "timeframe": plan.timeframe,
        "model_factories": reproducibility["factory_configuration"],
        "model_configuration": initial_model_configs,
        "runtime_environment": reproducibility["environment"],
        "reproducibility_manifest_sha256": reproducibility["manifest_sha256"],
    }
    input_fingerprint = hashlib.sha256(json.dumps(
        provenance_payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()

    run_contract = {
        "manifest": run_manifest,
        "input_fingerprint": input_fingerprint,
        "reproducibility_manifest_sha256": reproducibility["manifest_sha256"],
        "input_example_count": len(canonical_examples),
        "fold_partition_fingerprint": hashlib.sha256(json.dumps(
            canonical_folds, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "model_factory_ids": provenance_payload["model_factories"],
        "evaluation_contract": dict(evaluation_contract),
        "exclusions": dict(exclusions),
        "feature_names": list(feature_names),
        "model_names": sorted(model_factories),
        "symbol": plan.symbol,
        "timeframe": plan.timeframe,
        "fold_indices": fold_indices,
        "block_length": block_length,
        "bootstrap_resamples": bootstrap_resamples,
        "confidence_level": confidence_level,
        "random_seed": random_seed,
        "calibration_uncertainty": {
            "method": "moving_block_percentile_bootstrap_for_10_bin_ECE",
            "block_length": block_length,
            "bootstrap_resamples": max(100, bootstrap_resamples),
            "confidence_level": confidence_level,
            "minimum_sample_count": 30,
            "seed_derivation": "SHA256(random_seed:fold_index:partition) first 8 hex digits",
        },
    }
    expected_prediction_rows = sum(
        len(fold.validation) + len(fold.test) for fold in plan.folds
    ) * len(factories)
    ledger.start_run(run_id, run_contract, expected_prediction_rows=expected_prediction_rows)

    try:
        seen_model_instances: dict[str, list[BinaryResearchModel]] = {name: [] for name in factories}
        for fold_position, fold in enumerate(plan.folds):
            train_times = {_utc(bar.timestamp, "fold") for bar in fold.train}
            validation_times = {_utc(bar.timestamp, "fold") for bar in fold.validation}
            test_times = {_utc(bar.timestamp, "fold") for bar in fold.test}
            if not train_times or not validation_times or not test_times:
                raise ValueError("every fold must have non-empty train, validation, and test partitions")
            if (train_times & validation_times) or (train_times & test_times) or (validation_times & test_times):
                raise ValueError("fold partitions must not share observation timestamps")
            if max(train_times) >= min(validation_times) or max(validation_times) >= min(test_times):
                raise ValueError("fold partitions must be strictly chronological: train before validation before test")
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
                model = initial_models[name] if fold_position == 0 else factory()
                if any(model is previous for previous in seen_model_instances[name]):
                    raise ValueError(f"model factory reused an instance across folds for {name!r}")
                seen_model_instances[name].append(model)
                model_version = str(getattr(model, "model_version", "unknown"))
                prior_version = model_versions.setdefault(name, model_version)
                if prior_version != model_version:
                    raise ValueError(f"model version changed across folds for {name!r}")
                model_config = snapshot_model_configuration(model)
                if model_config["configuration_sha256"] != initial_model_configs[name]["configuration_sha256"]:
                    raise ValueError(f"model configuration changed across folds for {name!r}")
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
                            model_version=model_version,
                            probability=probability, label=row.label,
                            label_available_at=row.label_available_at, actionable=False,
                        ))
                    calibration_seed = int(hashlib.sha256(
                        f"{random_seed}:{fold.fold_index}:{partition}".encode("utf-8")
                    ).hexdigest()[:8], 16)
                    return _metrics(
                        [row.label for row in rows], probabilities,
                        observation_timestamps=[row.timestamp for row in rows],
                        calibration_block_length=min(block_length, len(rows)),
                        calibration_bootstrap_resamples=max(100, bootstrap_resamples),
                        calibration_confidence_level=confidence_level,
                        calibration_random_seed=calibration_seed,
                    )

                fold_results[name].append(ModelFoldResult(
                    fold_index=fold.fold_index,
                    model_type=str(getattr(model, "model_type", type(model).__name__)),
                    model_version=model_version,
                    train_rows_available=len(train),
                    train_rows_excluded_unmatured=len(train_all) - len(train),
                    validation=score(validation, "validation"),
                    test=score(test, "test"),
                    probability_semantics="uncalibrated research probability; not trading performance",
                    actionable=False,
                ))

        inserted = ledger.record_many(prediction_records)
        comparisons = []
        versions = model_versions
        missing_versions = set(factories) - set(versions)
        if missing_versions:
            raise ValueError(f"missing fitted model versions: {sorted(missing_versions)}")
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
            manifest=run_manifest,
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
        ledger.finish_run(run_id, report["report_sha256"])
        return OfflineExperimentResult(
            run_id=run_id, report=report, prediction_rows_inserted=inserted,
            ledger_path=ledger.database_path, report_path=saved_path, actionable=False,
        )
    except Exception as exc:
        # Preserve the original experiment error if the ledger itself is unavailable.
        try:
            ledger.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise

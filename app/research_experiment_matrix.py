"""Fair, fixed-fold experiment matrix for causal feature ablations and tabular models.

This orchestration compares research runs without selecting a winner. Every candidate uses
the same chronological purged folds and the same source dataset hash; test metrics are
reported only and must not be reused to choose a candidate for the same test period.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from app.causal_ta_features import feature_ablation_sets
from app.evaluation import ForexBar
from app.model_walk_forward import TimedBinaryExample
from app.optuna_walk_forward import run_optuna_walk_forward
from app.walk_forward import WalkForwardPlan, generate_walk_forward_folds


def _canonical_utc_z(value: str) -> str:
    """Normalize an ISO-8601 timestamp to the same UTC-Z form used by fold metrics."""
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("walk-forward test boundaries must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ExperimentMatrixReport:
    schema_version: int
    run_id: str
    dataset_sha256: str
    symbol: str
    timeframe: str
    feature_names: tuple[str, ...]
    model_types: tuple[str, ...]
    fold_count: int
    experiments: tuple[dict[str, Any], ...]
    common_fold_test_windows: tuple[dict[str, Any], ...]
    horizon_bars: int = 1
    class_1_means: str = "UP"
    test_metrics_used_for_selection: bool = False
    automatic_winner_selection: bool = False
    actionable: bool = False
    interpretation: str = (
        "Offline research comparison only. Candidates share fixed chronological folds; "
        "test metrics are descriptive and must not be used to select a winner on these same windows."
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        payload["model_types"] = list(self.model_types)
        payload["common_fold_test_windows"] = list(self.common_fold_test_windows)
        payload["experiments"] = list(self.experiments)
        return payload


def run_experiment_matrix(
    bars: Sequence[ForexBar],
    examples: Sequence[TimedBinaryExample],
    *,
    feature_names: Sequence[str],
    run_id: str,
    dataset_sha256: str,
    model_types: Sequence[str] = ("xgboost", "lightgbm"),
    initial_train_bars: int,
    validation_bars: int,
    test_bars: int,
    step_bars: int,
    horizon_bars: int,
    n_trials_per_fold: int = 20,
    random_seed: int = 20261021,
    bootstrap_resamples: int = 500,
    include_ablation: bool = True,
    max_experiments: int = 10,
    class_1_means: str = "UP",
) -> ExperimentMatrixReport:
    """Run each model/feature variant on one shared, immutable purged fold plan."""
    names = tuple(feature_names)
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if len(dataset_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in dataset_sha256.lower()):
        raise ValueError("dataset_sha256 must be a 64-character hexadecimal digest")
    if len(bars) != len(examples) or not bars:
        raise ValueError("bars and examples must be non-empty and have equal lengths")
    if tuple(model_types) != tuple(dict.fromkeys(model_types)) or not model_types:
        raise ValueError("model_types must be non-empty and unique")
    unsupported = set(model_types) - {"xgboost", "lightgbm"}
    if unsupported:
        raise ValueError(f"unsupported model types: {sorted(unsupported)}")
    if class_1_means not in {"UP", "DOWN"}:
        raise ValueError("class_1_means must be 'UP' or 'DOWN'")
    if isinstance(max_experiments, bool) or not isinstance(max_experiments, int) or max_experiments < 1:
        raise ValueError("max_experiments must be a positive integer")
    if any(set(row.features) != set(names) for row in examples):
        raise ValueError("each example must contain exactly the declared full feature schema")

    variants = feature_ablation_sets(names) if include_ablation else {"all_features": names}
    experiment_count = len(variants) * len(model_types)
    if experiment_count > max_experiments:
        raise ValueError(
            f"matrix contains {experiment_count} experiments, exceeding max_experiments={max_experiments}; "
            "reduce model_types or disable ablation variants"
        )

    plan: WalkForwardPlan = generate_walk_forward_folds(
        bars,
        initial_train_bars=initial_train_bars,
        validation_bars=validation_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        horizon_bars=horizon_bars,
    )
    bar_times = [bar.timestamp for bar in bars]
    example_times = [row.timestamp for row in examples]
    if bar_times != example_times:
        raise ValueError("bars and examples must have exactly aligned timestamps and order")

    experiment_results: list[dict[str, Any]] = []
    for model_index, model_type in enumerate(model_types):
        for variant_index, (variant_name, selected_features) in enumerate(variants.items()):
            selected = set(selected_features)
            variant_examples = [
                replace(row, features={name: row.features[name] for name in selected_features})
                for row in examples
            ]
            variant_run_id = f"{run_id}-{model_type}-{variant_name}"
            result = run_optuna_walk_forward(
                bars,
                variant_examples,
                feature_names=selected_features,
                run_id=variant_run_id,
                dataset_sha256=dataset_sha256,
                model_type=model_type,
                initial_train_bars=initial_train_bars,
                validation_bars=validation_bars,
                test_bars=test_bars,
                step_bars=step_bars,
                horizon_bars=horizon_bars,
                n_trials_per_fold=n_trials_per_fold,
                random_seed=random_seed + model_index * 1000 + variant_index,
                bootstrap_resamples=bootstrap_resamples,
                plan=plan,
                class_1_means=class_1_means,
            )
            experiment_results.append({
                "experiment_id": variant_run_id,
                "model_type": model_type,
                "feature_variant": variant_name,
                "feature_names": list(selected_features),
                "feature_count": len(selected_features),
                "feature_schema_sha256": result.feature_schema_sha256,
                "fold_count": result.fold_count,
                "fold_results": [asdict(fold) for fold in result.folds],
                "test_metrics_used_for_tuning": False,
                "used_common_fold_plan": True,
                "actionable": False,
            })

    common_windows = tuple({
        "fold_index": fold.fold_index,
        "test_start_utc": _canonical_utc_z(fold.test_start),
        "test_end_utc": _canonical_utc_z(fold.test_end),
        "test_rows": len(fold.test),
    } for fold in plan.folds)
    return ExperimentMatrixReport(
        schema_version=1,
        run_id=run_id,
        dataset_sha256=dataset_sha256,
        symbol=bars[0].symbol,
        timeframe=bars[0].timeframe,
        horizon_bars=horizon_bars,
        class_1_means=class_1_means,
        feature_names=names,
        model_types=tuple(model_types),
        fold_count=len(plan.folds),
        experiments=tuple(experiment_results),
        common_fold_test_windows=common_windows,
        actionable=False,
    )


def write_experiment_matrix_report(report: ExperimentMatrixReport, path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["actionable"] = False
    payload["test_metrics_used_for_selection"] = False
    payload["automatic_winner_selection"] = False
    payload["report_warning"] = (
        "Offline research only. Comparing many candidates can create selection bias. "
        "Do not select a production model from the same test windows used in this report."
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["report_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return str(destination)

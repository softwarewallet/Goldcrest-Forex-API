"""Paired, time-series-aware baseline analysis for offline research runs.

Baselines use only labels matured before each held-out test window. This module never
ranks/promotes models or connects research results to trading execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.research_results_summary import summarize_research_run

_EPS = 1e-15
_BASELINES = {
    "constant_probability_0_5": "constant_probability_class_1",
    "matured_refit_prevalence": "matured_refit_prevalence_probability_class_1",
    "lagged_bar_direction": "lagged_direction_probability_class_1",
}
_SAFETY_FLAGS = ("actionable", "automatic_winner_selection", "test_metrics_used_for_selection")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return result


def _label(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f"{field} must be integer 0 or 1")
    return value


def _log_loss(label: int, probability: float) -> float:
    probability = min(1.0 - _EPS, max(_EPS, probability))
    return -(label * math.log(probability) + (1 - label) * math.log(1 - probability))


def _brier(label: int, probability: float) -> float:
    return (probability - label) ** 2


def _metric_summary(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    if not labels or len(labels) != len(probabilities):
        raise ValueError("metrics require non-empty aligned labels and probabilities")
    n = len(labels)
    return {
        "sample_count": n,
        "log_loss": sum(_log_loss(y, p) for y, p in zip(labels, probabilities)) / n,
        "brier_score": sum(_brier(y, p) for y, p in zip(labels, probabilities)) / n,
        "accuracy_at_fixed_half": sum(int((p >= 0.5) == bool(y)) for y, p in zip(labels, probabilities)) / n,
        "label_prevalence": sum(labels) / n,
        "mean_predicted_probability": sum(probabilities) / n,
    }


def _quantile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    position = (len(sorted_values) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def _block_bootstrap(
    fold_differences: Sequence[Sequence[float]],
    *,
    block_length: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired circular moving-block bootstrap; preserve chronological dependence per fold."""
    folds = [list(values) for values in fold_differences if values]
    if not folds:
        raise ValueError("bootstrap requires at least one non-empty fold")
    if isinstance(block_length, bool) or not isinstance(block_length, int) or block_length < 1:
        raise ValueError("block_length must be a positive integer")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or not 100 <= resamples <= 20000:
        raise ValueError("resamples must be from 100 to 20000")
    total_n = sum(len(values) for values in folds)
    observed = sum(sum(values) for values in folds) / total_n
    rng = random.Random(seed)
    distribution: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for values in folds:
            n = len(values)
            length = min(block_length, n)
            sample: list[float] = []
            while len(sample) < n:
                start = rng.randrange(n)
                sample.extend(values[(start + offset) % n] for offset in range(length))
            total += sum(sample[:n])
        distribution.append(total / total_n)
    distribution.sort()
    return {
        "observed_mean_improvement": observed,
        "confidence_level": 0.95,
        "confidence_interval_95": [_quantile(distribution, 0.025), _quantile(distribution, 0.975)],
        "bootstrap_probability_improvement_positive": sum(value > 0 for value in distribution) / len(distribution),
        "bootstrap_resamples": resamples,
        "block_length_observations": block_length,
        "method": "paired_circular_moving_block_bootstrap_stratified_by_walk_forward_fold",
        "interpretation": "Improvement is baseline loss minus model loss; positive values favor the model. This is descriptive uncertainty, not proof of profitability.",
    }


def analyze_prediction_matrix(
    matrix: Mapping[str, Any],
    *,
    bootstrap_resamples: int = 1000,
    block_length: int | None = None,
    random_seed: int = 20261023,
) -> dict[str, Any]:
    """Validate stored OOS predictions and compare them with non-actionable baselines."""
    errors: list[str] = []
    warnings: list[str] = []
    for flag in _SAFETY_FLAGS:
        if matrix.get(flag) is not False:
            errors.append(f"{flag}_must_be_false")

    experiments = matrix.get("experiments")
    windows = matrix.get("common_fold_test_windows")
    if not isinstance(experiments, list) or not experiments:
        experiments = []
        errors.append("experiments_must_be_nonempty_list")
    if not isinstance(windows, list) or not windows:
        windows = []
        errors.append("common_fold_test_windows_must_be_nonempty_list")

    horizon = matrix.get("horizon_bars")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        errors.append("horizon_bars_missing_or_invalid")
        horizon = 1
    chosen_block_length = horizon if block_length is None else block_length
    if isinstance(chosen_block_length, bool) or not isinstance(chosen_block_length, int) or chosen_block_length < 1:
        errors.append("block_length_must_be_positive_integer")
        chosen_block_length = horizon
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or not 100 <= bootstrap_resamples <= 20000:
        errors.append("bootstrap_resamples_must_be_between_100_and_20000")

    windows_by_fold: dict[int, dict[str, Any]] = {}
    for position, window in enumerate(windows):
        if not isinstance(window, dict):
            errors.append(f"common_window_{position}_must_be_object")
            continue
        index = window.get("fold_index", position)
        if isinstance(index, bool) or not isinstance(index, int) or index in windows_by_fold:
            errors.append(f"common_window_{position}_invalid_or_duplicate_fold_index")
            continue
        try:
            start = _parse_utc(window.get("test_start_utc"), "test_start_utc")
            end = _parse_utc(window.get("test_end_utc"), "test_end_utc")
            rows = window.get("test_rows")
            if end < start or isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
                raise ValueError("invalid test window bounds or row count")
            windows_by_fold[index] = {"start": start, "end": end, "rows": rows}
        except (TypeError, ValueError) as exc:
            errors.append(f"common_window_{position}_invalid:{type(exc).__name__}")

    analyzed: list[dict[str, Any]] = []
    repeated_timestamps = 0
    for exp_position, experiment in enumerate(experiments):
        prefix = f"experiment_{exp_position}_"
        before_errors = len(errors)
        if not isinstance(experiment, dict):
            errors.append(prefix + "must_be_object")
            continue
        if experiment.get("actionable") is not False:
            errors.append(prefix + "actionable_must_be_false")
        if experiment.get("test_metrics_used_for_tuning") is not False:
            errors.append(prefix + "test_metrics_used_for_tuning_must_be_false")
        folds = experiment.get("fold_results")
        if not isinstance(folds, list) or not folds:
            errors.append(prefix + "fold_results_missing")
            continue
        if len(folds) != len(windows_by_fold):
            errors.append(prefix + "fold_count_mismatch")

        parsed_folds: list[dict[str, Any]] = []
        used_indices: set[int] = set()
        timestamps_across_folds: set[datetime] = set()
        for fold_position, fold in enumerate(folds):
            fold_prefix = prefix + f"fold_{fold_position}_"
            if not isinstance(fold, dict):
                errors.append(fold_prefix + "must_be_object")
                continue
            fold_index = fold.get("fold_index", fold_position)
            if isinstance(fold_index, bool) or not isinstance(fold_index, int) or fold_index in used_indices:
                errors.append(fold_prefix + "invalid_or_duplicate_index")
                continue
            used_indices.add(fold_index)
            window = windows_by_fold.get(fold_index)
            if window is None:
                errors.append(fold_prefix + "missing_common_window")
                continue
            if fold.get("actionable") is not False:
                errors.append(fold_prefix + "actionable_must_be_false")
            diagnostics = fold.get("baseline_diagnostics")
            if not isinstance(diagnostics, dict) or diagnostics.get("baseline_labels_used_test_data") is not False:
                errors.append(fold_prefix + "baseline_provenance_invalid")
            predictions = fold.get("test_predictions")
            if not isinstance(predictions, list) or not predictions:
                errors.append(fold_prefix + "test_predictions_missing")
                continue
            test_metrics = fold.get("test_metrics")
            expected_count = test_metrics.get("sample_count") if isinstance(test_metrics, dict) else None
            if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count != len(predictions):
                errors.append(fold_prefix + "prediction_count_mismatch")
            if len(predictions) != window["rows"]:
                errors.append(fold_prefix + "common_window_count_mismatch")

            rows: list[dict[str, Any]] = []
            row_times: list[datetime] = []
            for pred_position, row in enumerate(predictions):
                try:
                    if not isinstance(row, dict):
                        raise ValueError("prediction record must be an object")
                    timestamp = _parse_utc(row.get("timestamp_utc"), "timestamp_utc")
                    label = _label(row.get("label"), "label")
                    model_p = _probability(row.get("model_probability_class_1"), "model_probability_class_1")
                    baseline_p = {
                        name: _probability(row.get(field), field)
                        for name, field in _BASELINES.items()
                    }
                    majority = _label(row.get("matured_refit_majority_class"), "matured_refit_majority_class")
                    lagged_class = _label(row.get("lagged_direction_prediction"), "lagged_direction_prediction")
                    if not window["start"] <= timestamp <= window["end"]:
                        raise ValueError("prediction timestamp outside test window")
                    rows.append({
                        "timestamp": timestamp,
                        "label": label,
                        "model_p": model_p,
                        "baseline_p": baseline_p,
                        "majority": majority,
                        "lagged_class": lagged_class,
                    })
                    row_times.append(timestamp)
                    if timestamp in timestamps_across_folds:
                        repeated_timestamps += 1
                    timestamps_across_folds.add(timestamp)
                except (TypeError, ValueError, OverflowError) as exc:
                    errors.append(fold_prefix + f"prediction_{pred_position}_invalid:{type(exc).__name__}")
            if row_times != sorted(row_times) or len(set(row_times)) != len(row_times):
                errors.append(fold_prefix + "timestamps_not_unique_chronological")
            if rows:
                parsed_folds.append({"fold_index": fold_index, "rows": rows})

        if not parsed_folds or len(errors) > before_errors:
            continue

        labels_by_fold = [[row["label"] for row in fold["rows"]] for fold in sorted(parsed_folds, key=lambda f: f["fold_index"])]
        model_p_by_fold = [[row["model_p"] for row in fold["rows"]] for fold in sorted(parsed_folds, key=lambda f: f["fold_index"])]
        baseline_p_by_fold = {
            name: [[row["baseline_p"][name] for row in fold["rows"]] for fold in sorted(parsed_folds, key=lambda f: f["fold_index"])]
            for name in _BASELINES
        }
        labels = [value for fold_values in labels_by_fold for value in fold_values]
        model_probabilities = [value for fold_values in model_p_by_fold for value in fold_values]
        model_metrics = _metric_summary(labels, model_probabilities)
        baseline_metrics = {
            name: _metric_summary(
                labels,
                [value for fold_values in baseline_p_by_fold[name] for value in fold_values],
            )
            for name in _BASELINES
        }
        ordered_folds = sorted(parsed_folds, key=lambda f: f["fold_index"])
        fold_indices = [fold["fold_index"] for fold in ordered_folds]
        majority_predictions_by_fold = [[row["majority"] for row in fold["rows"]] for fold in ordered_folds]
        lagged_predictions_by_fold = [[row["lagged_class"] for row in fold["rows"]] for fold in ordered_folds]
        majority_accuracy = sum(
            int(pred == label)
            for label_fold, pred_fold in zip(labels_by_fold, majority_predictions_by_fold)
            for label, pred in zip(label_fold, pred_fold)
        ) / len(labels)
        lagged_accuracy = sum(
            int(pred == label)
            for label_fold, pred_fold in zip(labels_by_fold, lagged_predictions_by_fold)
            for label, pred in zip(label_fold, pred_fold)
        ) / len(labels)

        # Keep fold-level diagnostics: aggregate metrics can hide regime-dependent failures.
        fold_stability: list[dict[str, Any]] = []
        for fold in ordered_folds:
            fold_index = fold["fold_index"]
            fold_rows = fold["rows"]
            fold_labels = [row["label"] for row in fold_rows]
            fold_model_p = [row["model_p"] for row in fold_rows]
            fold_model_metrics = _metric_summary(fold_labels, fold_model_p)
            fold_baseline_metrics = {
                name: _metric_summary(
                    fold_labels, [row["baseline_p"][name] for row in fold_rows]
                )
                for name in _BASELINES
            }
            fold_accuracy_comparisons = {
                "matured_refit_majority_class": sum(
                    int(row["majority"] == row["label"]) for row in fold_rows
                ) / len(fold_rows),
                "lagged_bar_direction": sum(
                    int(row["lagged_class"] == row["label"]) for row in fold_rows
                ) / len(fold_rows),
            }
            fold_improvements: dict[str, Any] = {}
            for name in _BASELINES:
                baseline_probs = [row["baseline_p"][name] for row in fold_rows]
                fold_improvements[name] = {
                    "log_loss_improvement": sum(
                        _log_loss(y, p0) - _log_loss(y, p1)
                        for y, p0, p1 in zip(fold_labels, baseline_probs, fold_model_p)
                    ) / len(fold_rows),
                    "brier_improvement": sum(
                        _brier(y, p0) - _brier(y, p1)
                        for y, p0, p1 in zip(fold_labels, baseline_probs, fold_model_p)
                    ) / len(fold_rows),
                }
            test_window = windows_by_fold[fold_index]
            fold_stability.append({
                "fold_index": fold_index,
                "test_start_utc": test_window["start"].isoformat().replace("+00:00", "Z"),
                "test_end_utc": test_window["end"].isoformat().replace("+00:00", "Z"),
                "sample_count": len(fold_rows),
                "model_metrics": fold_model_metrics,
                "baseline_metrics": fold_baseline_metrics,
                "probability_loss_improvements": fold_improvements,
                "accuracy_by_baseline": {
                    "model": fold_model_metrics["accuracy_at_fixed_half"],
                    **fold_accuracy_comparisons,
                },
                "actionable": False,
            })

        comparisons: list[dict[str, Any]] = []
        baseline_index = 0
        for baseline_name in _BASELINES:
            metric_comparisons: dict[str, Any] = {}
            for metric_name, loss_function, metric_offset in (
                ("log_loss", _log_loss, 0),
                ("brier_score", _brier, 1),
            ):
                paired_by_fold: list[list[float]] = []
                flat_model_losses: list[float] = []
                flat_baseline_losses: list[float] = []
                for label_fold, model_fold, baseline_fold in zip(
                    labels_by_fold, model_p_by_fold, baseline_p_by_fold[baseline_name]
                ):
                    paired_by_fold.append([
                        loss_function(y, baseline_p) - loss_function(y, model_p)
                        for y, baseline_p, model_p in zip(label_fold, baseline_fold, model_fold)
                    ])
                    flat_model_losses.extend(loss_function(y, p) for y, p in zip(label_fold, model_fold))
                    flat_baseline_losses.extend(loss_function(y, p) for y, p in zip(label_fold, baseline_fold))
                try:
                    boot = _block_bootstrap(
                        paired_by_fold,
                        block_length=chosen_block_length,
                        resamples=bootstrap_resamples,
                        seed=random_seed + exp_position * 1009 + baseline_index * 97 + metric_offset,
                    )
                    mean_improvement = sum(sum(fold_values) for fold_values in paired_by_fold) / sum(len(fold_values) for fold_values in paired_by_fold)
                    fold_means = [
                        sum(values) / len(values)
                        for values in paired_by_fold if values
                    ]
                    interval_low, interval_high = boot["confidence_interval_95"]
                    interval_evidence = (
                        "positive_interval_excludes_zero" if interval_low > 0
                        else "negative_interval_excludes_zero" if interval_high < 0
                        else "interval_includes_zero"
                    )
                    metric_comparisons[metric_name] = {
                        "model_mean": sum(flat_model_losses) / len(flat_model_losses),
                        "baseline_mean": sum(flat_baseline_losses) / len(flat_baseline_losses),
                        "improvement_baseline_minus_model": mean_improvement,
                        "confidence_interval_95": boot["confidence_interval_95"],
                        "interval_evidence": interval_evidence,
                        "bootstrap_probability_improvement_positive": boot["bootstrap_probability_improvement_positive"],
                        "bootstrap_resamples": boot["bootstrap_resamples"],
                        "block_length_observations": boot["block_length_observations"],
                        "folds_with_positive_mean_improvement": sum(value > 0 for value in fold_means),
                        "folds_with_negative_mean_improvement": sum(value < 0 for value in fold_means),
                        "folds_with_zero_mean_improvement": sum(value == 0 for value in fold_means),
                        "folds_compared": len(fold_means),
                        "mean_fold_improvement": sum(fold_means) / len(fold_means) if fold_means else None,
                        "min_fold_improvement": min(fold_means) if fold_means else None,
                        "max_fold_improvement": max(fold_means) if fold_means else None,
                        "per_fold_improvements": [
                            {"fold_index": fold_indices[index], "improvement_baseline_minus_model": value}
                            for index, value in enumerate(fold_means)
                        ],
                    }
                except (TypeError, ValueError) as exc:
                    errors.append(prefix + f"{baseline_name}_{metric_name}_bootstrap_failed:{type(exc).__name__}")
            comparisons.append({"baseline": baseline_name, "metrics": metric_comparisons})
            baseline_index += 1

        analyzed.append({
            "experiment_id": experiment.get("experiment_id"),
            "model_type": experiment.get("model_type"),
            "feature_variant": experiment.get("feature_variant"),
            "fold_count": len(parsed_folds),
            "test_prediction_count_counted_by_fold": len(labels),
            "model_metrics": model_metrics,
            "baseline_metrics": baseline_metrics,
            "fold_stability": fold_stability,
            "probability_loss_comparisons": comparisons,
            "accuracy_comparisons": [
                {
                    "baseline": "matured_refit_majority_class",
                    "model_accuracy_at_fixed_half": model_metrics["accuracy_at_fixed_half"],
                    "baseline_accuracy": majority_accuracy,
                    "accuracy_improvement": model_metrics["accuracy_at_fixed_half"] - majority_accuracy,
                },
                {
                    "baseline": "lagged_bar_direction",
                    "model_accuracy_at_fixed_half": model_metrics["accuracy_at_fixed_half"],
                    "baseline_accuracy": lagged_accuracy,
                    "accuracy_improvement": model_metrics["accuracy_at_fixed_half"] - lagged_accuracy,
                },
            ],
            "ranked_or_selected": False,
            "actionable": False,
        })

    if repeated_timestamps:
        warnings.append("held_out_timestamps_repeat_across_folds; forecast errors may be dependent across fold fits")
    if not analyzed:
        errors.append("no_valid_prediction_experiments")
    status = "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS"
    return {
        "schema_version": 1,
        "scope": "GOLDCREST_TIME_SERIES_BASELINE_ANALYSIS",
        "status": status,
        "run_id": matrix.get("run_id"),
        "symbol": matrix.get("symbol"),
        "timeframe": matrix.get("timeframe"),
        "horizon_bars": horizon,
        "class_1_means": matrix.get("class_1_means"),
        "fold_count": len(windows_by_fold),
        "experiment_count": len(analyzed),
        "baseline_definitions": {
            "constant_probability_0_5": "Fixed 0.5 class-1 probability reference.",
            "matured_refit_prevalence": "Class-1 prevalence from refit labels matured before test start.",
            "matured_refit_majority_class": "Majority class from matured refit labels; used for accuracy comparison only.",
            "lagged_bar_direction": "Previous-close direction with Laplace-smoothed hit rate learned from matured refit labels; flat/first rows fall back to refit prevalence.",
        },
        "bootstrap": {
            "method": "paired circular moving-block bootstrap stratified by walk-forward fold",
            "resamples": bootstrap_resamples,
            "block_length_observations": chosen_block_length,
            "block_length_note": "Defaults to the label horizon. This is a pragmatic dependence adjustment, not proof of independence.",
            "seed": random_seed,
        },
        "experiments": analyzed,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "automatic_model_selection": False,
        "test_metrics_used_for_selection": False,
        "actionable": False,
        "execution_connected": False,
        "interpretation": "Descriptive paired held-out comparisons only. Positive loss improvement means model loss is lower than baseline loss; uncertainty intervals do not certify profitability or production readiness.",
    }


def analyze_research_run(
    manifest_path: str | Path,
    *,
    bootstrap_resamples: int = 1000,
    block_length: int | None = None,
    random_seed: int = 20261023,
) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    integrity = summarize_research_run(source)
    if integrity.get("status") == "FAIL":
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_TIME_SERIES_BASELINE_ANALYSIS",
            "status": "FAIL",
            "manifest_path": str(source),
            "errors": ["research_run_integrity_check_failed", *integrity.get("errors", [])],
            "warnings": integrity.get("warnings", []),
            "automatic_model_selection": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "execution_connected": False,
        }
    try:
        manifest = _read_json(source)
        matrix_value = manifest.get("experiment_matrix_report")
        if not isinstance(matrix_value, str) or not matrix_value:
            raise ValueError("experiment_matrix_report_path_missing")
        matrix_path = Path(matrix_value)
        if not matrix_path.is_absolute():
            matrix_path = (source.parent / matrix_path).resolve()
        if not matrix_path.is_file():
            raise ValueError("experiment_matrix_report_not_found")
        matrix_bytes = matrix_path.read_bytes()
        matrix_hash = hashlib.sha256(matrix_bytes).hexdigest()
        if matrix_hash != manifest.get("experiment_matrix_report_sha256"):
            raise ValueError("experiment_matrix_report_sha256_mismatch")
        matrix = json.loads(matrix_bytes.decode("utf-8"))
        if not isinstance(matrix, dict):
            raise ValueError("experiment_matrix_report_not_object")
        if matrix.get("run_id") != manifest.get("run_id"):
            raise ValueError("experiment_matrix_run_id_mismatch")
        result = analyze_prediction_matrix(
            matrix, bootstrap_resamples=bootstrap_resamples,
            block_length=block_length, random_seed=random_seed,
        )
        result["manifest_path"] = str(source)
        result["experiment_matrix_path"] = str(matrix_path.resolve())
        result["experiment_matrix_sha256"] = matrix_hash
        result["experiment_matrix_hash_verified"] = True
        result["research_run_validation_status"] = integrity.get("validation_status")
        result["research_run_provenance_status"] = integrity.get("provenance_audit_status")
        result["raw_ohlc_audit_status"] = integrity.get("raw_ohlc_audit_status")
        result["warnings"] = sorted(set(result.get("warnings", []) + integrity.get("warnings", [])))
        if result["warnings"] and result["status"] == "PASS":
            result["status"] = "PASS_WITH_WARNINGS"
        return result
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as exc:
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_TIME_SERIES_BASELINE_ANALYSIS",
            "status": "FAIL",
            "manifest_path": str(source),
            "errors": [f"baseline_analysis_input_invalid:{type(exc).__name__}:{exc}"],
            "warnings": integrity.get("warnings", []),
            "automatic_model_selection": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "execution_connected": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare stored held-out predictions with causal baselines and paired block-bootstrap uncertainty."
    )
    parser.add_argument("--manifest", required=True, help="Path to <run-id>_pipeline_manifest.json")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000, help="Bootstrap draws, from 100 to 20000")
    parser.add_argument("--block-length", type=int, help="Contiguous observations per block; defaults to the label horizon")
    parser.add_argument("--seed", type=int, default=20261023, help="Deterministic random seed")
    args = parser.parse_args(argv)
    result = analyze_research_run(
        args.manifest,
        bootstrap_resamples=args.bootstrap_resamples,
        block_length=args.block_length,
        random_seed=args.seed,
    )
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 3 if result.get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Descriptive, point-in-time volatility-regime diagnostics for held-out research predictions.

This tool joins persisted out-of-sample forecasts to the exact OHLC input, labels volatility
regimes using thresholds estimated strictly before each test window, and reports conditional
metrics. It never ranks models, chooses a strategy, or connects to trading execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.research_baseline_analysis import (
    _brier,
    _log_loss,
    _metric_summary,
    _parse_utc,
    analyze_prediction_matrix,
)
from app.research_results_summary import summarize_research_run

_REGIMES = ("low_volatility", "mid_volatility", "high_volatility")
_PREDICTION_BASELINES = {
    "constant_probability_0_5": "constant_probability_class_1",
    "matured_refit_prevalence": "matured_refit_prevalence_probability_class_1",
    "lagged_bar_direction": "lagged_direction_probability_class_1",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute volatility threshold from empty history")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _parse_ohlc(rows: Sequence[Mapping[str, Any]], *, symbol: str, timeframe: str) -> tuple[list[datetime], list[float]]:
    if not rows:
        raise ValueError("ohlc_rows_empty")
    timestamps: list[datetime] = []
    closes: list[float] = []
    for index, row in enumerate(rows):
        required = {"timestamp", "open", "high", "low", "close"}
        if not required.issubset(row):
            raise ValueError(f"ohlc_row_{index}_missing_columns")
        timestamp = _parse_utc(row.get("timestamp"), "OHLC timestamp")
        values: dict[str, float] = {}
        for name in ("open", "high", "low", "close"):
            raw = row.get(name)
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ohlc_row_{index}_{name}_not_numeric") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"ohlc_row_{index}_{name}_not_positive_finite")
            values[name] = value
        if (
            values["low"] > min(values["open"], values["close"])
            or values["high"] < max(values["open"], values["close"])
            or values["high"] < values["low"]
        ):
            raise ValueError(f"ohlc_row_{index}_invalid_ohlc_relationship")
        row_symbol = str(row.get("symbol", "")).strip().upper()
        row_timeframe = str(row.get("timeframe", "")).strip().upper()
        if row_symbol and row_symbol != symbol.upper():
            raise ValueError(f"ohlc_row_{index}_symbol_mismatch")
        if row_timeframe and row_timeframe != timeframe.upper():
            raise ValueError(f"ohlc_row_{index}_timeframe_mismatch")
        if timestamps and timestamp <= timestamps[-1]:
            raise ValueError("ohlc_timestamps_must_be_strictly_increasing_unique")
        timestamps.append(timestamp)
        closes.append(values["close"])
    return timestamps, closes


def _rolling_log_return_volatility(closes: Sequence[float], window: int) -> list[float | None]:
    """Use only returns available at each bar close; no centered/future window."""
    returns: list[float | None] = [None]
    returns.extend(math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)))
    result: list[float | None] = [None] * len(closes)
    for index in range(window, len(closes)):
        sample = returns[index - window + 1:index + 1]
        if len(sample) == window and all(value is not None and math.isfinite(value) for value in sample):
            result[index] = statistics.pstdev([float(value) for value in sample])
    return result


def _regime(value: float, low_threshold: float, high_threshold: float) -> str:
    # Strict comparisons leave tied/degenerate thresholds in the middle bucket instead
    # of accidentally classifying every observation as low volatility.
    if value < low_threshold:
        return "low_volatility"
    if value > high_threshold:
        return "high_volatility"
    return "mid_volatility"


def _conditional_block_bootstrap(
    fold_rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    target_regime: str,
    baseline_field: str,
    metric_name: str,
    block_length: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Resample full chronological fold rows, then score only rows in the target regime.

    Resampling an already-filtered regime sequence would incorrectly treat distant
    same-regime observations as adjacent. This function keeps original chronology by
    drawing blocks from each complete held-out fold and conditions each draw on rows
    belonging to the requested regime.
    """
    folds = [list(rows) for rows in fold_rows if rows]
    if not folds:
        raise ValueError("conditional bootstrap requires non-empty fold rows")
    if target_regime not in _REGIMES:
        raise ValueError("unknown volatility regime")
    if metric_name not in {"log_loss", "brier_score"}:
        raise ValueError("unsupported loss metric")
    if block_length < 1 or resamples < 100:
        raise ValueError("invalid conditional bootstrap parameters")

    def improvement(row: Mapping[str, Any]) -> float:
        label = int(row["label"])
        model_probability = float(row["model_probability"])
        baseline_probability = float(row["baseline_probabilities"][baseline_field])
        if metric_name == "log_loss":
            return _log_loss(label, baseline_probability) - _log_loss(label, model_probability)
        return _brier(label, baseline_probability) - _brier(label, model_probability)

    observed_values = [
        improvement(row)
        for fold in folds
        for row in fold
        if row.get("regime") == target_regime
    ]
    if not observed_values:
        raise ValueError("conditional bootstrap has no observations in the requested regime")

    observed = sum(observed_values) / len(observed_values)
    rng = random.Random(seed)
    distribution: list[float] = []
    attempts = 0
    max_attempts = resamples * 20
    while len(distribution) < resamples and attempts < max_attempts:
        attempts += 1
        total = 0.0
        count = 0
        for rows in folds:
            n = len(rows)
            length = min(block_length, n)
            sample_indices: list[int] = []
            while len(sample_indices) < n:
                start = rng.randrange(n)
                sample_indices.extend((start + offset) % n for offset in range(length))
            for index in sample_indices[:n]:
                row = rows[index]
                if row.get("regime") == target_regime:
                    total += improvement(row)
                    count += 1
        # A draw with zero target-regime observations has no conditional metric.
        if count:
            distribution.append(total / count)
    if len(distribution) != resamples:
        raise ValueError("could_not_generate_enough_nonempty_conditional_bootstrap_draws")
    distribution.sort()
    low = _quantile(distribution, 0.025)
    high = _quantile(distribution, 0.975)
    return {
        "observed_mean_improvement": observed,
        "confidence_level": 0.95,
        "confidence_interval_95": [low, high],
        "bootstrap_probability_improvement_positive": sum(value > 0 for value in distribution) / len(distribution),
        "bootstrap_resamples": len(distribution),
        "requested_bootstrap_resamples": resamples,
        "block_length_observations": block_length,
        "method": "paired_circular_moving_block_bootstrap_stratified_by_walk_forward_fold_conditioned_on_regime",
        "regime_sample_count": len(observed_values),
        "interpretation": "Conditional descriptive uncertainty; positive values favor the model, not proof of significance or profitability.",
    }


def analyze_prediction_regimes(
    matrix: Mapping[str, Any],
    ohlc_rows: Sequence[Mapping[str, Any]],
    *,
    volatility_window: int = 20,
    bootstrap_resamples: int = 500,
    block_length: int | None = None,
    random_seed: int = 20261023,
) -> dict[str, Any]:
    """Analyze held-out model/baseline quality by historical volatility regime."""
    errors: list[str] = []
    warnings: list[str] = []
    if isinstance(volatility_window, bool) or not isinstance(volatility_window, int) or not 2 <= volatility_window <= 500:
        errors.append("volatility_window_must_be_between_2_and_500")
        volatility_window = 20
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or not 100 <= bootstrap_resamples <= 20000:
        errors.append("bootstrap_resamples_must_be_between_100_and_20000")
    horizon_value = matrix.get("horizon_bars", 1)
    if isinstance(horizon_value, bool) or not isinstance(horizon_value, int) or horizon_value < 1:
        horizon_value = 1
    chosen_block_length = horizon_value if block_length is None else block_length
    if isinstance(chosen_block_length, bool) or not isinstance(chosen_block_length, int) or chosen_block_length < 1:
        errors.append("block_length_must_be_positive_integer")
        chosen_block_length = horizon_value
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        errors.append("random_seed_must_be_integer")
        random_seed = 20261023

    # Reuse the established fail-closed prediction/schema/provenance checks.
    validation = analyze_prediction_matrix(matrix, bootstrap_resamples=100)
    if validation.get("status") == "FAIL":
        errors.extend(["prediction_matrix_validation_failed", *validation.get("errors", [])])
    warnings.extend(validation.get("warnings", []))

    symbol = str(matrix.get("symbol", "")).strip().upper()
    timeframe = str(matrix.get("timeframe", "")).strip().upper()
    if not symbol or not timeframe:
        errors.append("matrix_symbol_or_timeframe_missing")

    try:
        timestamps, closes = _parse_ohlc(ohlc_rows, symbol=symbol, timeframe=timeframe)
    except (TypeError, ValueError) as exc:
        timestamps, closes = [], []
        errors.append(f"ohlc_validation_failed:{exc}")

    if errors:
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_VOLATILITY_REGIME_ANALYSIS",
            "status": "FAIL",
            "run_id": matrix.get("run_id"),
            "symbol": symbol or None,
            "timeframe": timeframe or None,
            "errors": sorted(set(errors)),
            "warnings": sorted(set(warnings)),
            "automatic_model_selection": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "execution_connected": False,
        }

    timestamp_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    volatility = _rolling_log_return_volatility(closes, volatility_window)
    windows = matrix.get("common_fold_test_windows", [])
    windows_by_fold: dict[int, dict[str, Any]] = {}
    for position, window in enumerate(windows):
        if isinstance(window, dict):
            fold_index = window.get("fold_index", position)
            if isinstance(fold_index, int) and not isinstance(fold_index, bool):
                windows_by_fold[fold_index] = window

    fold_assignments: dict[int, dict[datetime, str]] = {}
    threshold_reports: list[dict[str, Any]] = []
    for fold_index, window in sorted(windows_by_fold.items()):
        try:
            test_start = _parse_utc(window.get("test_start_utc"), "test_start_utc")
            test_end = _parse_utc(window.get("test_end_utc"), "test_end_utc")
        except (TypeError, ValueError):
            errors.append(f"fold_{fold_index}_invalid_test_window")
            continue

        historical_volatility = [
            float(value)
            for timestamp, value in zip(timestamps, volatility)
            if timestamp < test_start and value is not None and math.isfinite(value)
        ]
        # A quartile split is not meaningful with a tiny calibration history.
        if len(historical_volatility) < max(8, volatility_window):
            errors.append(f"fold_{fold_index}_insufficient_pretest_volatility_history")
            continue
        low_threshold = _quantile(historical_volatility, 0.25)
        high_threshold = _quantile(historical_volatility, 0.75)
        assignments: dict[datetime, str] = {}
        for timestamp, value in zip(timestamps, volatility):
            if test_start <= timestamp <= test_end and value is not None:
                assignments[timestamp] = _regime(float(value), low_threshold, high_threshold)
        fold_assignments[fold_index] = assignments
        threshold_reports.append({
            "fold_index": fold_index,
            "test_start_utc": test_start.isoformat().replace("+00:00", "Z"),
            "test_end_utc": test_end.isoformat().replace("+00:00", "Z"),
            "pretest_volatility_observations": len(historical_volatility),
            "volatility_window_bars": volatility_window,
            "low_threshold_q25": low_threshold,
            "high_threshold_q75": high_threshold,
            "regime_counts": {regime: 0 for regime in _REGIMES},
            "predictions_without_warmup_volatility": 0,
        })

    if errors:
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_VOLATILITY_REGIME_ANALYSIS",
            "status": "FAIL",
            "run_id": matrix.get("run_id"),
            "symbol": symbol,
            "timeframe": timeframe,
            "errors": sorted(set(errors)),
            "warnings": sorted(set(warnings)),
            "automatic_model_selection": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "execution_connected": False,
        }

    threshold_by_fold = {row["fold_index"]: row for row in threshold_reports}
    experiments_result: list[dict[str, Any]] = []
    skipped_without_volatility = 0
    for experiment_index, experiment in enumerate(matrix.get("experiments", [])):
        # Fold regime counts are shared-window diagnostics, so count each timestamp only
        # once rather than once per model/feature experiment.
        record_fold_counts = experiment_index == 0
        buckets: dict[str, dict[str, Any]] = {
            regime: {
                "labels": [],
                "model": [],
                **{name: [] for name in _PREDICTION_BASELINES.values()},
                "fold_indices": [],
                "rows_by_fold": {},
            }
            for regime in _REGIMES
        }
        full_fold_rows: dict[int, list[dict[str, Any]]] = {}
        for fold in experiment.get("fold_results", []):
            fold_index = fold.get("fold_index")
            if fold_index not in fold_assignments:
                errors.append(f"experiment_{experiment.get('experiment_id')}_fold_{fold_index}_threshold_missing")
                continue
            for prediction in fold.get("test_predictions", []):
                try:
                    timestamp = _parse_utc(prediction.get("timestamp_utc"), "prediction timestamp")
                    if timestamp not in timestamp_to_index:
                        errors.append(f"prediction_timestamp_missing_from_ohlc:{timestamp.isoformat()}")
                        continue
                    regime = fold_assignments[fold_index].get(timestamp)
                    if regime is None:
                        if record_fold_counts:
                            skipped_without_volatility += 1
                            threshold_by_fold[fold_index]["predictions_without_warmup_volatility"] += 1
                    label = prediction["label"]
                    probability = float(prediction["model_probability_class_1"])
                    if label not in (0, 1) or isinstance(label, bool) or not 0 <= probability <= 1 or not math.isfinite(probability):
                        errors.append(f"experiment_{experiment.get('experiment_id')}_invalid_prediction_row")
                        continue
                    baseline_probabilities: dict[str, float] = {}
                    valid_baselines = True
                    for baseline_name in _PREDICTION_BASELINES.values():
                        baseline_probability = float(prediction[baseline_name])
                        if not math.isfinite(baseline_probability) or not 0 <= baseline_probability <= 1:
                            errors.append(f"experiment_{experiment.get('experiment_id')}_invalid_baseline_probability")
                            valid_baselines = False
                            break
                        baseline_probabilities[baseline_name] = baseline_probability
                    if not valid_baselines:
                        continue
                    full_row = {
                        "timestamp": timestamp,
                        "regime": regime,
                        "label": int(label),
                        "model_probability": probability,
                        "baseline_probabilities": baseline_probabilities,
                    }
                    full_fold_rows.setdefault(fold_index, []).append(full_row)
                    if regime is not None:
                        bucket = buckets[regime]
                        bucket["labels"].append(int(label))
                        bucket["model"].append(probability)
                        bucket["fold_indices"].append(fold_index)
                        bucket["rows_by_fold"].setdefault(fold_index, []).append(full_row)
                        for baseline_name, baseline_probability in baseline_probabilities.items():
                            bucket[baseline_name].append(baseline_probability)
                        if record_fold_counts:
                            threshold_by_fold[fold_index]["regime_counts"][regime] += 1
                except (KeyError, TypeError, ValueError):
                    errors.append(f"experiment_{experiment.get('experiment_id')}_malformed_prediction_row")

        regime_metrics: dict[str, Any] = {}
        for regime in _REGIMES:
            bucket = buckets[regime]
            labels = bucket["labels"]
            if not labels:
                regime_metrics[regime] = {
                    "sample_count": 0,
                    "fold_count": 0,
                    "model": None,
                    "baselines": {},
                    "loss_improvements_baseline_minus_model": {},
                    "accuracy_improvements_model_minus_baseline": {},
                    "paired_bootstrap_uncertainty": {},
                }
                continue
            model_metrics = _metric_summary(labels, bucket["model"])
            baseline_metrics = {
                name: _metric_summary(labels, bucket[name])
                for name in _PREDICTION_BASELINES.values()
            }
            loss_improvements: dict[str, dict[str, float]] = {}
            accuracy_improvements: dict[str, float] = {}
            paired_uncertainty: dict[str, dict[str, Any]] = {}
            display_names = {
                "constant_probability_0_5": "constant_probability_0_5",
                "matured_refit_prevalence": "matured_refit_prevalence",
                "lagged_bar_direction": "lagged_bar_direction",
            }
            for baseline_index, (baseline_display, baseline_field) in enumerate(_PREDICTION_BASELINES.items()):
                baseline_metric = baseline_metrics[baseline_field]
                loss_improvements[display_names[baseline_display]] = {
                    "log_loss": baseline_metric["log_loss"] - model_metrics["log_loss"],
                    "brier_score": baseline_metric["brier_score"] - model_metrics["brier_score"],
                }
                accuracy_improvements[display_names[baseline_display]] = (
                    model_metrics["accuracy_at_fixed_half"] - baseline_metric["accuracy_at_fixed_half"]
                )

                metric_uncertainty: dict[str, Any] = {}
                for metric_index, metric_name in enumerate(("log_loss", "brier_score")):
                    selected_folds = [
                        rows for _, rows in sorted(full_fold_rows.items())
                        if any(row.get("regime") == regime for row in rows)
                    ]
                    seed = random_seed + experiment_index * 10000 + _REGIMES.index(regime) * 1000 + baseline_index * 10 + metric_index
                    bootstrap = _conditional_block_bootstrap(
                        selected_folds,
                        target_regime=regime,
                        baseline_field=baseline_field,
                        metric_name=metric_name,
                        block_length=chosen_block_length,
                        resamples=bootstrap_resamples,
                        seed=seed,
                    )
                    interval_low, interval_high = bootstrap["confidence_interval_95"]
                    interval_evidence = (
                        "positive_interval_excludes_zero" if interval_low > 0
                        else "negative_interval_excludes_zero" if interval_high < 0
                        else "interval_includes_zero"
                    )
                    metric_uncertainty[metric_name] = {
                        **bootstrap,
                        "interval_evidence": interval_evidence,
                        "folds_compared": sum(
                            any(row.get("regime") == regime for row in rows)
                            for rows in selected_folds
                        ),
                    }
                paired_uncertainty[display_names[baseline_display]] = metric_uncertainty

            regime_metrics[regime] = {
                "sample_count": len(labels),
                "fold_count": len(set(bucket["fold_indices"])),
                "model": model_metrics,
                "baselines": baseline_metrics,
                "loss_improvements_baseline_minus_model": loss_improvements,
                "accuracy_improvements_model_minus_baseline": accuracy_improvements,
                "paired_bootstrap_uncertainty": paired_uncertainty,
            }
        experiments_result.append({
            "experiment_id": experiment.get("experiment_id"),
            "model_type": experiment.get("model_type"),
            "feature_variant": experiment.get("feature_variant"),
            "regime_metrics": regime_metrics,
            "ranked_or_selected": False,
            "actionable": False,
        })

    if errors:
        status = "FAIL"
    else:
        status = "PASS_WITH_WARNINGS" if warnings or skipped_without_volatility else "PASS"
    if skipped_without_volatility:
        warnings.append("some_held_out_predictions_lacked_a_full_past_only_volatility_window")
        status = "PASS_WITH_WARNINGS"

    return {
        "schema_version": 1,
        "scope": "GOLDCREST_VOLATILITY_REGIME_ANALYSIS",
        "status": status,
        "run_id": matrix.get("run_id"),
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon_bars": matrix.get("horizon_bars"),
        "volatility_definition": (
            "Population standard deviation of trailing close-to-close log returns; "
            "uses only bars available at each prediction timestamp."
        ),
        "regime_threshold_definition": (
            "Per-fold 25th/75th percentile thresholds estimated only from volatility "
            "observations strictly before that fold's test start. Regimes are descriptive."
        ),
        "volatility_window_bars": volatility_window,
        "bootstrap": {
            "method": "paired_circular_moving_block_bootstrap_stratified_by_walk_forward_fold_conditioned_on_regime",
            "resamples": bootstrap_resamples,
            "block_length_observations": chosen_block_length,
            "block_length_note": "Defaults to label horizon; descriptive dependence adjustment, not proof of independence.",
            "confidence_level": 0.95,
            "seed": random_seed,
        },
        "fold_thresholds": threshold_reports,
        "experiments": experiments_result,
        "held_out_predictions_without_volatility": skipped_without_volatility,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "automatic_model_selection": False,
        "test_metrics_used_for_selection": False,
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "Conditional held-out metrics are descriptive diagnostics, not statistical proof, "
            "profitability estimates, a model ranking, or production-readiness evidence. "
            "Do not select a model using these same test windows."
        ),
    }


def analyze_research_run(
    manifest_path: str | Path,
    ohlc_csv_path: str | Path,
    *,
    volatility_window: int = 20,
    bootstrap_resamples: int = 500,
    block_length: int | None = None,
    random_seed: int = 20261023,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    ohlc_file = Path(ohlc_csv_path).resolve()
    integrity = summarize_research_run(manifest_file)
    if integrity.get("status") == "FAIL":
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_VOLATILITY_REGIME_ANALYSIS",
            "status": "FAIL",
            "manifest_path": str(manifest_file),
            "errors": ["research_run_integrity_check_failed", *integrity.get("errors", [])],
            "warnings": integrity.get("warnings", []),
            "automatic_model_selection": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "execution_connected": False,
        }
    try:
        manifest = _read_json(manifest_file)
        if not ohlc_file.is_file():
            raise ValueError("raw_ohlc_file_not_found")
        ohlc_digest = _sha256(ohlc_file)
        if ohlc_digest != manifest.get("raw_ohlc_sha256"):
            raise ValueError("raw_ohlc_sha256_does_not_match_pipeline_manifest")
        matrix_name = manifest.get("experiment_matrix_report")
        if not isinstance(matrix_name, str) or not matrix_name:
            raise ValueError("experiment_matrix_report_path_missing")
        matrix_file = Path(matrix_name)
        if not matrix_file.is_absolute():
            matrix_file = (manifest_file.parent / matrix_file).resolve()
        if not matrix_file.is_file():
            raise ValueError("experiment_matrix_report_not_found")
        matrix_digest = _sha256(matrix_file)
        if matrix_digest != manifest.get("experiment_matrix_report_sha256"):
            raise ValueError("experiment_matrix_report_sha256_mismatch")
        matrix = _read_json(matrix_file)
        if matrix.get("run_id") != manifest.get("run_id"):
            raise ValueError("experiment_matrix_run_id_mismatch")
        with ohlc_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        result = analyze_prediction_regimes(
            matrix, rows, volatility_window=volatility_window,
            bootstrap_resamples=bootstrap_resamples, block_length=block_length,
            random_seed=random_seed,
        )
        result["manifest_path"] = str(manifest_file)
        result["ohlc_path"] = str(ohlc_file)
        result["ohlc_sha256"] = ohlc_digest
        result["ohlc_hash_verified"] = True
        result["experiment_matrix_path"] = str(matrix_file)
        result["experiment_matrix_sha256"] = matrix_digest
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
            "scope": "GOLDCREST_VOLATILITY_REGIME_ANALYSIS",
            "status": "FAIL",
            "manifest_path": str(manifest_file),
            "ohlc_path": str(ohlc_file),
            "errors": [f"regime_analysis_input_invalid:{type(exc).__name__}:{exc}"],
            "warnings": integrity.get("warnings", []),
            "automatic_model_selection": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "execution_connected": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze held-out research predictions by causal historical-volatility regimes."
    )
    parser.add_argument("--manifest", required=True, help="Path to <run-id>_pipeline_manifest.json")
    parser.add_argument("--ohlc-csv", required=True, help="Exact raw OHLC CSV whose SHA-256 is recorded in the manifest")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--volatility-window", type=int, default=20, help="Trailing close-to-close return window (2-500 bars)")
    parser.add_argument("--bootstrap-resamples", type=int, default=500, help="Paired block-bootstrap draws per regime comparison (100-20000)")
    parser.add_argument("--block-length", type=int, help="Contiguous observations per block; defaults to label horizon")
    parser.add_argument("--seed", type=int, default=20261023, help="Deterministic bootstrap seed")
    args = parser.parse_args(argv)
    result = analyze_research_run(
        args.manifest, args.ohlc_csv, volatility_window=args.volatility_window,
        bootstrap_resamples=args.bootstrap_resamples, block_length=args.block_length,
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

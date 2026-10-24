"""Consolidated, non-actionable evidence report for a completed research run.

Combines the run-integrity receipt, validation, provenance, temporal calibration, and any
optional baseline/regime/gap analyses already produced. This module does not retrain models,
recompute their metrics, rank candidates, or connect to broker execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

from app.research_run_audit import audit_research_run_manifest

_SCHEMA_VERSION = 1
_REPORT_SCOPE = "GOLDCREST_CONSOLIDATED_RESEARCH_EVIDENCE"
_ALLOWED_STATUSES = {"PASS", "PASS_WITH_WARNINGS", "FAIL"}
_SAFETY_FLAGS = (
    "actionable",
    "execution_connected",
    "automatic_winner_selection",
    "automatic_model_selection",
    "test_metrics_used_for_selection",
    "test_metrics_used_for_tuning",
)
_OPTIONAL_REPORTS = {
    "baseline_analysis": ("GOLDCREST_TIME_SERIES_BASELINE_ANALYSIS", "_baseline_analysis.json"),
    "regime_analysis": ("GOLDCREST_VOLATILITY_REGIME_ANALYSIS", "_regime_analysis.json"),
    "temporal_stability": (
        "GOLDCREST_TEMPORAL_STABILITY_GAP_SENSITIVITY",
        "_temporal_stability.json",
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_path(value: object, base_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        relative = (base_dir / candidate).resolve()
        if relative.is_file():
            return relative
    return candidate.resolve()


def _number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return value


def _add_issue(target: list[str], message: str) -> None:
    if message not in target:
        target.append(message)


def _summary_for_validation(value: Mapping[str, Any]) -> dict[str, Any]:
    dataset = value.get("dataset_quality")
    matrix = value.get("experiment_quality")
    runtime = value.get("runtime_preflight")
    return {
        "dataset_status": dataset.get("status") if isinstance(dataset, dict) else None,
        "dataset_row_count": dataset.get("row_count") if isinstance(dataset, dict) else None,
        "feature_count": dataset.get("feature_count") if isinstance(dataset, dict) else None,
        "experiment_status": matrix.get("status") if isinstance(matrix, dict) else None,
        "experiment_count": matrix.get("experiment_count") if isinstance(matrix, dict) else None,
        "fold_count": matrix.get("fold_count") if isinstance(matrix, dict) else None,
        "runtime_status": runtime.get("status") if isinstance(runtime, dict) else None,
        "error_count": len(value.get("errors", [])) if isinstance(value.get("errors"), list) else None,
        "warning_count": len(value.get("warnings", [])) if isinstance(value.get("warnings"), list) else None,
    }


def _summary_for_calibration(value: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = value.get("comparisons")
    summaries = []
    if isinstance(comparisons, list):
        for row in comparisons:
            if not isinstance(row, dict):
                continue
            summaries.append({
                "model_name": row.get("model_name"),
                "partition": row.get("partition"),
                "period_count": row.get("period_count"),
                "ece_first": row.get("first_period_ece"),
                "ece_last": row.get("last_period_ece"),
                "ece_delta_first_to_last": row.get("first_to_last_ece_delta"),
                "ece_min": row.get("ece_min"),
                "ece_max": row.get("ece_max"),
                "interval_status": row.get("status"),
                "actionable": False,
            })
    return {
        "comparison_count": len(comparisons) if isinstance(comparisons, list) else 0,
        "matrix_payload_hash_verified": value.get("experiment_matrix_payload_hash_verified") is True,
        "comparisons": summaries,
    }


def _summary_for_baseline(value: Mapping[str, Any]) -> dict[str, Any]:
    experiments = value.get("experiments")
    summaries = []
    if isinstance(experiments, list):
        for experiment in experiments:
            if not isinstance(experiment, dict):
                continue
            comparisons = []
            raw_comparisons = experiment.get("probability_loss_comparisons")
            if isinstance(raw_comparisons, list):
                for comparison in raw_comparisons:
                    if not isinstance(comparison, dict):
                        continue
                    metric_summaries = {}
                    raw_metrics = comparison.get("metrics")
                    if isinstance(raw_metrics, dict):
                        for metric_name, metric in raw_metrics.items():
                            if not isinstance(metric, dict):
                                continue
                            metric_summaries[metric_name] = {
                                "improvement_baseline_minus_model": metric.get(
                                    "improvement_baseline_minus_model"
                                ),
                                "confidence_interval_95": metric.get("confidence_interval_95"),
                                "interval_evidence": metric.get("interval_evidence"),
                                "folds_compared": metric.get("folds_compared"),
                                "folds_with_positive_mean_improvement": metric.get(
                                    "folds_with_positive_mean_improvement"
                                ),
                                "folds_with_negative_mean_improvement": metric.get(
                                    "folds_with_negative_mean_improvement"
                                ),
                            }
                    comparisons.append({
                        "baseline": comparison.get("baseline"),
                        "metrics": metric_summaries,
                    })
            summaries.append({
                "experiment_id": experiment.get("experiment_id"),
                "model_type": experiment.get("model_type"),
                "feature_variant": experiment.get("feature_variant"),
                "sample_count_counted_by_fold": experiment.get("test_prediction_count_counted_by_fold"),
                "fold_count": experiment.get("fold_count"),
                "model_metrics": experiment.get("model_metrics"),
                "baseline_comparisons": comparisons,
                "ranked_or_selected": False,
            })
    return {
        "experiment_count": len(summaries),
        "fold_count": value.get("fold_count"),
        "repeated_held_out_timestamps_warning": any(
            "timestamps_repeat_across_folds" in str(item) for item in value.get("warnings", [])
        ),
        "experiments": summaries,
    }


def _summary_for_regimes(value: Mapping[str, Any]) -> dict[str, Any]:
    experiments = value.get("experiments")
    summaries = []
    if isinstance(experiments, list):
        for experiment in experiments:
            if not isinstance(experiment, dict):
                continue
            regimes: dict[str, Any] = {}
            raw_regimes = experiment.get("regime_metrics")
            if isinstance(raw_regimes, dict):
                for regime_name, regime in raw_regimes.items():
                    if not isinstance(regime, dict):
                        continue
                    regimes[regime_name] = {
                        "sample_count": regime.get("sample_count"),
                        "fold_count": regime.get("fold_count"),
                        "model_metrics": regime.get("model"),
                        "loss_improvements_baseline_minus_model": regime.get(
                            "loss_improvements_baseline_minus_model"
                        ),
                        "paired_bootstrap_uncertainty": regime.get("paired_bootstrap_uncertainty"),
                    }
            summaries.append({
                "experiment_id": experiment.get("experiment_id"),
                "model_type": experiment.get("model_type"),
                "feature_variant": experiment.get("feature_variant"),
                "regimes": regimes,
                "ranked_or_selected": False,
            })
    return {
        "experiment_count": len(summaries),
        "volatility_window_bars": value.get("volatility_window_bars"),
        "held_out_predictions_without_volatility": value.get("held_out_predictions_without_volatility"),
        "fold_threshold_count": len(value.get("fold_thresholds", []))
        if isinstance(value.get("fold_thresholds"), list) else None,
        "experiments": summaries,
    }


def _summary_for_temporal_stability(value: Mapping[str, Any]) -> dict[str, Any]:
    experiments = value.get("experiments")
    summaries = []
    if isinstance(experiments, list):
        for experiment in experiments:
            if not isinstance(experiment, dict):
                continue
            exposure = experiment.get("gap_exposure")
            summaries.append({
                "experiment_id": experiment.get("experiment_id"),
                "model_type": experiment.get("model_type"),
                "fold_count": experiment.get("fold_count"),
                "sample_count": experiment.get("sample_count"),
                "fold_consistency": experiment.get("fold_consistency"),
                "gap_exposure": (
                    {
                        "horizon_crosses_long_gap_count": exposure.get("horizon_crosses_long_gap_count"),
                        "horizon_does_not_cross_long_gap_count": exposure.get(
                            "horizon_does_not_cross_long_gap_count"
                        ),
                        "horizon_crosses_weekend_gap_count": exposure.get(
                            "horizon_crosses_weekend_gap_count"
                        ),
                        "horizon_crosses_non_weekend_gap_count": exposure.get(
                            "horizon_crosses_non_weekend_gap_count"
                        ),
                        "horizon_crosses_holiday_context_gap_count": exposure.get(
                            "horizon_crosses_holiday_context_gap_count"
                        ),
                    }
                    if isinstance(exposure, dict) else None
                ),
            })
    return {
        "long_gap_count": value.get("long_gap_count"),
        "weekend_closure_like_gap_count": value.get("weekend_closure_like_gap_count"),
        "non_weekend_long_gap_count": value.get("non_weekend_long_gap_count"),
        "holiday_context_gap_count": value.get("holiday_context_gap_count"),
        "prediction_count": value.get("prediction_count"),
        "unique_prediction_timestamp_count": value.get("unique_prediction_timestamp_count"),
        "prediction_horizon_crosses_long_gap_count": value.get("prediction_horizon_crosses_long_gap_count"),
        "prediction_horizon_crosses_long_gap_fraction": value.get("prediction_horizon_crosses_long_gap_fraction"),
        "experiments": summaries,
    }


def _artifact_record(
    *,
    section_name: str,
    path: Path,
    expected_scope: str | None,
    expected_run_id: str,
    expected_symbol: str | None,
    expected_timeframe: str | None,
    expected_manifest: Path,
    expected_matrix_sha256: str | None,
    expected_sha256: str | None = None,
    mandatory: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not path.is_file():
        message = f"{section_name}_artifact_missing"
        (errors if mandatory else warnings).append(message)
        return ({
            "status": "MISSING" if mandatory else "NOT_PROVIDED",
            "path": str(path),
            "sha256": None,
            "scope": expected_scope,
        }, None, errors, warnings)

    try:
        report = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return ({
            "status": "FAIL", "path": str(path), "sha256": _sha256(path),
            "scope": expected_scope,
        }, None, [f"{section_name}_artifact_unreadable:{type(exc).__name__}"], warnings)

    digest = _sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        errors.append(f"{section_name}_sha256_mismatch")
    if expected_scope and report.get("scope") != expected_scope:
        errors.append(f"{section_name}_scope_mismatch")
    if report.get("run_id") != expected_run_id:
        errors.append(f"{section_name}_run_id_mismatch")
    if expected_symbol and report.get("symbol") is not None:
        if str(report.get("symbol")).strip().upper() != expected_symbol.upper():
            errors.append(f"{section_name}_symbol_mismatch")
    if expected_timeframe and report.get("timeframe") is not None:
        if str(report.get("timeframe")).strip().upper() != expected_timeframe.upper():
            errors.append(f"{section_name}_timeframe_mismatch")

    reported_manifest = report.get("manifest_path")
    if reported_manifest:
        reported_path = Path(str(reported_manifest))
        if not reported_path.is_absolute():
            reported_path = (path.parent / reported_path).resolve()
        if reported_path.resolve() != expected_manifest.resolve():
            errors.append(f"{section_name}_manifest_path_mismatch")

    reported_matrix_hash = report.get("experiment_matrix_sha256")
    if reported_matrix_hash is None:
        reported_matrix_hash = report.get("source_matrix_sha256")
    if reported_matrix_hash is not None and expected_matrix_sha256 is not None:
        if reported_matrix_hash != expected_matrix_sha256:
            errors.append(f"{section_name}_source_matrix_hash_mismatch")
    elif section_name in _OPTIONAL_REPORTS and expected_matrix_sha256 is not None:
        warnings.append(f"{section_name}_source_matrix_hash_not_reported")

    for flag in _SAFETY_FLAGS:
        if flag in report and report.get(flag) is not False:
            errors.append(f"{section_name}_{flag}_must_be_false")
    status = report.get("status")
    if status not in _ALLOWED_STATUSES:
        errors.append(f"{section_name}_status_invalid")
    elif status == "FAIL":
        errors.append(f"{section_name}_reported_fail")
    elif status == "PASS_WITH_WARNINGS":
        warnings.append(f"{section_name}_reported_warnings")
    for item in report.get("errors", []) if isinstance(report.get("errors"), list) else []:
        _add_issue(errors, f"{section_name}_reported_error:{item}")
    for item in report.get("warnings", []) if isinstance(report.get("warnings"), list) else []:
        _add_issue(warnings, f"{section_name}_reported_warning:{item}")

    record = {
        "status": status if status in _ALLOWED_STATUSES else "FAIL",
        "path": str(path.resolve()),
        "sha256": digest,
        "scope": report.get("scope"),
    }
    if errors:
        record["status"] = "FAIL"
    return record, report, errors, warnings


def build_evidence_interpretation(
    *,
    baseline: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None,
    regimes: Mapping[str, Any] | None,
    temporal_stability: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Summarize cross-diagnostic patterns without ranking or promoting experiments."""
    baseline_counts = {
        "comparisons_evaluated": 0,
        "intervals_excluding_zero_positive": 0,
        "intervals_excluding_zero_negative": 0,
        "intervals_including_zero": 0,
        "intervals_unavailable": 0,
        "positive_point_estimates": 0,
        "negative_point_estimates": 0,
        "zero_point_estimates": 0,
    }
    if isinstance(baseline, dict) and baseline.get("status") in {"PASS", "PASS_WITH_WARNINGS"}:
        experiments = baseline.get("experiments")
        for experiment in experiments if isinstance(experiments, list) else []:
            if not isinstance(experiment, dict):
                continue
            comparisons = experiment.get("probability_loss_comparisons")
            for comparison in comparisons if isinstance(comparisons, list) else []:
                if not isinstance(comparison, dict):
                    continue
                metrics = comparison.get("metrics")
                for metric_name in ("log_loss", "brier_score"):
                    metric = metrics.get(metric_name) if isinstance(metrics, dict) else None
                    if not isinstance(metric, dict):
                        continue
                    improvement = _number(metric.get("improvement_baseline_minus_model"))
                    if improvement is not None:
                        if improvement > 0:
                            baseline_counts["positive_point_estimates"] += 1
                        elif improvement < 0:
                            baseline_counts["negative_point_estimates"] += 1
                        else:
                            baseline_counts["zero_point_estimates"] += 1
                    interval = metric.get("confidence_interval_95")
                    if (
                        not isinstance(interval, list)
                        or len(interval) != 2
                        or _number(interval[0]) is None
                        or _number(interval[1]) is None
                        or float(interval[0]) > float(interval[1])
                    ):
                        baseline_counts["intervals_unavailable"] += 1
                        continue
                    low, high = float(interval[0]), float(interval[1])
                    baseline_counts["comparisons_evaluated"] += 1
                    if low > 0:
                        baseline_counts["intervals_excluding_zero_positive"] += 1
                    elif high < 0:
                        baseline_counts["intervals_excluding_zero_negative"] += 1
                    else:
                        baseline_counts["intervals_including_zero"] += 1
    if baseline_counts["comparisons_evaluated"] == 0:
        baseline_status = "UNAVAILABLE"
    elif baseline_counts["intervals_excluding_zero_positive"] == 0 and baseline_counts["intervals_excluding_zero_negative"] == 0:
        baseline_status = "NO_CLEAR_INTERVAL_SEPARATION_FROM_TESTED_BASELINES"
    elif baseline_counts["intervals_excluding_zero_positive"] and baseline_counts["intervals_excluding_zero_negative"]:
        baseline_status = "MIXED_INTERVAL_EVIDENCE"
    elif baseline_counts["intervals_excluding_zero_positive"]:
        baseline_status = "POSITIVE_INTERVAL_EVIDENCE_PRESENT"
    else:
        baseline_status = "NEGATIVE_INTERVAL_EVIDENCE_PRESENT"

    calibration_counts = {
        "comparisons_evaluated": 0,
        "upward_ece_changes": 0,
        "downward_ece_changes": 0,
        "approximately_flat_ece_changes": 0,
        "deltas_unavailable": 0,
    }
    calibration_rows: list[dict[str, Any]] = []
    if isinstance(calibration, dict) and calibration.get("status") in {"PASS", "PASS_WITH_WARNINGS"}:
        raw_comparisons = calibration.get("comparisons")
        for row in raw_comparisons if isinstance(raw_comparisons, list) else []:
            if not isinstance(row, dict) or row.get("partition") not in (None, "test"):
                continue
            delta = _number(row.get("first_to_last_ece_delta"))
            if delta is None:
                calibration_counts["deltas_unavailable"] += 1
                continue
            calibration_counts["comparisons_evaluated"] += 1
            if delta > 1e-12:
                calibration_counts["upward_ece_changes"] += 1
                direction = "UPWARD"
            elif delta < -1e-12:
                calibration_counts["downward_ece_changes"] += 1
                direction = "DOWNWARD"
            else:
                calibration_counts["approximately_flat_ece_changes"] += 1
                direction = "APPROXIMATELY_FLAT"
            calibration_rows.append({
                "model_name": row.get("model_name"),
                "partition": row.get("partition", "test"),
                "first_to_last_ece_delta": delta,
                "direction": direction,
            })
    if not calibration_counts["comparisons_evaluated"]:
        calibration_status = "UNAVAILABLE"
    elif calibration_counts["upward_ece_changes"] == calibration_counts["comparisons_evaluated"]:
        calibration_status = "UPWARD_ECE_DRIFT_OBSERVED_IN_ALL_COMPARISONS"
    elif calibration_counts["downward_ece_changes"] == calibration_counts["comparisons_evaluated"]:
        calibration_status = "DOWNWARD_ECE_CHANGE_OBSERVED_IN_ALL_COMPARISONS"
    elif calibration_counts["upward_ece_changes"] or calibration_counts["downward_ece_changes"]:
        calibration_status = "MIXED_OR_NONMONOTONIC_ECE_CHANGES"
    else:
        calibration_status = "NO_MATERIAL_FIRST_TO_LAST_ECE_CHANGE_OBSERVED"

    regime_groups: list[dict[str, Any]] = []
    if isinstance(regimes, dict) and regimes.get("status") in {"PASS", "PASS_WITH_WARNINGS"}:
        experiments = regimes.get("experiments")
        for experiment in experiments if isinstance(experiments, list) else []:
            if not isinstance(experiment, dict):
                continue
            regime_metrics = experiment.get("regime_metrics")
            if not isinstance(regime_metrics, dict):
                continue
            by_baseline_metric: dict[tuple[str, str], list[tuple[str, float]]] = {}
            for regime_name, regime in regime_metrics.items():
                if not isinstance(regime, dict):
                    continue
                improvements = regime.get("loss_improvements_baseline_minus_model")
                if not isinstance(improvements, dict):
                    continue
                for baseline_name, metric_values in improvements.items():
                    if not isinstance(metric_values, dict):
                        continue
                    for metric_name in ("log_loss", "brier_score"):
                        value = _number(metric_values.get(metric_name))
                        if value is not None:
                            by_baseline_metric.setdefault((str(baseline_name), metric_name), []).append(
                                (str(regime_name), float(value))
                            )
            for (baseline_name, metric_name), rows in sorted(by_baseline_metric.items()):
                positive = sum(value > 0 for _, value in rows)
                negative = sum(value < 0 for _, value in rows)
                neutral = len(rows) - positive - negative
                if len(rows) < 2:
                    consistency = "INSUFFICIENT_REGIME_COVERAGE"
                elif positive and negative:
                    consistency = "MIXED_BY_VOLATILITY_REGIME"
                elif positive:
                    consistency = "POSITIVE_IN_ALL_OBSERVED_REGIMES"
                elif negative:
                    consistency = "NEGATIVE_IN_ALL_OBSERVED_REGIMES"
                else:
                    consistency = "ZERO_IMPROVEMENT_IN_ALL_OBSERVED_REGIMES"
                regime_groups.append({
                    "experiment_id": experiment.get("experiment_id"),
                    "model_type": experiment.get("model_type"),
                    "baseline": baseline_name,
                    "metric": metric_name,
                    "regimes_evaluated": len(rows),
                    "positive_regime_improvements": positive,
                    "negative_regime_improvements": negative,
                    "zero_regime_improvements": neutral,
                    "consistency": consistency,
                })
    mixed_regime_groups = sum(
        item["consistency"] == "MIXED_BY_VOLATILITY_REGIME" for item in regime_groups
    )
    if not regime_groups:
        regime_status = "UNAVAILABLE"
    elif mixed_regime_groups:
        regime_status = "MIXED_BY_VOLATILITY_REGIME_OBSERVED"
    elif all(item["consistency"] == "POSITIVE_IN_ALL_OBSERVED_REGIMES" for item in regime_groups):
        regime_status = "POSITIVE_IMPROVEMENTS_ACROSS_OBSERVED_REGIMES"
    elif all(item["consistency"] == "NEGATIVE_IN_ALL_OBSERVED_REGIMES" for item in regime_groups):
        regime_status = "NEGATIVE_IMPROVEMENTS_ACROSS_OBSERVED_REGIMES"
    else:
        regime_status = "REGIME_EVIDENCE_INCONCLUSIVE_OR_INCOMPLETE"

    gap_summary: dict[str, Any] = {
        "status": "UNAVAILABLE",
        "long_gap_count": None,
        "weekend_closure_like_gap_count": None,
        "non_weekend_long_gap_count": None,
        "prediction_horizon_crosses_long_gap_count": None,
        "prediction_horizon_crosses_long_gap_fraction": None,
    }
    if isinstance(temporal_stability, dict) and temporal_stability.get("status") in {"PASS", "PASS_WITH_WARNINGS"}:
        gap_summary = {
            "status": temporal_stability.get("status"),
            "long_gap_count": temporal_stability.get("long_gap_count"),
            "weekend_closure_like_gap_count": temporal_stability.get("weekend_closure_like_gap_count"),
            "non_weekend_long_gap_count": temporal_stability.get("non_weekend_long_gap_count"),
            "prediction_horizon_crosses_long_gap_count": temporal_stability.get(
                "prediction_horizon_crosses_long_gap_count"
            ),
            "prediction_horizon_crosses_long_gap_fraction": temporal_stability.get(
                "prediction_horizon_crosses_long_gap_fraction"
            ),
        }

    return {
        "baseline_comparison": {
            "status": baseline_status,
            **baseline_counts,
            "interpretation": (
                "Baseline loss improvement is baseline loss minus model loss. A confidence interval that includes zero "
                "does not show clear separation in this descriptive comparison; it is not a formal profitability test."
            ),
        },
        "temporal_calibration": {
            "status": calibration_status,
            **calibration_counts,
            "comparisons": calibration_rows,
            "interpretation": (
                "ECE direction describes first-to-last test-period change only. It is a drift diagnostic, "
                "not a significance test or proof of calibration failure."
            ),
        },
        "volatility_regime_consistency": {
            "status": regime_status,
            "comparison_group_count": len(regime_groups),
            "mixed_comparison_group_count": mixed_regime_groups,
            "comparisons": regime_groups,
            "interpretation": (
                "Positive baseline-minus-model loss improvement favors the model. Mixed signs across regimes "
                "indicate conditional heterogeneity; regime summaries are descriptive and may have dependent samples."
            ),
        },
        "label_gap_exposure": gap_summary,
        "research_conclusion": "DESCRIPTIVE_ONLY_NO_MODEL_SELECTED",
        "promotion_readiness": "NOT_ESTABLISHED_BY_THIS_REPORT",
        "automatic_model_selection": False,
        "test_metrics_used_for_selection": False,
        "actionable": False,
        "execution_connected": False,
    }


def build_research_evidence_report(
    manifest_path: str | Path,
    *,
    baseline_analysis_path: str | Path | None = None,
    regime_analysis_path: str | Path | None = None,
    temporal_stability_path: str | Path | None = None,
    validation_report_path: str | Path | None = None,
    temporal_calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and summarize evidence artifacts attached to one immutable research run."""
    manifest_file = Path(manifest_path).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest = _read_json(manifest_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "schema_version": _SCHEMA_VERSION,
            "scope": _REPORT_SCOPE,
            "status": "FAIL",
            "manifest_path": str(manifest_file),
            "errors": [f"pipeline_manifest_unreadable:{type(exc).__name__}"],
            "warnings": [],
            "actionable": False,
            "execution_connected": False,
            "automatic_model_selection": False,
            "test_metrics_used_for_selection": False,
        }

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        errors.append("pipeline_manifest_run_id_missing")
        run_id = ""
    for flag in ("actionable", "execution_connected", "automatic_winner_selection", "test_metrics_used_for_selection"):
        if manifest.get(flag) is not False:
            errors.append(f"pipeline_manifest_{flag}_must_be_false")

    integrity = audit_research_run_manifest(manifest_file)
    if integrity.get("status") == "FAIL":
        errors.extend(f"run_integrity:{item}" for item in integrity.get("errors", []))
    elif integrity.get("status") == "PASS_WITH_WARNINGS":
        warnings.extend(f"run_integrity:{item}" for item in integrity.get("warnings", []))

    symbol = str(manifest.get("symbol", "")).strip().upper() or None
    timeframe = str(manifest.get("timeframe", "")).strip().upper() or None
    matrix_hash = manifest.get("experiment_matrix_report_sha256")
    sections: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any] | None] = {}

    # These two receipts are explicitly referenced and hash-bound by the pipeline manifest.
    for name, path_field, hash_field, scope in (
        ("provenance_audit", "provenance_audit_path", "provenance_audit_sha256", None),
    ):
        path = _resolve_path(manifest.get(path_field), manifest_file.parent)
        declared = bool(manifest.get(path_field) or manifest.get(hash_field))
        if path is None:
            sections[name] = {
                "status": "MISSING" if declared else "NOT_PROVIDED",
                "path": None,
                "sha256": None,
                "scope": scope,
            }
            records[name] = None
            (errors if declared else warnings).append(f"{name}_path_missing" if declared else f"{name}_not_recorded_in_legacy_manifest")
            continue
        record, section_report, section_errors, section_warnings = _artifact_record(
            section_name=name,
            path=path,
            expected_scope=scope,
            expected_run_id=run_id,
            expected_symbol=symbol,
            expected_timeframe=timeframe,
            expected_manifest=manifest_file,
            expected_matrix_sha256=matrix_hash if isinstance(matrix_hash, str) else None,
            expected_sha256=manifest.get(hash_field) if isinstance(manifest.get(hash_field), str) else None,
            mandatory=declared,
        )
        sections[name] = record
        records[name] = section_report
        errors.extend(section_errors)
        warnings.extend(section_warnings)

    standard_temporal_path = manifest_file.parent / f"{run_id}_temporal_calibration_stability.json"
    temporal_path = (
        Path(temporal_calibration_path).resolve()
        if temporal_calibration_path is not None
        else _resolve_path(manifest.get("temporal_calibration_audit_path"), manifest_file.parent)
    )
    temporal_declared = bool(
        temporal_calibration_path is not None
        or manifest.get("temporal_calibration_audit_path")
        or manifest.get("temporal_calibration_audit_sha256")
    )
    if temporal_path is None and temporal_calibration_path is None and standard_temporal_path.is_file():
        temporal_path = standard_temporal_path.resolve()
    if temporal_path is None:
        sections["temporal_calibration"] = {
            "status": "MISSING" if temporal_declared else "NOT_PROVIDED",
            "path": None if not temporal_declared else str(standard_temporal_path),
            "sha256": None,
            "scope": "GOLDCREST_TEMPORAL_CALIBRATION_STABILITY_AUDIT",
        }
        records["temporal_calibration"] = None
        (errors if temporal_declared else warnings).append(
            "temporal_calibration_path_missing" if temporal_declared else "temporal_calibration_not_recorded_in_legacy_run"
        )
    else:
        temporal_record, temporal_report, section_errors, section_warnings = _artifact_record(
            section_name="temporal_calibration",
            path=temporal_path,
            expected_scope="GOLDCREST_TEMPORAL_CALIBRATION_STABILITY_AUDIT",
            expected_run_id=run_id,
            expected_symbol=symbol,
            expected_timeframe=timeframe,
            expected_manifest=manifest_file,
            expected_matrix_sha256=matrix_hash if isinstance(matrix_hash, str) else None,
            expected_sha256=(
                manifest.get("temporal_calibration_audit_sha256")
                if temporal_path == _resolve_path(manifest.get("temporal_calibration_audit_path"), manifest_file.parent)
                and isinstance(manifest.get("temporal_calibration_audit_sha256"), str)
                else None
            ),
            mandatory=temporal_declared,
        )
        sections["temporal_calibration"] = temporal_record
        records["temporal_calibration"] = temporal_report
        errors.extend(section_errors)
        warnings.extend(section_warnings)

    validation_path = (
        Path(validation_report_path).resolve()
        if validation_report_path is not None
        else (manifest_file.parent / f"{run_id}_validation_report.json")
    )
    validation_record, validation_report, section_errors, section_warnings = _artifact_record(
        section_name="validation",
        path=validation_path,
        expected_scope="GOLDCREST_COMPLETED_RUN_VALIDATION",
        expected_run_id=run_id,
        expected_symbol=symbol,
        expected_timeframe=timeframe,
        expected_manifest=manifest_file,
        expected_matrix_sha256=matrix_hash if isinstance(matrix_hash, str) else None,
        mandatory=validation_report_path is not None,
    )
    sections["validation"] = validation_record
    records["validation"] = validation_report
    errors.extend(section_errors)
    warnings.extend(section_warnings)

    for name, (scope, suffix) in _OPTIONAL_REPORTS.items():
        supplied = {
            "baseline_analysis": baseline_analysis_path,
            "regime_analysis": regime_analysis_path,
            "temporal_stability": temporal_stability_path,
        }[name]
        path = Path(supplied).resolve() if supplied is not None else (manifest_file.parent / f"{run_id}{suffix}")
        explicitly_supplied = supplied is not None
        if supplied is None and not path.is_file():
            sections[name] = {
                "status": "NOT_PROVIDED",
                "path": str(path),
                "sha256": None,
                "scope": scope,
            }
            records[name] = None
            _add_issue(warnings, f"{name}_not_provided")
            continue
        record, section_report, section_errors, section_warnings = _artifact_record(
            section_name=name,
            path=path,
            expected_scope=scope,
            expected_run_id=run_id,
            expected_symbol=symbol,
            expected_timeframe=timeframe,
            expected_manifest=manifest_file,
            expected_matrix_sha256=matrix_hash if isinstance(matrix_hash, str) else None,
            mandatory=explicitly_supplied,
        )
        sections[name] = record
        records[name] = section_report
        errors.extend(section_errors)
        warnings.extend(section_warnings)

    summary: dict[str, Any] = {
        "validation": _summary_for_validation(records["validation"])
        if records.get("validation") is not None else None,
        "provenance_audit": {
            "status": records["provenance_audit"].get("status"),
            "experiment_count": records["provenance_audit"].get("experiment_count"),
            "errors": records["provenance_audit"].get("errors", []),
            "warnings": records["provenance_audit"].get("warnings", []),
        } if records.get("provenance_audit") is not None else None,
        "temporal_calibration": _summary_for_calibration(records["temporal_calibration"])
        if records.get("temporal_calibration") is not None else None,
        "baseline_analysis": _summary_for_baseline(records["baseline_analysis"])
        if records.get("baseline_analysis") is not None else None,
        "regime_analysis": _summary_for_regimes(records["regime_analysis"])
        if records.get("regime_analysis") is not None else None,
        "temporal_stability": _summary_for_temporal_stability(records["temporal_stability"])
        if records.get("temporal_stability") is not None else None,
    }

    evidence_interpretation = build_evidence_interpretation(
        baseline=records.get("baseline_analysis"),
        calibration=records.get("temporal_calibration"),
        regimes=records.get("regime_analysis"),
        temporal_stability=records.get("temporal_stability"),
    )

    validation_status = sections["validation"].get("status")
    if validation_status == "FAIL":
        _add_issue(errors, "validation_report_failed")
    provenance_status = sections["provenance_audit"].get("status")
    if provenance_status == "FAIL" or provenance_status == "MISSING":
        _add_issue(errors, "provenance_audit_not_pass")
    elif provenance_status == "NOT_PROVIDED":
        _add_issue(warnings, "provenance_audit_not_recorded_in_manifest")
    temporal_status = sections["temporal_calibration"].get("status")
    if temporal_status == "FAIL" or temporal_status == "MISSING":
        _add_issue(errors, "temporal_calibration_not_pass")
    elif temporal_status == "NOT_PROVIDED":
        _add_issue(warnings, "temporal_calibration_not_recorded_in_manifest")

    available_sections = [name for name, section in sections.items() if section.get("status") in _ALLOWED_STATUSES]
    missing_optional_sections = [name for name in _OPTIONAL_REPORTS if sections[name].get("status") == "NOT_PROVIDED"]
    status = "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS"
    return {
        "schema_version": _SCHEMA_VERSION,
        "scope": _REPORT_SCOPE,
        "status": status,
        "run_id": run_id or None,
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon_bars": manifest.get("label_horizon_bars"),
        "manifest_path": str(manifest_file),
        "manifest_sha256": _sha256(manifest_file),
        "source_matrix_sha256": matrix_hash,
        "run_integrity": {
            "status": integrity.get("status"),
            "checked_artifact_count": integrity.get("checked_artifact_count"),
            "errors": integrity.get("errors", []),
            "warnings": integrity.get("warnings", []),
        },
        "evidence_coverage": {
            "available_sections": available_sections,
            "missing_optional_sections": missing_optional_sections,
            "available_optional_section_count": sum(
                sections[name].get("status") in _ALLOWED_STATUSES for name in _OPTIONAL_REPORTS
            ),
            "optional_section_count": len(_OPTIONAL_REPORTS),
        },
        "sections": sections,
        "summary": summary,
        "evidence_interpretation": evidence_interpretation,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "test_metrics_used_for_selection": False,
        "interpretation": (
            "Consolidated integrity and descriptive research evidence only. This report does not "
            "recompute model metrics, claim statistical significance or profitability, rank or select "
            "models, authorize deployment, or connect to broker execution. Missing optional analyses "
            "indicate incomplete evidence coverage, not a negative finding."
        ),
    }


def write_research_evidence_report(
    manifest_path: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the report and create it without replacing an existing artifact."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = build_research_evidence_report(manifest_path, **kwargs)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    report["report_path"] = str(destination.resolve())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate integrity, calibration, baseline, regime, and gap-sensitivity evidence for one research run."
    )
    parser.add_argument("--manifest", required=True, help="Path to <run-id>_pipeline_manifest.json")
    parser.add_argument("--output", help="Output JSON path; defaults beside the manifest")
    parser.add_argument("--validation-report", help="Validation report path if not using the standard run-scoped name")
    parser.add_argument("--temporal-calibration-audit", help="Calibration audit report path, useful for a legacy run without an embedded receipt")
    parser.add_argument("--baseline-analysis", help="Existing time-series baseline analysis report")
    parser.add_argument("--regime-analysis", help="Existing volatility-regime analysis report")
    parser.add_argument("--temporal-stability", help="Existing temporal stability / long-gap sensitivity report")
    args = parser.parse_args(argv)
    manifest_file = Path(args.manifest).resolve()
    try:
        manifest = _read_json(manifest_file)
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("pipeline manifest must contain a non-empty run_id")
        output = Path(args.output).resolve() if args.output else (
            manifest_file.parent / f"{run_id}_consolidated_evidence.json"
        )
        result = write_research_evidence_report(
            manifest_file,
            output,
            validation_report_path=args.validation_report,
            temporal_calibration_path=args.temporal_calibration_audit,
            baseline_analysis_path=args.baseline_analysis,
            regime_analysis_path=args.regime_analysis,
            temporal_stability_path=args.temporal_stability,
        )
    except FileExistsError as exc:
        print(f"research evidence report not written: refusing to overwrite existing file: {exc}", file=sys.stderr)
        return 2
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"research evidence report failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 3 if result.get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

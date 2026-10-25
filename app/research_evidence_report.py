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
from app.oos_period_robustness_audit import build_oos_period_robustness_audit

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



def _verify_matrix_payload_hash(path: Path) -> dict[str, Any]:
    """Verify the canonical self-hash embedded in an experiment-matrix report."""
    if not path.is_file():
        return {
            "status": "SOURCE_MATRIX_MISSING",
            "path": str(path),
            "declared_payload_sha256": None,
            "computed_payload_sha256": None,
        }
    try:
        payload = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "SOURCE_MATRIX_UNREADABLE",
            "path": str(path.resolve()),
            "declared_payload_sha256": None,
            "computed_payload_sha256": None,
            "error": type(exc).__name__,
        }

    declared = payload.get("report_payload_sha256")
    if not isinstance(declared, str) or not declared.strip():
        return {
            "status": "LEGACY_UNVERIFIED",
            "path": str(path.resolve()),
            "declared_payload_sha256": None,
            "computed_payload_sha256": None,
        }

    canonical_payload = dict(payload)
    canonical_payload.pop("report_payload_sha256", None)
    try:
        canonical = json.dumps(
            canonical_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        return {
            "status": "PAYLOAD_NOT_CANONICALIZABLE",
            "path": str(path.resolve()),
            "declared_payload_sha256": declared,
            "computed_payload_sha256": None,
        }
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "status": "VERIFIED" if computed == declared else "HASH_MISMATCH",
        "path": str(path.resolve()),
        "declared_payload_sha256": declared,
        "computed_payload_sha256": computed,
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
    identity_fields_present: list[str] = ["run_id"] if report.get("run_id") else []
    identity_fields_missing: list[str] = []
    if expected_symbol:
        if report.get("symbol") is None:
            identity_fields_missing.append("symbol")
            warnings.append(f"{section_name}_symbol_not_reported")
        else:
            identity_fields_present.append("symbol")
            if str(report.get("symbol")).strip().upper() != expected_symbol.upper():
                errors.append(f"{section_name}_symbol_mismatch")
    if expected_timeframe:
        if report.get("timeframe") is None:
            identity_fields_missing.append("timeframe")
            warnings.append(f"{section_name}_timeframe_not_reported")
        else:
            identity_fields_present.append("timeframe")
            if str(report.get("timeframe")).strip().upper() != expected_timeframe.upper():
                errors.append(f"{section_name}_timeframe_mismatch")

    reported_manifest = report.get("manifest_path")
    if reported_manifest:
        identity_fields_present.append("manifest_path")
        reported_path = Path(str(reported_manifest))
        if not reported_path.is_absolute():
            reported_path = (path.parent / reported_path).resolve()
        if reported_path.resolve() != expected_manifest.resolve():
            errors.append(f"{section_name}_manifest_path_mismatch")
    else:
        identity_fields_missing.append("manifest_path")
        warnings.append(f"{section_name}_manifest_path_not_reported")

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
    # Receipt diagnostics are part of the machine-readable contract. Do not
    # silently ignore malformed fields or non-string entries: doing so could
    # make a corrupt receipt look clean to downstream aggregation.
    diagnostic_schema_errors: list[str] = []
    for field_name, target in (("errors", errors), ("warnings", warnings)):
        if field_name not in report:
            continue
        values = report.get(field_name)
        if not isinstance(values, list):
            issue = f"{section_name}_{field_name}_must_be_list"
            _add_issue(errors, issue)
            diagnostic_schema_errors.append(issue)
            continue
        for index, item in enumerate(values):
            if not isinstance(item, str) or not item.strip():
                issue = f"{section_name}_{field_name}_item_{index}_invalid"
                _add_issue(errors, issue)
                diagnostic_schema_errors.append(issue)
                continue
            prefix = "reported_error" if field_name == "errors" else "reported_warning"
            _add_issue(target, f"{section_name}_{prefix}:{item}")

    count_specs = {
        "temporal_calibration": ("comparison_count", "comparisons"),
        "baseline_analysis": ("experiment_count", "experiments"),
        "regime_analysis": ("experiment_count", "experiments"),
        "temporal_stability": ("experiment_count", "experiments"),
    }
    declared_count_integrity: dict[str, Any] | None = None
    if section_name in count_specs:
        count_field, list_field = count_specs[section_name]
        declared_count = report.get(count_field)
        rows = report.get(list_field)
        if count_field in report and (
            isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count < 0
        ):
            errors.append(f"{section_name}_{count_field}_invalid")
            declared_count_integrity = {
                "status": "INVALID_DECLARED_COUNT",
                "declared_count": declared_count,
                "observed_count": len(rows) if isinstance(rows, list) else None,
            }
        elif isinstance(declared_count, int) and not isinstance(declared_count, bool):
            if isinstance(rows, list):
                if declared_count != len(rows):
                    errors.append(f"{section_name}_{count_field}_mismatch")
                    declared_count_integrity = {
                        "status": "MISMATCH",
                        "declared_count": declared_count,
                        "observed_count": len(rows),
                    }
                else:
                    declared_count_integrity = {
                        "status": "VERIFIED",
                        "declared_count": declared_count,
                        "observed_count": len(rows),
                    }
            else:
                errors.append(f"{section_name}_{list_field}_missing_for_declared_count")
                declared_count_integrity = {
                    "status": "UNVERIFIABLE",
                    "declared_count": declared_count,
                    "observed_count": None,
                }
        elif isinstance(rows, list):
            warnings.append(f"{section_name}_{count_field}_not_reported")
            declared_count_integrity = {
                "status": "DECLARED_COUNT_MISSING",
                "declared_count": None,
                "observed_count": len(rows),
            }

    record = {
        "status": status if status in _ALLOWED_STATUSES else "FAIL",
        "path": str(path.resolve()),
        "sha256": digest,
        "scope": report.get("scope"),
        "reported_status": status if status in _ALLOWED_STATUSES else None,
        "identity_metadata": {
            "status": "COMPLETE" if not identity_fields_missing else "PARTIAL",
            "fields_present": identity_fields_present,
            "fields_missing": identity_fields_missing,
        },
        "diagnostic_schema": {
            "status": "INVALID" if diagnostic_schema_errors else "VALID",
            "errors": diagnostic_schema_errors,
        },
    }
    if declared_count_integrity is not None:
        record["declared_count_integrity"] = declared_count_integrity
    if errors:
        record["status"] = "FAIL"
    return record, report, errors, warnings


def build_evidence_interpretation(
    *,
    baseline: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None,
    regimes: Mapping[str, Any] | None,
    temporal_stability: Mapping[str, Any] | None,
    oos_period_robustness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize cross-diagnostic patterns without ranking or promoting experiments."""
    baseline_counts = {
        "comparisons_evaluated": 0,
        "intervals_excluding_zero_positive": 0,
        "intervals_excluding_zero_negative": 0,
        "intervals_including_zero": 0,
        "intervals_unavailable": 0,
        "intervals_dependence_limited": 0,
        "positive_point_estimates": 0,
        "negative_point_estimates": 0,
        "zero_point_estimates": 0,
    }
    baseline_dependence_limited = bool(
        isinstance(baseline, dict)
        and isinstance(baseline.get("warnings"), list)
        and any(
            "held_out_timestamps_repeat_across_folds" in str(item)
            for item in baseline.get("warnings", [])
        )
    )
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
                    if baseline_dependence_limited:
                        # Keep the numeric interval in the source report, but do not classify
                        # it as evidence when the same held-out timestamps recur across folds.
                        baseline_counts["intervals_dependence_limited"] += 1
                        continue
                    low, high = float(interval[0]), float(interval[1])
                    baseline_counts["comparisons_evaluated"] += 1
                    if low > 0:
                        baseline_counts["intervals_excluding_zero_positive"] += 1
                    elif high < 0:
                        baseline_counts["intervals_excluding_zero_negative"] += 1
                    else:
                        baseline_counts["intervals_including_zero"] += 1
    if baseline_counts["comparisons_evaluated"] == 0 and baseline_counts["intervals_dependence_limited"] > 0:
        baseline_status = "DEPENDENCE_LIMITED_INTERVALS_NOT_CLASSIFIED"
    elif baseline_counts["intervals_dependence_limited"] > 0:
        baseline_status = "PARTIAL_DEPENDENCE_LIMITED_INTERVAL_EVIDENCE"
    elif baseline_counts["comparisons_evaluated"] == 0:
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
    regime_uncertainty_counts = {
        "intervals_evaluated": 0,
        "intervals_excluding_zero_positive": 0,
        "intervals_excluding_zero_negative": 0,
        "intervals_including_zero": 0,
        "intervals_unavailable": 0,
    }
    if isinstance(regimes, dict) and regimes.get("status") in {"PASS", "PASS_WITH_WARNINGS"}:
        experiments = regimes.get("experiments")
        for experiment in experiments if isinstance(experiments, list) else []:
            if not isinstance(experiment, dict):
                continue
            regime_metrics = experiment.get("regime_metrics")
            if not isinstance(regime_metrics, dict):
                continue
            by_baseline_metric: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for regime_name, regime in regime_metrics.items():
                if not isinstance(regime, dict):
                    continue
                improvements = regime.get("loss_improvements_baseline_minus_model")
                if not isinstance(improvements, dict):
                    continue
                uncertainty_by_baseline = regime.get("paired_bootstrap_uncertainty")
                for baseline_name, metric_values in improvements.items():
                    if not isinstance(metric_values, dict):
                        continue
                    baseline_uncertainty = (
                        uncertainty_by_baseline.get(baseline_name)
                        if isinstance(uncertainty_by_baseline, dict) else None
                    )
                    for metric_name in ("log_loss", "brier_score"):
                        value = _number(metric_values.get(metric_name))
                        if value is None:
                            continue
                        metric_uncertainty = (
                            baseline_uncertainty.get(metric_name)
                            if isinstance(baseline_uncertainty, dict) else None
                        )
                        by_baseline_metric.setdefault((str(baseline_name), metric_name), []).append({
                            "regime": str(regime_name),
                            "improvement": float(value),
                            "uncertainty": metric_uncertainty if isinstance(metric_uncertainty, dict) else None,
                        })
            for (baseline_name, metric_name), rows in sorted(by_baseline_metric.items()):
                positive = sum(row["improvement"] > 0 for row in rows)
                negative = sum(row["improvement"] < 0 for row in rows)
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

                group_uncertainty = {
                    "intervals_evaluated": 0,
                    "intervals_excluding_zero_positive": 0,
                    "intervals_excluding_zero_negative": 0,
                    "intervals_including_zero": 0,
                    "intervals_unavailable": 0,
                }
                regime_uncertainty_rows: list[dict[str, Any]] = []
                for row in rows:
                    uncertainty = row["uncertainty"]
                    interval = uncertainty.get("confidence_interval_95") if isinstance(uncertainty, dict) else None
                    valid_interval = (
                        isinstance(interval, list)
                        and len(interval) == 2
                        and _number(interval[0]) is not None
                        and _number(interval[1]) is not None
                        and float(interval[0]) <= float(interval[1])
                    )
                    evidence = "unavailable"
                    normalized_interval = None
                    if not valid_interval:
                        group_uncertainty["intervals_unavailable"] += 1
                        regime_uncertainty_counts["intervals_unavailable"] += 1
                    else:
                        low, high = float(interval[0]), float(interval[1])
                        normalized_interval = [low, high]
                        group_uncertainty["intervals_evaluated"] += 1
                        regime_uncertainty_counts["intervals_evaluated"] += 1
                        if low > 0:
                            evidence = "positive_interval_excludes_zero"
                            group_uncertainty["intervals_excluding_zero_positive"] += 1
                            regime_uncertainty_counts["intervals_excluding_zero_positive"] += 1
                        elif high < 0:
                            evidence = "negative_interval_excludes_zero"
                            group_uncertainty["intervals_excluding_zero_negative"] += 1
                            regime_uncertainty_counts["intervals_excluding_zero_negative"] += 1
                        else:
                            evidence = "interval_includes_zero"
                            group_uncertainty["intervals_including_zero"] += 1
                            regime_uncertainty_counts["intervals_including_zero"] += 1
                    regime_uncertainty_rows.append({
                        "regime": row["regime"],
                        "improvement_baseline_minus_model": row["improvement"],
                        "confidence_interval_95": normalized_interval,
                        "interval_evidence": evidence,
                        "bootstrap_resamples": (
                            uncertainty.get("bootstrap_resamples")
                            if isinstance(uncertainty, dict) else None
                        ),
                    })

                if group_uncertainty["intervals_evaluated"] == 0:
                    group_uncertainty_status = "UNCERTAINTY_UNAVAILABLE"
                elif (
                    group_uncertainty["intervals_excluding_zero_positive"]
                    and group_uncertainty["intervals_excluding_zero_negative"]
                ):
                    group_uncertainty_status = "MIXED_INTERVAL_EVIDENCE"
                elif group_uncertainty["intervals_excluding_zero_positive"]:
                    group_uncertainty_status = "POSITIVE_INTERVAL_EVIDENCE_PRESENT"
                elif group_uncertainty["intervals_excluding_zero_negative"]:
                    group_uncertainty_status = "NEGATIVE_INTERVAL_EVIDENCE_PRESENT"
                else:
                    group_uncertainty_status = "NO_REGIME_INTERVAL_EXCLUDES_ZERO"

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
                    "paired_bootstrap_uncertainty": {
                        **group_uncertainty,
                        "status": group_uncertainty_status,
                        "regimes": regime_uncertainty_rows,
                    },
                })
    mixed_regime_groups = sum(
        item["consistency"] == "MIXED_BY_VOLATILITY_REGIME" for item in regime_groups
    )
    if regime_uncertainty_counts["intervals_evaluated"] == 0:
        regime_uncertainty_status = "UNCERTAINTY_UNAVAILABLE"
    elif (
        regime_uncertainty_counts["intervals_excluding_zero_positive"]
        and regime_uncertainty_counts["intervals_excluding_zero_negative"]
    ):
        regime_uncertainty_status = "MIXED_INTERVAL_EVIDENCE"
    elif regime_uncertainty_counts["intervals_excluding_zero_positive"]:
        regime_uncertainty_status = "POSITIVE_INTERVAL_EVIDENCE_PRESENT"
    elif regime_uncertainty_counts["intervals_excluding_zero_negative"]:
        regime_uncertainty_status = "NEGATIVE_INTERVAL_EVIDENCE_PRESENT"
    else:
        regime_uncertainty_status = "NO_REGIME_INTERVAL_EXCLUDES_ZERO"
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

    robustness_rows: list[dict[str, Any]] = []
    if isinstance(oos_period_robustness, Mapping):
        raw_candidates = oos_period_robustness.get("candidates")
        for candidate in raw_candidates if isinstance(raw_candidates, list) else []:
            if not isinstance(candidate, Mapping):
                continue
            raw_periods = candidate.get("periods")
            periods = []
            for period in raw_periods if isinstance(raw_periods, list) else []:
                if not isinstance(period, Mapping):
                    continue
                periods.append({
                    "period": period.get("period"),
                    "fold_indices": period.get("fold_indices"),
                    "fold_count": period.get("fold_count"),
                    "median_test_brier_score": period.get("median_test_brier_score"),
                    "median_test_log_loss": period.get("median_test_log_loss"),
                    "median_test_expected_calibration_error": period.get(
                        "median_test_expected_calibration_error"
                    ),
                    "ece_fold_count": period.get("ece_fold_count"),
                })
            robustness_rows.append({
                "candidate_name": candidate.get("candidate_name"),
                "fold_count": candidate.get("fold_count"),
                "test_brier_range": candidate.get("test_brier_range"),
                "test_log_loss_range": candidate.get("test_log_loss_range"),
                "test_expected_calibration_error_range": candidate.get(
                    "test_expected_calibration_error_range"
                ),
                "expected_calibration_error_fold_count": candidate.get(
                    "expected_calibration_error_fold_count"
                ),
                "periods": periods,
            })
    if not isinstance(oos_period_robustness, Mapping):
        robustness_status = "UNAVAILABLE"
        robustness_fold_indices: list[int] = []
        robustness_candidate_count = len(robustness_rows)
        robustness_common_fold_index_count = 0
        robustness_complete_test_fold_count = 0
        robustness_candidate_coverage: list[dict[str, Any]] = []
    else:
        robustness_status = str(oos_period_robustness.get("status", "UNAVAILABLE"))
        raw_indices = oos_period_robustness.get("common_test_fold_indices")
        robustness_fold_indices = raw_indices if isinstance(raw_indices, list) else []
        raw_candidate_count = oos_period_robustness.get("candidate_count")
        robustness_candidate_count = (
            raw_candidate_count
            if isinstance(raw_candidate_count, int) and not isinstance(raw_candidate_count, bool)
            else len(robustness_rows)
        )
        robustness_common_fold_index_count = oos_period_robustness.get("common_fold_index_count", len(robustness_fold_indices))
        robustness_complete_test_fold_count = oos_period_robustness.get("complete_test_fold_count", len(robustness_fold_indices))
        raw_coverage = oos_period_robustness.get("candidate_fold_coverage")
        robustness_candidate_coverage = raw_coverage if isinstance(raw_coverage, list) else []

    # The report aggregates several diagnostic families and may contain many
    # unadjusted interval/period comparisons. Make that multiplicity visible
    # rather than allowing isolated interval exclusions to read as confirmation.
    baseline_interval_count = (
        baseline_counts["comparisons_evaluated"]
        + baseline_counts["intervals_dependence_limited"]
        + baseline_counts["intervals_unavailable"]
    )
    regime_interval_count = (
        regime_uncertainty_counts["intervals_evaluated"]
        + regime_uncertainty_counts["intervals_unavailable"]
    )
    multiplicity_families = {
        "baseline_probability_loss_intervals": {
            "comparison_count": baseline_interval_count,
            "intervals_classified": baseline_counts["comparisons_evaluated"],
            "intervals_dependence_limited": baseline_counts["intervals_dependence_limited"],
            "intervals_unavailable": baseline_counts["intervals_unavailable"],
            "adjustment_applied": False,
        },
        "temporal_calibration_test_period_changes": {
            "comparison_count": calibration_counts["comparisons_evaluated"]
            + calibration_counts["deltas_unavailable"],
            "deltas_available": calibration_counts["comparisons_evaluated"],
            "deltas_unavailable": calibration_counts["deltas_unavailable"],
            "adjustment_applied": False,
        },
        "volatility_regime_intervals": {
            "comparison_count": regime_interval_count,
            "comparison_groups": len(regime_groups),
            "adjustment_applied": False,
        },
        "chronological_oos_candidates": {
            "comparison_count": robustness_candidate_count,
            "period_group_count": (
                oos_period_robustness.get("period_group_count")
                if isinstance(oos_period_robustness, Mapping) else 0
            ),
            "adjustment_applied": False,
        },
    }
    family_comparison_counts = [
        row["comparison_count"] for row in multiplicity_families.values()
        if isinstance(row.get("comparison_count"), int) and row["comparison_count"] > 0
    ]
    total_diagnostic_comparisons = sum(family_comparison_counts)
    multiplicity_status = (
        "MULTIPLE_DIAGNOSTIC_COMPARISONS_UNADJUSTED"
        if total_diagnostic_comparisons > 1
        else "SINGLE_OR_NO_DIAGNOSTIC_COMPARISON"
    )

    return {
        "chronological_oos_robustness": {
            "status": robustness_status,
            "candidate_count": robustness_candidate_count,
            "summary_candidate_count": len(robustness_rows),
            "candidate_fold_coverage": robustness_candidate_coverage,
            "common_fold_index_count": robustness_common_fold_index_count,
            "complete_test_fold_count": robustness_complete_test_fold_count,
            "common_test_fold_indices": robustness_fold_indices,
            "period_group_count": (
                oos_period_robustness.get("period_group_count")
                if isinstance(oos_period_robustness, Mapping) else 0
            ),
            "chronology_integrity": (
                oos_period_robustness.get("chronology_integrity")
                if isinstance(oos_period_robustness, Mapping) else None
            ),
            "candidates": robustness_rows,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "interpretation": (
                "Early/middle/late summaries are derived from stored chronological walk-forward test-fold metrics. "
                "They are descriptive diagnostics only, not independent samples, a model ranking, tuning evidence, "
                "statistical significance, or profitability. Reported ECE summaries help locate calibration drift "
                "but do not fit or apply a calibration transform."
            ),
        },
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
            "paired_bootstrap_uncertainty": {
                "status": regime_uncertainty_status,
                **regime_uncertainty_counts,
                "interpretation": (
                    "Summarizes precomputed conditional paired moving-block bootstrap intervals only; positive "
                    "improvement favors the model, but an interval excluding zero is not a profitability test. "
                    "Intervals that include zero do not show clear conditional separation. This report does not "
                    "recompute bootstrap samples, and conditional observations/folds may remain dependent."
                ),
            },
            "interpretation": (
                "Positive baseline-minus-model loss improvement favors the model. Mixed signs across regimes "
                "indicate conditional heterogeneity; regime summaries are descriptive and may have dependent samples."
            ),
        },
        "label_gap_exposure": gap_summary,
        "multiplicity_context": {
            "status": multiplicity_status,
            "diagnostic_family_count": sum(count > 0 for count in family_comparison_counts),
            "total_diagnostic_comparison_count": total_diagnostic_comparisons,
            "families": multiplicity_families,
            "adjustment_applied": False,
            "confirmatory_inference_supported": False,
            "interpretation": (
                "Counts are an audit inventory across distinct descriptive diagnostic families, not a single "
                "statistical testing family. No multiplicity correction is applied. Confidence intervals and "
                "period/regime summaries are exploratory and must not be treated as confirmatory significance "
                "claims; an appropriate pre-specified testing family and method would be needed for that."
            ),
        },
        "research_conclusion": "DESCRIPTIVE_ONLY_NO_MODEL_SELECTED",
        "promotion_readiness": "NOT_ESTABLISHED_BY_THIS_REPORT",
        "automatic_model_selection": False,
        "test_metrics_used_for_selection": False,
        "actionable": False,
        "execution_connected": False,
    }


def _derive_oos_period_robustness(
    matrix_path: Path,
    *,
    expected_run_id: str,
    expected_matrix_sha256: str | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Derive a chronological period diagnostic from the hash-bound saved matrix."""
    errors: list[str] = []
    warnings: list[str] = []
    if not matrix_path.is_file():
        return ({
            "status": "SOURCE_MATRIX_UNAVAILABLE",
            "candidate_count": 0,
            "common_test_fold_indices": [],
            "period_group_count": 0,
            "candidates": [],
            "model_selection_performed": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "interpretation": "The source experiment matrix is not available; no period diagnostic was derived.",
        }, ["oos_period_robustness_source_matrix_missing"], warnings)
    digest = _sha256(matrix_path)
    if expected_matrix_sha256 and digest != expected_matrix_sha256:
        return ({
            "status": "SOURCE_MATRIX_HASH_MISMATCH",
            "source_matrix_path": str(matrix_path.resolve()),
            "source_matrix_sha256": digest,
            "candidate_count": 0,
            "common_test_fold_indices": [],
            "period_group_count": 0,
            "candidates": [],
            "model_selection_performed": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "interpretation": "Matrix fingerprint did not match the manifest; no period diagnostic was derived.",
        }, ["oos_period_robustness_source_matrix_hash_mismatch"], warnings)
    try:
        matrix = _read_json(matrix_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return ({
            "status": "SOURCE_MATRIX_UNREADABLE",
            "source_matrix_path": str(matrix_path.resolve()),
            "source_matrix_sha256": digest,
            "candidate_count": 0,
            "common_test_fold_indices": [],
            "period_group_count": 0,
            "candidates": [],
            "model_selection_performed": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "interpretation": "The source experiment matrix could not be parsed; no period diagnostic was derived.",
        }, [f"oos_period_robustness_source_matrix_unreadable:{type(exc).__name__}"], warnings)

    if matrix.get("run_id") != expected_run_id:
        return ({
            "status": "SOURCE_MATRIX_RUN_ID_MISMATCH",
            "source_matrix_path": str(matrix_path.resolve()),
            "source_matrix_sha256": digest,
            "candidate_count": 0,
            "common_test_fold_indices": [],
            "period_group_count": 0,
            "candidates": [],
            "model_selection_performed": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "interpretation": "The source experiment matrix belongs to another run; no period diagnostic was derived.",
        }, ["oos_period_robustness_source_matrix_run_id_mismatch"], warnings)

    experiments = matrix.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        return ({
            "status": "SOURCE_MATRIX_STRUCTURE_UNAVAILABLE",
            "source_matrix_path": str(matrix_path.resolve()),
            "source_matrix_sha256": digest,
            "candidate_count": 0,
            "common_test_fold_indices": [],
            "period_group_count": 0,
            "candidates": [],
            "model_selection_performed": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "interpretation": "The source matrix has no experiment list; no period diagnostic was derived.",
        }, ["oos_period_robustness_source_matrix_experiments_invalid"], warnings)

    fold_results: dict[str, list[Mapping[str, Any]]] = {}
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            errors.append(f"oos_period_robustness_experiment_{index}_invalid")
            continue
        experiment_id = experiment.get("experiment_id")
        folds = experiment.get("fold_results")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            errors.append(f"oos_period_robustness_experiment_{index}_id_missing")
            continue
        if experiment_id in fold_results:
            errors.append(f"oos_period_robustness_experiment_id_duplicate:{experiment_id}")
            continue
        if not isinstance(folds, list) or not folds:
            errors.append(f"oos_period_robustness_fold_results_missing:{experiment_id}")
            continue
        fold_results[experiment_id] = folds

    # Some legacy matrix writers stored fold starts only in this matrix-level table.
    # Join those immutable boundaries by fold_index so the chronology audit can verify
    # ordering without altering metric values or inventing timestamps.
    raw_windows = matrix.get("common_fold_test_windows")
    window_starts: dict[int, object] = {}
    if isinstance(raw_windows, list):
        for window in raw_windows:
            if not isinstance(window, dict):
                continue
            index = window.get("fold_index")
            start = window.get("test_start_utc", window.get("test_start"))
            if (
                isinstance(index, int)
                and not isinstance(index, bool)
                and index >= 0
                and isinstance(start, str)
                and start.strip()
            ):
                if index in window_starts and window_starts[index] != start:
                    errors.append(
                        f"oos_period_robustness_conflicting_common_fold_window:{index}"
                    )
                else:
                    window_starts[index] = start

    for folds in fold_results.values():
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            if fold.get("test_start_utc") is not None or fold.get("test_start") is not None:
                continue
            index = fold.get("fold_index")
            if isinstance(index, int) and not isinstance(index, bool):
                start = window_starts.get(index)
                if start is not None:
                    fold["test_start_utc"] = start

    if errors or not fold_results:
        return ({
            "status": "SOURCE_MATRIX_STRUCTURE_INVALID",
            "source_matrix_path": str(matrix_path.resolve()),
            "source_matrix_sha256": digest,
            "candidate_count": len(fold_results),
            "common_test_fold_indices": [],
            "period_group_count": 0,
            "candidates": [],
            "model_selection_performed": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "interpretation": "The source matrix structure is incomplete; no period diagnostic was derived.",
        }, errors or ["oos_period_robustness_no_valid_experiments"], warnings)

    audit = build_oos_period_robustness_audit(fold_results, [])
    if audit.get("status") == "CHRONOLOGY_INVALID":
        warnings.append("oos_period_robustness_chronology_invalid_period_summaries_withheld")
    audit["source_matrix_path"] = str(matrix_path.resolve())
    audit["source_matrix_sha256"] = digest
    audit["source_run_id"] = expected_run_id
    return audit, errors, warnings



def _check_manifest_receipt_consistency(
    *,
    section_name: str,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any] | None,
    errors: list[str],
) -> None:
    """Ensure manifest summaries of hash-bound receipts match the receipt itself.

    This checks the audit trail's internal consistency; it does not replace the
    artifact hash, scope, run, or source-matrix provenance checks.
    """
    if receipt is None:
        return

    status_field = f"{section_name}_status"
    if status_field in manifest and manifest.get(status_field) != receipt.get("status"):
        _add_issue(errors, f"{section_name}_manifest_status_mismatch")

    for suffix in ("errors", "warnings"):
        manifest_field = f"{section_name}_{suffix}"
        if manifest_field not in manifest:
            continue
        declared = manifest.get(manifest_field)
        actual = receipt.get(suffix, [])
        if not isinstance(declared, list) or not isinstance(actual, list):
            _add_issue(errors, f"{section_name}_manifest_{suffix}_mismatch")
            continue
        if declared != actual:
            _add_issue(errors, f"{section_name}_manifest_{suffix}_mismatch")

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
    matrix_path = _resolve_path(manifest.get("experiment_matrix_report"), manifest_file.parent)
    if matrix_path is None:
        matrix_path = manifest_file.parent / f"{run_id}_experiment_matrix.json"
    matrix_payload_integrity = _verify_matrix_payload_hash(matrix_path)
    if matrix_payload_integrity["status"] == "HASH_MISMATCH":
        errors.append("experiment_matrix_payload_hash_mismatch")
        oos_period_robustness = {
            "status": "SOURCE_MATRIX_PAYLOAD_HASH_MISMATCH",
            "source_matrix_path": str(matrix_path.resolve()),
            "candidate_count": 0,
            "common_test_fold_indices": [],
            "period_group_count": 0,
            "candidates": [],
            "model_selection_performed": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "interpretation": "The embedded canonical matrix payload hash did not match; period summaries were withheld.",
        }
        oos_errors, oos_warnings = [], []
    else:
        if matrix_payload_integrity["status"] == "LEGACY_UNVERIFIED":
            warnings.append("experiment_matrix_payload_hash_not_available_for_legacy_artifact")
        elif matrix_payload_integrity["status"] != "VERIFIED":
            errors.append(
                "experiment_matrix_payload_integrity_" + str(matrix_payload_integrity["status"]).lower()
            )
        oos_period_robustness, oos_errors, oos_warnings = _derive_oos_period_robustness(
            matrix_path,
            expected_run_id=run_id,
            expected_matrix_sha256=matrix_hash if isinstance(matrix_hash, str) else None,
        )
        errors.extend(oos_errors)
        warnings.extend(oos_warnings)
    sections: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any] | None] = {}

    # These two receipts are explicitly referenced and hash-bound by the pipeline manifest.
    for name, path_field, hash_field, scope in (
        ("provenance_audit", "provenance_audit_path", "provenance_audit_sha256", None),
        (
            "model_evaluation_gate",
            "model_evaluation_gate_path",
            "model_evaluation_gate_sha256",
            "GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE",
        ),
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


    # The manifest repeats summary fields for these hash-bound receipts. Check
    # those copies against the actual receipt so stale status/error metadata
    # cannot contradict the artifact used by this consolidated report.
    for receipt_name in ("provenance_audit", "model_evaluation_gate"):
        _check_manifest_receipt_consistency(
            section_name=receipt_name,
            manifest=manifest,
            receipt=records.get(receipt_name),
            errors=errors,
        )


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


    declared_temporal_path = _resolve_path(
        manifest.get("temporal_calibration_audit_path"), manifest_file.parent
    )
    if declared_temporal_path is not None and temporal_path == declared_temporal_path:
        _check_manifest_receipt_consistency(
            section_name="temporal_calibration_audit",
            manifest=manifest,
            receipt=records.get("temporal_calibration"),
            errors=errors,
        )


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
        "validation": {
            **_summary_for_validation(records["validation"]),
            "effective_status": sections["validation"].get("status"),
            "reported_status": records["validation"].get("status"),
        } if records.get("validation") is not None else None,
        "provenance_audit": {
            "status": sections["provenance_audit"].get("status"),
            "reported_status": records["provenance_audit"].get("status"),
            "experiment_count": records["provenance_audit"].get("experiment_count"),
            "errors": records["provenance_audit"].get("errors", []),
            "warnings": records["provenance_audit"].get("warnings", []),
        } if records.get("provenance_audit") is not None else None,
        "model_evaluation_gate": {
            "status": sections["model_evaluation_gate"].get("status"),
            "reported_status": records["model_evaluation_gate"].get("status"),
            "experiment_count": records["model_evaluation_gate"].get("experiment_count"),
            "folds_checked": records["model_evaluation_gate"].get("folds_checked"),
            "fold_metric_objects_checked": records["model_evaluation_gate"].get("fold_metric_objects_checked"),
            "errors": records["model_evaluation_gate"].get("errors", []),
            "warnings": records["model_evaluation_gate"].get("warnings", []),
            "actionable": False,
        } if records.get("model_evaluation_gate") is not None else None,
        "temporal_calibration": {
            **_summary_for_calibration(records["temporal_calibration"]),
            "effective_status": sections["temporal_calibration"].get("status"),
            "reported_status": records["temporal_calibration"].get("status"),
        } if records.get("temporal_calibration") is not None else None,
        "baseline_analysis": {
            **_summary_for_baseline(records["baseline_analysis"]),
            "effective_status": sections["baseline_analysis"].get("status"),
            "reported_status": records["baseline_analysis"].get("status"),
        } if records.get("baseline_analysis") is not None else None,
        "regime_analysis": {
            **_summary_for_regimes(records["regime_analysis"]),
            "effective_status": sections["regime_analysis"].get("status"),
            "reported_status": records["regime_analysis"].get("status"),
        } if records.get("regime_analysis") is not None else None,
        "temporal_stability": {
            **_summary_for_temporal_stability(records["temporal_stability"]),
            "effective_status": sections["temporal_stability"].get("status"),
            "reported_status": records["temporal_stability"].get("status"),
        } if records.get("temporal_stability") is not None else None,
        "oos_period_robustness": oos_period_robustness,
    }

    # Expose effective (integrity-adjusted) section statuses separately from
    # each receipt's self-reported status, so a hash/provenance failure cannot
    # leave downstream consumers reading a stale PASS in the summary.
    summary["section_statuses"] = {
        name: section.get("status")
        for name, section in sections.items()
    }

    evidence_interpretation = build_evidence_interpretation(
        baseline=records.get("baseline_analysis"),
        calibration=records.get("temporal_calibration"),
        regimes=records.get("regime_analysis"),
        temporal_stability=records.get("temporal_stability"),
        oos_period_robustness=oos_period_robustness,
    )

    validation_status = sections["validation"].get("status")
    if validation_status == "FAIL":
        _add_issue(errors, "validation_report_failed")
    evaluation_gate_status = sections["model_evaluation_gate"].get("status")
    if evaluation_gate_status == "FAIL" or evaluation_gate_status == "MISSING":
        _add_issue(errors, "model_evaluation_gate_not_pass")
    elif evaluation_gate_status == "NOT_PROVIDED":
        _add_issue(warnings, "model_evaluation_gate_not_recorded_in_legacy_manifest")
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
    section_status_counts: dict[str, int] = {}
    identity_complete_count = 0
    identity_partial_count = 0
    declared_count_mismatch_count = 0
    diagnostic_schema_invalid_count = 0
    effective_status_mismatch_count = 0
    for section in sections.values():
        section_status = str(section.get("status", "UNKNOWN"))
        section_status_counts[section_status] = section_status_counts.get(section_status, 0) + 1
        identity_metadata = section.get("identity_metadata")
        if isinstance(identity_metadata, dict):
            if identity_metadata.get("status") == "COMPLETE":
                identity_complete_count += 1
            elif identity_metadata.get("status") == "PARTIAL":
                identity_partial_count += 1
        count_integrity = section.get("declared_count_integrity")
        if isinstance(count_integrity, dict) and count_integrity.get("status") == "MISMATCH":
            declared_count_mismatch_count += 1
        diagnostic_schema = section.get("diagnostic_schema")
        if isinstance(diagnostic_schema, dict) and diagnostic_schema.get("status") == "INVALID":
            diagnostic_schema_invalid_count += 1
        reported_status = section.get("reported_status")
        if reported_status is not None and reported_status != section.get("status"):
            effective_status_mismatch_count += 1

    integrity_summary = {
        "artifact_section_count": len(sections),
        "section_status_counts": section_status_counts,
        "identity_metadata_complete_section_count": identity_complete_count,
        "identity_metadata_partial_section_count": identity_partial_count,
        "declared_count_mismatch_section_count": declared_count_mismatch_count,
        "diagnostic_schema_invalid_section_count": diagnostic_schema_invalid_count,
        "effective_status_mismatch_section_count": effective_status_mismatch_count,
        "run_integrity_status": integrity.get("status"),
        "matrix_payload_integrity_status": matrix_payload_integrity.get("status"),
        "has_integrity_errors": bool(errors),
        "has_warnings": bool(warnings),
        "interpretation": (
            "Aggregated integrity inventory only. Section counts and metadata completeness do not "
            "establish model quality, leakage absence, statistical significance, or profitability."
        ),
    }
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
        "matrix_payload_integrity": matrix_payload_integrity,
        "integrity_summary": integrity_summary,
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
            "recompute model predictions or rank or select models, claim statistical significance or profitability, "
            "authorize deployment, or connect to broker execution. The chronological period diagnostic summarizes "
            "stored per-fold test metrics only. Missing optional analyses "
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
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Return exit code 4 when the report has warnings, while still writing the report.",
    )
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
    if result.get("status") == "FAIL":
        return 3
    if args.strict_warnings and result.get("warnings"):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

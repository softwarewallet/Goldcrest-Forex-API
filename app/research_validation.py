"""End-to-end validation and consolidated reporting for offline research runs.

This module validates completed artifacts and summarizes their descriptive test metrics.
It never selects a model, promotes signals, contacts a broker, or enables execution.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.research_run_audit import audit_research_run_manifest
from app.runtime_diagnostics import collect_runtime_diagnostics

_CONTRACT_COLUMNS = {
    "symbol", "timeframe", "timestamp", "open", "high", "low", "close",
    "label", "label_available_at", "label_elapsed_hours", "label_eligible",
}
_METRIC_KEYS = (
    "brier_score", "log_loss", "accuracy_at_fixed_half",
    "expected_calibration_error", "prevalence", "mean_predicted_probability",
)
_REQUIRED_RESEARCH_MODULES = ("NumPy", "scikit-learn", "SciPy", "XGBoost", "LightGBM", "Optuna")


def _parse_utc(value: str, field: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _resolve_artifact(value: object, manifest_path: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        relative = (manifest_path.parent / candidate).resolve()
        if relative.is_file():
            return relative
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def validate_prepared_dataset(
    dataset_path: str | Path,
    *,
    expected_symbol: str | None = None,
    expected_timeframe: str | None = None,
    max_label_elapsed_hours: float | None = None,
) -> dict[str, Any]:
    """Check prepared rows without sorting, imputing, dropping, or changing the data."""
    source = Path(dataset_path)
    errors: dict[str, int] = {}
    warnings: list[str] = []
    row_count = 0
    feature_names: list[str] = []
    labels: list[int] = []
    timestamps: list[datetime] = []
    symbols: set[str] = set()
    timeframes: set[str] = set()
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    invalid_ohlc_rows = 0
    invalid_feature_rows = 0
    invalid_label_rows = 0
    missing_feature_cells = 0
    label_elapsed_metadata_count = 0
    eligible_label_rows = 0
    ineligible_label_rows = 0

    def fail(code: str) -> None:
        errors[code] = errors.get(code, 0) + 1

    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            required_contract_columns = _CONTRACT_COLUMNS - {"label_elapsed_hours", "label_eligible"}
            missing = required_contract_columns - set(columns)
            if missing:
                return {
                    "status": "FAIL", "path": str(source), "row_count": 0,
                    "errors": [f"missing_columns:{name}" for name in sorted(missing)],
                    "warnings": [], "actionable": False,
                }
            feature_names = [name for name in columns if name not in _CONTRACT_COLUMNS]
            if not feature_names:
                fail("no_feature_columns")
            has_elapsed = "label_elapsed_hours" in columns
            has_eligibility = "label_eligible" in columns
            if has_elapsed != has_eligibility:
                fail("label_eligibility_metadata_incomplete")
            if max_label_elapsed_hours is not None and not (has_elapsed and has_eligibility):
                fail("label_eligibility_metadata_missing_for_elapsed_cap")
            previous_timestamp: datetime | None = None
            seen_timestamps: set[datetime] = set()
            previous_label_available_timestamp: datetime | None = None
            seen_label_available_timestamps: set[datetime] = set()
            for row in reader:
                row_count += 1
                symbol = (row.get("symbol") or "").strip().upper()
                timeframe = (row.get("timeframe") or "").strip().upper()
                symbols.add(symbol)
                timeframes.add(timeframe)
                try:
                    ts = _parse_utc(row.get("timestamp") or "", "timestamp")
                    available = _parse_utc(row.get("label_available_at") or "", "label_available_at")
                    # A label cannot be known at or before its feature timestamp.
                    # Equality is invalid too: it would imply a zero-duration/future-outcome label.
                    if available <= ts:
                        fail("label_available_not_after_feature")
                    if available in seen_label_available_timestamps:
                        fail("duplicate_label_available_timestamp")
                    if (
                        previous_label_available_timestamp is not None
                        and available <= previous_label_available_timestamp
                    ):
                        fail("label_available_timestamps_not_strictly_chronological")
                    seen_label_available_timestamps.add(available)
                    previous_label_available_timestamp = available
                    if has_elapsed and has_eligibility:
                        label_elapsed_metadata_count += 1
                        try:
                            declared_elapsed = float(row.get("label_elapsed_hours") or "")
                            actual_elapsed = (available - ts).total_seconds() / 3600.0
                            if (
                                not math.isfinite(declared_elapsed)
                                or declared_elapsed <= 0
                                or abs(declared_elapsed - actual_elapsed) > 1e-5
                            ):
                                raise ValueError
                            eligibility_text = (row.get("label_eligible") or "").strip()
                            if eligibility_text not in {"0", "1"}:
                                raise ValueError
                            is_eligible = eligibility_text == "1"
                            if max_label_elapsed_hours is not None and (
                                is_eligible != (declared_elapsed <= max_label_elapsed_hours + 1e-9)
                            ):
                                fail("label_eligibility_cap_mismatch")
                            eligible_label_rows += int(is_eligible)
                            ineligible_label_rows += int(not is_eligible)
                        except (TypeError, ValueError, OverflowError):
                            fail("invalid_label_eligibility_metadata")
                    if ts in seen_timestamps:
                        fail("duplicate_timestamp")
                    if previous_timestamp is not None and ts <= previous_timestamp:
                        fail("timestamps_not_strictly_chronological")
                    seen_timestamps.add(ts)
                    previous_timestamp = ts
                    timestamps.append(ts)
                    if first_timestamp is None:
                        first_timestamp = ts.isoformat().replace("+00:00", "Z")
                    last_timestamp = ts.isoformat().replace("+00:00", "Z")
                except ValueError:
                    fail("invalid_timestamp")
                    continue

                if expected_symbol and symbol != expected_symbol.strip().upper():
                    fail("symbol_mismatch")
                if expected_timeframe and timeframe != expected_timeframe.strip().upper():
                    fail("timeframe_mismatch")
                try:
                    ohlc = {key: float(row.get(key) or "") for key in ("open", "high", "low", "close")}
                    if any(not math.isfinite(value) or value <= 0 for value in ohlc.values()):
                        raise ValueError
                    if (
                        ohlc["low"] > min(ohlc["open"], ohlc["close"])
                        or ohlc["high"] < max(ohlc["open"], ohlc["close"])
                        or ohlc["high"] < ohlc["low"]
                    ):
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    invalid_ohlc_rows += 1
                    fail("invalid_ohlc")
                label_text = (row.get("label") or "").strip()
                if label_text not in {"0", "1"}:
                    invalid_label_rows += 1
                    fail("invalid_label")
                else:
                    labels.append(int(label_text))
                for name in feature_names:
                    raw = row.get(name)
                    if raw is None or not raw.strip():
                        missing_feature_cells += 1
                        invalid_feature_rows += 1
                        continue
                    try:
                        value = float(raw)
                        if not math.isfinite(value):
                            raise ValueError
                    except (TypeError, ValueError, OverflowError):
                        invalid_feature_rows += 1
                if invalid_feature_rows and invalid_feature_rows > 0:
                    # Count failures per cell above; do not mutate or quarantine source rows.
                    pass
    except OSError as exc:
        return {
            "status": "FAIL", "path": str(source), "row_count": 0,
            "errors": [f"dataset_unreadable:{type(exc).__name__}"],
            "warnings": [], "actionable": False,
        }

    if row_count == 0:
        fail("empty_dataset")
    if len(symbols) > 1 or "" in symbols:
        fail("dataset_must_contain_one_nonempty_symbol")
    if len(timeframes) > 1 or "" in timeframes:
        fail("dataset_must_contain_one_nonempty_timeframe")
    if invalid_ohlc_rows:
        warnings.append("OHLC validation failed for one or more rows; details are in errors.")
    if invalid_feature_rows:
        fail("invalid_or_missing_features")
    if label_elapsed_metadata_count and label_elapsed_metadata_count != row_count:
        fail("label_eligibility_metadata_row_count_mismatch")
    if max_label_elapsed_hours is not None and row_count and label_elapsed_metadata_count != row_count:
        fail("label_eligibility_metadata_row_count_mismatch")
    if not label_elapsed_metadata_count:
        eligible_label_rows = row_count
        ineligible_label_rows = 0
    label_count = len(labels)
    positive = sum(labels)
    negative = label_count - positive
    prevalence = (positive / label_count) if label_count else None
    if prevalence is not None and (prevalence < 0.10 or prevalence > 0.90):
        warnings.append("label distribution is highly imbalanced; compare against suitable baselines.")
    if row_count and timestamps and len(timestamps) != row_count:
        fail("some_rows_have_invalid_timestamps")

    return {
        "status": "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS",
        "path": str(source.resolve()),
        "row_count": row_count,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "symbols": sorted(symbols),
        "timeframes": sorted(timeframes),
        "first_timestamp_utc": first_timestamp,
        "last_timestamp_utc": last_timestamp,
        "label_distribution": {
            "valid_label_count": label_count,
            "positive_count": positive,
            "negative_count": negative,
            "positive_prevalence": prevalence,
        },
        "label_eligibility": {
            "mode": "elapsed_time_cap" if label_elapsed_metadata_count else "all_labels_included",
            "max_label_elapsed_hours": max_label_elapsed_hours,
            "eligible_row_count": eligible_label_rows,
            "ineligible_row_count": ineligible_label_rows,
            "metadata_row_count": label_elapsed_metadata_count,
        },
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "invalid_feature_cells_or_rows": invalid_feature_rows,
        "missing_feature_cells": missing_feature_cells,
        "invalid_label_rows": invalid_label_rows,
        "errors": [f"{key}:{count}" for key, count in sorted(errors.items())],
        "warnings": warnings,
        "actionable": False,
    }


def summarize_experiment_matrix(report_path: str | Path) -> dict[str, Any]:
    """Validate basic experiment-report structure and summarize fold-level test metrics.

    Aggregates are descriptive only. No candidate is ranked or selected; overlapping
    test windows are explicitly flagged because their observations are not independent.
    """
    source = Path(report_path)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        report = _read_json(source)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "FAIL", "path": str(source),
            "errors": [f"experiment_report_unreadable:{type(exc).__name__}"],
            "warnings": [], "actionable": False,
        }

    for field in ("actionable", "automatic_winner_selection", "test_metrics_used_for_selection"):
        if report.get(field) is not False:
            errors.append(f"{field}_must_be_false")
    experiments = report.get("experiments")
    windows = report.get("common_fold_test_windows")
    if not isinstance(experiments, list) or not experiments:
        errors.append("experiments_must_be_nonempty_list")
        experiments = []
    if not isinstance(windows, list) or not windows:
        errors.append("common_fold_test_windows_must_be_nonempty_list")
        windows = []
    if isinstance(report.get("fold_count"), bool) or not isinstance(report.get("fold_count"), int) or report.get("fold_count", 0) < 1:
        errors.append("invalid_fold_count")

    parsed_windows: list[tuple[datetime, datetime]] = []
    window_rows_by_index: dict[int, int] = {}
    for index, window in enumerate(windows):
        try:
            start = _parse_utc(str(window["test_start_utc"]), "test_start_utc")
            end = _parse_utc(str(window["test_end_utc"]), "test_end_utc")
            if end < start:
                raise ValueError("test end precedes start")
            parsed_windows.append((start, end))
            fold_index = window.get("fold_index", index)
            test_rows = window.get("test_rows")
            if isinstance(fold_index, bool) or not isinstance(fold_index, int):
                raise ValueError("fold index must be an integer")
            if isinstance(test_rows, bool) or not isinstance(test_rows, int) or test_rows < 1:
                errors.append(f"invalid_common_test_window_row_count_{index}")
            else:
                window_rows_by_index[fold_index] = test_rows
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid_common_test_window_{index}")

    sorted_windows = sorted(parsed_windows)
    overlapping_windows = 0
    furthest_end: datetime | None = None
    for start, end in sorted_windows:
        if furthest_end is not None and start <= furthest_end:
            overlapping_windows += 1
        if furthest_end is None or end > furthest_end:
            furthest_end = end
    if overlapping_windows:
        warnings.append(
            f"{overlapping_windows} test window(s) overlap; fold metrics should not be pooled as independent observations."
        )

    experiment_summaries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for exp_index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            errors.append(f"experiment_{exp_index}_must_be_object")
            continue
        experiment_id = experiment.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            errors.append(f"experiment_{exp_index}_missing_id")
        elif experiment_id in seen_ids:
            errors.append(f"experiment_{exp_index}_duplicate_id")
        else:
            seen_ids.add(experiment_id)
        if experiment.get("actionable") is not False:
            errors.append(f"experiment_{exp_index}_actionable_must_be_false")
        if experiment.get("test_metrics_used_for_tuning") is not False:
            errors.append(f"experiment_{exp_index}_test_metrics_used_for_tuning_must_be_false")
        folds = experiment.get("fold_results")
        if not isinstance(folds, list) or not folds:
            errors.append(f"experiment_{exp_index}_fold_results_must_be_nonempty")
            continue
        if len(folds) != report.get("fold_count"):
            errors.append(f"experiment_{exp_index}_fold_count_mismatch")

        metric_values: dict[str, list[float]] = {key: [] for key in _METRIC_KEYS}
        total_test_rows = 0
        for fold_index, fold in enumerate(folds):
            if not isinstance(fold, dict):
                errors.append(f"experiment_{exp_index}_fold_{fold_index}_must_be_object")
                continue
            try:
                train_end = _parse_utc(str(fold["train_end_utc"]), "train_end_utc")
                validation_start = _parse_utc(str(fold["validation_start_utc"]), "validation_start_utc")
                validation_end = _parse_utc(str(fold["validation_end_utc"]), "validation_end_utc")
                test_start = _parse_utc(str(fold["test_start_utc"]), "test_start_utc")
                test_end = _parse_utc(str(fold["test_end_utc"]), "test_end_utc")
                if not train_end < validation_start < validation_end < test_start <= test_end:
                    errors.append(f"experiment_{exp_index}_fold_{fold_index}_nonchronological_boundaries")
            except (KeyError, TypeError, ValueError):
                errors.append(f"experiment_{exp_index}_fold_{fold_index}_invalid_boundaries")
            if fold.get("test_metrics_used_for_tuning") is True:
                errors.append(f"experiment_{exp_index}_fold_{fold_index}_test_leakage_flagged")
            metrics = fold.get("test_metrics")
            if not isinstance(metrics, dict):
                errors.append(f"experiment_{exp_index}_fold_{fold_index}_missing_test_metrics")
                continue
            sample_count = metrics.get("sample_count")
            positive = metrics.get("positive_count")
            negative = metrics.get("negative_count")
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (sample_count, positive, negative)):
                errors.append(f"experiment_{exp_index}_fold_{fold_index}_invalid_class_counts")
                continue
            if sample_count < 1 or positive < 0 or negative < 0 or positive + negative != sample_count:
                errors.append(f"experiment_{exp_index}_fold_{fold_index}_inconsistent_class_counts")
                continue
            fold_identity = fold.get("fold_index", fold_index)
            expected_rows = window_rows_by_index.get(fold_identity)
            if expected_rows is None:
                errors.append(f"experiment_{exp_index}_fold_{fold_index}_missing_common_test_window")
            elif report.get("label_eligibility_mode", "all_labels_included") == "elapsed_time_cap":
                available_before_filter = fold.get("test_rows_available_before_eligibility")
                excluded_by_filter = fold.get("test_rows_excluded_ineligible")
                if (
                    isinstance(available_before_filter, bool)
                    or not isinstance(available_before_filter, int)
                    or isinstance(excluded_by_filter, bool)
                    or not isinstance(excluded_by_filter, int)
                    or available_before_filter != expected_rows
                    or excluded_by_filter < 0
                    or excluded_by_filter >= available_before_filter
                ):
                    errors.append(f"experiment_{exp_index}_fold_{fold_index}_invalid_eligibility_counts")
                elif sample_count != available_before_filter - excluded_by_filter:
                    errors.append(f"experiment_{exp_index}_fold_{fold_index}_eligible_test_sample_count_mismatch")
            elif sample_count != expected_rows:
                errors.append(f"experiment_{exp_index}_fold_{fold_index}_test_sample_count_mismatch")
            test_predictions = fold.get("test_predictions")
            if isinstance(test_predictions, list) and len(test_predictions) != sample_count:
                errors.append(f"experiment_{exp_index}_fold_{fold_index}_prediction_count_mismatch")
            total_test_rows += sample_count
            for name in _METRIC_KEYS:
                value = metrics.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    errors.append(f"experiment_{exp_index}_fold_{fold_index}_invalid_metric_{name}")
                    continue
                numeric = float(value)
                if name in {"brier_score", "accuracy_at_fixed_half", "expected_calibration_error", "prevalence", "mean_predicted_probability"} and not 0 <= numeric <= 1:
                    errors.append(f"experiment_{exp_index}_fold_{fold_index}_metric_out_of_range_{name}")
                    continue
                if name == "log_loss" and numeric < 0:
                    errors.append(f"experiment_{exp_index}_fold_{fold_index}_metric_out_of_range_log_loss")
                    continue
                metric_values[name].append(numeric)

        summary_metrics: dict[str, Any] = {}
        for name, values in metric_values.items():
            if values:
                summary_metrics[name] = {
                    "mean_across_folds": round(statistics.fmean(values), 8),
                    "min_across_folds": round(min(values), 8),
                    "max_across_folds": round(max(values), 8),
                    "fold_values": [round(value, 8) for value in values],
                }
        experiment_summaries.append({
            "experiment_id": experiment_id,
            "model_type": experiment.get("model_type"),
            "feature_variant": experiment.get("feature_variant"),
            "fold_count": len(folds),
            "test_rows_counted_by_fold": total_test_rows,
            "metrics": summary_metrics,
            "ranked_or_selected": False,
            "actionable": False,
        })

    if not experiment_summaries:
        errors.append("no_experiment_summaries_created")
    status = "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS"
    return {
        "status": status,
        "path": str(source.resolve()),
        "run_id": report.get("run_id"),
        "symbol": report.get("symbol"),
        "timeframe": report.get("timeframe"),
        "fold_count": report.get("fold_count"),
        "experiment_count": len(experiment_summaries),
        "overlapping_test_window_count": overlapping_windows,
        "experiment_summaries": experiment_summaries,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
        "errors": sorted(set(errors)),
        "warnings": warnings,
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "Descriptive fold-level test metrics only. These summaries do not rank candidates, "
            "establish profitability, or prove leakage-free data."
        ),
    }


def _ctrader_config_status() -> dict[str, Any]:
    """Validate DEMO credentials without returning or logging any credential value."""
    from app.ctrader_history_ingest import load_config

    required = ("CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET", "CTRADER_ACCESS_TOKEN", "CTRADER_ACCOUNT_ID")
    try:
        config = load_config()  # Loads the repository-root .env when present.
    except Exception as exc:
        presence = {name: bool(os.getenv(name, "").strip()) for name in required}
        host = os.getenv("CTRADER_HOST", "demo").strip().lower() or "demo"
        return {
            "status": "FAIL",
            "host": "demo" if host == "demo" else "rejected_non_demo",
            "required_variables_present": presence,
            "env_file_checked": str((Path(__file__).resolve().parent.parent / ".env")),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "credential_values_exposed": False,
            "network_checked": False,
            "orders_possible_from_this_check": False,
        }

    presence = {name: bool(os.getenv(name, "").strip()) for name in required}
    return {
        "status": "PASS",
        "host": config.host,
        "required_variables_present": presence,
        "env_file_checked": str((Path(__file__).resolve().parent.parent / ".env")),
        "credential_values_exposed": False,
        "network_checked": False,
        "orders_possible_from_this_check": False,
    }


def _runtime_summary() -> dict[str, Any]:
    raw = collect_runtime_diagnostics()
    packages = raw.get("python_packages", {})
    required = {
        label: bool(packages.get(label, {}).get("available"))
        for label in _REQUIRED_RESEARCH_MODULES
    }
    return {
        "status": "PASS" if all(required.values()) else "WARN",
        "required_research_modules": required,
        "resource_profile": raw.get("hardware", {}).get("resource_profile", {}),
        "llm_gateway": raw.get("llm_gateway", {}),
        "unverified_components": raw.get("unverified_components", {}),
        "note": "Module discovery and host inventory do not prove imports, drivers, served-model health, or GPU acceleration.",
    }


def collect_research_preflight(*, check_ctrader_config: bool = False) -> dict[str, Any]:
    """Inspect research prerequisites without starting model training or network calls."""
    runtime = _runtime_summary()
    missing = [name for name, present in runtime["required_research_modules"].items() if not present]
    warnings = []
    errors = []
    if missing:
        warnings.append("required research modules not detected: " + ", ".join(missing))
    ctrader = _ctrader_config_status() if check_ctrader_config else {
        "status": "NOT_CHECKED",
        "network_checked": False,
        "credential_values_exposed": False,
    }
    if check_ctrader_config and ctrader["status"] == "FAIL":
        errors.append("cTrader DEMO configuration is incomplete or invalid")
    return {
        "schema_version": 1,
        "scope": "GOLDCREST_RESEARCH_PREFLIGHT",
        "status": "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS",
        "runtime_preflight": runtime,
        "ctrader_demo_config": ctrader,
        "errors": errors,
        "warnings": warnings,
        "actionable": False,
        "execution_connected": False,
        "network_checked": False,
        "credential_values_exposed": False,
        "interpretation": (
            "Dependency discovery and optional cTrader DEMO environment-variable validation only. "
            "This preflight does not test API connectivity, served-model health, model accuracy, or profitability."
        ),
    }


def validate_completed_run(
    manifest_file: str | Path,
    *,
    output_path: str | Path | None = None,
    check_ctrader_config: bool = False,
) -> dict[str, Any]:
    """Build a consolidated quality and integrity report for an existing run."""
    manifest_path = Path(manifest_file)
    audit = audit_research_run_manifest(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []
    manifest: dict[str, Any] = {}
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError):
        errors.append("pipeline_manifest_unreadable")

    dataset_summary: dict[str, Any] | None = None
    matrix_summary: dict[str, Any] | None = None
    dataset_path = _resolve_artifact(manifest.get("dataset_path"), manifest_path)
    if dataset_path and dataset_path.is_file():
        dataset_summary = validate_prepared_dataset(
            dataset_path,
            expected_symbol=manifest.get("symbol"),
            expected_timeframe=manifest.get("timeframe"),
            max_label_elapsed_hours=manifest.get("max_label_elapsed_hours"),
        )
        expected_eligibility_mode = manifest.get("label_eligibility_mode", "all_labels_included")
        if dataset_summary.get("label_eligibility", {}).get("mode") != expected_eligibility_mode:
            errors.append("dataset:label_eligibility_mode_does_not_match_manifest")
        if dataset_summary["status"] == "FAIL":
            errors.extend(f"dataset:{item}" for item in dataset_summary["errors"])
        elif dataset_summary["status"] == "PASS_WITH_WARNINGS":
            warnings.extend(f"dataset:{item}" for item in dataset_summary["warnings"])
    else:
        errors.append("prepared_dataset_not_found")

    matrix_path = _resolve_artifact(manifest.get("experiment_matrix_report"), manifest_path)
    if matrix_path and matrix_path.is_file():
        matrix_summary = summarize_experiment_matrix(matrix_path)
        if matrix_summary["status"] == "FAIL":
            errors.extend(f"experiment_matrix:{item}" for item in matrix_summary["errors"])
        elif matrix_summary["status"] == "PASS_WITH_WARNINGS":
            warnings.extend(f"experiment_matrix:{item}" for item in matrix_summary["warnings"])
    else:
        errors.append("experiment_matrix_report_not_found")

    if audit.get("status") == "FAIL":
        errors.extend(f"integrity:{item}" for item in audit.get("errors", []))
    elif audit.get("status") == "PASS_WITH_WARNINGS":
        warnings.extend(f"integrity:{item}" for item in audit.get("warnings", []))

    temporal_calibration_summary: dict[str, Any] | None = None
    temporal_path_value = manifest.get("temporal_calibration_audit_path")
    if temporal_path_value or manifest.get("temporal_calibration_audit_sha256"):
        temporal_path = _resolve_artifact(temporal_path_value, manifest_path)
        if temporal_path is not None and temporal_path.is_file():
            try:
                temporal_receipt = _read_json(temporal_path)
                temporal_calibration_summary = {
                    "status": temporal_receipt.get("status"),
                    "run_id": temporal_receipt.get("run_id"),
                    "experiment_matrix_sha256": temporal_receipt.get("experiment_matrix_sha256"),
                    "experiment_matrix_payload_hash_verified": temporal_receipt.get(
                        "experiment_matrix_payload_hash_verified"
                    ),
                    "comparison_count": temporal_receipt.get("comparison_count"),
                    "errors": temporal_receipt.get("errors", []),
                    "warnings": temporal_receipt.get("warnings", []),
                    "actionable": temporal_receipt.get("actionable") is False,
                    "execution_connected": temporal_receipt.get("execution_connected") is False,
                }
            except (OSError, json.JSONDecodeError, ValueError):
                temporal_calibration_summary = {
                    "status": "UNAVAILABLE",
                    "comparison_count": 0,
                    "errors": ["temporal_calibration_audit_receipt_unreadable"],
                    "warnings": [],
                    "actionable": False,
                    "execution_connected": False,
                }
        else:
            temporal_calibration_summary = {
                "status": "UNAVAILABLE",
                "comparison_count": 0,
                "errors": ["temporal_calibration_audit_receipt_missing"],
                "warnings": [],
                "actionable": False,
                "execution_connected": False,
            }

    runtime = _runtime_summary()
    if runtime["status"] != "PASS":
        missing = [name for name, present in runtime["required_research_modules"].items() if not present]
        warnings.append("required research modules not detected in current environment: " + ", ".join(missing))
    cTrader = _ctrader_config_status() if check_ctrader_config else {
        "status": "NOT_CHECKED",
        "network_checked": False,
        "credential_values_exposed": False,
        "note": "Use --check-ctrader-config to inspect required variable presence; it does not connect to cTrader.",
    }
    if check_ctrader_config and cTrader["status"] == "FAIL":
        errors.append("cTrader DEMO configuration is incomplete or invalid")

    status = "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS"
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "GOLDCREST_COMPLETED_RUN_VALIDATION",
        "status": status,
        "run_id": manifest.get("run_id"),
        "manifest_path": str(manifest_path.resolve()),
        "integrity_audit": audit,
        "dataset_quality": dataset_summary,
        "experiment_quality": matrix_summary,
        "temporal_calibration_stability": temporal_calibration_summary,
        "runtime_preflight": runtime,
        "ctrader_demo_config": cTrader,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "actionable": False,
        "execution_connected": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
        "interpretation": (
            "Run-integrity and research-quality review only. A passing report does not prove predictive edge, "
            "economic profitability, deployment readiness, or absence of all data leakage."
        ),
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        result["report_path"] = str(target.resolve())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preflight research prerequisites or validate a completed offline research run."
    )
    parser.add_argument(
        "manifest", nargs="?", help="Path to <run-id>_pipeline_manifest.json; omit for preflight only"
    )
    parser.add_argument("--output", help="Output path for the JSON report")
    parser.add_argument(
        "--check-ctrader-config", action="store_true",
        help="Check cTrader DEMO environment-variable presence only; no network requests are made",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = parser.parse_args(argv)
    if args.manifest:
        report = validate_completed_run(
            args.manifest, output_path=args.output, check_ctrader_config=args.check_ctrader_config
        )
    else:
        report = collect_research_preflight(check_ctrader_config=args.check_ctrader_config)
        if args.output:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            report["report_path"] = str(target.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Research validation: {report['status']}")
        if report.get("run_id"):
            print(f"Run ID: {report['run_id']}")
        dataset = report.get("dataset_quality") or {}
        if dataset:
            print(f"Dataset rows: {dataset.get('row_count', 'unavailable')}")
        matrix = report.get("experiment_quality") or {}
        if matrix:
            print(f"Experiments/folds: {matrix.get('experiment_count', 'unavailable')}/{matrix.get('fold_count', 'unavailable')}")
        for item in report.get("errors", []):
            print(f"ERROR: {item}")
        for item in report.get("warnings", []):
            print(f"WARNING: {item}")
        if report.get("report_path"):
            print(f"Report: {report['report_path']}")
        print("Actionable: false")
    return 3 if report["status"] == "FAIL" else 0

if __name__ == "__main__":
    raise SystemExit(main())

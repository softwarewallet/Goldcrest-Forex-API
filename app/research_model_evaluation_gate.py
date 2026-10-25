"""Independent structural and metric-integrity gate for offline model evaluations.

This audit checks persisted walk-forward experiment matrices. It does not retrain models,
infer profitability, establish statistical significance, or authorize trading/execution.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


_METRICS = {
    "sample_count": "positive_int",
    "positive_count": "nonnegative_int",
    "negative_count": "nonnegative_int",
    "prevalence": "unit_interval",
    "brier_score": "unit_interval",
    "log_loss": "nonnegative",
    "accuracy_at_fixed_half": "unit_interval",
    "mean_predicted_probability": "unit_interval",
    "expected_calibration_error": "unit_interval",
}
_TIMESTAMP_FIELDS = ("train_start_utc", "train_end_utc", "validation_start_utc",
                     "validation_end_utc", "test_start_utc", "test_end_utc")


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def audit_model_evaluation_report(report_path: str | Path) -> dict[str, Any]:
    """Audit a persisted experiment matrix without modifying the source artifact."""
    source = Path(report_path)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    def error(code: str, location: str, detail: str) -> None:
        errors.append({"code": code, "location": location, "detail": detail})

    def warning(code: str, location: str, detail: str) -> None:
        warnings.append({"code": code, "location": location, "detail": detail})

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "schema_version": 1, "status": "FAIL", "source_path": str(source),
            "errors": [{"code": "report_unreadable", "location": "$", "detail": str(exc)}],
            "warnings": [], "actionable": False,
        }

    if not isinstance(payload, dict):
        return {
            "schema_version": 1, "status": "FAIL", "source_path": str(source),
            "errors": [{"code": "root_not_object", "location": "$", "detail": "Expected a JSON object."}],
            "warnings": [], "actionable": False,
        }

    for flag in ("test_metrics_used_for_selection", "automatic_winner_selection", "actionable"):
        if payload.get(flag) is not False:
            error("unsafe_or_unrecorded_selection_flag", flag,
                  f"{flag} must be explicitly false for this offline evaluation gate.")

    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        error("experiments_missing", "experiments", "Expected a non-empty experiments array.")
        experiments = []

    seen_experiment_ids: set[str] = set()
    checked_folds = 0
    checked_metrics = 0
    expected_fold_count = payload.get("fold_count")
    if isinstance(expected_fold_count, bool) or not isinstance(expected_fold_count, int) or expected_fold_count <= 0:
        error("invalid_fold_count", "fold_count", "fold_count must be a positive integer.")
        expected_fold_count = None

    for exp_index, experiment in enumerate(experiments):
        exp_loc = f"experiments[{exp_index}]"
        if not isinstance(experiment, dict):
            error("experiment_not_object", exp_loc, "Each experiment must be a JSON object.")
            continue
        experiment_id = experiment.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            error("experiment_id_missing", f"{exp_loc}.experiment_id", "A non-empty experiment_id is required.")
        elif experiment_id in seen_experiment_ids:
            error("duplicate_experiment_id", f"{exp_loc}.experiment_id", experiment_id)
        else:
            seen_experiment_ids.add(experiment_id)

        if experiment.get("actionable") is not False:
            error("experiment_actionable_flag_not_false", f"{exp_loc}.actionable",
                  "Research experiment must explicitly remain non-actionable.")
        if experiment.get("test_metrics_used_for_tuning") is not False:
            error("test_metrics_used_for_tuning", f"{exp_loc}.test_metrics_used_for_tuning",
                  "Test metrics must be explicitly excluded from tuning.")

        folds = experiment.get("fold_results")
        if not isinstance(folds, list) or not folds:
            error("fold_results_missing", f"{exp_loc}.fold_results", "Expected non-empty fold_results.")
            continue
        if expected_fold_count is not None and len(folds) != expected_fold_count:
            error("fold_count_mismatch", f"{exp_loc}.fold_results",
                  f"Expected {expected_fold_count} folds, found {len(folds)}.")

        seen_fold_indices: set[int] = set()
        for fold_index, fold in enumerate(folds):
            fold_loc = f"{exp_loc}.fold_results[{fold_index}]"
            checked_folds += 1
            if not isinstance(fold, dict):
                error("fold_not_object", fold_loc, "Each fold must be a JSON object.")
                continue
            index = fold.get("fold_index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                error("invalid_fold_index", f"{fold_loc}.fold_index", "fold_index must be a non-negative integer.")
            elif index in seen_fold_indices:
                error("duplicate_fold_index", f"{fold_loc}.fold_index", str(index))
            else:
                seen_fold_indices.add(index)

            if fold.get("test_metrics_used_for_tuning") is not False:
                error("test_metrics_used_for_tuning", f"{fold_loc}.test_metrics_used_for_tuning",
                      "Test metrics must be explicitly excluded from tuning.")

            parsed_times: dict[str, datetime] = {}
            for field in _TIMESTAMP_FIELDS:
                if field not in fold:
                    error("fold_timestamp_missing", f"{fold_loc}.{field}",
                          "Every fold must record all train, validation, and test boundaries.")
                    continue
                parsed = _parse_timestamp(fold.get(field))
                if parsed is None:
                    error("invalid_fold_timestamp", f"{fold_loc}.{field}",
                          "Timestamp must be valid ISO-8601 and timezone-aware.")
                else:
                    parsed_times[field] = parsed
            ordered_pairs = (
                ("train_start_utc", "train_end_utc"),
                ("validation_start_utc", "validation_end_utc"),
                ("test_start_utc", "test_end_utc"),
            )
            for start_field, end_field in ordered_pairs:
                if start_field in parsed_times and end_field in parsed_times:
                    if parsed_times[start_field] >= parsed_times[end_field]:
                        error("invalid_fold_window", fold_loc,
                              f"{start_field} must precede {end_field}.")

            metrics = fold.get("test_metrics")
            if not isinstance(metrics, dict):
                error("test_metrics_missing", f"{fold_loc}.test_metrics",
                      "Every fold must persist a test_metrics object.")
                continue
            checked_metrics += 1
            values: dict[str, object] = {}
            for key, kind in _METRICS.items():
                value = metrics.get(key)
                values[key] = value
                if not _finite_number(value):
                    error("metric_missing_or_nonfinite", f"{fold_loc}.test_metrics.{key}",
                          f"{key} must be a finite numeric value.")
                    continue
                numeric = float(value)
                if kind == "positive_int" and (isinstance(value, bool) or int(value) != value or value <= 0):
                    error("metric_out_of_range", f"{fold_loc}.test_metrics.{key}",
                          "sample_count must be a positive integer.")
                elif kind == "nonnegative_int" and (isinstance(value, bool) or int(value) != value or value < 0):
                    error("metric_out_of_range", f"{fold_loc}.test_metrics.{key}",
                          f"{key} must be a non-negative integer.")
                elif kind == "unit_interval" and not 0 <= numeric <= 1:
                    error("metric_out_of_range", f"{fold_loc}.test_metrics.{key}",
                          f"{key} must be in [0, 1].")
                elif kind == "nonnegative" and numeric < 0:
                    error("metric_out_of_range", f"{fold_loc}.test_metrics.{key}",
                          f"{key} must be non-negative.")

            sample_count, positive_count, negative_count = (
                values["sample_count"], values["positive_count"], values["negative_count"]
            )
            if all(_finite_number(v) for v in (sample_count, positive_count, negative_count)):
                if positive_count + negative_count != sample_count:
                    error("class_counts_do_not_sum", f"{fold_loc}.test_metrics",
                          "positive_count + negative_count must equal sample_count.")
                if sample_count > 0 and positive_count >= 0:
                    expected_prevalence = float(positive_count) / float(sample_count)
                    prevalence = values["prevalence"]
                    if _finite_number(prevalence) and abs(float(prevalence) - expected_prevalence) > 1e-8:
                        error("prevalence_inconsistent", f"{fold_loc}.test_metrics.prevalence",
                              "prevalence must equal positive_count / sample_count.")

    result = {
        "schema_version": 1,
        "scope": "GOLDCREST_MODEL_EVALUATION_INTEGRITY_GATE",
        "status": "FAIL" if errors else "PASS",
        "run_id": payload.get("run_id"),
        "symbol": payload.get("symbol"),
        "timeframe": payload.get("timeframe"),
        "source_path": str(source.resolve()),
        "source_matrix_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
        "experiment_count": len(experiments),
        "folds_checked": checked_folds,
        "fold_metric_objects_checked": checked_metrics,
        "errors": errors,
        "warnings": warnings,
        "actionable": False,
        "interpretation": (
            "PASS means only that the checked report structure, recorded safeguards, and metric ranges "
            "are internally consistent. It does not prove leakage-free research, model quality, statistical "
            "significance, trading profitability, or execution readiness."
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-matrix", required=True, help="Persisted experiment-matrix JSON.")
    parser.add_argument("--output", help="Optional new JSON receipt path; existing files are never overwritten.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON audit result.")
    args = parser.parse_args()
    result = audit_model_evaluation_report(args.experiment_matrix)
    if args.output:
        output = Path(args.output)
        if output.exists():
            parser.error(f"refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"Model evaluation gate: {result['status']}")
        print(f"Experiments: {result.get('experiment_count', 0)}; folds checked: {result.get('folds_checked', 0)}")
        for item in result["errors"]:
            print(f"ERROR {item['code']} at {item['location']}: {item['detail']}")
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())

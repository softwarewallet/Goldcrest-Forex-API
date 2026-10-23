"""Compact, integrity-aware metrics summary for an offline research run.

This command summarizes held-out fold metrics for human review. It does not rank or select
models, choose thresholds, promote a model, or connect to trading execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from app.research_validation import summarize_experiment_matrix

_CONSTANT_HALF_BRIER = 0.25
_CONSTANT_HALF_LOG_LOSS = math.log(2.0)
_METRICS = (
    "log_loss",
    "brier_score",
    "accuracy_at_fixed_half",
    "expected_calibration_error",
    "prevalence",
    "mean_predicted_probability",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_path(value: object, parent: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        relative = (parent / candidate).resolve()
        if relative.is_file():
            return relative
    return candidate


def _compact_metric(metric: object, baseline: float | None = None) -> dict[str, Any] | None:
    if not isinstance(metric, dict):
        return None
    result = {
        name: metric.get(name)
        for name in ("mean_across_folds", "min_across_folds", "max_across_folds")
    }
    fold_values = metric.get("fold_values")
    if baseline is not None and isinstance(fold_values, list):
        numeric = [float(value) for value in fold_values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        result["folds_below_constant_half_baseline"] = sum(value < baseline for value in numeric)
        result["folds_compared"] = len(numeric)
    return result


def summarize_research_run(
    manifest_path: str | Path,
    *,
    validation_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify core fingerprints and return a concise, non-ranking metrics summary."""
    source = Path(manifest_path).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest = _read_json(source)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "schema_version": 1,
            "status": "FAIL",
            "manifest_path": str(source),
            "errors": [f"pipeline_manifest_unreadable:{type(exc).__name__}"],
            "warnings": [],
            "actionable": False,
            "execution_connected": False,
            "automatic_model_selection": False,
        }

    expected_manifest_hash = manifest.get("pipeline_manifest_sha256")
    hash_payload = dict(manifest)
    hash_payload.pop("pipeline_manifest_sha256", None)
    try:
        canonical = json.dumps(hash_payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        actual_manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if not isinstance(expected_manifest_hash, str) or expected_manifest_hash != actual_manifest_hash:
            errors.append("pipeline_manifest_sha256_mismatch")
    except (TypeError, ValueError):
        errors.append("pipeline_manifest_not_canonical_json")

    run_id = manifest.get("run_id")
    matrix_path = _resolve_path(manifest.get("experiment_matrix_report"), source.parent)
    if matrix_path is None or not matrix_path.is_file():
        errors.append("experiment_matrix_report_not_found")
        matrix_summary: dict[str, Any] = {"status": "FAIL", "errors": ["experiment_matrix_report_not_found"]}
        matrix_hash_match = False
    else:
        matrix_hash = _sha256(matrix_path)
        expected_matrix_hash = manifest.get("experiment_matrix_report_sha256")
        matrix_hash_match = isinstance(expected_matrix_hash, str) and matrix_hash == expected_matrix_hash
        if not matrix_hash_match:
            errors.append("experiment_matrix_report_sha256_mismatch")
        try:
            report = _read_json(matrix_path)
            if report.get("run_id") != run_id:
                errors.append("experiment_matrix_run_id_mismatch")
            if report.get("actionable") is not False:
                errors.append("experiment_matrix_must_be_non_actionable")
            matrix_summary = summarize_experiment_matrix(matrix_path)
            if matrix_summary.get("status") == "FAIL":
                errors.extend(f"experiment_matrix:{item}" for item in matrix_summary.get("errors", []))
            warnings.extend(f"experiment_matrix:{item}" for item in matrix_summary.get("warnings", []))
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            matrix_summary = {
                "status": "FAIL",
                "errors": [f"experiment_matrix_summary_failed:{type(exc).__name__}"],
            }
            errors.extend(matrix_summary["errors"])

    if validation_report_path:
        validation_path = Path(validation_report_path).resolve()
    elif source.name.endswith("_pipeline_manifest.json"):
        validation_path = source.with_name(source.name.replace("_pipeline_manifest.json", "_validation_report.json"))
    else:
        validation_path = source.parent / f"{run_id}_validation_report.json"

    validation: dict[str, Any] = {}
    try:
        validation = _read_json(validation_path)
        if validation.get("run_id") not in (None, run_id):
            errors.append("validation_report_run_id_mismatch")
        validation_status = validation.get("status", "UNKNOWN")
        errors.extend(f"validation:{item}" for item in validation.get("errors", []) if isinstance(item, str))
        warnings.extend(f"validation:{item}" for item in validation.get("warnings", []) if isinstance(item, str))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        validation_status = "UNAVAILABLE"
        errors.append(f"validation_report_unavailable:{type(exc).__name__}")

    if manifest.get("actionable") is not False or manifest.get("execution_connected") is not False:
        errors.append("pipeline_safety_flags_invalid")
    if manifest.get("automatic_winner_selection") is not False or manifest.get("test_metrics_used_for_selection") is not False:
        errors.append("pipeline_model_selection_safety_flags_invalid")
    if manifest.get("provenance_audit_status") != "PASS":
        errors.append("pipeline_provenance_audit_not_pass")
    if manifest.get("raw_ohlc_audit_status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        errors.append("pipeline_raw_ohlc_audit_not_pass")

    models: list[dict[str, Any]] = []
    for experiment in matrix_summary.get("experiment_summaries", []):
        if not isinstance(experiment, dict):
            continue
        metric_payload = experiment.get("metrics", {})
        if not isinstance(metric_payload, dict):
            metric_payload = {}
        log_loss = metric_payload.get("log_loss", {})
        brier = metric_payload.get("brier_score", {})
        log_loss_mean = log_loss.get("mean_across_folds") if isinstance(log_loss, dict) else None
        brier_mean = brier.get("mean_across_folds") if isinstance(brier, dict) else None
        model_metrics: dict[str, Any] = {}
        for name in _METRICS:
            baseline = _CONSTANT_HALF_LOG_LOSS if name == "log_loss" else _CONSTANT_HALF_BRIER if name == "brier_score" else None
            compact = _compact_metric(metric_payload.get(name), baseline)
            if compact is not None:
                model_metrics[name] = compact
        models.append({
            "experiment_id": experiment.get("experiment_id"),
            "model_type": experiment.get("model_type"),
            "feature_variant": experiment.get("feature_variant"),
            "fold_count": experiment.get("fold_count"),
            "test_rows_counted_by_fold": experiment.get("test_rows_counted_by_fold"),
            "log_loss_improvement_vs_constant_0_5": (
                round(_CONSTANT_HALF_LOG_LOSS - float(log_loss_mean), 8)
                if isinstance(log_loss_mean, (int, float)) and not isinstance(log_loss_mean, bool) else None
            ),
            "brier_improvement_vs_constant_0_5": (
                round(_CONSTANT_HALF_BRIER - float(brier_mean), 8)
                if isinstance(brier_mean, (int, float)) and not isinstance(brier_mean, bool) else None
            ),
            "metrics": model_metrics,
            "ranked_or_selected": False,
            "actionable": False,
        })

    validation_status_lower = str(validation_status).upper()
    if validation_status_lower == "FAIL":
        errors.append("consolidated_validation_failed")
    if validation_status_lower == "PASS_WITH_WARNINGS":
        warnings.extend(
            f"raw_ohlc_audit:{warning}"
            for warning in manifest.get("raw_ohlc_audit_warnings", [])
            if isinstance(warning, str)
        )
        if not manifest.get("raw_ohlc_audit_warnings"):
            warnings.append("consolidated_validation_passed_with_warnings")
    if matrix_summary.get("status") == "FAIL":
        errors.append("experiment_matrix_summary_failed")

    status = "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings or validation_status_lower == "PASS_WITH_WARNINGS" or matrix_summary.get("status") == "PASS_WITH_WARNINGS" else "PASS"
    return {
        "schema_version": 1,
        "scope": "GOLDCREST_COMPACT_OFFLINE_RESEARCH_RESULTS",
        "status": status,
        "run_id": run_id,
        "symbol": manifest.get("symbol"),
        "timeframe": manifest.get("timeframe"),
        "validation_status": validation_status,
        "raw_ohlc_audit_status": manifest.get("raw_ohlc_audit_status"),
        "raw_ohlc_audit_warnings": manifest.get("raw_ohlc_audit_warnings", []),
        "provenance_audit_status": manifest.get("provenance_audit_status"),
        "provenance_audit_errors": manifest.get("provenance_audit_errors", []),
        "fold_count": manifest.get("fold_count"),
        "experiment_count": manifest.get("experiment_count"),
        "experiment_matrix_hash_verified": matrix_hash_match,
        "validation_report_path": str(validation_path),
        "models": models,
        "constant_probability_reference": {
            "predicted_probability": 0.5,
            "brier_score": _CONSTANT_HALF_BRIER,
            "log_loss": round(_CONSTANT_HALF_LOG_LOSS, 8),
            "interpretation": (
                "Fixed uninformative reference only. Positive improvement values mean a lower "
                "mean fold metric than this reference; this is not a strategy or profitability benchmark."
            ),
        },
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "automatic_model_selection": False,
        "test_metrics_used_for_selection": False,
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "Descriptive held-out fold metrics only. Baseline comparisons are not model rankings and "
            "do not establish profitability, statistical significance, or production readiness."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print a compact, non-ranking summary of a completed Goldcrest research run."
    )
    parser.add_argument("--manifest", required=True, help="Path to <run-id>_pipeline_manifest.json")
    parser.add_argument("--validation-report", help="Optional path to the consolidated validation report")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args(argv)

    result = summarize_research_run(args.manifest, validation_report_path=args.validation_report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 3 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

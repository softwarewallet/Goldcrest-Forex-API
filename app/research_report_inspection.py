"""Integrity-checked, read-only summaries of offline experiment reports.

This module never accesses market data, fits models, mutates a ledger, or authorizes
trading. It validates the report's self-contained canonical fingerprint and preserves
fold/partition boundaries rather than aggregating overlapping test windows.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from app.research_strict_json import strict_json_loads
from app.research_reproducibility import verify_reproducibility_manifest
from app.model_selection_audit import build_model_selection_audit
from app.oos_period_robustness_audit import build_oos_period_robustness_audit
from app.regime_uncertainty_audit import build_regime_uncertainty_audit
from app.fold_stability import summarize_fold_stability
from app.block_length_sensitivity_audit import build_block_length_sensitivity_audit
from app.temporal_calibration_stability import build_temporal_calibration_stability_audit
from app.calibration_sample_adequacy import build_calibration_sample_adequacy_audit


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"report contains values that cannot be canonically fingerprinted: {exc}") from exc


def _metrics(value: Any, *, context: str) -> dict[str, int | float | bool | str | None]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} metrics must be a JSON object")
    output: dict[str, int | float | bool | str | None] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{context} metric names must be strings")
        if item is None or isinstance(item, (str, bool, int)):
            output[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            output[key] = item
        elif isinstance(item, (dict, list)):
            _canonical({"value": item})
            output[key] = item
        else:
            raise ValueError(f"{context} contains a non-finite or unsupported metric: {key}")
    return output


def inspect_experiment_report(report_path: str | Path) -> dict[str, Any]:
    """Load, fingerprint-check, and summarize a completed offline research report.

    Returns per-model/per-fold validation and test metrics without pooled scores.
    The self-fingerprint proves only report integrity, not source-data authenticity.
    """
    path = Path(report_path)
    try:
        report = strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"report cannot be read as strict JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    supplied = report.get("report_sha256")
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise ValueError("report_sha256 must be a 64-character SHA-256 digest")
    try:
        bytes.fromhex(supplied)
    except ValueError as exc:
        raise ValueError("report_sha256 must be hexadecimal") from exc
    payload = dict(report)
    payload.pop("report_sha256")
    actual = hashlib.sha256(_canonical(payload)).hexdigest()
    if actual != supplied:
        raise ValueError("report fingerprint mismatch; artifact may have been modified")

    run_id = report.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("report must contain a non-empty run_id")
    if report.get("schema_version") != 1:
        raise ValueError("unsupported experiment report schema_version")

    manifest = report.get("manifest", {})
    if not isinstance(manifest, dict):
        raise ValueError("report manifest must be a JSON object")
    reproducibility = manifest.get("reproducibility")
    if reproducibility is not None:
        if not isinstance(reproducibility, dict):
            raise ValueError("reproducibility manifest must be a JSON object")
        verify_reproducibility_manifest(reproducibility)
    interpretation = report.get("interpretation")
    if not isinstance(interpretation, dict) or interpretation.get("actionable") is not False:
        raise ValueError("report must explicitly declare actionable=false")
    if interpretation.get("scope") != "OFFLINE_RESEARCH_ONLY":
        raise ValueError("report scope must be OFFLINE_RESEARCH_ONLY")

    fold_results = report.get("fold_results")
    if not isinstance(fold_results, dict) or not fold_results:
        raise ValueError("fold_results must contain at least one model")
    models: list[dict[str, Any]] = []
    fold_index_by_model: dict[str, set[int]] = {}
    metrics_by_model_fold: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for model_name, folds in sorted(fold_results.items()):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model names must be non-empty strings")
        if not isinstance(folds, list) or not folds:
            raise ValueError(f"model {model_name!r} must contain at least one fold")
        seen: set[int] = set()
        summarized_folds: list[dict[str, Any]] = []
        for fold in folds:
            if not isinstance(fold, dict):
                raise ValueError(f"model {model_name!r} contains a malformed fold")
            index = fold.get("fold_index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen:
                raise ValueError(f"model {model_name!r} contains an invalid or duplicate fold index")
            seen.add(index)
            if fold.get("actionable", False) is not False:
                raise ValueError("every fold must remain actionable=false")
            if "validation" not in fold or "test" not in fold:
                raise ValueError(f"fold {index} must contain separate validation and test metrics")
            validation = _metrics(fold["validation"], context=f"{model_name} fold {index} validation")
            test = _metrics(fold["test"], context=f"{model_name} fold {index} test")
            for partition_name, metric in (("validation", validation), ("test", test)):
                uncertainty = metric.get("calibration_uncertainty")
                if uncertainty is not None:
                    context = f"{model_name} fold {index} {partition_name} calibration_uncertainty"
                    if not isinstance(uncertainty, dict):
                        raise ValueError(f"{context} must be a JSON object")
                    low, high = uncertainty.get("interval_low"), uncertainty.get("interval_high")
                    if (low is None) != (high is None):
                        raise ValueError(f"{context} interval bounds must both be present or both be null")
                    if low is not None:
                        if (isinstance(low, bool) or not isinstance(low, (int, float))
                                or isinstance(high, bool) or not isinstance(high, (int, float))
                                or not math.isfinite(float(low)) or not math.isfinite(float(high))
                                or not 0 <= low <= high <= 1):
                            raise ValueError(f"{context} interval bounds must be ordered finite values in [0, 1]")
                    sample_count = metric.get("sample_count")
                    if uncertainty.get("sample_count") != sample_count:
                        raise ValueError(f"{context} sample_count must match the scored partition")
                    block_length = uncertainty.get("block_length")
                    resamples = uncertainty.get("bootstrap_resamples")
                    confidence = uncertainty.get("confidence_level")
                    if (isinstance(block_length, bool) or not isinstance(block_length, int)
                            or block_length < 1 or block_length > sample_count):
                        raise ValueError(f"{context} block_length must be between 1 and sample_count")
                    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
                        raise ValueError(f"{context} bootstrap_resamples must be an integer >= 100")
                    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                            or not math.isfinite(float(confidence)) or not 0 < confidence < 1):
                        raise ValueError(f"{context} confidence_level must be between 0 and 1")
                    status = uncertainty.get("status")
                    if low is None and status != "WITHHELD_INSUFFICIENT_SAMPLE_COUNT":
                        raise ValueError(f"{context} null interval requires an explicit withheld status")
                    if low is not None and status != "DESCRIPTIVE_MOVING_BLOCK_PERCENTILE_INTERVAL":
                        raise ValueError(f"{context} present interval has an unexpected status")
            metrics_by_model_fold[(model_name, index)] = {"validation": validation, "test": test}
            summarized_folds.append({
                "fold_index": index, "validation": validation, "test": test, "actionable": False,
            })
        fold_index_by_model[model_name] = seen
        models.append({"model_name": model_name, "folds": sorted(summarized_folds, key=lambda row: row["fold_index"])})

    exclusions = report.get("exclusions", {})
    if not isinstance(exclusions, dict):
        raise ValueError("exclusions must be a JSON object")
    normalized_exclusions: dict[str, int] = {}
    for name, count in exclusions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("exclusion names must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("exclusion counts must be non-negative integers")
        normalized_exclusions[name] = count

    comparisons = report.get("paired_comparisons", [])
    if not isinstance(comparisons, list) or any(not isinstance(item, dict) for item in comparisons):
        raise ValueError("paired_comparisons must be a list of objects")
    seen_comparisons: set[tuple[str, str, int, str]] = set()
    for comparison in comparisons:
        if comparison.get("actionable", False) is not False:
            raise ValueError("paired comparisons must remain actionable=false")
        candidate = comparison.get("candidate_name")
        baseline = comparison.get("baseline_name")
        fold_index = comparison.get("fold_index")
        partition = comparison.get("partition")
        sample_count = comparison.get("sample_count")
        if not isinstance(candidate, str) or candidate not in fold_index_by_model:
            raise ValueError("paired comparison candidate_name must reference a reported model")
        if not isinstance(baseline, str) or baseline not in fold_index_by_model:
            raise ValueError("paired comparison baseline_name must reference a reported model")
        if candidate == baseline:
            raise ValueError("paired comparison candidate and baseline must differ")
        if isinstance(fold_index, bool) or not isinstance(fold_index, int) or fold_index < 0:
            raise ValueError("paired comparison fold_index must be a non-negative integer")
        if partition not in {"validation", "test"}:
            raise ValueError("paired comparison partition must be validation or test")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError("paired comparison sample_count must be a positive integer")
        key = (candidate, baseline, fold_index, partition)
        if key in seen_comparisons:
            raise ValueError("duplicate paired comparison identity")
        seen_comparisons.add(key)
        paired_metrics = []
        for model_name, role in ((candidate, "candidate"), (baseline, "baseline")):
            fold_metrics = metrics_by_model_fold.get((model_name, fold_index))
            if fold_metrics is None:
                raise ValueError(f"paired comparison {role} references a missing fold")
            metric = fold_metrics[partition]
            metric_count = metric.get("sample_count")
            if isinstance(metric_count, bool) or not isinstance(metric_count, int) or metric_count != sample_count:
                raise ValueError(f"paired comparison sample_count does not match {role} fold metrics")
            paired_metrics.append(metric)

        candidate_identity = paired_metrics[0].get("observation_identity_sha256")
        baseline_identity = paired_metrics[1].get("observation_identity_sha256")
        comparison_identity = comparison.get("observation_identity_sha256")
        for label, digest in (
            ("candidate metric", candidate_identity),
            ("baseline metric", baseline_identity),
            ("paired comparison", comparison_identity),
        ):
            if digest is not None:
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError(f"{label} observation identity fingerprint must be a SHA-256 digest")
                try:
                    bytes.fromhex(digest)
                except ValueError as exc:
                    raise ValueError(f"{label} observation identity fingerprint must be hexadecimal") from exc
        if (candidate_identity is None) != (baseline_identity is None):
            raise ValueError("paired metric observation identity fingerprints must both be present or both absent")
        if candidate_identity is not None and candidate_identity != baseline_identity:
            raise ValueError("candidate and baseline observation identity fingerprints do not match")
        expected_identity = candidate_identity
        if comparison_identity is not None:
            if expected_identity is None:
                raise ValueError("paired comparison identity fingerprint lacks matching fold metric fingerprints")
            if comparison_identity != expected_identity:
                raise ValueError("paired comparison observation identity fingerprint does not match fold metrics")

    block_sensitivity_audit = report.get("block_length_sensitivity_audit")
    if block_sensitivity_audit is not None:
        if not isinstance(block_sensitivity_audit, dict):
            raise ValueError("block_length_sensitivity_audit must be a JSON object")
        expected_block_sensitivity = build_block_length_sensitivity_audit(comparisons)
        if _canonical(block_sensitivity_audit) != _canonical(expected_block_sensitivity):
            raise ValueError("block_length_sensitivity_audit does not match paired comparison sensitivity metadata")

    selection_audit = report.get("model_selection_audit")
    if selection_audit is not None:
        if not isinstance(selection_audit, dict):
            raise ValueError("model_selection_audit must be a JSON object")
        expected_selection_audit = build_model_selection_audit(fold_results, comparisons)
        if _canonical(selection_audit) != _canonical(expected_selection_audit):
            raise ValueError("model_selection_audit does not match validation-only fold metrics")

    oos_robustness = report.get("oos_period_robustness_audit")
    if oos_robustness is not None:
        if not isinstance(oos_robustness, dict):
            raise ValueError("oos_period_robustness_audit must be a JSON object")
        expected_oos_robustness = build_oos_period_robustness_audit(fold_results, comparisons)
        if _canonical(oos_robustness) != _canonical(expected_oos_robustness):
            raise ValueError("oos_period_robustness_audit does not match out-of-sample fold metrics")

    regime_uncertainty = report.get("regime_uncertainty_audit")
    if regime_uncertainty is not None:
        if not isinstance(regime_uncertainty, dict):
            raise ValueError("regime_uncertainty_audit must be a JSON object")
        expected_regime_uncertainty = build_regime_uncertainty_audit(fold_results, comparisons, manifest.get("fold_regime_annotations"), manifest.get("fold_temporal_boundaries"))
        if _canonical(regime_uncertainty) != _canonical(expected_regime_uncertainty):
            raise ValueError("regime_uncertainty_audit does not match declared annotations and fold metrics")

    temporal_calibration = report.get("temporal_calibration_stability_audit")
    if temporal_calibration is not None:
        if not isinstance(temporal_calibration, dict):
            raise ValueError("temporal_calibration_stability_audit must be a JSON object")
        expected_temporal_calibration = build_temporal_calibration_stability_audit(
            fold_results, manifest.get("fold_temporal_boundaries")
        )
        if _canonical(temporal_calibration) != _canonical(expected_temporal_calibration):
            raise ValueError("temporal_calibration_stability_audit does not match fold metrics and declared temporal boundaries")

    sample_adequacy = report.get("calibration_sample_adequacy_audit")
    if sample_adequacy is not None:
        if not isinstance(sample_adequacy, dict):
            raise ValueError("calibration_sample_adequacy_audit must be a JSON object")
        expected_sample_adequacy = build_calibration_sample_adequacy_audit(fold_results)
        if _canonical(sample_adequacy) != _canonical(expected_sample_adequacy):
            raise ValueError("calibration_sample_adequacy_audit does not match fold metrics")

    warnings = report.get("warnings", [])
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise ValueError("warnings must be a list of strings")

    return {
        "verified": True, "run_id": run_id, "report_sha256": supplied,
        "schema_version": report["schema_version"], "scope": "OFFLINE_RESEARCH_ONLY",
        "actionable": False, "models": models, "paired_comparisons": comparisons,
        "model_selection_audit": selection_audit,
        "oos_period_robustness_audit": oos_robustness,
        "regime_uncertainty_audit": regime_uncertainty,
        "temporal_calibration_stability_audit": temporal_calibration,
        "calibration_sample_adequacy_audit": sample_adequacy,
        "fold_stability": summarize_fold_stability(fold_results, comparisons),
        "exclusions": dict(sorted(normalized_exclusions.items())), "warnings": warnings,
        "evaluation_contract": report.get("evaluation_contract", {}),
        "interpretation": {
            "pooled_test_metrics": False, "limitations": interpretation.get("limitations", []),
            "source_data_authenticated": False,
            "meaning": "Fingerprint verifies report integrity only; it does not establish data correctness, calibration, profitability, or generalization.",
        },
    }

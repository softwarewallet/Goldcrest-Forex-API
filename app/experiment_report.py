"""Deterministic report assembly for offline GoldcrestAI experiments.

The report consolidates already-computed artifacts; it does not fetch data, train
models, infer missing assumptions, promote models, or authorize trading. Per-fold
metrics are preserved rather than pooled across potentially overlapping test windows.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.model_selection_audit import build_model_selection_audit
from app.oos_period_robustness_audit import build_oos_period_robustness_audit
from app.regime_uncertainty_audit import build_regime_uncertainty_audit


REPORT_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("report values must not contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported report value type: {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def build_experiment_report(
    *,
    manifest: Any,
    fold_results: Mapping[str, Sequence[Any]],
    paired_comparisons: Sequence[Any] = (),
    exclusions: Mapping[str, int] | None = None,
    evaluation_contract: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a deterministic report from a manifest and precomputed research artifacts.

    Expected fold results are ModelFoldResult-like records with fold_index, validation,
    test, and actionable fields. Paired comparisons are LedgerComparison-like records.
    This function intentionally does not calculate pooled cross-fold test scores.
    """
    manifest_data = _jsonable(manifest)
    if not isinstance(manifest_data, dict):
        raise ValueError("manifest must be a mapping or dataclass")
    run_id = manifest_data.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("manifest must contain a non-empty run_id")
    if not fold_results:
        raise ValueError("fold_results must include at least one model")
    if any(not isinstance(name, str) or not name.strip() for name in fold_results):
        raise ValueError("model names must be non-empty strings")

    serialized_folds: dict[str, list[dict[str, Any]]] = {}
    for model_name, results in sorted(fold_results.items()):
        if not results:
            raise ValueError(f"model {model_name!r} has no fold results")
        model_rows: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        for result in results:
            row = _jsonable(result)
            if not isinstance(row, dict) or "fold_index" not in row:
                raise ValueError("each fold result must expose fold_index")
            if row.get("actionable", False) is not False:
                raise ValueError("experiment report accepts research-only actionable=False results")
            index = row["fold_index"]
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("fold_index must be a non-negative integer")
            if index in seen_indices:
                raise ValueError(f"duplicate fold index for model {model_name!r}")
            seen_indices.add(index)
            if "validation" not in row or "test" not in row:
                raise ValueError("each fold result must contain separate validation and test metrics")
            model_rows.append(row)
        serialized_folds[model_name] = sorted(model_rows, key=lambda row: row["fold_index"])

    comparisons: list[dict[str, Any]] = []
    for comparison in paired_comparisons:
        row = _jsonable(comparison)
        if not isinstance(row, dict) or row.get("actionable", False) is not False:
            raise ValueError("paired comparisons must remain actionable=False")
        if not isinstance(row.get("sample_count"), int) or row["sample_count"] < 1:
            raise ValueError("paired comparisons require a positive sample_count")
        comparisons.append(row)
    comparisons.sort(key=lambda row: (
        str(row.get("candidate_name", "")),
        int(row.get("fold_index", -1)),
        str(row.get("partition", "")),
        str(row.get("baseline_name", "")),
        int(row.get("sample_count", 0)),
    ))

    normalized_exclusions: dict[str, int] = {}
    for name, count in (exclusions or {}).items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("exclusion names must be non-empty")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("exclusion counts must be non-negative integers")
        normalized_exclusions[name] = count

    selection_audit = build_model_selection_audit(serialized_folds, comparisons)
    oos_robustness_audit = build_oos_period_robustness_audit(serialized_folds, comparisons)
    regime_uncertainty_audit = build_regime_uncertainty_audit(serialized_folds, comparisons, manifest_data.get("fold_regime_annotations"))

    contract = _jsonable(dict(evaluation_contract or {}))
    warning_list = [str(item) for item in warnings]
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "manifest": manifest_data,
        "fold_results": serialized_folds,
        "paired_comparisons": comparisons,
        "model_selection_audit": selection_audit,
        "oos_period_robustness_audit": oos_robustness_audit,
        "regime_uncertainty_audit": regime_uncertainty_audit,
        "exclusions": dict(sorted(normalized_exclusions.items())),
        "evaluation_contract": contract,
        "warnings": warning_list,
        "interpretation": {
            "scope": "OFFLINE_RESEARCH_ONLY",
            "actionable": False,
            "test_metrics": "Per-fold only; do not pool overlapping test windows.",
            "limitations": [
                "Synthetic tests validate software mechanics, not market performance.",
                "Uncertainty depends on the chosen dependence model and block length.",
                "No profitability, calibration, or generalization guarantee is implied.",
                "Final-test metrics must not drive model selection.",
            ],
        },
    }
    report_fingerprint = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return {**payload, "report_sha256": report_fingerprint}


def write_experiment_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write canonical JSON atomically; report hash is verified before writing."""
    data = _jsonable(report)
    if not isinstance(data, dict):
        raise ValueError("report must be a mapping")
    fingerprint = data.get("report_sha256")
    payload = dict(data)
    payload.pop("report_sha256", None)
    actual = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    if not isinstance(fingerprint, str) or fingerprint != actual:
        raise ValueError("report fingerprint missing or invalid")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False,
                                        allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination

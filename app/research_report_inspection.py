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
            # Keep structured diagnostics intact, but validate their JSON compatibility.
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
        report = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {token}")))
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
    interpretation = report.get("interpretation")
    if not isinstance(interpretation, dict) or interpretation.get("actionable") is not False:
        raise ValueError("report must explicitly declare actionable=false")
    if interpretation.get("scope") != "OFFLINE_RESEARCH_ONLY":
        raise ValueError("report scope must be OFFLINE_RESEARCH_ONLY")

    fold_results = report.get("fold_results")
    if not isinstance(fold_results, dict) or not fold_results:
        raise ValueError("fold_results must contain at least one model")
    models: list[dict[str, Any]] = []
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
            summarized_folds.append({
                "fold_index": index,
                "validation": _metrics(fold["validation"], context=f"{model_name} fold {index} validation"),
                "test": _metrics(fold["test"], context=f"{model_name} fold {index} test"),
                "actionable": False,
            })
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
    for comparison in comparisons:
        if comparison.get("actionable", False) is not False:
            raise ValueError("paired comparisons must remain actionable=false")

    warnings = report.get("warnings", [])
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise ValueError("warnings must be a list of strings")

    return {
        "verified": True,
        "run_id": run_id,
        "report_sha256": supplied,
        "schema_version": report["schema_version"],
        "scope": "OFFLINE_RESEARCH_ONLY",
        "actionable": False,
        "models": models,
        "paired_comparisons": comparisons,
        "exclusions": dict(sorted(normalized_exclusions.items())),
        "warnings": warnings,
        "evaluation_contract": report.get("evaluation_contract", {}),
        "interpretation": {
            "pooled_test_metrics": False,
            "limitations": interpretation.get("limitations", []),
            "source_data_authenticated": False,
            "meaning": "Fingerprint verifies report integrity only; it does not establish data correctness, calibration, profitability, or generalization.",
        },
    }

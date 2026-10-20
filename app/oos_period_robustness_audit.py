"""Chronological out-of-sample period robustness diagnostics for offline research reports.

This is a descriptive post-selection audit. It never chooses a model, ranks candidates,
pools observations across folds, infers market regimes, or authorizes execution.
Fold indices are used only as the report's declared chronological order; overlapping
test windows are not independent observations.
"""
from __future__ import annotations

from math import isfinite
from statistics import median
from typing import Any, Mapping, Sequence


def _finite_metric(fold: Mapping[str, Any], key: str) -> float | None:
    test = fold.get("test")
    if not isinstance(test, Mapping):
        return None
    value = test.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        return None
    return float(value)


def build_oos_period_robustness_audit(
    fold_results: Mapping[str, Sequence[Mapping[str, Any]]],
    paired_comparisons: Sequence[Mapping[str, Any]],
    *,
    minimum_test_folds: int = 3,
) -> dict[str, Any]:
    """Describe per-candidate stability across early/middle/late test-fold blocks.

    No candidate is selected or ranked. The audit uses common test-fold indices with
    finite Brier and log-loss for every candidate, then splits those ordered folds into
    three non-overlapping fold groups. It does not claim underlying test observations
    are independent, and does not inspect validation metrics.
    """
    if isinstance(minimum_test_folds, bool) or not isinstance(minimum_test_folds, int) or minimum_test_folds < 3:
        raise ValueError("minimum_test_folds must be an integer of at least 3")

    baseline_names = {
        item.get("baseline_name") for item in paired_comparisons
        if isinstance(item, Mapping) and isinstance(item.get("baseline_name"), str)
    }
    candidates = sorted(name for name in fold_results if name not in baseline_names)
    indexed: dict[str, dict[int, Mapping[str, Any]]] = {}
    for name in candidates:
        indexed[name] = {
            int(row["fold_index"]): row
            for row in fold_results[name]
            if isinstance(row, Mapping)
            and isinstance(row.get("fold_index"), int)
            and not isinstance(row.get("fold_index"), bool)
            and row["fold_index"] >= 0
        }

    common = set.intersection(*(set(indexed[name]) for name in candidates)) if candidates else set()
    complete = [
        index for index in sorted(common)
        if all(
            _finite_metric(indexed[name][index], "brier_score") is not None
            and _finite_metric(indexed[name][index], "log_loss") is not None
            for name in candidates
        )
    ]
    enough = len(complete) >= minimum_test_folds
    period_groups: list[tuple[str, list[int]]] = []
    if enough:
        quotient, remainder = divmod(len(complete), 3)
        cursor = 0
        for position, label in enumerate(("EARLY", "MIDDLE", "LATE")):
            size = quotient + (1 if position < remainder else 0)
            period_groups.append((label, complete[cursor:cursor + size]))
            cursor += size

    rows = []
    if enough:
        for name in candidates:
            per_fold = []
            for index in complete:
                fold = indexed[name][index]
                test = fold.get("test")
                per_fold.append({
                    "fold_index": index,
                    "sample_count": test.get("sample_count") if isinstance(test, Mapping) else None,
                    "brier_score": _finite_metric(fold, "brier_score"),
                    "log_loss": _finite_metric(fold, "log_loss"),
                    "accuracy_at_fixed_half": _finite_metric(fold, "accuracy_at_fixed_half"),
                    "expected_calibration_error": _finite_metric(fold, "expected_calibration_error"),
                })
            periods = []
            for label, indices in period_groups:
                selected = [item for item in per_fold if item["fold_index"] in indices]
                periods.append({
                    "period": label,
                    "fold_indices": indices,
                    "fold_count": len(selected),
                    "median_test_brier_score": float(median(item["brier_score"] for item in selected)),
                    "median_test_log_loss": float(median(item["log_loss"] for item in selected)),
                    "worst_fold_brier_score": max(item["brier_score"] for item in selected),
                    "worst_fold_log_loss": max(item["log_loss"] for item in selected),
                })
            briers = [item["brier_score"] for item in per_fold]
            losses = [item["log_loss"] for item in per_fold]
            rows.append({
                "candidate_name": name,
                "fold_count": len(per_fold),
                "per_fold": per_fold,
                "periods": periods,
                "test_brier_range": max(briers) - min(briers),
                "test_log_loss_range": max(losses) - min(losses),
                "status": "DESCRIPTIVE_PERIOD_DIAGNOSTIC",
            })

    return {
        "protocol": "CHRONOLOGICAL_EARLY_MIDDLE_LATE_TEST_FOLD_DIAGNOSTIC",
        "minimum_test_folds": minimum_test_folds,
        "candidate_count": len(candidates),
        "common_test_fold_indices": complete,
        "period_group_count": len(period_groups),
        "candidates": rows,
        "status": "PERIOD_DIAGNOSTIC_AVAILABLE" if enough and candidates else (
            "NO_CANDIDATES" if not candidates else "INSUFFICIENT_COMMON_TEST_FOLDS"
        ),
        "model_selection_performed": False,
        "test_metrics_used_for_selection": False,
        "actionable": False,
        "interpretation": (
            "Post-selection descriptive robustness check only. Fold indices define the declared chronological "
            "order; groups are not proof of market regimes or statistically independent samples. Test metrics "
            "must not be used to tune or select models. No pooled score, significance test, P&L, or profitability "
            "claim is produced. Overlapping test windows may reuse observations."
        ),
    }

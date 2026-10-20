"""Descriptive fold-to-fold stability diagnostics for offline model reports.

These summaries compare paired model metrics within matching folds. They are not
significance tests, pooled scores, trading returns, or a model-selection rule.
"""
from __future__ import annotations

from math import isfinite
from statistics import median
from typing import Any, Mapping, Sequence


def _finite_metric(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        return None
    return float(value)


def _metric_summary(differences: Sequence[float]) -> dict[str, int | float | str]:
    wins = sum(value < 0 for value in differences)
    losses = sum(value > 0 for value in differences)
    ties = len(differences) - wins - losses
    if wins > losses:
        direction = "CANDIDATE_LOWER_IN_MORE_FOLDS"
    elif losses > wins:
        direction = "BASELINE_LOWER_IN_MORE_FOLDS"
    else:
        direction = "NO_MAJORITY_DIRECTION"
    return {
        "folds_with_metric": len(differences),
        "candidate_wins": wins,
        "ties": ties,
        "candidate_losses": losses,
        "median_candidate_minus_baseline": float(median(differences)),
        "direction": direction,
    }


def summarize_fold_stability(
    fold_results: Mapping[str, Sequence[Mapping[str, Any]]],
    paired_comparisons: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize metric direction across common folds for each reported model pair.

    Negative Brier/log-loss differences favor the candidate. Only matching folds with
    equal sample counts are compared. Results remain per-fold and descriptive; repeated
    or overlapping test windows are not independent observations.
    """
    pair_partitions = sorted({
        (
            str(item["candidate_name"]),
            str(item["baseline_name"]),
            str(item["partition"]),
        )
        for item in paired_comparisons
    })
    indexed: dict[str, dict[int, Mapping[str, Any]]] = {}
    for model_name, folds in fold_results.items():
        indexed[model_name] = {
            int(fold["fold_index"]): fold
            for fold in folds
        }

    summaries: list[dict[str, Any]] = []
    for candidate, baseline, partition in pair_partitions:
        candidate_folds = indexed.get(candidate, {})
        baseline_folds = indexed.get(baseline, {})
        common_indices = sorted(set(candidate_folds) & set(baseline_folds))
        differences: dict[str, list[float]] = {"brier_score": [], "log_loss": []}
        compared_fold_indices: list[int] = []
        for fold_index in common_indices:
            candidate_metrics = candidate_folds[fold_index].get(partition)
            baseline_metrics = baseline_folds[fold_index].get(partition)
            if not isinstance(candidate_metrics, Mapping) or not isinstance(baseline_metrics, Mapping):
                continue
            candidate_count = candidate_metrics.get("sample_count")
            baseline_count = baseline_metrics.get("sample_count")
            if candidate_count != baseline_count:
                raise ValueError(
                    f"fold stability sample counts differ for {candidate}/{baseline} "
                    f"fold {fold_index} {partition}"
                )
            compared_fold_indices.append(fold_index)
            for metric_name in differences:
                candidate_value = _finite_metric(candidate_metrics, metric_name)
                baseline_value = _finite_metric(baseline_metrics, metric_name)
                if candidate_value is not None and baseline_value is not None:
                    differences[metric_name].append(candidate_value - baseline_value)

        if not compared_fold_indices:
            continue
        metrics = {
            name: _metric_summary(values)
            for name, values in differences.items()
            if values
        }
        summaries.append({
            "candidate_name": candidate,
            "baseline_name": baseline,
            "partition": partition,
            "fold_indices": compared_fold_indices,
            "fold_count": len(compared_fold_indices),
            "metrics": metrics,
            "interpretation": "DESCRIPTIVE_FOLD_STABILITY_NOT_SIGNIFICANCE_TEST",
            "actionable": False,
        })
    return summaries

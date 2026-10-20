"""Validation-only model-selection audit for offline research reports.

The audit ranks candidates using validation metrics only. It never reads test metrics,
does not promote models for execution, and withholds a selection when fewer than two
common validation folds have complete metrics.
"""
from __future__ import annotations

from math import isfinite
from statistics import median
from typing import Any, Mapping, Sequence


def _metric(fold: Mapping[str, Any], key: str) -> float | None:
    validation = fold.get("validation")
    if not isinstance(validation, Mapping):
        return None
    value = validation.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        return None
    return float(value)


def build_model_selection_audit(
    fold_results: Mapping[str, Sequence[Mapping[str, Any]]],
    paired_comparisons: Sequence[Mapping[str, Any]],
    *,
    minimum_validation_folds: int = 2,
) -> dict[str, Any]:
    """Build a deterministic selection audit using validation log-loss then Brier.

    Candidate/baseline roles come from paired-comparison metadata when available.
    Only fold indices with finite validation log-loss and Brier for every candidate
    are included. Final test metrics are deliberately never inspected.
    """
    if isinstance(minimum_validation_folds, bool) or not isinstance(minimum_validation_folds, int) or minimum_validation_folds < 2:
        raise ValueError("minimum_validation_folds must be an integer of at least 2")

    baseline_names = {
        item.get("baseline_name") for item in paired_comparisons
        if isinstance(item, Mapping) and isinstance(item.get("baseline_name"), str)
    }
    candidate_names = sorted(name for name in fold_results if name not in baseline_names)
    indexed: dict[str, dict[int, Mapping[str, Any]]] = {}
    for name in candidate_names:
        indexed[name] = {
            int(fold["fold_index"]): fold
            for fold in fold_results[name]
            if isinstance(fold, Mapping)
            and isinstance(fold.get("fold_index"), int)
            and not isinstance(fold.get("fold_index"), bool)
            and fold["fold_index"] >= 0
        }

    if not candidate_names:
        return {
            "protocol": "VALIDATION_ONLY_MEDIAN_LOG_LOSS_THEN_BRIER",
            "minimum_validation_folds": minimum_validation_folds,
            "candidate_count": 0,
            "common_validation_fold_indices": [],
            "ranking": [],
            "selected_candidate": None,
            "status": "NO_CANDIDATES",
            "test_metrics_used_for_selection": False,
            "actionable": False,
        }

    common = set.intersection(*(set(indexed[name]) for name in candidate_names)) if candidate_names else set()
    complete_indices = []
    for index in sorted(common):
        if all(
            _metric(indexed[name][index], "log_loss") is not None
            and _metric(indexed[name][index], "brier_score") is not None
            for name in candidate_names
        ):
            complete_indices.append(index)

    ranking = []
    if len(complete_indices) >= minimum_validation_folds:
        for name in candidate_names:
            log_losses = [_metric(indexed[name][i], "log_loss") for i in complete_indices]
            briers = [_metric(indexed[name][i], "brier_score") for i in complete_indices]
            # Completeness was checked above; these casts preserve the validated values.
            ranking.append({
                "candidate_name": name,
                "validation_fold_count": len(complete_indices),
                "median_validation_log_loss": float(median(value for value in log_losses if value is not None)),
                "median_validation_brier_score": float(median(value for value in briers if value is not None)),
                "rank": 0,
            })
        ranking.sort(key=lambda row: (
            row["median_validation_log_loss"],
            row["median_validation_brier_score"],
            row["candidate_name"],
        ))
        for rank, row in enumerate(ranking, start=1):
            row["rank"] = rank

    enough = len(complete_indices) >= minimum_validation_folds
    return {
        "protocol": "VALIDATION_ONLY_MEDIAN_LOG_LOSS_THEN_BRIER",
        "minimum_validation_folds": minimum_validation_folds,
        "candidate_count": len(candidate_names),
        "common_validation_fold_indices": complete_indices,
        "ranking": ranking,
        "selected_candidate": ranking[0]["candidate_name"] if enough and ranking else None,
        "status": "VALIDATION_RANKING_AVAILABLE" if enough else "INSUFFICIENT_COMMON_VALIDATION_FOLDS",
        "test_metrics_used_for_selection": False,
        "actionable": False,
        "interpretation": (
            "Research-only selection audit; selection ranks candidates by median validation "
            "log-loss, then median validation Brier score. Final test metrics are excluded "
            "from ranking and must remain untouched until the protocol is frozen."
        ),
    }

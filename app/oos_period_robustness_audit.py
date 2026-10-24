"""Chronological out-of-sample period robustness diagnostics for offline research reports.

This is a descriptive post-selection audit. It never chooses a model, ranks candidates,
pools observations across folds, infers market regimes, or authorizes execution.
Fold indices are used only as the report's declared chronological order; overlapping
test windows are not independent observations.
"""
from __future__ import annotations

from datetime import datetime, timezone
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
    number = float(value)
    if key in {"brier_score", "accuracy_at_fixed_half", "expected_calibration_error"}:
        if not 0.0 <= number <= 1.0:
            return None
    elif key == "log_loss" and number < 0.0:
        return None
    return number


def _parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        normalized = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _chronology_integrity(
    indexed: Mapping[str, Mapping[int, Mapping[str, Any]]],
    candidates: Sequence[str],
    fold_indices: Sequence[int],
) -> dict[str, Any]:
    """Cross-check declared fold order against stored test-window timestamps when available."""
    expected = len(candidates) * len(fold_indices)
    observed = 0
    invalid = 0
    missing = 0
    timestamps_by_fold: dict[int, list[datetime]] = {index: [] for index in fold_indices}
    for candidate in candidates:
        for index in fold_indices:
            fold = indexed[candidate][index]
            raw = fold.get("test_start_utc", fold.get("test_start"))
            if raw is None:
                missing += 1
                continue
            parsed = _parse_utc_timestamp(raw)
            if parsed is None:
                invalid += 1
                continue
            observed += 1
            timestamps_by_fold[index].append(parsed)

    if expected == 0:
        status = "NOT_EVALUATED"
    elif observed == 0 and invalid == 0:
        status = "DECLARED_FOLD_ORDER_ONLY"
    elif invalid:
        status = "INVALID_TEST_START_TIMESTAMPS"
    elif missing:
        status = "PARTIAL_TIMESTAMP_COVERAGE"
    else:
        cross_candidate_mismatch = any(
            len(timestamps_by_fold[index]) != len(candidates)
            or len(set(timestamps_by_fold[index])) != 1
            for index in fold_indices
        )
        if cross_candidate_mismatch:
            status = "CROSS_CANDIDATE_WINDOW_MISMATCH"
        else:
            ordered = [timestamps_by_fold[index][0] for index in fold_indices]
            status = (
                "VERIFIED_TEST_START_ORDER"
                if all(left < right for left, right in zip(ordered, ordered[1:]))
                else "FOLD_ORDER_TIMESTAMP_MISMATCH"
            )

    fold_starts = []
    for index in fold_indices:
        values = timestamps_by_fold[index]
        fold_starts.append({
            "fold_index": index,
            "test_start_utc": (
                values[0].strftime("%Y-%m-%dT%H:%M:%SZ")
                if values and len(set(values)) == 1 else None
            ),
            "candidate_timestamp_count": len(values),
        })
    return {
        "status": status,
        "expected_candidate_fold_timestamps": expected,
        "observed_candidate_fold_timestamps": observed,
        "missing_timestamp_count": missing,
        "invalid_timestamp_count": invalid,
        "fold_test_starts": fold_starts,
        "interpretation": (
            "Verified test-window starts establish that fold-index ordering matches strictly increasing UTC test starts."
            if status == "VERIFIED_TEST_START_ORDER"
            else "Without complete consistent UTC test-start timestamps, chronology relies on declared fold indices; "
                 "malformed or contradictory timestamps invalidate the period summaries."
        ),
    }


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
    chronology = _chronology_integrity(indexed, candidates, complete)
    invalid_chronology = chronology["status"] in {
        "INVALID_TEST_START_TIMESTAMPS",
        "CROSS_CANDIDATE_WINDOW_MISMATCH",
        "FOLD_ORDER_TIMESTAMP_MISMATCH",
    }
    if invalid_chronology:
        return {
            "protocol": "CHRONOLOGICAL_EARLY_MIDDLE_LATE_TEST_FOLD_DIAGNOSTIC",
            "minimum_test_folds": minimum_test_folds,
            "candidate_count": len(candidates),
            "common_test_fold_indices": complete,
            "period_group_count": 0,
            "candidates": [],
            "chronology_integrity": chronology,
            "status": "CHRONOLOGY_INVALID",
            "model_selection_performed": False,
            "test_metrics_used_for_selection": False,
            "actionable": False,
            "interpretation": (
                "Period summaries were withheld because stored test-start timestamps are malformed, inconsistent "
                "between candidates, or not strictly increasing with fold index. Correct the source matrix before "
                "using chronological period diagnostics."
            ),
        }

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
                ece_values = [
                    item["expected_calibration_error"]
                    for item in selected
                    if item["expected_calibration_error"] is not None
                ]
                periods.append({
                    "period": label,
                    "fold_indices": indices,
                    "fold_count": len(selected),
                    "median_test_brier_score": float(median(item["brier_score"] for item in selected)),
                    "median_test_log_loss": float(median(item["log_loss"] for item in selected)),
                    "worst_fold_brier_score": max(item["brier_score"] for item in selected),
                    "worst_fold_log_loss": max(item["log_loss"] for item in selected),
                    "ece_fold_count": len(ece_values),
                    "median_test_expected_calibration_error": float(median(ece_values)) if ece_values else None,
                    "maximum_test_expected_calibration_error": max(ece_values) if ece_values else None,
                })
            briers = [item["brier_score"] for item in per_fold]
            losses = [item["log_loss"] for item in per_fold]
            ece_values = [
                item["expected_calibration_error"]
                for item in per_fold
                if item["expected_calibration_error"] is not None
            ]
            rows.append({
                "candidate_name": name,
                "fold_count": len(per_fold),
                "per_fold": per_fold,
                "periods": periods,
                "test_brier_range": max(briers) - min(briers),
                "test_log_loss_range": max(losses) - min(losses),
                "test_expected_calibration_error_range": max(ece_values) - min(ece_values) if ece_values else None,
                "expected_calibration_error_fold_count": len(ece_values),
                "status": "DESCRIPTIVE_PERIOD_DIAGNOSTIC",
            })

    return {
        "protocol": "CHRONOLOGICAL_EARLY_MIDDLE_LATE_TEST_FOLD_DIAGNOSTIC",
        "minimum_test_folds": minimum_test_folds,
        "candidate_count": len(candidates),
        "common_test_fold_indices": complete,
        "period_group_count": len(period_groups),
        "candidates": rows,
        "chronology_integrity": chronology,
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

"""Descriptive temporal stability audit for out-of-sample calibration error.

Requires explicit chronological test boundaries. Does not infer regimes or claim
statistical significance; overlapping folds may not be independent.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


def build_temporal_calibration_stability_audit(
    fold_results: Mapping[str, Any],
    fold_temporal_boundaries: Any,
) -> dict[str, Any]:
    if not isinstance(fold_temporal_boundaries, list):
        fold_temporal_boundaries = []
    boundaries: dict[int, tuple[str, str]] = {}
    for item in fold_temporal_boundaries:
        if not isinstance(item, dict):
            continue
        index = item.get("fold_index")
        start, end = item.get("test_start_utc"), item.get("test_end_utc")
        if (isinstance(index, bool) or not isinstance(index, int) or index < 0
                or not isinstance(start, str) or not isinstance(end, str)):
            continue
        if index in boundaries:
            raise ValueError("duplicate fold index in temporal calibration boundaries")
        if start >= end:
            raise ValueError("temporal calibration test boundaries must have start before end")
        boundaries[index] = (start, end)

    rows: list[dict[str, Any]] = []
    for model_name, folds in sorted(fold_results.items()):
        if not isinstance(model_name, str) or not isinstance(folds, list):
            continue
        for partition in ("validation", "test"):
            periods = []
            for fold in folds:
                if not isinstance(fold, dict):
                    continue
                index = fold.get("fold_index")
                boundary = boundaries.get(index) if isinstance(index, int) and not isinstance(index, bool) else None
                metric = fold.get(partition)
                if boundary is None or not isinstance(metric, dict):
                    continue
                ece = metric.get("expected_calibration_error")
                if isinstance(ece, bool) or not isinstance(ece, (int, float)) or not math.isfinite(float(ece)) or not 0 <= ece <= 1:
                    continue
                uncertainty = metric.get("calibration_uncertainty")
                if not isinstance(uncertainty, dict):
                    uncertainty = {}
                low, high = uncertainty.get("interval_low"), uncertainty.get("interval_high")
                if (isinstance(low, bool) or not isinstance(low, (int, float))
                        or isinstance(high, bool) or not isinstance(high, (int, float))
                        or not math.isfinite(float(low)) or not math.isfinite(float(high))
                        or not 0 <= low <= high <= 1):
                    low, high = None, None
                periods.append({
                    "fold_index": index, "test_start_utc": boundary[0], "test_end_utc": boundary[1],
                    "expected_calibration_error": float(ece),
                    "interval_low": float(low) if low is not None else None,
                    "interval_high": float(high) if high is not None else None,
                    "sample_count": metric.get("sample_count"),
                })
            periods.sort(key=lambda row: (row["test_start_utc"], row["fold_index"]))
            eces = [row["expected_calibration_error"] for row in periods]
            if len(periods) < 3:
                status = "INSUFFICIENT_TEMPORAL_COVERAGE"
            else:
                first, last = periods[0], periods[-1]
                if first["interval_low"] is None or last["interval_low"] is None:
                    status = "NO_CLEAR_ENDPOINT_SEPARATION_INTERVALS_UNAVAILABLE"
                elif last["interval_low"] > first["interval_high"]:
                    status = "ENDPOINT_ECE_INCREASE_INTERVALS_DISJOINT"
                elif last["interval_high"] < first["interval_low"]:
                    status = "ENDPOINT_ECE_DECREASE_INTERVALS_DISJOINT"
                else:
                    status = "NO_CLEAR_ENDPOINT_SEPARATION"
            adjacent_overlap = 0
            adjacent_comparable = 0
            for left, right in zip(periods, periods[1:]):
                if left["interval_low"] is None or right["interval_low"] is None:
                    continue
                adjacent_comparable += 1
                if max(left["interval_low"], right["interval_low"]) <= min(left["interval_high"], right["interval_high"]):
                    adjacent_overlap += 1
            rows.append({
                "model_name": model_name,
                "partition": partition,
                "period_count": len(periods),
                "fold_indices": [row["fold_index"] for row in periods],
                "periods": periods,
                "first_period_ece": eces[0] if eces else None,
                "last_period_ece": eces[-1] if eces else None,
                "first_to_last_ece_delta": eces[-1] - eces[0] if len(eces) >= 2 else None,
                "ece_min": min(eces) if eces else None,
                "ece_max": max(eces) if eces else None,
                "ece_range": max(eces) - min(eces) if eces else None,
                "adjacent_interval_overlap_count": adjacent_overlap,
                "adjacent_interval_comparison_count": adjacent_comparable,
                "status": status,
                "actionable": False,
            })
    return {
        "comparisons": rows,
        "interpretation": (
            "Descriptive chronological ECE changes only. Endpoint interval separation is not a significance test; "
            "folds may overlap and are not necessarily independent. No regime, calibration, profitability, or "
            "generalization guarantee is inferred. Rows without explicit temporal boundaries are omitted."
        ),
        "actionable": False,
    }

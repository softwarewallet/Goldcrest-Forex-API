"""Derived audit of moving-block bootstrap interval sensitivity."""
from __future__ import annotations
import math
from typing import Any, Mapping, Sequence


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _direction(low: float, high: float, field: str) -> str:
    if low > high:
        raise ValueError(f"{field} lower bound must not exceed upper bound")
    return "CANDIDATE_LOWER" if high < 0 else "CANDIDATE_HIGHER" if low > 0 else "INCLUDES_ZERO"


def build_block_length_sensitivity_audit(paired_comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = []
    for row in paired_comparisons:
        if not isinstance(row, Mapping):
            raise ValueError("paired comparisons must be objects")
        comparison = row.get("comparison")
        if not isinstance(comparison, Mapping):
            continue
        raw = comparison.get("block_length_sensitivity")
        if raw is None:
            continue
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("block_length_sensitivity must be a non-empty list")
        primary_length = comparison.get("block_length")
        if isinstance(primary_length, bool) or not isinstance(primary_length, int) or primary_length < 1:
            raise ValueError("primary block_length must be a positive integer")
        entries, seen = [], set()
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("each block-length sensitivity result must be an object")
            length = item.get("block_length")
            if isinstance(length, bool) or not isinstance(length, int) or length < 1 or length in seen:
                raise ValueError("sensitivity block lengths must be unique positive integers")
            seen.add(length)
            blo = _number(item.get("brier_difference_ci_low"), "Brier lower interval bound")
            bhi = _number(item.get("brier_difference_ci_high"), "Brier upper interval bound")
            llo = _number(item.get("log_loss_difference_ci_low"), "log-loss lower interval bound")
            lhi = _number(item.get("log_loss_difference_ci_high"), "log-loss upper interval bound")
            bd, ld = _direction(blo, bhi, "Brier"), _direction(llo, lhi, "log-loss")
            if item.get("brier_direction") != bd or item.get("log_loss_direction") != ld:
                raise ValueError("block-length sensitivity direction does not match its interval bounds")
            entries.append({"block_length": length, "brier_difference_ci_low": blo,
                "brier_difference_ci_high": bhi, "log_loss_difference_ci_low": llo,
                "log_loss_difference_ci_high": lhi, "brier_direction": bd, "log_loss_direction": ld})
        if primary_length not in seen:
            raise ValueError("sensitivity results must include the primary block length")
        primary = next(item for item in entries if item["block_length"] == primary_length)
        for low_key, high_key in (("brier_difference_ci_low", "brier_difference_ci_high"),
                                  ("log_loss_difference_ci_low", "log_loss_difference_ci_high")):
            if not math.isclose(_number(comparison.get(low_key), low_key), primary[low_key], rel_tol=0, abs_tol=1e-12):
                raise ValueError("primary interval does not match its block-length sensitivity entry")
            if not math.isclose(_number(comparison.get(high_key), high_key), primary[high_key], rel_tol=0, abs_tol=1e-12):
                raise ValueError("primary interval does not match its block-length sensitivity entry")
        entries.sort(key=lambda item: item["block_length"])
        if len(entries) < 2:
            status = "INSUFFICIENT_BLOCK_LENGTH_VARIATION"
        elif len({item["brier_direction"] for item in entries}) == 1 and len({item["log_loss_direction"] for item in entries}) == 1:
            status = "DIRECTION_STABLE_ACROSS_TESTED_BLOCK_LENGTHS"
        else:
            status = "SENSITIVE_TO_BLOCK_LENGTH_CHOICE"
        output.append({"candidate_name": row.get("candidate_name"), "baseline_name": row.get("baseline_name"),
            "fold_index": row.get("fold_index"), "partition": row.get("partition"), "sample_count": row.get("sample_count"),
            "tested_block_lengths": [item["block_length"] for item in entries], "status": status,
            "sensitivity": entries, "actionable": False})
    return {"protocol": "PAIRED_LOSS_MOVING_BLOCK_LENGTH_SENSITIVITY", "comparison_count": len(output),
        "direction_stable_count": sum(x["status"] == "DIRECTION_STABLE_ACROSS_TESTED_BLOCK_LENGTHS" for x in output),
        "sensitive_count": sum(x["status"] == "SENSITIVE_TO_BLOCK_LENGTH_CHOICE" for x in output),
        "insufficient_variation_count": sum(x["status"] == "INSUFFICIENT_BLOCK_LENGTH_VARIATION" for x in output),
        "comparisons": output, "model_selection_performed": False, "test_metrics_used_for_selection": False,
        "actionable": False,
        "interpretation": ("The same paired out-of-sample loss differences are resampled using shorter, configured, and longer "
        "contiguous block lengths where available. Direction stability means only that percentile intervals have the same "
        "sign classification at the tested lengths; it is not statistical significance or proof of robustness to all "
        "dependence structures. This does not correct selection bias, multiple testing, nonstationarity, overlapping folds, "
        "data defects, or feature leakage. It does not select a block length or model, pool folds, estimate trading P&L, "
        "or authorize execution.")}

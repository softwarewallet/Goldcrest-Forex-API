"""Audit sample adequacy and class balance for calibration diagnostics."""
from __future__ import annotations
from typing import Any, Mapping


def build_calibration_sample_adequacy_audit(fold_results: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model_name, folds in sorted(fold_results.items()):
        if not isinstance(model_name, str) or not isinstance(folds, list):
            continue
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            fold_index = fold.get("fold_index")
            if isinstance(fold_index, bool) or not isinstance(fold_index, int) or fold_index < 0:
                continue
            for partition in ("validation", "test"):
                metric = fold.get(partition)
                if not isinstance(metric, dict):
                    continue
                n, positive, negative = metric.get("sample_count"), metric.get("positive_count"), metric.get("negative_count")
                valid_n = isinstance(n, int) and not isinstance(n, bool) and n > 0
                valid_classes = (valid_n and isinstance(positive, int) and not isinstance(positive, bool)
                    and isinstance(negative, int) and not isinstance(negative, bool)
                    and positive >= 0 and negative >= 0 and positive + negative == n)
                bins = metric.get("reliability_bins")
                bin_counts = []
                if isinstance(bins, list):
                    for item in bins:
                        if isinstance(item, dict):
                            count = item.get("count")
                            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                                bin_counts.append(count)
                populated_bins = [count for count in bin_counts if count > 0]
                sparse_bins = sum(count < 5 for count in populated_bins)
                if not valid_n: status = "SAMPLE_COUNT_UNAVAILABLE_OR_INVALID"
                elif n < 30: status = "LOW_SAMPLE_COUNT"
                elif not valid_classes: status = "CLASS_COUNTS_UNAVAILABLE_OR_INCONSISTENT"
                elif min(positive, negative) < 10: status = "LOW_MINOR_CLASS_COUNT"
                elif populated_bins and sparse_bins: status = "SPARSE_RELIABILITY_BINS"
                else: status = "COUNTS_MEET_MINIMUM_SCREEN"
                rows.append({
                    "model_name": model_name, "fold_index": fold_index, "partition": partition,
                    "sample_count": n if valid_n else None,
                    "positive_count": positive if valid_classes else None,
                    "negative_count": negative if valid_classes else None,
                    "positive_prevalence": positive / n if valid_classes else None,
                    "populated_reliability_bins": len(populated_bins),
                    "sparse_populated_reliability_bins": sparse_bins,
                    "status": status, "actionable": False,
                })
    rows.sort(key=lambda row: (row["model_name"], row["fold_index"], row["partition"]))
    return {
        "minimum_sample_count_screen": 30, "minimum_minor_class_count_screen": 10,
        "minimum_populated_bin_count_screen": 5, "comparisons": rows,
        "interpretation": ("Descriptive sample-size, class-balance, and populated-bin screening only. "
            "Passing these screens does not prove calibration; ECE remains bin-dependent, and low-count "
            "bins can make apparent calibration unstable. No samples are discarded, no thresholds are tuned, "
            "and no model is promoted."),
        "actionable": False,
    }

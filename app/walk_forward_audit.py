"""Auditable temporal-integrity summary for offline walk-forward research.

The audit records fold boundaries, label maturity at each training cutoff, and repeated
out-of-sample timestamps. It does not prove that externally supplied features are causal
or that the source dataset is authentic.
"""
from __future__ import annotations

from datetime import timezone
from typing import Mapping, Sequence

from app.model_walk_forward import TimedBinaryExample
from app.walk_forward import WalkForwardPlan


def _utc_iso(value) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("walk-forward timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def audit_walk_forward_plan(
    plan: WalkForwardPlan,
    examples: Sequence[TimedBinaryExample],
) -> dict[str, object]:
    """Describe split boundaries and identify repeated test observations.

    A repeated test timestamp across folds is reported, not silently deduplicated:
    overlapping test windows are possible when step_bars is smaller than test_bars.
    Fold-level metrics must not be pooled as if those observations were independent.
    """
    if not plan.folds:
        raise ValueError("walk-forward plan contains no folds")
    by_time: dict[object, TimedBinaryExample] = {}
    for example in examples:
        timestamp = example.timestamp
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("example timestamps must be timezone-aware")
        key = timestamp.astimezone(timezone.utc)
        if key in by_time:
            raise ValueError("examples contain duplicate normalized timestamps")
        by_time[key] = example

    seen_fold_indices: set[int] = set()
    test_timestamp_counts: dict[object, int] = {}
    fold_reports: list[dict[str, object]] = []
    total_test_observations = 0
    total_train_rows = 0
    total_matured_train_rows = 0
    total_unmatured_train_rows = 0
    total_ineligible_train_rows = 0
    total_ineligible_validation_rows = 0
    total_ineligible_test_rows = 0

    for fold in plan.folds:
        if fold.fold_index in seen_fold_indices:
            raise ValueError("walk-forward plan contains duplicate fold indices")
        seen_fold_indices.add(fold.fold_index)
        train_times = [bar.timestamp.astimezone(timezone.utc) for bar in fold.train]
        validation_times = [bar.timestamp.astimezone(timezone.utc) for bar in fold.validation]
        test_times = [bar.timestamp.astimezone(timezone.utc) for bar in fold.test]
        if not train_times or not validation_times or not test_times:
            raise ValueError("each fold requires non-empty train, validation, and test partitions")
        if len(set(train_times + validation_times + test_times)) != len(train_times + validation_times + test_times):
            raise ValueError("fold partitions share observation timestamps")
        if max(train_times) >= min(validation_times) or max(validation_times) >= min(test_times):
            raise ValueError("fold partitions are not strictly chronological")
        required = set(train_times + validation_times + test_times)
        missing = required - by_time.keys()
        if missing:
            raise ValueError(f"fold references {len(missing)} timestamps missing from examples")

        train_cutoff = max(train_times)
        train_examples = [by_time[t] for t in train_times]
        eligible_train = [row for row in train_examples if row.label_eligible]
        matured = [
            row for row in eligible_train
            if row.label_available_at.astimezone(timezone.utc) <= train_cutoff
        ]
        unmatured = len(eligible_train) - len(matured)
        validation_examples = [by_time[t] for t in validation_times]
        test_examples = [by_time[t] for t in test_times]
        ineligible_validation = sum(not row.label_eligible for row in validation_examples)
        ineligible_test = sum(not row.label_eligible for row in test_examples)
        for timestamp in test_times:
            test_timestamp_counts[timestamp] = test_timestamp_counts.get(timestamp, 0) + 1

        total_test_observations += len(test_times)
        total_train_rows += len(train_examples)
        total_matured_train_rows += len(matured)
        total_unmatured_train_rows += unmatured
        total_ineligible_train_rows += len(train_examples) - len(eligible_train)
        total_ineligible_validation_rows += ineligible_validation
        total_ineligible_test_rows += ineligible_test
        fold_reports.append({
            "fold_index": fold.fold_index,
            "train": {
                "start_utc": _utc_iso(fold.train[0].timestamp),
                "end_utc": _utc_iso(fold.train[-1].timestamp),
                "rows_available": len(train_examples),
                "rows_label_eligible": len(eligible_train),
                "rows_excluded_ineligible": len(train_examples) - len(eligible_train),
                "rows_label_matured_at_cutoff": len(matured),
                "rows_excluded_unmatured": unmatured,
                "label_maturity_cutoff_utc": _utc_iso(fold.train[-1].timestamp),
            },
            "validation": {
                "start_utc": _utc_iso(fold.validation[0].timestamp),
                "end_utc": _utc_iso(fold.validation[-1].timestamp),
                "rows": len(validation_times),
                "rows_eligible": len(validation_examples) - ineligible_validation,
                "rows_excluded_ineligible": ineligible_validation,
            },
            "test": {
                "start_utc": _utc_iso(fold.test[0].timestamp),
                "end_utc": _utc_iso(fold.test[-1].timestamp),
                "rows": len(test_times),
                "rows_eligible": len(test_examples) - ineligible_test,
                "rows_excluded_ineligible": ineligible_test,
            },
            "purged_rows": int(fold.purged_rows),
            "partitions_disjoint": True,
            "strictly_chronological": True,
            "actionable": False,
        })

    unique_test_timestamps = len(test_timestamp_counts)
    repeated_test_timestamps = total_test_observations - unique_test_timestamps
    overlapping_timestamps = sorted(
        (_utc_iso(timestamp) for timestamp, count in test_timestamp_counts.items() if count > 1)
    )
    return {
        "schema_version": 1,
        "scope": "OFFLINE_RESEARCH_ONLY",
        "fold_count": len(fold_reports),
        "folds": fold_reports,
        "training_rows_available_across_folds": total_train_rows,
        "training_rows_label_matured_at_cutoff_across_folds": total_matured_train_rows,
        "training_rows_excluded_unmatured_across_folds": total_unmatured_train_rows,
        "training_rows_excluded_ineligible_across_folds": total_ineligible_train_rows,
        "validation_rows_excluded_ineligible_across_folds": total_ineligible_validation_rows,
        "test_rows_excluded_ineligible_across_folds": total_ineligible_test_rows,
        "test_observations_across_folds": total_test_observations,
        "unique_test_timestamps": unique_test_timestamps,
        "repeated_test_timestamp_observations": repeated_test_timestamps,
        "test_windows_overlap": repeated_test_timestamps > 0,
        "overlapping_test_timestamps_utc": overlapping_timestamps[:100],
        "overlapping_test_timestamps_truncated": len(overlapping_timestamps) > 100,
        "interpretation": (
            "Test timestamps repeat across folds; fold metrics are not independent and "
            "must not be pooled as if each observation appeared once."
            if repeated_test_timestamps
            else "No test timestamp repeats across folds; this does not remove all model-selection or dataset risks."
        ),
        "feature_causality_verified": False,
        "provider_authenticity_verified": False,
        "actionable": False,
    }

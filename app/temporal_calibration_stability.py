"""Descriptive temporal stability audit for out-of-sample calibration error.

Requires explicit chronological test boundaries. Does not infer regimes or claim
statistical significance; overlapping folds may not be independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_temporal_calibration_stability_audit(
    fold_results: Mapping[str, Any],
    fold_temporal_boundaries: Any,
    *,
    partitions: Sequence[str] = ("validation", "test"),
) -> dict[str, Any]:
    """Describe ECE stability in explicitly requested validation/test partitions."""
    if isinstance(partitions, (str, bytes)) or not isinstance(partitions, Sequence):
        raise ValueError("partitions must be a sequence containing validation and/or test")
    requested_partitions = tuple(partitions)
    if (
        not requested_partitions
        or any(
            not isinstance(item, str) or item not in {"validation", "test"}
            for item in requested_partitions
        )
        or len(set(requested_partitions)) != len(requested_partitions)
    ):
        raise ValueError("partitions must contain unique values from validation and test")
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
        for partition in requested_partitions:
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


def _parse_utc_boundary(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _read_experiment_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment matrix must contain a JSON object")
    return payload


def build_temporal_calibration_stability_report(
    matrix: Mapping[str, Any],
    *,
    matrix_path: str | Path,
) -> dict[str, Any]:
    """Adapt the repository's stored experiment-matrix schema to this audit."""
    errors: list[str] = []
    warnings: list[str] = []
    matrix_file = Path(matrix_path)
    matrix_sha256 = hashlib.sha256(matrix_file.read_bytes()).hexdigest()

    for field in ("actionable", "automatic_winner_selection", "test_metrics_used_for_selection"):
        if matrix.get(field) is not False:
            errors.append(f"{field}_must_be_false")

    claimed_payload_hash = matrix.get("report_payload_sha256")
    verified_payload_hash: bool | None = None
    if isinstance(claimed_payload_hash, str):
        canonical_payload = dict(matrix)
        canonical_payload.pop("report_payload_sha256", None)
        try:
            canonical = json.dumps(
                canonical_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            verified_payload_hash = (
                hashlib.sha256(canonical.encode("utf-8")).hexdigest() == claimed_payload_hash
            )
        except (TypeError, ValueError):
            verified_payload_hash = False
        if verified_payload_hash is False:
            errors.append("experiment_matrix_report_payload_sha256_mismatch")
    else:
        warnings.append("experiment_matrix_report_payload_sha256_missing")

    raw_windows = matrix.get("common_fold_test_windows")
    boundaries: list[dict[str, Any]] = []
    windows_by_index: dict[int, dict[str, Any]] = {}
    if not isinstance(raw_windows, list) or not raw_windows:
        errors.append("common_fold_test_windows_must_be_nonempty_list")
        raw_windows = []
    for position, window in enumerate(raw_windows):
        if not isinstance(window, dict):
            errors.append(f"common_fold_test_window_{position}_must_be_object")
            continue
        fold_index = window.get("fold_index", position)
        start = window.get("test_start_utc")
        end = window.get("test_end_utc")
        if isinstance(fold_index, bool) or not isinstance(fold_index, int) or fold_index < 0:
            errors.append(f"common_fold_test_window_{position}_invalid_fold_index")
            continue
        if fold_index in windows_by_index:
            errors.append(f"duplicate_common_fold_index:{fold_index}")
            continue
        try:
            parsed_start = _parse_utc_boundary(start, "test_start_utc")
            parsed_end = _parse_utc_boundary(end, "test_end_utc")
            if parsed_start >= parsed_end:
                raise ValueError("test_start_utc must be before test_end_utc")
        except ValueError as exc:
            errors.append(f"common_fold_test_window_{position}_invalid_boundary:{exc}")
            continue
        normalized = {
            "fold_index": fold_index,
            "test_start_utc": str(start),
            "test_end_utc": str(end),
        }
        boundaries.append(normalized)
        windows_by_index[fold_index] = normalized

    raw_experiments = matrix.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        errors.append("experiments_must_be_nonempty_list")
        raw_experiments = []

    normalized_experiments: dict[str, list[dict[str, Any]]] = {}
    for experiment_position, experiment in enumerate(raw_experiments):
        if not isinstance(experiment, dict):
            errors.append(f"experiment_{experiment_position}_must_be_object")
            continue
        experiment_id = experiment.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            errors.append(f"experiment_{experiment_position}_missing_id")
            continue
        if experiment_id in normalized_experiments:
            errors.append(f"duplicate_experiment_id:{experiment_id}")
            continue
        if experiment.get("actionable") is not False:
            errors.append(f"experiment_{experiment_id}_actionable_must_be_false")
        if experiment.get("test_metrics_used_for_tuning") is not False:
            errors.append(f"experiment_{experiment_id}_test_metrics_used_for_tuning_must_be_false")
        raw_folds = experiment.get("fold_results")
        if not isinstance(raw_folds, list) or not raw_folds:
            errors.append(f"experiment_{experiment_id}_fold_results_must_be_nonempty_list")
            continue

        converted_folds: list[dict[str, Any]] = []
        seen_fold_indices: set[int] = set()
        for fold_position, fold in enumerate(raw_folds):
            if not isinstance(fold, dict):
                errors.append(f"experiment_{experiment_id}_fold_{fold_position}_must_be_object")
                continue
            fold_index = fold.get("fold_index", fold_position)
            if isinstance(fold_index, bool) or not isinstance(fold_index, int) or fold_index < 0:
                errors.append(f"experiment_{experiment_id}_fold_{fold_position}_invalid_fold_index")
                continue
            if fold_index in seen_fold_indices:
                errors.append(f"experiment_{experiment_id}_duplicate_fold_index:{fold_index}")
                continue
            seen_fold_indices.add(fold_index)
            common_window = windows_by_index.get(fold_index)
            if common_window is None:
                errors.append(f"experiment_{experiment_id}_fold_{fold_index}_missing_common_test_window")
                continue
            try:
                fold_start = _parse_utc_boundary(fold.get("test_start_utc"), "fold test_start_utc")
                fold_end = _parse_utc_boundary(fold.get("test_end_utc"), "fold test_end_utc")
                common_start = _parse_utc_boundary(common_window["test_start_utc"], "common test_start_utc")
                common_end = _parse_utc_boundary(common_window["test_end_utc"], "common test_end_utc")
                if fold_start != common_start or fold_end != common_end:
                    raise ValueError("fold boundary does not match common test window")
            except ValueError as exc:
                errors.append(f"experiment_{experiment_id}_fold_{fold_index}_boundary_mismatch:{exc}")
                continue
            metrics = fold.get("test_metrics")
            if not isinstance(metrics, dict):
                errors.append(f"experiment_{experiment_id}_fold_{fold_index}_test_metrics_missing")
                continue
            converted_folds.append({"fold_index": fold_index, "test": metrics})
        normalized_experiments[experiment_id] = converted_folds

    comparisons: list[dict[str, Any]] = []
    if not errors:
        try:
            audit = build_temporal_calibration_stability_audit(
                normalized_experiments,
                boundaries,
                partitions=("test",),
            )
            comparisons = audit["comparisons"]
            for item in comparisons:
                if item["period_count"] < 3:
                    warnings.append(
                        f"insufficient_temporal_coverage:{item['model_name']}:test"
                    )
                if item["status"] in {
                    "NO_CLEAR_ENDPOINT_SEPARATION_INTERVALS_UNAVAILABLE",
                }:
                    warnings.append(
                        f"calibration_uncertainty_intervals_unavailable:{item['model_name']}:test"
                    )
        except (TypeError, ValueError) as exc:
            errors.append(f"temporal_calibration_audit_failed:{type(exc).__name__}:{exc}")

    status = "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS"
    return {
        "schema_version": 1,
        "scope": "GOLDCREST_TEMPORAL_CALIBRATION_STABILITY_AUDIT",
        "status": status,
        "run_id": matrix.get("run_id"),
        "symbol": matrix.get("symbol"),
        "timeframe": matrix.get("timeframe"),
        "horizon_bars": matrix.get("horizon_bars"),
        "experiment_matrix_path": str(matrix_file.resolve()),
        "experiment_matrix_sha256": matrix_sha256,
        "experiment_matrix_payload_hash_verified": verified_payload_hash,
        "experiment_count": len(normalized_experiments),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "automatic_model_selection": False,
        "test_metrics_used_for_selection": False,
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "Descriptive chronological test-set calibration-error stability only. "
            "Endpoint interval separation is not a significance test; folds may be dependent. "
            "This audit does not fit a calibrator, rank models, estimate profitability, or authorize execution."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit chronological stability of out-of-sample calibration error from a "
            "stored Goldcrest experiment-matrix report. This is descriptive only."
        )
    )
    parser.add_argument(
        "--experiment-matrix",
        required=True,
        help="Path to <run-id>_experiment_matrix.json produced by the offline research pipeline.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for a new JSON report; existing files are never overwritten.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.experiment_matrix).expanduser()
    try:
        matrix = _read_experiment_matrix(input_path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        parser.error(f"cannot read valid experiment matrix: {exc}")

    output_path: Path | None = Path(args.output).expanduser() if args.output else None
    if output_path is not None and output_path.exists():
        parser.error(f"refusing to overwrite existing report: {output_path}")

    try:
        report = build_temporal_calibration_stability_report(matrix, matrix_path=input_path)
    except OSError as exc:
        parser.error(f"cannot fingerprint experiment matrix: {exc}")

    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Report written to: {output_path}", file=sys.stderr)
    print(rendered)
    return 3 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

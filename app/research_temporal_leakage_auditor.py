"""Offline temporal boundary audit for research experiment-matrix reports."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a non-empty ISO-8601 string")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include timezone")
    return parsed


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def audit_temporal_integrity(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    declared_hash = payload.get("report_payload_sha256")
    hash_payload = dict(payload)
    hash_payload.pop("report_payload_sha256", None)
    if not isinstance(declared_hash, str) or declared_hash != _canonical_hash(hash_payload):
        errors.append("report_payload_sha256_mismatch")
    for field in ("actionable", "automatic_winner_selection", "test_metrics_used_for_selection"):
        if payload.get(field) is not False:
            errors.append(f"report_{field}_must_be_false")

    windows = payload.get("common_fold_test_windows")
    if not isinstance(windows, list):
        errors.append("common_fold_test_windows_must_be_list")
        windows = []
    normalized: dict[int, tuple[datetime, datetime]] = {}
    for pos, window in enumerate(windows):
        if not isinstance(window, dict):
            errors.append(f"common_window_{pos}_must_be_object")
            continue
        idx = window.get("fold_index")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
            errors.append(f"common_window_{pos}_invalid_fold_index")
            continue
        if idx in normalized:
            errors.append(f"duplicate_common_fold_index:{idx}")
            continue
        try:
            start, end = _timestamp(window.get("test_start_utc")), _timestamp(window.get("test_end_utc"))
            if end < start:
                errors.append(f"common_window_{idx}_end_before_start")
            normalized[idx] = (start, end)
        except (TypeError, ValueError):
            errors.append(f"common_window_{idx}_invalid_timestamp")
    if normalized and sorted(normalized) != list(range(len(normalized))):
        errors.append("common_fold_indices_not_contiguous_from_zero")

    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        errors.append("experiments_must_be_list")
        experiments = []
    if not experiments:
        warnings.append("no_experiments_to_audit")
    for ei, exp in enumerate(experiments):
        prefix = f"experiment_{ei}"
        if not isinstance(exp, dict):
            errors.append(f"{prefix}_must_be_object")
            continue
        if exp.get("actionable") is not False:
            errors.append(f"{prefix}_actionable_must_be_false")
        if exp.get("test_metrics_used_for_tuning") is not False:
            errors.append(f"{prefix}_test_metrics_used_for_tuning_must_be_false")
        folds = exp.get("fold_results")
        if not isinstance(folds, list):
            errors.append(f"{prefix}_fold_results_must_be_list")
            continue
        seen: set[int] = set()
        for fi, fold in enumerate(folds):
            fp = f"{prefix}_fold_{fi}"
            if not isinstance(fold, dict):
                errors.append(f"{fp}_must_be_object")
                continue
            idx = fold.get("fold_index")
            if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
                errors.append(f"{fp}_invalid_fold_index")
                continue
            if idx in seen:
                errors.append(f"{fp}_duplicate_fold_index")
            seen.add(idx)
            try:
                ts = [_timestamp(fold.get(k)) for k in (
                    "train_start_utc", "train_end_utc", "validation_start_utc",
                    "validation_end_utc", "test_start_utc", "test_end_utc")]
            except (TypeError, ValueError):
                errors.append(f"{fp}_invalid_or_timezone_naive_boundary")
                continue
            if not (ts[0] <= ts[1] < ts[2] <= ts[3] < ts[4] <= ts[5]):
                errors.append(f"{fp}_partition_boundaries_overlap_or_out_of_order")
            if idx not in normalized:
                errors.append(f"{fp}_missing_common_test_window")
            elif (ts[4], ts[5]) != normalized[idx]:
                errors.append(f"{fp}_test_window_differs_from_common_plan")
            for field in ("train_rows_excluded_unmatured", "refit_rows_excluded_unmatured"):
                if field in fold:
                    value = fold[field]
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        errors.append(f"{fp}_invalid_{field}")
        if normalized and seen != set(normalized):
            errors.append(f"{prefix}_fold_indices_do_not_match_common_plan")

    return {
        "schema_version": 1, "run_id": payload.get("run_id"),
        "status": "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS",
        "errors": sorted(set(errors)), "warnings": sorted(set(warnings)),
        "audited_experiment_count": len(experiments), "audited_common_fold_count": len(normalized),
        "actionable": False, "execution_connected": False,
        "interpretation": "Declared temporal boundaries and report integrity only; not proof of row-level feature availability, absence of all leakage, or profitability."
    }


def audit_temporal_integrity_file(source: str | Path, destination: str | Path | None = None) -> dict[str, Any]:
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment matrix report must be a JSON object")
    result = audit_temporal_integrity(payload)
    result["source_report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if destination is not None:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit temporal boundaries in an offline experiment report.")
    parser.add_argument("--input-report", required=True)
    parser.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_temporal_integrity_file(args.input_report, args.receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())

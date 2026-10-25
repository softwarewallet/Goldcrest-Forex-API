"""Offline audit of research-pipeline leakage boundaries.

Input is a JSON manifest describing chronological splits, fitted preprocessing, and
label/target horizons. This tool reports structural risks only; it does not inspect
model weights, train models, or connect to trading execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def audit_pipeline_leakage_boundaries(manifest: dict[str, Any], *, source_sha256: str) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest_must_be_object")
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in source_sha256)):
        raise ValueError("source_sha256_invalid")

    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("splits_must_be_object")

    normalized: dict[str, dict[str, int]] = {}
    for name in ("train", "validation", "test"):
        segment = splits.get(name)
        if segment is None and name != "train":
            continue
        if not isinstance(segment, dict):
            errors.append({"code": "split_missing_or_invalid", "split": name})
            continue
        start, end = segment.get("start_index"), segment.get("end_index")
        if not _nonnegative_int(start) or not _nonnegative_int(end) or end < start:
            errors.append({"code": "split_bounds_invalid", "split": name})
            continue
        normalized[name] = {"start_index": start, "end_index": end}

    ordered_names = [name for name in ("train", "validation", "test") if name in normalized]
    for previous, current in zip(ordered_names, ordered_names[1:]):
        left, right = normalized[previous], normalized[current]
        if right["start_index"] <= left["end_index"]:
            errors.append({
                "code": "split_overlap_or_nonchronological",
                "previous_split": previous,
                "current_split": current,
                "previous_end_index": left["end_index"],
                "current_start_index": right["start_index"],
            })
        elif right["start_index"] > left["end_index"] + 1:
            warnings.append(
                f"gap_between_splits:{previous}:{current}:"
                f"{right['start_index'] - left['end_index'] - 1}_rows"
            )

    train = normalized.get("train")
    preprocessing_results: list[dict[str, Any]] = []
    preprocessing = manifest.get("preprocessing", [])
    if not isinstance(preprocessing, list):
        errors.append({"code": "preprocessing_must_be_list"})
        preprocessing = []
    for index, item in enumerate(preprocessing):
        if not isinstance(item, dict):
            errors.append({"code": "preprocessing_record_invalid", "record_index": index})
            continue
        name = str(item.get("name") or f"preprocessing_{index}")
        fit_start, fit_end = item.get("fit_start_index"), item.get("fit_end_index")
        record_errors = []
        if not _nonnegative_int(fit_start) or not _nonnegative_int(fit_end) or fit_end < fit_start:
            record_errors.append("fit_bounds_invalid")
        elif train and fit_end > train["end_index"]:
            record_errors.append("fit_uses_rows_after_training_boundary")
        elif train and fit_start < train["start_index"]:
            record_errors.append("fit_starts_before_training_split")
        applied = item.get("applied_to_splits", [])
        if not isinstance(applied, list) or any(value not in ("train", "validation", "test") for value in applied):
            record_errors.append("applied_to_splits_invalid")
            applied = []
        if any(value in ("validation", "test") for value in applied) and not item.get("fit_only_on_train", True):
            record_errors.append("preprocessing_fit_scope_not_train_only")
        for code in record_errors:
            errors.append({"code": code, "preprocessing": name, "record_index": index})
        preprocessing_results.append({
            "name": name,
            "fit_start_index": fit_start,
            "fit_end_index": fit_end,
            "applied_to_splits": applied,
            "passed": not record_errors,
            "errors": record_errors,
        })

    horizon = manifest.get("label_horizon_bars", 0)
    if not _nonnegative_int(horizon):
        errors.append({"code": "label_horizon_bars_invalid"})
        horizon = 0
    label_results: list[dict[str, Any]] = []
    labels = manifest.get("labels", [])
    if not isinstance(labels, list):
        errors.append({"code": "labels_must_be_list"})
        labels = []
    for index, item in enumerate(labels):
        if not isinstance(item, dict):
            errors.append({"code": "label_record_invalid", "record_index": index})
            continue
        source_index, target_index = item.get("source_index"), item.get("target_index")
        split_name = item.get("split")
        record_errors = []
        if not _nonnegative_int(source_index) or not _nonnegative_int(target_index) or target_index < source_index:
            record_errors.append("label_indices_invalid")
        elif target_index - source_index < horizon:
            record_errors.append("label_horizon_shorter_than_declared")
        segment = normalized.get(split_name) if isinstance(split_name, str) else None
        if segment and _nonnegative_int(source_index) and _nonnegative_int(target_index):
            if not (segment["start_index"] <= source_index <= segment["end_index"]):
                record_errors.append("label_source_outside_declared_split")
            if not (segment["start_index"] <= target_index <= segment["end_index"]):
                record_errors.append("label_target_crosses_split_boundary")
        elif split_name not in normalized:
            record_errors.append("label_split_unknown")
        for code in record_errors:
            errors.append({"code": code, "label_index": index, "split": split_name})
        label_results.append({
            "source_index": source_index,
            "target_index": target_index,
            "split": split_name,
            "passed": not record_errors,
            "errors": record_errors,
        })

    if not preprocessing:
        warnings.append("no_preprocessing_fit_records_supplied")
    if not labels:
        warnings.append("no_label_records_supplied; label-boundary checks are limited to declared horizon metadata")

    return {
        "schema_version": 1,
        "audit_name": "pipeline_leakage_boundary",
        "source_manifest_sha256": source_sha256,
        "split_results": normalized,
        "preprocessing_results": preprocessing_results,
        "label_horizon_bars": horizon,
        "label_results": label_results,
        "errors": errors,
        "warnings": warnings,
        "status": "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS"),
        "actionable": False,
        "inferential": False,
        "model_training_performed": False,
        "execution_connected": False,
        "automatic_repair_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON pipeline manifest")
    parser.add_argument("--output", type=Path, help="New JSON report path; existing files are never overwritten")
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
        digest = hashlib.sha256(raw).hexdigest()
        report = audit_pipeline_leakage_boundaries(manifest, source_sha256=digest)
        output = args.output or args.input.with_name(args.input.stem + "_pipeline_leakage_audit.json")
        if output.exists():
            raise ValueError(f"output_already_exists:{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "report_path": str(output.resolve()),
            "error_count": len(report["errors"]),
            "warning_count": len(report["warnings"]),
            "actionable": False,
            "execution_connected": False,
        }, indent=2))
        return 0 if report["status"] != "FAIL" else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

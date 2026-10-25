"""Offline audit of repeated-seed and walk-forward evaluation robustness evidence.

This tool summarizes already-produced metric records. It does not train models, infer
statistical significance, choose models, or connect to trading execution. Its results
are descriptive stability evidence only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

METRIC_DIRECTIONS = {"maximize", "minimize", "descriptive"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _metric_summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    mean = statistics.fmean(values)
    standard_deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "observations": len(values),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "range": ordered[-1] - ordered[0],
        "mean": mean,
        "median": statistics.median(values),
        "population_standard_deviation": standard_deviation,
        "coefficient_of_variation": (
            standard_deviation / abs(mean) if abs(mean) > 1e-15 else None
        ),
    }


def audit_walk_forward_robustness(
    manifest: dict[str, Any],
    *,
    manifest_sha256: str,
    minimum_unique_seeds: int = 3,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate provenance/coverage and describe variation across seeds and folds."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest_must_be_object")
    if not _valid_sha256(manifest_sha256):
        raise ValueError("manifest_sha256_invalid")
    if not isinstance(minimum_unique_seeds, int) or isinstance(minimum_unique_seeds, bool) or minimum_unique_seeds < 2:
        raise ValueError("minimum_unique_seeds_must_be_at_least_two")

    records = manifest.get("evaluations")
    if not isinstance(records, list) or not records:
        raise ValueError("evaluations_must_be_nonempty_list")
    expected_folds = manifest.get("expected_folds")
    if expected_folds is not None:
        if (
            not isinstance(expected_folds, list)
            or not expected_folds
            or any(not isinstance(item, (str, int)) or isinstance(item, bool) for item in expected_folds)
            or len({str(item) for item in expected_folds}) != len(expected_folds)
        ):
            raise ValueError("expected_folds_invalid")
        expected_fold_keys = {str(item) for item in expected_folds}
    else:
        expected_fold_keys = set()

    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    normalized: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    seen_seed_folds: set[tuple[int, str]] = set()
    dataset_hashes: set[str] = set()
    metric_names: set[str] = set()
    metric_names_by_record: list[set[str]] = []

    for index, item in enumerate(records):
        item_errors: list[str] = []
        if not isinstance(item, dict):
            errors.append({"code": "evaluation_record_invalid", "index": index})
            continue
        run_id, seed, fold_id = item.get("run_id"), item.get("seed"), item.get("fold_id")
        dataset_hash, metrics = item.get("dataset_sha256"), item.get("metrics")
        if not isinstance(run_id, str) or not run_id.strip():
            item_errors.append("run_id_invalid")
            run_id = f"invalid_{index}"
        if run_id in seen_run_ids:
            item_errors.append("duplicate_run_id")
        seen_run_ids.add(run_id)
        if not isinstance(seed, int) or isinstance(seed, bool):
            item_errors.append("seed_invalid")
        if not isinstance(fold_id, (str, int)) or isinstance(fold_id, bool) or str(fold_id) == "":
            item_errors.append("fold_id_invalid")
            fold_key = f"invalid_{index}"
        else:
            fold_key = str(fold_id)
        if isinstance(seed, int) and not isinstance(seed, bool):
            key = (seed, fold_key)
            if key in seen_seed_folds:
                item_errors.append("duplicate_seed_fold")
            seen_seed_folds.add(key)
        if not _valid_sha256(dataset_hash):
            item_errors.append("dataset_sha256_invalid")
        else:
            dataset_hashes.add(dataset_hash.lower())
        if not isinstance(metrics, dict) or not metrics:
            item_errors.append("metrics_must_be_nonempty_object")
            metrics = {}
        clean_metrics: dict[str, float] = {}
        for metric_name, value in metrics.items():
            if not isinstance(metric_name, str) or not metric_name.strip():
                item_errors.append("metric_name_invalid")
                continue
            if not _finite_number(value):
                item_errors.append(f"metric_value_not_finite_number:{metric_name}")
                continue
            clean_metrics[metric_name] = float(value)
            metric_names.add(metric_name)
        metric_names_by_record.append(set(clean_metrics))
        for code in item_errors:
            errors.append({"code": code, "index": index, "run_id": run_id})
        normalized.append({
            "run_id": run_id,
            "seed": seed,
            "fold_id": fold_key,
            "dataset_sha256": dataset_hash.lower() if _valid_sha256(dataset_hash) else None,
            "metrics": clean_metrics,
            "passed_validation": not item_errors,
            "errors": item_errors,
        })

    if len(dataset_hashes) > 1:
        errors.append({
            "code": "dataset_fingerprint_mismatch",
            "dataset_sha256_values": sorted(dataset_hashes),
        })
    if not dataset_hashes:
        errors.append({"code": "no_valid_dataset_fingerprint"})

    valid_seed_values = {
        item["seed"] for item in normalized
        if isinstance(item["seed"], int) and not isinstance(item["seed"], bool)
    }
    if len(valid_seed_values) < minimum_unique_seeds:
        errors.append({
            "code": "insufficient_unique_seeds",
            "observed": len(valid_seed_values),
            "required": minimum_unique_seeds,
        })

    if metric_names_by_record:
        reference_metrics = metric_names_by_record[0]
        for index, names in enumerate(metric_names_by_record[1:], start=1):
            if names != reference_metrics:
                errors.append({
                    "code": "metric_set_mismatch",
                    "index": index,
                    "missing_metrics": sorted(reference_metrics - names),
                    "unexpected_metrics": sorted(names - reference_metrics),
                })
    if not metric_names:
        errors.append({"code": "no_valid_metrics"})

    observed_folds = {item["fold_id"] for item in normalized if item["fold_id"]}
    if expected_fold_keys:
        missing_folds = sorted(expected_fold_keys - observed_folds)
        unexpected_folds = sorted(observed_folds - expected_fold_keys)
        if missing_folds:
            errors.append({"code": "expected_folds_missing", "fold_ids": missing_folds})
        if unexpected_folds:
            errors.append({"code": "unexpected_folds_present", "fold_ids": unexpected_folds})
        for seed in sorted(valid_seed_values):
            seed_folds = {
                item["fold_id"] for item in normalized if item["seed"] == seed
            }
            missing_for_seed = sorted(expected_fold_keys - seed_folds)
            if missing_for_seed:
                errors.append({
                    "code": "seed_fold_coverage_incomplete",
                    "seed": seed,
                    "missing_fold_ids": missing_for_seed,
                })
    else:
        warnings.append("expected_folds_not_declared; fold completeness cannot be certified")

    directions = manifest.get("metric_directions", {})
    if not isinstance(directions, dict):
        errors.append({"code": "metric_directions_must_be_object"})
        directions = {}
    for metric_name, direction in directions.items():
        if metric_name not in metric_names:
            errors.append({"code": "metric_direction_for_unknown_metric", "metric": metric_name})
        elif direction not in METRIC_DIRECTIONS:
            errors.append({"code": "metric_direction_invalid", "metric": metric_name})

    metric_summaries: dict[str, Any] = {}
    for metric_name in sorted(metric_names):
        values = [
            item["metrics"][metric_name]
            for item in normalized
            if item["passed_validation"] and metric_name in item["metrics"]
        ]
        if values:
            metric_summaries[metric_name] = _metric_summary(values)

    by_seed: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_fold: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item in normalized:
        if not item["passed_validation"]:
            continue
        seed_key = str(item["seed"])
        fold_key = item["fold_id"]
        for metric_name, value in item["metrics"].items():
            by_seed[seed_key][metric_name].append(value)
            by_fold[fold_key][metric_name].append(value)

    seed_summaries = {
        seed: {metric: _metric_summary(values) for metric, values in sorted(metrics.items())}
        for seed, metrics in sorted(by_seed.items(), key=lambda pair: pair[0])
    }
    fold_summaries = {
        fold: {metric: _metric_summary(values) for metric, values in sorted(metrics.items())}
        for fold, metrics in sorted(by_fold.items())
    }

    if len(valid_seed_values) < 5:
        warnings.append("small_seed_count; metric stability summaries may be sensitive to selected seeds")
    if any(len(values) < len(normalized) for values in (
        [item["metrics"].get(metric) for item in normalized if item["passed_validation"]]
        for metric in metric_names
    )):
        warnings.append("metric_summary_excludes_invalid_evaluation_records")

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "walk_forward_robustness",
        "manifest_sha256": manifest_sha256.lower(),
        "evaluations_supplied": len(records),
        "evaluations_validated": sum(item["passed_validation"] for item in normalized),
        "unique_seeds": sorted(valid_seed_values),
        "unique_seed_count": len(valid_seed_values),
        "observed_folds": sorted(observed_folds),
        "expected_folds": sorted(expected_fold_keys),
        "dataset_sha256_values": sorted(dataset_hashes),
        "metric_summaries": metric_summaries,
        "per_seed_summaries": seed_summaries,
        "per_fold_summaries": fold_summaries,
        "errors": errors,
        "warnings": warnings,
        "status": status,
        "actionable": False,
        "inferential": False,
        "statistical_significance_claimed": False,
        "automatic_model_selection": False,
        "model_training_performed": False,
        "execution_connected": False,
        "automatic_repair_performed": False,
        "scope_limitations": [
            "Summarizes supplied metric records; it does not rerun the evaluation pipeline.",
            "Metrics are descriptive and are not statistical significance tests.",
            "Passing does not establish profitability, predictive validity, or deployment readiness.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON manifest with repeated-seed/fold metrics")
    parser.add_argument("--output", type=Path, help="New JSON report path; existing files are never overwritten")
    parser.add_argument("--minimum-unique-seeds", type=int, default=3)
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
        report = audit_walk_forward_robustness(
            manifest,
            manifest_sha256=_sha256(raw),
            minimum_unique_seeds=args.minimum_unique_seeds,
            base_dir=args.input.resolve().parent,
        )
        output = args.output or args.input.with_name(args.input.stem + "_walk_forward_robustness_audit.json")
        if output.exists():
            raise ValueError(f"output_already_exists:{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "status": report["status"],
            "report_path": str(output.resolve()),
            "evaluations_validated": report["evaluations_validated"],
            "unique_seed_count": report["unique_seed_count"],
            "error_count": len(report["errors"]),
            "warning_count": len(report["warnings"]),
            "execution_connected": False,
        }, indent=2))
        return 0 if report["status"] != "FAIL" else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

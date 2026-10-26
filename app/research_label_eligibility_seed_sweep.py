"""Repeated-seed sensitivity analysis for matched label-eligibility replay pairs.

Each seed runs a fresh uncapped/capped pair with matched settings and immutable source data.
The report summarizes consistency across optimizer seeds; all runs reuse the same market
history and test windows, so the seed repetitions are not independent market observations.
This module never selects or promotes a model and never connects to execution services.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.research_label_eligibility_paired_replay import run_controlled_paired_replay
from app.research_strict_json import strict_json_loads


DEFAULT_SEEDS = (20261021, 20261022, 20261023, 20261024, 20261025)
_MAX_SEED = 2**32 - 1
_METRIC_DIRECTIONS = {
    "log_loss": "lower_is_better",
    "brier_score": "lower_is_better",
    "accuracy_at_fixed_half": "higher_is_better",
}


def parse_random_seeds(value: str | Sequence[int]) -> list[int]:
    """Parse at least two distinct non-negative 32-bit seeds."""
    if isinstance(value, str):
        raw_parts = value.split(",")
        if any(not part.strip() for part in raw_parts):
            raise ValueError("seeds must be a comma-separated list of integers")
        try:
            seeds = [int(part.strip(), 10) for part in raw_parts]
        except ValueError as exc:
            raise ValueError("seeds must be a comma-separated list of integers") from exc
    else:
        seeds = list(value)

    if len(seeds) < 2:
        raise ValueError("at least two distinct random seeds are required")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > _MAX_SEED:
            raise ValueError(f"each seed must be an integer between 0 and {_MAX_SEED}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("random seeds must be unique; duplicate seeds are not independent reruns")
    return seeds


def _safe_prefix(value: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,69}", value)
        or ".." in value
    ):
        raise ValueError("run_id_prefix must be a safe filename token of 1-70 characters")
    # Leave room for "-seed-" and the longest supported seed before the paired runner's
    # own 90-character prefix limit is applied.
    if len(value) + len(f"-seed-{_MAX_SEED}") > 90:
        raise ValueError("run_id_prefix is too long after adding the seed identifier")
    return value


def choose_seed_sweep_report_path(output_dir: str | Path, prefix: str) -> Path:
    """Return a free report path; never replace a previous seed-sweep result."""
    safe = _safe_prefix(prefix)
    destination = Path(output_dir)
    for suffix in range(1, 10002):
        rerun = "" if suffix == 1 else f"-rerun-{suffix}"
        candidate = destination / f"{safe}{rerun}_seed_sweep_report.json"
        if not candidate.exists():
            return candidate
    raise ValueError("unable_to_find_free_seed_sweep_report_path_after_10000_attempts")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = strict_json_loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"comparison_report_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"comparison_report_not_object:{path}")
    return value


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"invalid_numeric_value:{field}")
    return float(value)


def _experiment_key(experiment: Mapping[str, Any]) -> tuple[str, str]:
    model_type = experiment.get("model_type")
    feature_variant = experiment.get("feature_variant")
    if not isinstance(model_type, str) or not model_type or not isinstance(feature_variant, str) or not feature_variant:
        raise ValueError("comparison_experiment_missing_model_or_feature_variant")
    return model_type, feature_variant


def _fold_signature(comparison: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    """Return chronological test-window identity, excluding cap/seed-dependent row counts.

    Eligibility counts are useful diagnostics, but they are not fold geometry. Including
    eligible_test_rows here incorrectly conflates sample eligibility with the identity of
    a held-out test window and can reject otherwise matched seed-sweep runs.
    """
    experiments = comparison.get("experiments")
    if not isinstance(experiments, list) or not experiments or not isinstance(experiments[0], dict):
        raise ValueError("comparison_experiments_missing")
    folds = experiments[0].get("fold_results")
    if not isinstance(folds, list) or not folds:
        raise ValueError("comparison_fold_results_missing")
    signature: list[tuple[Any, ...]] = []
    seen_indices: set[int] = set()
    for fold in folds:
        if not isinstance(fold, dict):
            raise ValueError("comparison_fold_entry_invalid")
        fold_index = fold.get("fold_index")
        start = fold.get("test_start_utc")
        end = fold.get("test_end_utc")
        if isinstance(fold_index, bool) or not isinstance(fold_index, int) or fold_index < 0:
            raise ValueError("comparison_fold_index_invalid")
        if fold_index in seen_indices:
            raise ValueError("comparison_fold_index_duplicate")
        if not isinstance(start, str) or not start.strip() or not isinstance(end, str) or not end.strip():
            raise ValueError("comparison_fold_test_window_invalid")
        seen_indices.add(fold_index)
        signature.append((fold_index, start, end))
    return tuple(signature)


def _fold_eligibility_counts(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract per-fold eligibility counts separately from fold-window identity."""
    experiments = comparison.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("comparison_experiments_missing_for_fold_eligibility_counts")

    shared: list[dict[str, Any]] | None = None
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("comparison_experiment_invalid_for_fold_eligibility_counts")
        folds = experiment.get("fold_results")
        if not isinstance(folds, list) or not folds:
            raise ValueError("comparison_fold_results_missing_for_fold_eligibility_counts")
        rows: list[dict[str, Any]] = []
        seen: set[int] = set()
        for fold in folds:
            if not isinstance(fold, dict):
                raise ValueError("comparison_fold_entry_invalid_for_fold_eligibility_counts")
            index = fold.get("fold_index")
            eligible = fold.get("eligible_test_rows")
            start, end = fold.get("test_start_utc"), fold.get("test_end_utc")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen:
                raise ValueError("comparison_fold_index_invalid_for_fold_eligibility_counts")
            if isinstance(eligible, bool) or not isinstance(eligible, int) or eligible < 0:
                raise ValueError("comparison_eligible_test_rows_invalid")
            if not isinstance(start, str) or not start.strip() or not isinstance(end, str) or not end.strip():
                raise ValueError("comparison_fold_window_invalid_for_fold_eligibility_counts")
            seen.add(index)
            rows.append({
                "fold_index": index,
                "test_start_utc": start,
                "test_end_utc": end,
                "eligible_test_rows": eligible,
            })
        rows.sort(key=lambda row: row["fold_index"])
        if shared is None:
            shared = rows
        elif rows != shared:
            raise ValueError("comparison_fold_eligibility_counts_differ_between_experiments")
    return shared or []


def _extract_seed_result(seed: int, replay: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, Any]:
    if comparison.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise ValueError(f"seed_{seed}_comparison_status_not_pass")
    if comparison.get("actionable") is not False or comparison.get("execution_connected") is not False:
        raise ValueError(f"seed_{seed}_comparison_safety_flags_invalid")
    if comparison.get("automatic_model_selection") is not False:
        raise ValueError(f"seed_{seed}_automatic_model_selection_flag_invalid")

    experiments = comparison.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"seed_{seed}_comparison_experiments_missing")
    normalized: dict[str, Any] = {}
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError(f"seed_{seed}_invalid_experiment")
        key = _experiment_key(experiment)
        key_text = f"{key[0]}/{key[1]}"
        if key_text in normalized:
            raise ValueError(f"seed_{seed}_duplicate_experiment:{key_text}")
        aggregate = experiment.get("mean_across_folds")
        if not isinstance(aggregate, dict):
            raise ValueError(f"seed_{seed}_aggregate_metrics_missing:{key_text}")
        baseline = aggregate.get("baseline_same_eligible_timestamps_metrics")
        capped = aggregate.get("variant_eligible_test_metrics")
        deltas = aggregate.get("paired_variant_minus_baseline_delta_mean_across_folds")
        if not all(isinstance(section, dict) for section in (baseline, capped, deltas)):
            raise ValueError(f"seed_{seed}_paired_metrics_missing:{key_text}")
        metrics: dict[str, Any] = {}
        for metric in _METRIC_DIRECTIONS:
            metrics[metric] = {
                "baseline": _finite_number(baseline.get(metric), f"{key_text}.{metric}.baseline"),
                "capped": _finite_number(capped.get(metric), f"{key_text}.{metric}.capped"),
                "delta": _finite_number(deltas.get(metric), f"{key_text}.{metric}.delta"),
            }
        normalized[key_text] = {
            "model_type": key[0],
            "feature_variant": key[1],
            "fold_count": experiment.get("fold_count"),
            "metrics": metrics,
        }

    return {
        "seed": seed,
        "status": replay.get("status"),
        "comparison_status": comparison.get("status"),
        "uncapped_replay_run_id": replay.get("uncapped_replay_run_id"),
        "capped_replay_run_id": replay.get("capped_replay_run_id"),
        "uncapped_replay_manifest_path": replay.get("uncapped_replay_manifest_path"),
        "capped_replay_manifest_path": replay.get("capped_replay_manifest_path"),
        "comparison_report_path": replay.get("paired_comparison_report_path"),
        "raw_ohlc_sha256": comparison.get("raw_ohlc_sha256"),
        "fold_signature": _fold_signature(comparison),
        "fold_eligibility_counts": _fold_eligibility_counts(comparison),
        "experiments": normalized,
        "warnings": list(replay.get("warnings") or []) + list(comparison.get("warnings") or []),
        "errors": list(replay.get("errors") or []) + list(comparison.get("errors") or []),
    }


def aggregate_seed_results(
    seed_results: Sequence[Mapping[str, Any]],
    *,
    neutral_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Aggregate paired deltas across optimizer seeds using descriptive summaries only."""
    if not math.isfinite(neutral_tolerance) or neutral_tolerance < 0:
        raise ValueError("neutral_tolerance must be finite and non-negative")
    successful = [item for item in seed_results if isinstance(item.get("experiments"), Mapping)]
    if not successful:
        return {
            "experiment_count": 0,
            "experiments": [],
            "interpretation": "No successful seed comparisons were available to aggregate.",
            "inferential": False,
            "automatic_model_selection": False,
            "actionable": False,
        }

    keys = set(successful[0]["experiments"].keys())
    for item in successful[1:]:
        if set(item["experiments"].keys()) != keys:
            raise ValueError("experiment_matrix_differs_between_seed_results")

    experiments: list[dict[str, Any]] = []
    for key_text in sorted(keys):
        first = successful[0]["experiments"][key_text]
        metrics: dict[str, Any] = {}
        for metric, direction in _METRIC_DIRECTIONS.items():
            observations = []
            for item in successful:
                metric_values = item["experiments"][key_text]["metrics"][metric]
                delta = _finite_number(metric_values["delta"], f"{key_text}.{metric}.delta")
                observations.append({
                    "seed": item["seed"],
                    "baseline": _finite_number(metric_values["baseline"], f"{key_text}.{metric}.baseline"),
                    "capped": _finite_number(metric_values["capped"], f"{key_text}.{metric}.capped"),
                    "delta": delta,
                })
            deltas = [row["delta"] for row in observations]
            if direction == "lower_is_better":
                favorable = sum(delta < -neutral_tolerance for delta in deltas)
                adverse = sum(delta > neutral_tolerance for delta in deltas)
            else:
                favorable = sum(delta > neutral_tolerance for delta in deltas)
                adverse = sum(delta < -neutral_tolerance for delta in deltas)
            metrics[metric] = {
                "direction": direction,
                "seed_count": len(observations),
                "mean_baseline_metric": round(statistics.fmean(row["baseline"] for row in observations), 10),
                "mean_capped_metric": round(statistics.fmean(row["capped"] for row in observations), 10),
                "mean_delta": round(statistics.fmean(deltas), 10),
                "median_delta": round(statistics.median(deltas), 10),
                "minimum_delta": round(min(deltas), 10),
                "maximum_delta": round(max(deltas), 10),
                "favorable_seed_count": favorable,
                "adverse_seed_count": adverse,
                "neutral_seed_count": len(deltas) - favorable - adverse,
                "per_seed": observations,
            }
        experiments.append({
            "model_type": first["model_type"],
            "feature_variant": first["feature_variant"],
            "fold_count_per_seed": first.get("fold_count"),
            "metrics": metrics,
            "interpretation": (
                "Seed-level changes summarize optimizer/retraining variability on the same fixed market "
                "history and test windows; seeds do not create independent market observations."
            ),
            "automatic_model_selection": False,
            "actionable": False,
        })

    return {
        "experiment_count": len(experiments),
        "experiments": experiments,
        "neutral_tolerance": neutral_tolerance,
        "interpretation": (
            "Descriptive repeated-seed sensitivity only. All seeds reuse the same historical OHLC snapshot "
            "and walk-forward time windows. Favorable/adverse seed counts describe sensitivity to the "
            "random seed; they are not inferential evidence, an independent validation set, or profitability "
            "evidence. No model is ranked or selected."
        ),
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
    }


def run_label_eligibility_seed_sweep(
    baseline_manifest_path: str | Path,
    *,
    seeds: str | Sequence[int] = DEFAULT_SEEDS,
    max_label_elapsed_hours: float = 1.0,
    run_id_prefix: str | None = None,
    bootstrap_resamples: int | None = None,
) -> dict[str, Any]:
    seed_values = parse_random_seeds(seeds)
    manifest_path = Path(baseline_manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"baseline_manifest_not_found:{manifest_path}")
    try:
        baseline_manifest = _read_json(manifest_path)
    except ValueError:
        raise
    baseline_run_id = baseline_manifest.get("run_id")
    if not isinstance(baseline_run_id, str) or not baseline_run_id:
        raise ValueError("baseline_manifest_missing_run_id")

    cap_token = format(float(max_label_elapsed_hours), "g").replace(".", "p")
    prefix = _safe_prefix(
        run_id_prefix or f"{baseline_run_id[:48]}-seedcap-{cap_token}h"
    )
    report_path = choose_seed_sweep_report_path(manifest_path.parent, prefix)
    results: list[dict[str, Any]] = []
    raw_hash: str | None = None
    fold_signature: tuple[tuple[Any, ...], ...] | None = None
    fold_eligibility_counts: list[dict[str, Any]] | None = None
    expected_experiments: set[str] | None = None
    warnings: list[str] = []
    errors: list[str] = []
    comparison_statuses: list[str] = []

    for seed in seed_values:
        seed_prefix = f"{prefix}-seed-{seed}"
        record: dict[str, Any] = {
            "seed": seed,
            "status": "FAIL",
            "errors": [],
            "warnings": [],
            "actionable": False,
            "execution_connected": False,
            "automatic_model_selection": False,
        }
        try:
            replay = run_controlled_paired_replay(
                str(manifest_path),
                max_label_elapsed_hours=max_label_elapsed_hours,
                run_id_prefix=seed_prefix,
                random_seed=seed,
                bootstrap_resamples=bootstrap_resamples,
            )
            if replay.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
                raise ValueError(
                    "paired_replay_failed:"
                    + ",".join(str(item) for item in replay.get("errors", []))
                )
            comparison_path = replay.get("paired_comparison_report_path")
            if not isinstance(comparison_path, str) or not comparison_path:
                raise ValueError("paired_comparison_report_path_missing")
            comparison = _read_json(comparison_path)
            normalized = _extract_seed_result(seed, replay, comparison)

            current_hash = normalized.get("raw_ohlc_sha256")
            if not isinstance(current_hash, str) or not current_hash:
                raise ValueError("seed_replay_raw_ohlc_hash_missing")
            current_signature = normalized["fold_signature"]
            current_fold_eligibility_counts = normalized["fold_eligibility_counts"]
            current_experiments = set(normalized["experiments"])
            if raw_hash is None:
                raw_hash = current_hash
                fold_signature = current_signature
                fold_eligibility_counts = current_fold_eligibility_counts
                expected_experiments = current_experiments
            else:
                if current_hash != raw_hash:
                    raise ValueError("raw_ohlc_hash_differs_between_seed_runs")
                if current_signature != fold_signature:
                    raise ValueError("fold_test_windows_differ_between_seed_runs")
                if current_fold_eligibility_counts != fold_eligibility_counts:
                    raise ValueError("fold_eligibility_counts_differ_between_seed_runs")
                if current_experiments != expected_experiments:
                    raise ValueError("model_feature_matrix_differs_between_seed_runs")

            record.update(normalized)
            record["status"] = replay.get("status")
            record["warnings"] = normalized["warnings"]
            record["errors"] = normalized["errors"]
            record["run_receipt_path"] = replay.get("receipt_path")
            record["actionable"] = False
            record["execution_connected"] = False
            record["automatic_model_selection"] = False
            comparison_statuses.append(str(comparison.get("status")))
            warnings.extend(str(item) for item in normalized["warnings"])
            results.append(record)
        except Exception as exc:
            message = f"seed_{seed}:{type(exc).__name__}:{exc}"
            record["errors"].append(message)
            errors.append(message)
            results.append(record)

    successful = [item for item in results if item.get("experiments")]
    if len(successful) != len(seed_values):
        errors.append(f"incomplete_seed_sweep:{len(successful)}_of_{len(seed_values)}_seeds_succeeded")

    aggregate: dict[str, Any]
    try:
        aggregate = aggregate_seed_results(successful)
    except Exception as exc:
        aggregate = {
            "experiment_count": 0,
            "experiments": [],
            "inferential": False,
            "automatic_model_selection": False,
            "actionable": False,
        }
        errors.append(f"aggregation_failed:{type(exc).__name__}:{exc}")

    # Preserve first occurrence order while avoiding repeated identical warning text.
    unique_warnings = list(dict.fromkeys(warnings))
    status = "FAIL" if errors else (
        "PASS_WITH_WARNINGS"
        if unique_warnings or any(value != "PASS" for value in comparison_statuses)
        else "PASS"
    )
    report = {
        "schema_version": 1,
        "scope": "GOLDCREST_LABEL_ELIGIBILITY_MULTI_SEED_SENSITIVITY",
        "status": status,
        "historical_baseline_run_id": baseline_run_id,
        "historical_baseline_manifest_path": str(manifest_path),
        "max_label_elapsed_hours": float(max_label_elapsed_hours),
        "requested_seeds": seed_values,
        "successful_seed_count": len(successful),
        "failed_seed_count": len(seed_values) - len(successful),
        "run_id_prefix": prefix,
        "raw_ohlc_sha256": raw_hash,
        "fold_eligibility_counts": fold_eligibility_counts,
        "shared_fold_test_windows_across_seeds": fold_signature is not None and len(successful) > 0,
        "same_seed_applied_to_uncapped_and_capped_within_each_pair": True,
        "seed_results": results,
        "aggregate": aggregate,
        "warnings": unique_warnings,
        "errors": errors,
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "This run quantifies optimizer-seed sensitivity using fresh matched uncapped/capped replay "
            "pairs on the same fixed historical snapshot and chronological folds. It does not test across "
            "independent market periods, recover the unknown historical baseline seed, prove statistical "
            "significance, establish profitability, or select a model."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path.resolve())
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated matched uncapped/capped research pairs across optimizer seeds; "
            "offline only, no model selection or execution."
        )
    )
    parser.add_argument("--baseline-manifest", required=True, help="Historical all-label manifest used to reconstruct settings")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="At least two distinct comma-separated random seeds (default: five seeds)",
    )
    parser.add_argument("--max-label-elapsed-hours", type=float, default=1.0)
    parser.add_argument("--run-id-prefix", help="Optional safe prefix for new immutable artifacts")
    parser.add_argument("--bootstrap-resamples", type=int, help="Shared bootstrap count per paired run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_label_eligibility_seed_sweep(
            args.baseline_manifest,
            seeds=args.seeds,
            max_label_elapsed_hours=args.max_label_elapsed_hours,
            run_id_prefix=args.run_id_prefix,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    except Exception as exc:
        report = {
            "status": "FAIL",
            "scope": "GOLDCREST_LABEL_ELIGIBILITY_MULTI_SEED_SENSITIVITY",
            "errors": [f"{type(exc).__name__}:{exc}"],
            "warnings": [],
            "inferential": False,
            "automatic_model_selection": False,
            "actionable": False,
            "execution_connected": False,
        }
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    print(json.dumps({
        "status": report["status"],
        "historical_baseline_run_id": report["historical_baseline_run_id"],
        "requested_seeds": report["requested_seeds"],
        "successful_seed_count": report["successful_seed_count"],
        "failed_seed_count": report["failed_seed_count"],
        "report_path": report["report_path"],
        "errors": report["errors"],
        "warnings": report["warnings"],
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
    }, indent=2, sort_keys=True))
    for experiment in report["aggregate"].get("experiments", []):
        print(f'{experiment["model_type"]}/{experiment["feature_variant"]}')
        for metric, values in experiment["metrics"].items():
            print(
                f"  {metric}: mean_delta={values['mean_delta']:+.6f}, "
                f"median_delta={values['median_delta']:+.6f}, "
                f"favorable_seeds={values['favorable_seed_count']}/{values['seed_count']}, "
                f"adverse_seeds={values['adverse_seed_count']}, "
                f"range=[{values['minimum_delta']:+.6f}, {values['maximum_delta']:+.6f}]"
            )
    return 0 if report["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

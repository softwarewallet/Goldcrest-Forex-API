"""Run a versioned elapsed-time label-eligibility sensitivity variant from a baseline run.

The runner reads only local, hash-verified research artifacts. It reconstructs the original
walk-forward geometry from the dataset timestamps and experiment matrix, preserves the raw
OHLC snapshot, and flags legacy settings that were not persisted by older manifests.
It never connects to a broker or selects a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.research_label_eligibility_comparison import compare_label_eligibility_runs
from app.research_pipeline import run_research_pipeline
from app.research_run_audit import audit_research_run_manifest


DEFAULT_RANDOM_SEED = 20261021
DEFAULT_BOOTSTRAP_RESAMPLES = 500
_EXPECTED_MODEL_TYPES = ["xgboost", "lightgbm"]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"json_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"json_not_object:{path}")
    return value


def _resolve_artifact(manifest_path: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest_artifact_path_missing:{name}")
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        beside_manifest = (manifest_path.parent / candidate).resolve()
        if beside_manifest.is_file():
            return beside_manifest
    raise ValueError(f"manifest_artifact_not_found:{name}:{candidate}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_utc(value: str, *, field: str) -> str:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid_timestamp:{field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timezone_required:{field}")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _dataset_timestamps(dataset_path: Path) -> list[str]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "timestamp" not in reader.fieldnames:
            raise ValueError("prepared_dataset_missing_timestamp_column")
        timestamps = [
            _canonical_utc(str(row.get("timestamp", "")), field=f"dataset row {index}")
            for index, row in enumerate(reader, start=2)
        ]
    if not timestamps:
        raise ValueError("prepared_dataset_has_no_timestamps")
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        raise ValueError("prepared_dataset_timestamps_not_strictly_increasing")
    return timestamps


def _infer_fold_geometry(
    dataset_path: Path,
    matrix: Mapping[str, Any],
    *,
    horizon_bars: int,
) -> dict[str, int]:
    """Recover pre-purge fold sizes and verify every persisted test window exactly."""
    timestamps = _dataset_timestamps(dataset_path)
    index_by_timestamp = {timestamp: index for index, timestamp in enumerate(timestamps)}
    windows = matrix.get("common_fold_test_windows")
    experiments = matrix.get("experiments")
    if not isinstance(windows, list) or len(windows) < 2:
        raise ValueError("cannot_reconstruct_fold_geometry_without_two_common_test_windows")
    if not isinstance(experiments, list) or not experiments or not isinstance(experiments[0], dict):
        raise ValueError("experiment_matrix_missing_experiments")
    first_folds = experiments[0].get("fold_results")
    if not isinstance(first_folds, list) or len(first_folds) != len(windows):
        raise ValueError("experiment_matrix_fold_count_mismatch")

    def position(value: object, field: str) -> int:
        if not isinstance(value, str):
            raise ValueError(f"fold_timestamp_missing:{field}")
        canonical = _canonical_utc(value, field=field)
        try:
            return index_by_timestamp[canonical]
        except KeyError as exc:
            raise ValueError(f"fold_timestamp_not_in_dataset:{field}:{canonical}") from exc

    first_fold = first_folds[0]
    validation_start = position(first_fold.get("validation_start_utc"), "validation_start_utc")
    test_start = position(first_fold.get("test_start_utc"), "test_start_utc")
    test_end = position(first_fold.get("test_end_utc"), "test_end_utc")
    initial_train_bars = validation_start - horizon_bars
    validation_bars = test_start - horizon_bars - initial_train_bars
    test_bars = test_end - test_start + 1 + horizon_bars
    if initial_train_bars < 1 or validation_bars < 1 or test_bars <= horizon_bars:
        raise ValueError("inferred_walk_forward_partition_sizes_invalid")

    second_fold = first_folds[1]
    step_bars = (
        position(second_fold.get("validation_start_utc"), "second_fold_validation_start")
        - validation_start
    )
    if step_bars < 1:
        raise ValueError("inferred_walk_forward_step_invalid")

    # Reconstruct the same deterministic fold plan and verify each stored window, including
    # row count, before any fitting is started.
    inferred_fold_count = 0
    train_end = initial_train_bars
    while train_end + validation_bars + test_bars <= len(timestamps):
        validation_end = train_end + validation_bars
        expected_test_start_idx = validation_end + horizon_bars
        expected_test_end_idx = validation_end + test_bars - 1
        if inferred_fold_count >= len(windows):
            raise ValueError("reconstructed_walk_forward_has_extra_fold")
        window = windows[inferred_fold_count]
        if not isinstance(window, dict):
            raise ValueError("common_fold_test_window_not_object")
        actual_start = _canonical_utc(str(window.get("test_start_utc", "")), field="common_test_start")
        actual_end = _canonical_utc(str(window.get("test_end_utc", "")), field="common_test_end")
        expected_start = timestamps[expected_test_start_idx]
        expected_end = timestamps[expected_test_end_idx]
        expected_rows = expected_test_end_idx - expected_test_start_idx + 1
        if (
            actual_start != expected_start
            or actual_end != expected_end
            or window.get("test_rows") != expected_rows
        ):
            raise ValueError(f"reconstructed_test_window_mismatch:fold_{inferred_fold_count}")
        for experiment in experiments:
            if not isinstance(experiment, dict):
                raise ValueError("experiment_entry_not_object")
            folds = experiment.get("fold_results")
            if not isinstance(folds, list) or len(folds) != len(windows):
                raise ValueError("experiment_fold_count_mismatch")
            fold = folds[inferred_fold_count]
            if (
                _canonical_utc(str(fold.get("test_start_utc", "")), field="experiment_test_start") != actual_start
                or _canonical_utc(str(fold.get("test_end_utc", "")), field="experiment_test_end") != actual_end
            ):
                raise ValueError(f"experiment_test_window_mismatch:fold_{inferred_fold_count}")
        inferred_fold_count += 1
        train_end += step_bars
    if inferred_fold_count != len(windows):
        raise ValueError("reconstructed_walk_forward_fold_count_mismatch")

    return {
        "initial_train_bars": initial_train_bars,
        "validation_bars": validation_bars,
        "test_bars": test_bars,
        "step_bars": step_bars,
        "fold_count": inferred_fold_count,
        "dataset_rows": len(timestamps),
    }



def _variant_run_collisions(
    output_dir: str | Path,
    run_id: str,
    *,
    symbol: str,
    timeframe: str,
) -> list[Path]:
    """List known pipeline and variant receipts already occupying a run namespace."""
    destination = Path(output_dir)
    file_instrument = re.sub(r"[^A-Z0-9_-]+", "_", symbol.strip().upper()).strip("_")
    filenames = (
        f"{run_id}_raw_ohlc_audit.json",
        f"{run_id}_{file_instrument}_{timeframe}_research_dataset.csv",
        f"{run_id}_{file_instrument}_{timeframe}_research_dataset.manifest.json",
        f"{run_id}_experiment_matrix.json",
        f"{run_id}_provenance_audit.json",
        f"{run_id}_temporal_calibration_stability.json",
        f"{run_id}_pipeline_manifest.json",
        f"{run_id}_validation_report.json",
        f"{run_id}_label_eligibility_comparison.json",
        f"{run_id}_label_eligibility_variant_setup.json",
    )
    return [destination / name for name in filenames if (destination / name).exists()]


def _choose_variant_run_id(
    output_dir: str | Path,
    requested_run_id: str,
    *,
    symbol: str,
    timeframe: str,
    allow_auto_versioning: bool,
) -> tuple[str, bool]:
    """Never reuse an artifact namespace; auto-version only an implicit default ID."""
    collisions = _variant_run_collisions(
        output_dir, requested_run_id, symbol=symbol, timeframe=timeframe
    )
    if not collisions:
        return requested_run_id, False
    if not allow_auto_versioning:
        paths = ", ".join(str(path.resolve()) for path in collisions)
        raise ValueError(
            "variant_run_id_artifacts_already_exist; refusing to overwrite: "
            f"{paths}. Pass a different --run-id."
        )

    for suffix in range(2, 10002):
        candidate = f"{requested_run_id}-rerun-{suffix}"
        if not _variant_run_collisions(
            output_dir, candidate, symbol=symbol, timeframe=timeframe
        ):
            return candidate, True
    raise ValueError(
        "unable_to_find_free_variant_run_id_after_10000_attempts; "
        "specify a unique --run-id"
    )

def build_variant_plan(
    baseline_manifest_path: str | Path,
    *,
    max_label_elapsed_hours: float = 1.0,
    run_id: str | None = None,
    random_seed: int | None = None,
    bootstrap_resamples: int | None = None,
) -> dict[str, Any]:
    """Verify artifacts and derive a reproducible sensitivity-run configuration."""
    if (
        isinstance(max_label_elapsed_hours, bool)
        or not isinstance(max_label_elapsed_hours, (int, float))
        or not math.isfinite(float(max_label_elapsed_hours))
        or max_label_elapsed_hours <= 0
    ):
        raise ValueError("max_label_elapsed_hours must be finite and positive")

    manifest_path = Path(baseline_manifest_path).expanduser().resolve()
    integrity = audit_research_run_manifest(manifest_path)
    if integrity.get("status") == "FAIL":
        raise ValueError("baseline_run_integrity_failed:" + ",".join(integrity.get("errors", [])))
    baseline = _read_json(manifest_path)
    baseline_run_id = baseline.get("run_id")
    if not isinstance(baseline_run_id, str) or not baseline_run_id:
        raise ValueError("baseline_run_id_missing")
    if baseline.get("label_eligibility_mode", "all_labels_included") != "all_labels_included":
        raise ValueError("baseline_must_include_all_labels")
    if baseline.get("max_label_elapsed_hours") is not None:
        raise ValueError("baseline_already_has_elapsed_time_cap")

    raw_path = _resolve_artifact(manifest_path, baseline.get("raw_ohlc_path"), name="raw_ohlc_path")
    dataset_path = _resolve_artifact(manifest_path, baseline.get("dataset_path"), name="dataset_path")
    dataset_manifest_path = _resolve_artifact(
        manifest_path, baseline.get("dataset_manifest_path"), name="dataset_manifest_path"
    )
    matrix_path = _resolve_artifact(
        manifest_path, baseline.get("experiment_matrix_report"), name="experiment_matrix_report"
    )
    raw_digest = _sha256(raw_path)
    if raw_digest != baseline.get("raw_ohlc_sha256"):
        raise ValueError("baseline_raw_ohlc_hash_mismatch")
    if _sha256(dataset_path) != baseline.get("dataset_sha256"):
        raise ValueError("baseline_dataset_hash_mismatch")
    matrix = _read_json(matrix_path)
    dataset_manifest = _read_json(dataset_manifest_path)
    if matrix.get("dataset_sha256") != baseline.get("dataset_sha256"):
        raise ValueError("baseline_matrix_dataset_hash_mismatch")
    if matrix.get("run_id") != baseline_run_id:
        raise ValueError("baseline_matrix_run_id_mismatch")
    if dataset_manifest.get("input_snapshot_sha256") != raw_digest:
        raise ValueError("baseline_dataset_manifest_raw_hash_mismatch")

    symbol = baseline.get("symbol")
    timeframe = baseline.get("timeframe")
    horizon_bars = baseline.get("label_horizon_bars")
    class_1_means = baseline.get("class_1_means")
    for value, name in ((symbol, "symbol"), (timeframe, "timeframe"), (class_1_means, "class_1_means")):
        if not isinstance(value, str) or not value:
            raise ValueError(f"baseline_{name}_missing")
    if isinstance(horizon_bars, bool) or not isinstance(horizon_bars, int) or horizon_bars < 1:
        raise ValueError("baseline_horizon_bars_invalid")

    model_types = matrix.get("model_types")
    if model_types != _EXPECTED_MODEL_TYPES:
        raise ValueError(f"unsupported_baseline_model_types:{model_types}")
    experiments = matrix.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("baseline_experiment_matrix_empty")
    feature_variants = [item.get("feature_variant") for item in experiments if isinstance(item, dict)]
    if len(feature_variants) != len(experiments) or not all(isinstance(name, str) for name in feature_variants):
        raise ValueError("baseline_feature_variant_metadata_missing")
    unique_variants = list(dict.fromkeys(feature_variants))
    include_ablation = any(name != "all_features" for name in unique_variants)
    if len(experiments) != len(model_types) * len(unique_variants):
        raise ValueError("baseline_experiment_count_not_compatible_with_pipeline")

    fold_sizes = _infer_fold_geometry(dataset_path, matrix, horizon_bars=horizon_bars)
    trial_counts: set[int] = set()
    for experiment in experiments:
        folds = experiment.get("fold_results", [])
        for fold in folds:
            trials = fold.get("trials_requested") if isinstance(fold, dict) else None
            if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
                raise ValueError("baseline_trial_count_not_recorded")
            trial_counts.add(trials)
    if len(trial_counts) != 1:
        raise ValueError("baseline_trial_count_differs_across_folds")
    trials_per_fold = next(iter(trial_counts))

    saved_config = baseline.get("research_config")
    assumptions: list[str] = []
    if isinstance(saved_config, dict):
        source_config = saved_config
        for field, inferred in fold_sizes.items():
            if field == "fold_count" or field == "dataset_rows":
                continue
            if source_config.get(field) != inferred:
                raise ValueError(f"saved_research_config_mismatch:{field}")
        if source_config.get("trials_per_fold") != trials_per_fold:
            raise ValueError("saved_research_config_mismatch:trials_per_fold")
    else:
        source_config = {}
        assumptions.append("legacy_manifest_does_not_record_all_research_settings; fold geometry and trial count were reconstructed from hashed artifacts")
        assumptions.append("random_seed was not persisted in the baseline; CLI default is used unless --random-seed overrides it")
        assumptions.append("bootstrap_resamples was not persisted in the baseline; CLI default is used unless --bootstrap-resamples overrides it")

    saved_seed = source_config.get("random_seed")
    chosen_seed = random_seed if random_seed is not None else (
        saved_seed if isinstance(saved_seed, int) and not isinstance(saved_seed, bool) else DEFAULT_RANDOM_SEED
    )
    if random_seed is not None and random_seed != saved_seed and isinstance(saved_config, dict):
        assumptions.append("random_seed explicitly overridden for this variant")
    if random_seed is not None and not isinstance(saved_seed, int):
        assumptions.append("random_seed explicitly supplied because legacy manifest did not persist it")

    saved_bootstrap = source_config.get("bootstrap_resamples")
    chosen_bootstrap = bootstrap_resamples if bootstrap_resamples is not None else (
        saved_bootstrap if isinstance(saved_bootstrap, int) and not isinstance(saved_bootstrap, bool)
        else DEFAULT_BOOTSTRAP_RESAMPLES
    )
    if bootstrap_resamples is not None and bootstrap_resamples != saved_bootstrap and isinstance(saved_config, dict):
        assumptions.append("bootstrap_resamples explicitly overridden for this variant")
    if bootstrap_resamples is not None and not isinstance(saved_bootstrap, int):
        assumptions.append("bootstrap_resamples explicitly supplied because legacy manifest did not persist it")
    if isinstance(chosen_seed, bool) or not isinstance(chosen_seed, int) or chosen_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    if isinstance(chosen_bootstrap, bool) or not isinstance(chosen_bootstrap, int) or chosen_bootstrap < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")

    feature_names = matrix.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names or not all(isinstance(name, str) for name in feature_names):
        raise ValueError("baseline_feature_schema_missing")
    if not isinstance(dataset_manifest.get("data_source"), str) or not dataset_manifest.get("data_source"):
        raise ValueError("dataset_manifest_data_source_missing")
    terms_reference = dataset_manifest.get("terms_reference")
    if not isinstance(terms_reference, str) or not terms_reference.strip():
        raise ValueError("dataset_manifest_terms_reference_missing")
    retrieved_at = baseline.get("retrieved_at_utc")
    if not isinstance(retrieved_at, str) or not retrieved_at.strip():
        retrieved_at = dataset_manifest.get("retrieved_at_utc")
    if not isinstance(retrieved_at, str) or not retrieved_at.strip():
        raise ValueError("baseline_retrieved_at_utc_missing")
    source_version = baseline.get("source_version") or dataset_manifest.get("source_version")
    if not isinstance(source_version, str) or not source_version.strip():
        raise ValueError("baseline_source_version_missing")

    cap_token = format(float(max_label_elapsed_hours), "g").replace(".", "p")
    requested_run_id = run_id or f"{baseline_run_id}-elapsed-cap-{cap_token}h"
    if not isinstance(requested_run_id, str) or not requested_run_id:
        raise ValueError("variant_run_id_missing")
    if requested_run_id == baseline_run_id:
        raise ValueError("variant_run_id_must_differ_from_baseline")
    derived_run_id, run_id_auto_versioned = _choose_variant_run_id(
        manifest_path.parent,
        requested_run_id,
        symbol=str(symbol),
        timeframe=str(timeframe),
        allow_auto_versioning=run_id is None,
    )

    # The current pipeline uses these two model types in this stable order.
    # Infer ablation from the actual matrix and reject anything the pipeline cannot reproduce.
    if include_ablation and len(unique_variants) < 2:
        raise ValueError("baseline_ablation_configuration_inconsistent")

    config = {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon_bars": horizon_bars,
        "class_1_means": class_1_means,
        "terms_reference": terms_reference,
        "source_version": source_version,
        "retrieved_at_utc": retrieved_at,
        "output_dir": str(manifest_path.parent),
        "run_id": derived_run_id,
        "initial_train_bars": fold_sizes["initial_train_bars"],
        "validation_bars": fold_sizes["validation_bars"],
        "test_bars": fold_sizes["test_bars"],
        "step_bars": fold_sizes["step_bars"],
        "trials_per_fold": trials_per_fold,
        "random_seed": chosen_seed,
        "bootstrap_resamples": chosen_bootstrap,
        "max_experiments": len(experiments),
        "include_ablation": include_ablation,
        "strict_provenance_audit": bool(
            source_config.get("strict_provenance_audit", baseline.get("strict_provenance_audit", False))
        ),
        "strict_validation": bool(source_config.get("strict_validation", False)),
        "strict_temporal_calibration_audit": bool(
            source_config.get(
                "strict_temporal_calibration_audit",
                baseline.get("strict_temporal_calibration_audit", False),
            )
        ),
        "max_label_elapsed_hours": float(max_label_elapsed_hours),
    }
    return {
        "baseline_manifest_path": str(manifest_path),
        "baseline_manifest_sha256": integrity.get("manifest_sha256"),
        "baseline_run_id": baseline_run_id,
        "requested_variant_run_id": requested_run_id,
        "run_id_auto_versioned": run_id_auto_versioned,
        "raw_ohlc_path": str(raw_path),
        "raw_ohlc_sha256": raw_digest,
        "dataset_path": str(dataset_path),
        "dataset_sha256": baseline.get("dataset_sha256"),
        "dataset_manifest_path": str(dataset_manifest_path),
        "experiment_matrix_path": str(matrix_path),
        "fold_geometry": fold_sizes,
        "feature_variant_names": unique_variants,
        "model_types": model_types,
        "settings": config,
        "assumptions": assumptions,
        "baseline_integrity_status": integrity.get("status"),
        "baseline_integrity_warnings": integrity.get("warnings", []),
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
    }


def _comparison_summary_lines(
    comparison: Mapping[str, Any],
    comparison_path: str | Path,
) -> list[str]:
    """Format a concise terminal summary of the paired offline comparison."""
    def number(value: object, *, integer: bool = False) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "n/a"
        if not math.isfinite(float(value)):
            return "n/a"
        return str(int(value)) if integer and float(value).is_integer() else (
            "n/a" if integer else f"{float(value):+.6f}"
        )

    lines = [
        f"Paired comparison status: {comparison.get('status', 'UNKNOWN')}",
        f"Paired comparison report: {Path(comparison_path).resolve()}",
    ]
    eligibility = comparison.get("eligibility_summary")
    if isinstance(eligibility, Mapping):
        lines.append(
            "Eligibility: "
            f"{number(eligibility.get('variant_eligible_rows'), integer=True)}/"
            f"{number(eligibility.get('variant_dataset_rows'), integer=True)} dataset rows eligible; "
            f"{number(eligibility.get('variant_ineligible_rows'), integer=True)} ineligible"
        )
        lines.append(
            "Paired held-out observations scored across folds: "
            f"{number(eligibility.get('model_rows_scored_across_experiments_and_folds'), integer=True)}; "
            "excluded observations across experiments/folds: "
            f"{number(eligibility.get('model_rows_excluded_across_experiments_and_folds'), integer=True)}"
        )

    experiments = comparison.get("experiments")
    if isinstance(experiments, list):
        for experiment in experiments:
            if not isinstance(experiment, Mapping):
                continue
            aggregate = experiment.get("mean_across_folds")
            if not isinstance(aggregate, Mapping):
                continue
            baseline = aggregate.get("baseline_same_eligible_timestamps_metrics")
            capped = aggregate.get("variant_eligible_test_metrics")
            deltas = aggregate.get("paired_variant_minus_baseline_delta_mean_across_folds")
            if not all(isinstance(item, Mapping) for item in (baseline, capped, deltas)):
                continue
            lines.append(
                f"  {experiment.get('model_type', 'unknown')}/{experiment.get('feature_variant', 'unknown')}: "
                f"rows-across-folds={number(aggregate.get('test_rows_counted_across_folds'), integer=True)}; "
                f"log_loss baseline/capped={number(baseline.get('log_loss'))}/"
                f"{number(capped.get('log_loss'))}, delta={number(deltas.get('log_loss'))}; "
                f"Brier baseline/capped={number(baseline.get('brier_score'))}/"
                f"{number(capped.get('brier_score'))}, delta={number(deltas.get('brier_score'))}; "
                f"accuracy@0.5 delta={number(deltas.get('accuracy_at_fixed_half'))}"
            )
            stability = experiment.get("fold_direction_stability")
            stability_metrics = stability.get("metrics") if isinstance(stability, Mapping) else None
            if isinstance(stability_metrics, Mapping):
                for metric_name in ("log_loss", "brier_score", "accuracy_at_fixed_half"):
                    metric_summary = stability_metrics.get(metric_name)
                    if not isinstance(metric_summary, Mapping):
                        continue
                    lines.append(
                        f"    fold direction {metric_name}: "
                        f"favorable={number(metric_summary.get('favorable_fold_count'), integer=True)}/"
                        f"{number(stability.get('fold_count'), integer=True)}, "
                        f"adverse={number(metric_summary.get('adverse_fold_count'), integer=True)}, "
                        f"neutral={number(metric_summary.get('neutral_fold_count'), integer=True)}; "
                        f"median delta={number(metric_summary.get('median_delta'))}, "
                        f"range=[{number(metric_summary.get('minimum_delta'))}, "
                        f"{number(metric_summary.get('maximum_delta'))}]"
                    )

    warning_diff = comparison.get("raw_ohlc_audit_warning_diff")
    if isinstance(warning_diff, Mapping):
        for side, label in (("baseline_only", "baseline-only"), ("variant_only", "variant-only")):
            items = warning_diff.get(side)
            if isinstance(items, list) and items:
                lines.append(f"Raw OHLC audit warnings ({label}, {len(items)}):")
                lines.extend(f"  - {item}" for item in items if isinstance(item, str))

    lines.append(
        "Delta = capped minus baseline on identical eligible held-out timestamps; "
        "negative log-loss/Brier and positive accuracy deltas are directionally favorable. "
        "Descriptive only; no model selection or execution."
    )
    return lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an elapsed-time label-eligibility sensitivity run from an existing, "
            "hash-verified offline research pipeline manifest."
        )
    )
    parser.add_argument("--baseline-manifest", required=True, help="Existing uncapped pipeline manifest")
    parser.add_argument("--max-label-elapsed-hours", type=float, default=1.0)
    parser.add_argument("--run-id", help="Optional unique output run ID; defaults to <baseline>-elapsed-cap-<hours>h")
    parser.add_argument("--random-seed", type=int, help="Override/recover the baseline seed when it was not persisted")
    parser.add_argument("--bootstrap-resamples", type=int, help="Override/recover the baseline bootstrap count")
    parser.add_argument("--plan-only", action="store_true", help="Print the reconstructed settings without running model experiments")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_variant_plan(
            args.baseline_manifest,
            max_label_elapsed_hours=args.max_label_elapsed_hours,
            run_id=args.run_id,
            random_seed=args.random_seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    except Exception as exc:
        report = {
            "status": "FAIL",
            "errors": [f"{type(exc).__name__}:{exc}"],
            "warnings": [],
            "actionable": False,
            "execution_connected": False,
            "automatic_model_selection": False,
        }
        print(json.dumps(report, indent=2, sort_keys=True) if args.json else report["errors"][0], file=sys.stderr)
        return 2

    if args.plan_only:
        report = {
            "status": "PASS_WITH_WARNINGS" if plan["assumptions"] else "PASS",
            "scope": "GOLDCREST_LABEL_ELIGIBILITY_VARIANT_PLAN",
            "requested_variant_run_id": plan["requested_variant_run_id"],
            "variant_run_id": plan["settings"]["run_id"],
            "run_id_auto_versioned": plan["run_id_auto_versioned"],
            "plan": plan,
            "errors": [],
            "warnings": plan["assumptions"],
            "actionable": False,
            "execution_connected": False,
            "automatic_model_selection": False,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    settings = plan["settings"]
    try:
        result = run_research_pipeline(
            symbol=settings["symbol"],
            timeframe=settings["timeframe"],
            horizon_bars=settings["horizon_bars"],
            class_1_means=settings["class_1_means"],
            terms_reference=settings["terms_reference"],
            source_version=settings["source_version"],
            output_dir=settings["output_dir"],
            run_id=settings["run_id"],
            retrieved_at_utc=settings["retrieved_at_utc"],
            ohlc_csv=plan["raw_ohlc_path"],
            fetch_ctrader=False,
            initial_train_bars=settings["initial_train_bars"],
            validation_bars=settings["validation_bars"],
            test_bars=settings["test_bars"],
            step_bars=settings["step_bars"],
            trials_per_fold=settings["trials_per_fold"],
            random_seed=settings["random_seed"],
            bootstrap_resamples=settings["bootstrap_resamples"],
            max_experiments=settings["max_experiments"],
            include_ablation=settings["include_ablation"],
            strict_provenance_audit=settings["strict_provenance_audit"],
            strict_validation=settings["strict_validation"],
            strict_temporal_calibration_audit=settings["strict_temporal_calibration_audit"],
            max_label_elapsed_hours=settings["max_label_elapsed_hours"],
        )
        variant_manifest_path = Path(settings["output_dir"]) / f"{settings['run_id']}_pipeline_manifest.json"
        variant_manifest = _read_json(variant_manifest_path)
        variant_matrix_path = _resolve_artifact(
            variant_manifest_path, variant_manifest.get("experiment_matrix_report"), name="variant_experiment_matrix_report"
        )
        variant_matrix = _read_json(variant_matrix_path)
        baseline_matrix = _read_json(Path(plan["experiment_matrix_path"]))
        windows_match = variant_matrix.get("common_fold_test_windows") == baseline_matrix.get("common_fold_test_windows")
        model_pairs_match = [
            (item.get("model_type"), item.get("feature_variant")) for item in variant_matrix.get("experiments", [])
        ] == [
            (item.get("model_type"), item.get("feature_variant")) for item in baseline_matrix.get("experiments", [])
        ]
        raw_hash_match = variant_manifest.get("raw_ohlc_sha256") == plan["raw_ohlc_sha256"]
        errors = []
        if not windows_match:
            errors.append("variant_test_windows_differ_from_baseline")
        if not model_pairs_match:
            errors.append("variant_model_feature_pairs_differ_from_baseline")
        if not raw_hash_match:
            errors.append("variant_raw_ohlc_hash_differs_from_baseline")
        for status_field, error_name in (
            ("validation_status", "variant_validation_failed"),
            ("provenance_audit_status", "variant_provenance_audit_failed"),
            ("temporal_calibration_audit_status", "variant_temporal_calibration_audit_failed"),
        ):
            if result.get(status_field) == "FAIL":
                errors.append(error_name)

        comparison = compare_label_eligibility_runs(
            plan["baseline_manifest_path"], str(variant_manifest_path)
        )
        comparison_path = (
            Path(settings["output_dir"])
            / f"{settings['run_id']}_label_eligibility_comparison.json"
        )
        comparison_path.write_text(
            json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if comparison.get("status") == "FAIL":
            errors.append("paired_label_eligibility_comparison_failed")

        result_warnings = result.get("validation_warnings") or []
        comparison_warnings = comparison.get("warnings") or []
        warnings = plan["assumptions"] + result_warnings + comparison_warnings
        setup_report = {
            "schema_version": 1,
            "scope": "GOLDCREST_LABEL_ELIGIBILITY_VARIANT_RUN",
            "status": "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS",
            "baseline_run_id": plan["baseline_run_id"],
            "variant_run_id": settings["run_id"],
            "requested_variant_run_id": plan["requested_variant_run_id"],
            "run_id_auto_versioned": plan["run_id_auto_versioned"],
            "baseline_manifest_path": plan["baseline_manifest_path"],
            "baseline_manifest_sha256": plan["baseline_manifest_sha256"],
            "variant_manifest_path": str(variant_manifest_path.resolve()),
            "variant_manifest_sha256": variant_manifest.get("pipeline_manifest_sha256"),
            "paired_comparison_report_path": str(comparison_path.resolve()),
            "paired_comparison_status": comparison.get("status"),
            "paired_comparison_errors": comparison.get("errors", []),
            "paired_comparison_warnings": comparison_warnings,
            "paired_comparison_eligibility_summary": comparison.get("eligibility_summary"),
            "raw_ohlc_sha256": plan["raw_ohlc_sha256"],
            "raw_ohlc_hash_match": raw_hash_match,
            "common_test_windows_match": windows_match,
            "model_feature_pairs_match": model_pairs_match,
            "variant_label_eligibility": variant_manifest.get("label_eligibility_metadata"),
            "variant_validation_status": result.get("validation_status"),
            "variant_validation_warnings": result_warnings,
            "replay_settings": settings,
            "assumptions": plan["assumptions"],
            "errors": errors,
            "warnings": warnings,
            "actionable": False,
            "execution_connected": False,
            "automatic_model_selection": False,
            "interpretation": (
                "Offline sensitivity analysis only. This runner preserves the source OHLC hash, verifies "
                "identical chronological test windows and model/feature pairs, and writes a paired metric "
                "comparison on identical eligible held-out timestamps. A capped variant is diagnostic, "
                "not an automatically preferred dataset or trading model."
            ),
        }
        output_path = Path(settings["output_dir"]) / f"{settings['run_id']}_label_eligibility_variant_setup.json"
        output_path.write_text(json.dumps(setup_report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(setup_report, indent=2, sort_keys=True))
        else:
            print(f"Elapsed-time eligibility variant: {setup_report['status']}")
            print(f"Variant manifest: {setup_report['variant_manifest_path']}")
            print(f"Setup report: {output_path.resolve()}")
            for line in _comparison_summary_lines(comparison, comparison_path):
                print(line)
            for warning in setup_report["warnings"]:
                print(f"WARNING: {warning}")
            for error in errors:
                print(f"ERROR: {error}")
            print("Actionable: false")
        return 3 if errors else 0
    except Exception as exc:
        report = {
            "status": "FAIL",
            "scope": "GOLDCREST_LABEL_ELIGIBILITY_VARIANT_RUN",
            "variant_run_id": settings["run_id"],
            "errors": [f"{type(exc).__name__}:{exc}"],
            "warnings": plan["assumptions"],
            "actionable": False,
            "execution_connected": False,
            "automatic_model_selection": False,
        }
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

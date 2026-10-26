"""Repeated-seed comparison of predefined elapsed-time label caps.

Runs fresh matched uncapped/capped pairs for the caps not covered by an optional reused
seed-sweep report. All thresholds must use the same raw OHLC hash, fold windows, seed list,
and model/feature matrix. Descriptive offline research only; no model selection or execution.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.research_label_eligibility_seed_sweep import (
    DEFAULT_SEEDS,
    parse_random_seeds,
    run_label_eligibility_seed_sweep,
)

DEFAULT_CAPS = (1.0, 1.5, 8.0)


def parse_elapsed_caps(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("at least two elapsed-time caps are required")
        parts = value.split(",")
        if any(not part.strip() for part in parts):
            raise ValueError("caps must be a comma-separated list of positive hours")
        try:
            caps = [float(part.strip()) for part in parts]
        except ValueError as exc:
            raise ValueError("caps must be a comma-separated list of positive hours") from exc
    else:
        caps = list(value)
    if len(caps) < 2:
        raise ValueError("at least two elapsed-time caps are required")
    normalized: list[float] = []
    for cap in caps:
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(float(cap)) or cap <= 0:
            raise ValueError("each elapsed-time cap must be finite and positive")
        normalized.append(float(cap))
    if len(set(normalized)) != len(normalized):
        raise ValueError("elapsed-time caps must be unique")
    return normalized


def _safe_prefix(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,47}", value) or ".." in value:
        raise ValueError("run_id_prefix must be a safe filename token of 1-48 characters")
    return value


def choose_threshold_report_path(output_dir: str | Path, prefix: str) -> Path:
    safe = _safe_prefix(prefix)
    destination = Path(output_dir)
    for suffix in range(1, 10002):
        rerun = "" if suffix == 1 else f"-rerun-{suffix}"
        path = destination / f"{safe}{rerun}_threshold_sweep_report.json"
        if not path.exists():
            return path
    raise ValueError("unable_to_find_free_threshold_sweep_report_path_after_10000_attempts")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects containing repeated keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    """Reject NaN and Infinity tokens, which are not valid RFC 8259 JSON."""
    raise ValueError(f"nonstandard_json_constant:{value}")


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"json_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"json_not_object:{path}")
    return value


def _fold_signature(report: Mapping[str, Any]) -> list[Any]:
    seed_results = report.get("seed_results")
    if not isinstance(seed_results, list) or not seed_results:
        raise ValueError("seed_sweep_report_has_no_seed_results")
    signatures = {
        json.dumps(item.get("fold_signature"), sort_keys=True, separators=(",", ":"))
        for item in seed_results
        if isinstance(item, dict) and item.get("experiments")
    }
    if len(signatures) != 1:
        raise ValueError("seed_sweep_report_fold_windows_not_consistent")
    signature = json.loads(next(iter(signatures)))
    if not isinstance(signature, list) or not signature:
        raise ValueError("seed_sweep_report_fold_signature_invalid")
    return signature


def _fold_window_signature(report: Mapping[str, Any]) -> list[Any]:
    """Return chronological fold boundaries, excluding cap-dependent eligible-row counts."""
    return [fold[:3] for fold in _fold_signature(report)]


def _experiment_keys(report: Mapping[str, Any]) -> list[str]:
    aggregate = report.get("aggregate")
    experiments = aggregate.get("experiments") if isinstance(aggregate, dict) else None
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("seed_sweep_aggregate_experiments_missing")
    keys: list[str] = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("seed_sweep_aggregate_experiment_invalid")
        model = experiment.get("model_type")
        variant = experiment.get("feature_variant")
        if not isinstance(model, str) or not model or not isinstance(variant, str) or not variant:
            raise ValueError("seed_sweep_experiment_identity_missing")
        key = f"{model}/{variant}"
        if key in keys:
            raise ValueError(f"seed_sweep_duplicate_experiment:{key}")
        keys.append(key)
    return sorted(keys)


def _validate_seed_sweep_report(
    report: Mapping[str, Any],
    *,
    cap_hours: float,
    expected_seeds: Sequence[int],
    label: str,
) -> None:
    if report.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise ValueError(f"{label}_seed_sweep_status_not_pass")
    if report.get("actionable") is not False or report.get("execution_connected") is not False:
        raise ValueError(f"{label}_safety_flags_invalid")
    if report.get("automatic_model_selection") is not False or report.get("inferential") is not False:
        raise ValueError(f"{label}_research_only_flags_invalid")
    recorded_cap = report.get("max_label_elapsed_hours")
    if isinstance(recorded_cap, bool) or not isinstance(recorded_cap, (int, float)) or not math.isclose(float(recorded_cap), cap_hours, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"{label}_recorded_cap_mismatch")
    if report.get("requested_seeds") != list(expected_seeds):
        raise ValueError(f"{label}_seed_list_mismatch")
    if report.get("failed_seed_count") != 0 or report.get("successful_seed_count") != len(expected_seeds):
        raise ValueError(f"{label}_seed_sweep_incomplete")
    if not isinstance(report.get("raw_ohlc_sha256"), str) or not report.get("raw_ohlc_sha256"):
        raise ValueError(f"{label}_raw_ohlc_hash_missing")
    _fold_signature(report)
    _experiment_keys(report)


def _cap_record(cap_hours: float, report_path: str | Path, report: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = report.get("aggregate")
    experiments = aggregate.get("experiments") if isinstance(aggregate, dict) else []
    metrics_rows: list[dict[str, Any]] = []
    for experiment in experiments or []:
        if not isinstance(experiment, dict):
            continue
        metrics = experiment.get("metrics")
        if not isinstance(metrics, dict):
            continue
        metrics_rows.append({
            "model_type": experiment.get("model_type"),
            "feature_variant": experiment.get("feature_variant"),
            "metrics": {
                metric: {
                    key: values.get(key)
                    for key in ("mean_delta", "median_delta", "minimum_delta", "maximum_delta",
                                "favorable_seed_count", "adverse_seed_count", "neutral_seed_count")
                }
                for metric, values in metrics.items() if isinstance(values, dict)
            },
        })
    return {
        "max_label_elapsed_hours": cap_hours,
        "status": report.get("status"),
        "seed_sweep_report_path": str(Path(report_path).resolve()),
        "successful_seed_count": report.get("successful_seed_count"),
        "failed_seed_count": report.get("failed_seed_count"),
        "raw_ohlc_sha256": report.get("raw_ohlc_sha256"),
        "fold_signature": _fold_signature(report),
        "fold_window_signature": _fold_window_signature(report),
        "fold_eligibility_counts": report.get("fold_eligibility_counts"),
        "experiment_keys": _experiment_keys(report),
        "metrics": metrics_rows,
        "warnings": list(report.get("warnings") or []),
        "errors": list(report.get("errors") or []),
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_elapsed_threshold_sweep(
    baseline_manifest_path: str | Path,
    *,
    caps: str | Sequence[float] = DEFAULT_CAPS,
    seeds: str | Sequence[int] = DEFAULT_SEEDS,
    run_id_prefix: str | None = None,
    reference_seed_sweep_report: str | Path | None = None,
    reference_cap_hours: float = 1.0,
    bootstrap_resamples: int | None = None,
) -> dict[str, Any]:
    seed_values = parse_random_seeds(seeds)
    cap_values = parse_elapsed_caps(caps)
    if isinstance(reference_cap_hours, bool) or not isinstance(reference_cap_hours, (int, float)) or not math.isfinite(float(reference_cap_hours)) or reference_cap_hours <= 0:
        raise ValueError("reference_cap_hours must be finite and positive")
    reference_cap = float(reference_cap_hours)

    manifest_path = Path(baseline_manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"baseline_manifest_not_found:{manifest_path}")
    baseline = _read_json(manifest_path)
    baseline_id = baseline.get("run_id")
    if not isinstance(baseline_id, str) or not baseline_id:
        raise ValueError("baseline_manifest_run_id_missing")

    reference_record: dict[str, Any] | None = None
    if reference_seed_sweep_report is not None:
        reference_path = Path(reference_seed_sweep_report).expanduser().resolve()
        reference_report = _read_json(reference_path)
        _validate_seed_sweep_report(reference_report, cap_hours=reference_cap, expected_seeds=seed_values, label="reference")
        reference_record = _cap_record(reference_cap, reference_path, reference_report)
        if reference_cap not in cap_values:
            cap_values = [reference_cap, *cap_values]

    caps_to_run = [cap for cap in cap_values if reference_record is None or not math.isclose(cap, reference_cap, rel_tol=0, abs_tol=1e-9)]
    prefix = _safe_prefix(run_id_prefix or f"{baseline_id[:24]}-cap-sweep")
    report_path = choose_threshold_report_path(manifest_path.parent, prefix)

    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "GOLDCREST_LABEL_ELIGIBILITY_THRESHOLD_SWEEP",
        "status": "RUNNING",
        "historical_baseline_run_id": baseline_id,
        "historical_baseline_manifest_path": str(manifest_path),
        "requested_caps_hours": list(cap_values),
        "executed_caps_hours": [],
        "reference_cap_hours": reference_cap if reference_record else None,
        "reference_seed_sweep_report_path": str(Path(reference_seed_sweep_report).expanduser().resolve()) if reference_seed_sweep_report else None,
        "seeds": seed_values,
        "requested_seed_count": len(seed_values),
        "run_id_prefix": prefix,
        "same_seeds_used_across_caps": True,
        "same_raw_ohlc_snapshot_across_caps": False,
        "same_fold_windows_across_caps": False,
        "cap_results": [],
        "threshold_comparison": [],
        "warnings": [],
        "errors": [],
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "Predeclared elapsed-time threshold sensitivity. Each cap uses matched uncapped/capped "
            "pairs with the same seed list and chronological test windows. All caps share the same "
            "historical OHLC snapshot; repeated seeds are optimizer-sensitivity checks, not independent "
            "market replications. No threshold or model is automatically selected."
        ),
    }

    global_hash: str | None = None
    global_folds: list[Any] | None = None
    global_keys: list[str] | None = None

    try:
        if reference_record is not None:
            report["cap_results"].append(reference_record)
            report["executed_caps_hours"].append(reference_cap)
            report["warnings"].extend(reference_record["warnings"])
            global_hash = reference_record["raw_ohlc_sha256"]
            global_folds = reference_record["fold_window_signature"]
            global_keys = reference_record["experiment_keys"]

        for cap in caps_to_run:
            token = format(float(cap), "g").replace(".", "p")
            try:
                cap_report = run_label_eligibility_seed_sweep(
                    str(manifest_path),
                    seeds=seed_values,
                    max_label_elapsed_hours=cap,
                    run_id_prefix=f"{prefix}-cap-{token}h",
                    bootstrap_resamples=bootstrap_resamples,
                )
                cap_path = cap_report.get("report_path")
                if not isinstance(cap_path, str) or not cap_path:
                    raise ValueError("cap_seed_sweep_report_path_missing")
                _validate_seed_sweep_report(cap_report, cap_hours=cap, expected_seeds=seed_values, label=f"cap_{token}h")
                record = _cap_record(cap, cap_path, cap_report)
                if global_hash is None:
                    global_hash, global_folds, global_keys = record["raw_ohlc_sha256"], record["fold_window_signature"], record["experiment_keys"]
                else:
                    if record["raw_ohlc_sha256"] != global_hash:
                        raise ValueError(f"cap_{token}h_raw_ohlc_hash_differs_across_thresholds")
                    if record["fold_window_signature"] != global_folds:
                        raise ValueError(f"cap_{token}h_fold_windows_differ_across_thresholds")
                    if record["experiment_keys"] != global_keys:
                        raise ValueError(f"cap_{token}h_experiment_matrix_differs_across_thresholds")
                report["cap_results"].append(record)
                report["executed_caps_hours"].append(cap)
                report["warnings"].extend(record["warnings"])
            except Exception as exc:
                report["errors"].append(f"cap_{token}h:{type(exc).__name__}:{exc}")
                report["cap_results"].append({
                    "max_label_elapsed_hours": cap, "status": "FAIL",
                    "errors": [report["errors"][-1]], "inferential": False,
                    "automatic_model_selection": False, "actionable": False,
                })

            good = [item for item in report["cap_results"] if item.get("status") in {"PASS", "PASS_WITH_WARNINGS"}]
            report["same_raw_ohlc_snapshot_across_caps"] = bool(global_hash) and all(item.get("raw_ohlc_sha256") == global_hash for item in good)
            report["same_fold_windows_across_caps"] = bool(global_folds) and all(item.get("fold_window_signature") == global_folds for item in good)
            _write_report(report_path, report)

        successful = [item for item in report["cap_results"] if item.get("status") in {"PASS", "PASS_WITH_WARNINGS"}]
        for key in sorted({(str(exp["model_type"]), str(exp["feature_variant"])) for rec in successful for exp in rec.get("metrics", [])}):
            by_metric: dict[str, list[dict[str, Any]]] = {}
            for rec in successful:
                exp = next((row for row in rec.get("metrics", []) if (str(row.get("model_type")), str(row.get("feature_variant"))) == key), None)
                if not exp:
                    continue
                for metric, values in exp.get("metrics", {}).items():
                    if isinstance(values, dict):
                        by_metric.setdefault(metric, []).append({"max_label_elapsed_hours": rec["max_label_elapsed_hours"], **values})
            report["threshold_comparison"].append({
                "model_type": key[0], "feature_variant": key[1], "metrics_by_cap": by_metric,
                "automatic_model_selection": False, "actionable": False,
            })

        report["raw_ohlc_sha256"] = global_hash
        report["shared_fold_signature"] = global_folds  # Chronological boundaries only; eligible-row counts vary by cap.
        report["shared_experiment_keys"] = global_keys
        report["warnings"] = list(dict.fromkeys(str(w) for w in report["warnings"]))
        if len(successful) != len(cap_values):
            report["errors"].append(f"incomplete_threshold_sweep:{len(successful)}_of_{len(cap_values)}_caps_succeeded")
        report["status"] = "FAIL" if report["errors"] else (
            "PASS_WITH_WARNINGS" if report["warnings"] or any(x.get("status") != "PASS" for x in successful) else "PASS"
        )
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}:{exc}")
        report["status"] = "FAIL"

    report["report_path"] = str(report_path.resolve())
    _write_report(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare elapsed-time label caps across matched multi-seed offline replay pairs.")
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--caps", default=",".join(format(cap, "g") for cap in DEFAULT_CAPS), help="Distinct positive cap thresholds in hours; default: 1,1.5,8")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS), help="At least two distinct random seeds")
    parser.add_argument("--run-id-prefix", help="Safe short prefix for new immutable artifacts")
    parser.add_argument("--reference-seed-sweep-report", help="Optional completed seed-sweep report reused instead of retraining its cap")
    parser.add_argument("--reference-cap-hours", type=float, default=1.0)
    parser.add_argument("--bootstrap-resamples", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_elapsed_threshold_sweep(
            args.baseline_manifest, caps=args.caps, seeds=args.seeds,
            run_id_prefix=args.run_id_prefix,
            reference_seed_sweep_report=args.reference_seed_sweep_report,
            reference_cap_hours=args.reference_cap_hours,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    except Exception as exc:
        report = {
            "status": "FAIL", "scope": "GOLDCREST_LABEL_ELIGIBILITY_THRESHOLD_SWEEP",
            "errors": [f"{type(exc).__name__}:{exc}"], "warnings": [],
            "inferential": False, "automatic_model_selection": False,
            "actionable": False, "execution_connected": False,
        }
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    print(json.dumps({
        "status": report["status"],
        "historical_baseline_run_id": report["historical_baseline_run_id"],
        "requested_caps_hours": report["requested_caps_hours"],
        "executed_caps_hours": report["executed_caps_hours"],
        "same_raw_ohlc_snapshot_across_caps": report["same_raw_ohlc_snapshot_across_caps"],
        "same_fold_windows_across_caps": report["same_fold_windows_across_caps"],
        "report_path": report["report_path"],
        "errors": report["errors"],
        "warnings": report["warnings"],
        "inferential": False, "automatic_model_selection": False,
        "actionable": False, "execution_connected": False,
    }, indent=2, sort_keys=True))
    for exp in report.get("threshold_comparison", []):
        print(f'{exp["model_type"]}/{exp["feature_variant"]}')
        for metric, rows in exp["metrics_by_cap"].items():
            for row in rows:
                seeds_n = row["favorable_seed_count"] + row["adverse_seed_count"] + row["neutral_seed_count"]
                print(
                    f'  cap={row["max_label_elapsed_hours"]:g}h {metric}: '
                    f'mean_delta={row["mean_delta"]:+.6f}, median_delta={row["median_delta"]:+.6f}, '
                    f'favorable_seeds={row["favorable_seed_count"]}/{seeds_n}, '
                    f'range=[{row["minimum_delta"]:+.6f}, {row["maximum_delta"]:+.6f}]'
                )
    return 0 if report["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

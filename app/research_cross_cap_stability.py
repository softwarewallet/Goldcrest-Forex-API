"""Descriptive five-seed stability audit across label-duration caps.

Consumes PASS common-cohort audits and their source comparison reports. It does not
select models, infer profitability, or connect to execution.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.research_label_eligibility_common_cohort import audit_common_heldout_cohort

CAPS_HOURS = (1.0, 1.5, 8.0)
REFERENCE_CAP_HOURS = 1.0
PRIMARY_METRICS = ("log_loss", "brier_score", "accuracy_at_fixed_half")
METRIC_DIRECTIONS = {
    "log_loss": "lower_is_better",
    "brier_score": "lower_is_better",
    "accuracy_at_fixed_half": "higher_is_better",
}
DEFAULT_EXPECTED_SEEDS = (20261021, 20261022, 20261023, 20261024, 20261025)
_NEUTRAL_TOLERANCE = 1e-12


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _source_path(value: object, audit_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{audit_path}: invalid source comparison-report path")
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        relative = (audit_path.parent / candidate).resolve()
        if relative.is_file():
            return relative
    raise ValueError(f"{audit_path}: source comparison report not found: {value}")


def _finite_metric(metrics: Mapping[str, Any], metric: str, context: str) -> float:
    value = metrics.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{context}: missing or invalid metric {metric}")
    return float(value)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _validate_safety(value: Mapping[str, Any], context: str) -> None:
    for field in ("automatic_model_selection", "actionable", "execution_connected"):
        if value.get(field) is not False:
            raise ValueError(f"{context}: {field} must be false")



def _nested_equal(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    """Compare nested metric records with a strict absolute tolerance for numeric leaves."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_nested_equal(left[key], right[key], tolerance=tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_nested_equal(a, b, tolerance=tolerance) for a, b in zip(left, right))
    return left == right


def _cap_equivalence_diagnostics(
    sources: Sequence[tuple[float, Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Compare cap-specific eligibility, predictions, and metrics without assuming model equivalence."""
    from itertools import combinations

    indexed: dict[float, dict[tuple[str, str], dict[int, Mapping[str, Any]]]] = {}
    for cap, path, report in sources:
        experiments: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = {}
        for experiment in report.get("experiments", []):
            key = (str(experiment.get("model_type")), str(experiment.get("feature_variant")))
            folds = experiment.get("fold_results")
            if not isinstance(folds, list):
                raise ValueError(f"{path}: fold results missing for cap equivalence diagnostics")
            fold_map = {}
            for fold in folds:
                if not isinstance(fold, dict) or not isinstance(fold.get("fold_index"), int):
                    raise ValueError(f"{path}: invalid fold in cap equivalence diagnostics")
                fold_map[fold["fold_index"]] = fold
            experiments[key] = fold_map
        indexed[cap] = experiments

    counts_by_pair: dict[str, dict[str, int]] = {}
    issues: list[dict[str, Any]] = []
    comparison_count = 0
    experiment_keys = set.intersection(*(set(indexed[cap]) for cap in indexed))
    for left_cap, right_cap in combinations(sorted(indexed), 2):
        pair_key = f"{left_cap:g}_vs_{right_cap:g}"
        pair_counts = {
            "equivalent_predictions_and_metrics": 0,
            "same_eligibility_but_predictions_or_metrics_differ": 0,
            "different_eligibility": 0,
        }
        for experiment_key in sorted(experiment_keys):
            left_folds = indexed[left_cap][experiment_key]
            right_folds = indexed[right_cap][experiment_key]
            if set(left_folds) != set(right_folds):
                raise ValueError(
                    f"cap equivalence fold matrix differs for {experiment_key} "
                    f"between {left_cap:g}h and {right_cap:g}h"
                )
            for fold_index in sorted(left_folds):
                left = left_folds[fold_index]
                right = right_folds[fold_index]
                left_rows = left.get("eligible_timestamp_predictions")
                right_rows = right.get("eligible_timestamp_predictions")
                if not isinstance(left_rows, list) or not isinstance(right_rows, list):
                    raise ValueError("cap equivalence diagnostics require per-timestamp predictions")
                left_by_ts = {row["timestamp_utc"]: row for row in left_rows}
                right_by_ts = {row["timestamp_utc"]: row for row in right_rows}
                same_timestamps = set(left_by_ts) == set(right_by_ts)
                comparison_count += 1
                if not same_timestamps:
                    pair_counts["different_eligibility"] += 1
                    continue

                timestamps = sorted(left_by_ts)
                labels_match = all(left_by_ts[ts].get("label") == right_by_ts[ts].get("label") for ts in timestamps)
                baseline_predictions_match = all(
                    _nested_equal(
                        left_by_ts[ts].get("baseline_probability_class_1"),
                        right_by_ts[ts].get("baseline_probability_class_1"),
                    ) for ts in timestamps
                )
                variant_predictions_match = all(
                    _nested_equal(
                        left_by_ts[ts].get("variant_probability_class_1"),
                        right_by_ts[ts].get("variant_probability_class_1"),
                    ) for ts in timestamps
                )
                metric_fields = (
                    "baseline_same_eligible_timestamps_metrics",
                    "variant_eligible_test_metrics",
                    "paired_variant_minus_baseline_delta",
                )
                metrics_match = all(
                    isinstance(left.get(field), Mapping)
                    and isinstance(right.get(field), Mapping)
                    and _nested_equal(left[field], right[field])
                    for field in metric_fields
                )
                if labels_match and baseline_predictions_match and variant_predictions_match and metrics_match:
                    pair_counts["equivalent_predictions_and_metrics"] += 1
                else:
                    pair_counts["same_eligibility_but_predictions_or_metrics_differ"] += 1
                    issues.append({
                        "cap_pair": pair_key,
                        "model_type": experiment_key[0],
                        "feature_variant": experiment_key[1],
                        "fold_index": fold_index,
                        "eligible_timestamp_count": len(timestamps),
                        "eligible_timestamp_fingerprint_sha256": left.get("eligible_timestamp_fingerprint_sha256"),
                        "labels_match": labels_match,
                        "baseline_predictions_match": baseline_predictions_match,
                        "variant_predictions_match": variant_predictions_match,
                        "reported_metrics_match": metrics_match,
                        "interpretation": (
                            "Eligible timestamps match, but prediction or metric records differ. "
                            "This is diagnostic only: distinct training artifacts can legitimately "
                            "produce different predictions, so review provenance before treating it as a defect."
                        ),
                    })
        counts_by_pair[pair_key] = pair_counts

    mismatches = sum(
        values["same_eligibility_but_predictions_or_metrics_differ"]
        for values in counts_by_pair.values()
    )
    return {
        "status": "PASS" if mismatches == 0 else "REQUIRES_REVIEW",
        "comparison_count": comparison_count,
        "numeric_metric_tolerance": 1e-12,
        "counts_by_cap_pair": counts_by_pair,
        "issues": issues,
        "interpretation": (
            "Eligibility is compared using the actual per-fold timestamp sets. When timestamp sets match, "
            "labels, baseline probabilities, variant probabilities, and reported metrics/deltas are compared. "
            "A mismatch is surfaced for review, not automatically treated as a model defect."
        ),
    }

def _validate_audit_and_sources(audit_path: Path, seed: int) -> dict[str, Any]:
    audit = _read_json(audit_path)
    if audit.get("status") != "PASS":
        raise ValueError(f"{audit_path}: common-cohort audit status must be PASS")
    _validate_safety(audit, str(audit_path))
    if audit.get("scope") != "GOLDCREST_COMMON_HELDOUT_COHORT_AUDIT":
        raise ValueError(f"{audit_path}: unexpected audit scope")
    if audit.get("baseline_identity_mode") != "dataset_hash_equivalence":
        raise ValueError(f"{audit_path}: expected dataset_hash_equivalence identity mode")
    if not _valid_sha256(audit.get("baseline_dataset_sha256")):
        raise ValueError(f"{audit_path}: invalid baseline dataset SHA-256")
    if not _valid_sha256(audit.get("raw_ohlc_sha256")):
        raise ValueError(f"{audit_path}: invalid raw OHLC SHA-256")
    caps = audit.get("caps_hours")
    if not isinstance(caps, list) or sorted(float(x) for x in caps) != list(CAPS_HOURS):
        raise ValueError(f"{audit_path}: caps must be exactly {list(CAPS_HOURS)}")
    source_values = audit.get("input_comparison_reports")
    if not isinstance(source_values, list) or len(source_values) != len(CAPS_HOURS):
        raise ValueError(f"{audit_path}: exactly three source comparison reports are required")
    sources = []
    for value in source_values:
        path = _source_path(value, audit_path)
        report = _read_json(path)
        if report.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
            raise ValueError(f"{path}: source comparison report is not passing")
        _validate_safety(report, str(path))
        cap = report.get("max_label_elapsed_hours")
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(float(cap)):
            raise ValueError(f"{path}: invalid max_label_elapsed_hours")
        sources.append((float(cap), path, report))
    sources.sort(key=lambda item: item[0])
    source_caps = [item[0] for item in sources]
    if source_caps != list(CAPS_HOURS):
        raise ValueError(f"{audit_path}: source reports do not cover the three required caps")
    source_hashes = {item[2].get("baseline_dataset_sha256") for item in sources}
    if len(source_hashes) != 1 or not _valid_sha256(next(iter(source_hashes))):
        raise ValueError(f"{audit_path}: source baseline dataset hashes do not match")
    if audit.get("baseline_dataset_sha256") != next(iter(source_hashes)):
        raise ValueError(f"{audit_path}: audit/source baseline dataset hash mismatch")
    for field in ("symbol", "timeframe", "label_horizon_bars", "raw_ohlc_sha256"):
        source_values_for_field = {json.dumps(item[2].get(field), sort_keys=True) for item in sources}
        if len(source_values_for_field) != 1 or audit.get(field) != sources[0][2].get(field):
            raise ValueError(f"{audit_path}: inconsistent source/audit provenance field {field}")
    expected_ids = list(dict.fromkeys(item[2].get("baseline_run_id") for item in sources))
    if audit.get("baseline_run_ids") != expected_ids:
        raise ValueError(f"{audit_path}: baseline run ID provenance does not match source reports")

    # Re-run the formal cohort audit against current source files. This detects stale audit
    # outputs if a comparison report was replaced after the original audit was generated.
    recomputed = audit_common_heldout_cohort(
        [str(path) for _, path, _ in sources],
        allow_baseline_id_variation=True,
    )
    if recomputed.get("status") != "PASS":
        source_errors = recomputed.get("errors", [])
        raise ValueError(f"{audit_path}: source reports no longer pass common-cohort audit: {source_errors}")
    for field in ("baseline_dataset_sha256", "baseline_run_ids", "symbol", "timeframe",
                  "label_horizon_bars", "raw_ohlc_sha256", "caps_hours", "experiments"):
        if audit.get(field) != recomputed.get(field):
            raise ValueError(f"{audit_path}: stored audit differs from recomputed source results for {field}")

    experiments = audit.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"{audit_path}: experiments missing")
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError(f"{audit_path}: invalid experiment record")
        if experiment.get("actionable") is not False:
            raise ValueError(f"{audit_path}: experiment actionable flag must be false")
        folds = experiment.get("fold_results")
        if not isinstance(folds, list) or not folds:
            raise ValueError(f"{audit_path}: fold results missing")
        seen_folds = set()
        for fold in folds:
            if not isinstance(fold, dict):
                raise ValueError(f"{audit_path}: invalid fold record")
            index = fold.get("fold_index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen_folds:
                raise ValueError(f"{audit_path}: invalid/duplicate fold index")
            seen_folds.add(index)
            if fold.get("common_timestamp_count") != fold.get("baseline_metrics_on_common_cohort", {}).get("sample_count"):
                raise ValueError(f"{audit_path}: common cohort count disagrees with baseline metric sample count")
            per_cap = fold.get("variant_metrics_on_common_cohort_by_cap")
            if not isinstance(per_cap, dict) or set(per_cap) != {str(cap) for cap in CAPS_HOURS}:
                raise ValueError(f"{audit_path}: per-cap metrics are incomplete")
            for cap in CAPS_HOURS:
                metrics = per_cap[str(cap)]
                if not isinstance(metrics, dict):
                    raise ValueError(f"{audit_path}: invalid metric record for cap {cap}")
                if fold.get("actionable") is not False:
                    raise ValueError(f"{audit_path}: fold actionable flag must be false")
                if metrics.get("sample_count") != fold.get("common_timestamp_count"):
                    raise ValueError(f"{audit_path}: cap {cap} was not scored on the common timestamp count")
                for metric in PRIMARY_METRICS:
                    _finite_metric(metrics, metric, f"{audit_path} seed {seed} fold {index} cap {cap}")
    equivalence = _cap_equivalence_diagnostics(sources)\n    return {"seed": seed, "audit_path": str(audit_path.resolve()), "audit": audit, "sources": sources, "cap_equivalence_diagnostics": equivalence}


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty value sequence")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def audit_cross_cap_stability(
    audit_paths: Sequence[str | Path],
    *,
    expected_seeds: Sequence[int] = DEFAULT_EXPECTED_SEEDS,
) -> dict[str, Any]:
    """Aggregate same-cohort cap differences, keeping seed-level and fold-level evidence distinct."""
    try:
        if len(audit_paths) != len(expected_seeds):
            raise ValueError(f"expected {len(expected_seeds)} common-cohort audit reports; received {len(audit_paths)}")
        loaded = []
        seen_seeds: set[int] = set()
        for raw_path in audit_paths:
            path = Path(raw_path).expanduser().resolve()
            match = re.search(r"seed_(\d+)\.json$", path.name)
            if not match:
                raise ValueError(f"{path}: filename must end in seed_<integer>.json")
            seed = int(match.group(1))
            if seed in seen_seeds:
                raise ValueError(f"duplicate seed audit: {seed}")
            seen_seeds.add(seed)
            loaded.append(_validate_audit_and_sources(path, seed))
        if seen_seeds != set(expected_seeds):
            raise ValueError(f"seed set mismatch: expected {sorted(expected_seeds)}, received {sorted(seen_seeds)}")
        loaded.sort(key=lambda item: item["seed"])
        first_audit = loaded[0]["audit"]
        for item in loaded[1:]:
            audit = item["audit"]
            seed_id = item["seed"]
            for field in ("symbol", "timeframe", "label_horizon_bars", "raw_ohlc_sha256", "caps_hours"):
                if audit.get(field) != first_audit.get(field):
                    raise ValueError(f"seed {seed_id}: cross-seed provenance mismatch for {field}")
        # baseline_dataset_sha256 and baseline_run_ids are intentionally validated within
        # each seed only; they are not required to match across independently seeded runs.

        reference_key = str(REFERENCE_CAP_HOURS)
        per_seed = []
        seed_delta_values: dict[tuple[str, str, str, str], list[float]] = {}
        experiment_keys: set[tuple[str, str]] | None = None
        for item in loaded:
            audit = item["audit"]
            current_keys = {(str(exp.get("model_type")), str(exp.get("feature_variant"))) for exp in audit["experiments"]}
            if experiment_keys is None:
                experiment_keys = current_keys
            elif current_keys != experiment_keys:
                raise ValueError(f"seed {item['seed']}: model/feature experiment matrix differs across seeds")
            seed_experiments = []
            for experiment in sorted(audit["experiments"], key=lambda exp: (str(exp["model_type"]), str(exp["feature_variant"]))):
                model = str(experiment["model_type"])
                feature = str(experiment["feature_variant"])
                folds_out = []
                fold_delta_by_metric_cap: dict[tuple[str, str], list[float]] = {}
                for fold in sorted(experiment["fold_results"], key=lambda row: row["fold_index"]):
                    cap_metrics = fold["variant_metrics_on_common_cohort_by_cap"]
                    reference = cap_metrics[reference_key]
                    fold_caps = {}
                    for cap in CAPS_HOURS:
                        cap_key = str(cap)
                        metrics = cap_metrics[cap_key]
                        deltas = {}
                        if cap != REFERENCE_CAP_HOURS:
                            for metric in PRIMARY_METRICS:
                                delta = _finite_metric(metrics, metric, "cap metrics") - _finite_metric(reference, metric, "reference metrics")
                                deltas[metric] = delta
                                fold_delta_by_metric_cap.setdefault((cap_key, metric), []).append(delta)
                        fold_caps[cap_key] = {
                            "metrics": {metric: metrics[metric] for metric in (
                                "sample_count", "positive_count", "negative_count", "prevalence",
                                "log_loss", "brier_score", "accuracy_at_fixed_half",
                                "mean_predicted_probability",
                            ) if metric in metrics},
                            "delta_vs_1h": deltas,
                        }
                    folds_out.append({
                        "fold_index": fold["fold_index"],
                        "test_start_utc": fold.get("test_start_utc"),
                        "test_end_utc": fold.get("test_end_utc"),
                        "common_timestamp_count": fold["common_timestamp_count"],
                        "common_cohort_timestamp_fingerprint_sha256": fold.get("common_cohort_timestamp_fingerprint_sha256"),
                        "baseline_metrics_on_common_cohort": fold["baseline_metrics_on_common_cohort"],
                        "by_cap": fold_caps,
                        "actionable": False,
                    })
                seed_cap_summaries = []
                for cap in CAPS_HOURS:
                    cap_key = str(cap)
                    if cap == REFERENCE_CAP_HOURS:
                        continue
                    for metric in PRIMARY_METRICS:
                        values = fold_delta_by_metric_cap.get((cap_key, metric), [])
                        if not values:
                            raise ValueError(f"seed {item['seed']} has no fold deltas for {model}/{feature}/{cap_key}/{metric}")
                        seed_mean_delta = statistics.fmean(values)
                        key = (model, feature, cap_key, metric)
                        seed_delta_values.setdefault(key, []).append(seed_mean_delta)
                        direction = METRIC_DIRECTIONS[metric]
                        favorable = seed_mean_delta < -_NEUTRAL_TOLERANCE if direction == "lower_is_better" else seed_mean_delta > _NEUTRAL_TOLERANCE
                        adverse = seed_mean_delta > _NEUTRAL_TOLERANCE if direction == "lower_is_better" else seed_mean_delta < -_NEUTRAL_TOLERANCE
                        seed_cap_summaries.append({
                            "cap_hours": cap,
                            "reference_cap_hours": REFERENCE_CAP_HOURS,
                            "metric": metric,
                            "direction": direction,
                            "mean_fold_delta": seed_mean_delta,
                            "median_fold_delta": statistics.median(values),
                            "minimum_fold_delta": min(values),
                            "maximum_fold_delta": max(values),
                            "favorable_fold_count": sum(v < -_NEUTRAL_TOLERANCE for v in values) if direction == "lower_is_better" else sum(v > _NEUTRAL_TOLERANCE for v in values),
                            "adverse_fold_count": sum(v > _NEUTRAL_TOLERANCE for v in values) if direction == "lower_is_better" else sum(v < -_NEUTRAL_TOLERANCE for v in values),
                            "neutral_fold_count": sum(abs(v) <= _NEUTRAL_TOLERANCE for v in values),
                            "seed_mean_direction": "favorable" if favorable else ("adverse" if adverse else "neutral"),
                            "fold_count": len(values),
                        })
                seed_experiments.append({
                    "model_type": model,
                    "feature_variant": feature,
                    "fold_count": len(folds_out),
                    "fold_results": folds_out,
                    "seed_level_cap_deltas": seed_cap_summaries,
                    "actionable": False,
                })
            per_seed.append({
                "seed": item["seed"],
                "audit_report": item["audit_path"],
                "baseline_dataset_sha256": audit["baseline_dataset_sha256"],
                "baseline_run_ids": audit["baseline_run_ids"],
                "caps_hours": audit["caps_hours"],
                "symbol": audit["symbol"],
                "timeframe": audit["timeframe"],
                "label_horizon_bars": audit["label_horizon_bars"],
                "raw_ohlc_sha256": audit["raw_ohlc_sha256"],
                "cap_equivalence_diagnostics": item["cap_equivalence_diagnostics"],
                "experiments": seed_experiments,
                "actionable": False,
            })

        cross_seed = []
        for (model, feature, cap_key, metric), values in sorted(seed_delta_values.items()):
            direction = METRIC_DIRECTIONS[metric]
            favorable = sum(v < -_NEUTRAL_TOLERANCE for v in values) if direction == "lower_is_better" else sum(v > _NEUTRAL_TOLERANCE for v in values)
            adverse = sum(v > _NEUTRAL_TOLERANCE for v in values) if direction == "lower_is_better" else sum(v < -_NEUTRAL_TOLERANCE for v in values)
            neutral = len(values) - favorable - adverse
            cross_seed.append({
                "model_type": model,
                "feature_variant": feature,
                "cap_hours": float(cap_key),
                "reference_cap_hours": REFERENCE_CAP_HOURS,
                "metric": metric,
                "direction": direction,
                "seed_level_mean_fold_delta_summary": _summary(values),
                "seed_favorable_count": favorable,
                "seed_adverse_count": adverse,
                "seed_neutral_count": neutral,
                "seed_direction_consistent": favorable == len(values) or adverse == len(values) or neutral == len(values),
                "seed_count": len(values),
                "interpretation": "Each value is one seed's mean of its fold-level deltas; folds are not treated as independent seeds.",
                "actionable": False,
            })
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_CROSS_CAP_STABILITY_AUDIT",
            "status": "PASS",
            "expected_seed_ids": sorted(expected_seeds),
            "observed_seed_ids": sorted(seen_seeds),
            "caps_hours": list(CAPS_HOURS),
            "reference_cap_hours": REFERENCE_CAP_HOURS,
            "primary_metrics": list(PRIMARY_METRICS),
            "metric_directions": METRIC_DIRECTIONS,
            "per_seed_results": per_seed,
            "cross_seed_summary": cross_seed,
            "warnings": [
                "Descriptive research diagnostics only; no statistical significance, causal effect, or profitability claim is made.",
                "Fold test windows may overlap; fold observations are summarized within each seed and are not treated as independent seeds.",
                "Cross-seed summaries use one mean fold delta per seed, so seeds—not folds—are the aggregation units.",
                "Matching baseline dataset hashes within a seed do not prove identical training configurations, training samples, artifacts, or training runs.",
                "Cap-equivalence diagnostics compare per-fold eligible timestamps, predictions, and metrics; any same-cohort mismatch is surfaced for review.",
                "No model selection, promotion, execution connectivity, or trading action is authorized.",
            ],
            "automatic_model_selection": False,
            "actionable": False,
            "execution_connected": False,
            "interpretation": "Metrics are recomputed by the source common-cohort audits on the same held-out timestamp intersection across all three caps within each seed and fold. Deltas are cap metric minus the 1-hour metric; lower log loss/Brier and higher accuracy are directionally favorable.",
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_CROSS_CAP_STABILITY_AUDIT",
            "status": "FAIL",
            "errors": [f"{type(exc).__name__}:{exc}"],
            "warnings": [],
            "automatic_model_selection": False,
            "actionable": False,
            "execution_connected": False,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-report", action="append", required=True,
                        help="PASS common-cohort audit JSON; repeat once per seed")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly permit replacing an existing output report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print(json.dumps({
            "status": "FAIL",
            "output": str(output),
            "errors": ["output already exists; pass --overwrite only if replacement is intended"],
            "actionable": False,
        }, indent=2))
        return 2
    report = audit_cross_cap_stability(args.audit_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "seeds": report.get("observed_seed_ids", []),
        "summary_rows": len(report.get("cross_seed_summary", [])),
        "errors": report.get("errors", []),
        "actionable": False,
    }, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

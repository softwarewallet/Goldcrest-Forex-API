"""Common held-out cohort audit across label-eligibility sensitivity reports.

Descriptive only. A common evaluation cohort does not imply identical training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


def _metrics(rows: Sequence[Mapping[str, Any]], probability_field: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("common held-out cohort is empty")
    labels = [row["label"] for row in rows]
    probs = [row[probability_field] for row in rows]
    if any(isinstance(y, bool) or y not in (0, 1) for y in labels):
        raise ValueError("labels must be binary integers")
    if any(isinstance(p, bool) or not isinstance(p, (int, float))
           or not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in probs):
        raise ValueError("probabilities must be finite and within [0, 1]")
    eps = 1e-15
    return {
        "sample_count": len(rows),
        "positive_count": sum(labels),
        "negative_count": len(rows) - sum(labels),
        "prevalence": statistics.fmean(labels),
        "log_loss": -statistics.fmean(
            y * math.log(max(eps, p)) + (1-y) * math.log(max(eps, 1-p))
            for y, p in zip(labels, probs)
        ),
        "brier_score": statistics.fmean((p-y) ** 2 for y, p in zip(labels, probs)),
        "accuracy_at_fixed_half": statistics.fmean(float((p >= 0.5) == bool(y))
                                                     for y, p in zip(labels, probs)),
        "mean_predicted_probability": statistics.fmean(probs),
    }


def _read_report(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}:
        raise ValueError(f"{path}: input comparison report is not a passing report")
    for field in ("baseline_run_id", "variant_run_id", "symbol", "timeframe",
                  "label_horizon_bars", "raw_ohlc_sha256", "experiments"):
        if field not in value:
            raise ValueError(f"{path}: missing required field {field}")
    for field in ("automatic_model_selection", "actionable", "execution_connected"):
        if value.get(field) is not False:
            raise ValueError(f"{path}: {field} must be false")
    baseline_dataset_sha256 = value.get("baseline_dataset_sha256")
    if baseline_dataset_sha256 is not None:
        if (not isinstance(baseline_dataset_sha256, str) or len(baseline_dataset_sha256) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in baseline_dataset_sha256)):
            raise ValueError(f"{path}: invalid baseline_dataset_sha256")
    cap = value.get("max_label_elapsed_hours")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(float(cap)) or cap <= 0:
        raise ValueError(f"{path}: invalid max_label_elapsed_hours")
    return value


def _experiment_map(report: Mapping[str, Any], path: str) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    experiments = report.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"{path}: experiments missing")
    for exp in experiments:
        if not isinstance(exp, dict):
            raise ValueError(f"{path}: invalid experiment record")
        key = (str(exp.get("model_type", "")), str(exp.get("feature_variant", "")))
        if not all(key) or key in result:
            raise ValueError(f"{path}: invalid/duplicate model-feature key")
        result[key] = exp
    return result


def _fold_map(experiment: Mapping[str, Any], path: str) -> dict[int, dict[str, Any]]:
    folds = experiment.get("fold_results")
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"{path}: fold_results missing")
    result = {}
    for fold in folds:
        if not isinstance(fold, dict):
            raise ValueError(f"{path}: invalid fold record")
        index = fold.get("fold_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in result:
            raise ValueError(f"{path}: invalid/duplicate fold index")
        start, end = fold.get("test_start_utc"), fold.get("test_end_utc")
        if not isinstance(start, str) or not isinstance(end, str):
            raise ValueError(f"{path}: fold boundaries missing")
        pairs = fold.get("eligible_timestamp_predictions")
        fingerprint = fold.get("eligible_timestamp_fingerprint_sha256")
        if not isinstance(pairs, list) or not isinstance(fingerprint, str):
            raise ValueError(f"{path}: per-timestamp paired predictions are missing; regenerate comparison report")
        normalized = []
        seen = set()
        for row in pairs:
            if not isinstance(row, dict):
                raise ValueError(f"{path}: invalid paired timestamp record")
            ts, label = row.get("timestamp_utc"), row.get("label")
            if not isinstance(ts, str) or ts in seen:
                raise ValueError(f"{path}: missing/duplicate paired timestamp")
            seen.add(ts)
            if isinstance(label, bool) or label not in (0, 1):
                raise ValueError(f"{path}: invalid paired label")
            normalized.append({
                "timestamp_utc": ts,
                "label": int(label),
                "baseline_probability_class_1": row.get("baseline_probability_class_1"),
                "variant_probability_class_1": row.get("variant_probability_class_1"),
            })
        expected = hashlib.sha256(("\n".join(sorted(seen)) + ("\n" if seen else "")).encode("utf-8")).hexdigest()
        if fingerprint != expected:
            raise ValueError(f"{path}: eligible timestamp fingerprint mismatch")
        result[index] = {"test_start_utc": start, "test_end_utc": end, "pairs": normalized}
    return result


def audit_common_heldout_cohort(
    report_paths: Sequence[str | Path],
    *,
    allow_baseline_id_variation: bool = False,
) -> dict[str, Any]:
    """Audit shared held-out timestamps; optionally allow distinct baseline IDs only when dataset hashes match."""
    try:
        if len(report_paths) < 2:
            raise ValueError("provide at least two comparison reports")
        loaded = [(str(path), _read_report(path)) for path in report_paths]
        base = loaded[0][1]
        if allow_baseline_id_variation:
            for path, report in loaded:
                dataset_hash = report.get("baseline_dataset_sha256")
                if (not isinstance(dataset_hash, str) or len(dataset_hash) != 64
                        or any(char not in "0123456789abcdefABCDEF" for char in dataset_hash)):
                    raise ValueError(
                        f"{path}: baseline_dataset_sha256 is required for baseline-ID variation mode"
                    )
            identity_fields = (
                "baseline_dataset_sha256", "symbol", "timeframe",
                "label_horizon_bars", "raw_ohlc_sha256",
            )
        else:
            identity_fields = (
                "baseline_run_id", "symbol", "timeframe",
                "label_horizon_bars", "raw_ohlc_sha256", "baseline_dataset_sha256",
            )
        for path, report in loaded[1:]:
            for field in identity_fields:
                if report.get(field) != base.get(field):
                    raise ValueError(f"{path}: incompatible {field}")
        caps = [float(report["max_label_elapsed_hours"]) for _, report in loaded]
        if len(set(caps)) != len(caps):
            raise ValueError("duplicate cap values supplied")
        experiment_maps = [(path, report, _experiment_map(report, path)) for path, report in loaded]
        expected_keys = set(experiment_maps[0][2])
        for path, _, mapping in experiment_maps[1:]:
            if set(mapping) != expected_keys:
                raise ValueError(f"{path}: model/feature experiment matrix differs")

        results = []
        warnings = [
            "common evaluation cohort does not imply identical training datasets or training samples",
            "metrics are descriptive held-out diagnostics only; no statistical inference or profitability claim is made",
            "fold observations may overlap; aggregate counts are observation counts across folds, not guaranteed unique timestamps",
        ]
        baseline_run_ids = list(dict.fromkeys(report["baseline_run_id"] for _, report in loaded))
        if allow_baseline_id_variation and len(baseline_run_ids) > 1:
            warnings.append(
                "baseline run IDs differ; matching baseline dataset hashes do not prove identical "
                "training configurations, training samples, model artifacts, or training runs"
            )
        for key in sorted(expected_keys):
            fold_maps = []
            for path, _, mapping in experiment_maps:
                fold_maps.append((path, _fold_map(mapping[key], path)))
            expected_folds = set(fold_maps[0][1])
            for path, fmap in fold_maps[1:]:
                if set(fmap) != expected_folds:
                    raise ValueError(f"{path}: fold index matrix differs for {key}")
            per_fold = []
            aggregate_by_cap: dict[str, list[dict[str, Any]]] = {
                str(cap): [] for cap in caps
            }
            for fold_index in sorted(expected_folds):
                folds = [fmap[fold_index] for _, fmap in fold_maps]
                boundaries = {(f["test_start_utc"], f["test_end_utc"]) for f in folds}
                if len(boundaries) != 1:
                    raise ValueError(f"fold {fold_index}: test boundaries differ")
                maps = []
                for (path, _), fold in zip(loaded, folds):
                    maps.append({p["timestamp_utc"]: p for p in fold["pairs"]})
                common = set(maps[0])
                for mapping in maps[1:]:
                    common.intersection_update(mapping)
                if not common:
                    raise ValueError(f"fold {fold_index}: common held-out timestamp intersection is empty")
                baseline_rows = []
                cap_rows = {str(cap): [] for cap in caps}
                for ts in sorted(common):
                    rows = [mapping[ts] for mapping in maps]
                    label_set = {row["label"] for row in rows}
                    if len(label_set) != 1:
                        raise ValueError(f"fold {fold_index}: label mismatch at {ts}")
                    baseline_probs = [row["baseline_probability_class_1"] for row in rows]
                    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
                           not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in baseline_probs):
                        raise ValueError(f"fold {fold_index}: invalid baseline probability at {ts}")
                    if max(baseline_probs) - min(baseline_probs) > 1e-12:
                        raise ValueError(f"fold {fold_index}: baseline prediction differs across reports at {ts}")
                    baseline_rows.append({"label": rows[0]["label"], "p": float(baseline_probs[0])})
                    for cap, row in zip(caps, rows):
                        cap_rows[str(cap)].append({"label": row["label"], "p": row["variant_probability_class_1"]})
                baseline_metrics = _metrics(baseline_rows, "p")
                cap_metrics = {str(cap): _metrics(cap_rows[str(cap)], "p") for cap in caps}
                for cap, (path, _), mapping in zip(caps, loaded, maps):
                    aggregate_by_cap[str(cap)].extend(cap_rows[str(cap)])
                per_fold.append({
                    "fold_index": fold_index,
                    "test_start_utc": folds[0]["test_start_utc"],
                    "test_end_utc": folds[0]["test_end_utc"],
                    "common_timestamp_count": len(common),
                    "per_cap_eligible_timestamp_count": {str(cap): len(mapping) for cap, mapping in zip(caps, maps)},
                    "per_cap_excluded_to_common_count": {str(cap): len(mapping) - len(common) for cap, mapping in zip(caps, maps)},
                    "common_cohort_timestamp_fingerprint_sha256": hashlib.sha256(
                        ("\n".join(sorted(common)) + "\n").encode("utf-8")
                    ).hexdigest(),
                    "baseline_metrics_on_common_cohort": baseline_metrics,
                    "variant_metrics_on_common_cohort_by_cap": cap_metrics,
                    "actionable": False,
                })
            # Flatten common rows across folds from per-fold metrics is not valid for exact
            # aggregate recomputation when folds overlap; retain fold-level metrics as primary.
            results.append({
                "model_type": key[0],
                "feature_variant": key[1],
                "fold_count": len(per_fold),
                "fold_results": per_fold,
                "interpretation": "Each cap and the baseline are scored on the intersection of eligible held-out timestamps shared by every supplied report within each fold.",
                "actionable": False,
            })
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_COMMON_HELDOUT_COHORT_AUDIT",
            "status": "PASS",
            "baseline_run_id": base["baseline_run_id"] if len(baseline_run_ids) == 1 else None,
            "baseline_run_ids": baseline_run_ids,
            "baseline_identity_mode": (
                "dataset_hash_equivalence" if allow_baseline_id_variation else "strict_run_id"
            ),
            "baseline_dataset_sha256": base.get("baseline_dataset_sha256"),
            "symbol": base["symbol"],
            "timeframe": base["timeframe"],
            "label_horizon_bars": base["label_horizon_bars"],
            "raw_ohlc_sha256": base["raw_ohlc_sha256"],
            "input_comparison_reports": [path for path, _ in loaded],
            "caps_hours": caps,
            "experiments": results,
            "warnings": warnings,
            "automatic_model_selection": False,
            "actionable": False,
            "execution_connected": False,
            "interpretation": "Common-cohort comparison is descriptive only. It does not imply identical training data, prove causal benefit, or establish profitability.",
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return {
            "schema_version": 1,
            "scope": "GOLDCREST_COMMON_HELDOUT_COHORT_AUDIT",
            "status": "FAIL",
            "errors": [f"{type(exc).__name__}:{exc}"],
            "warnings": [],
            "automatic_model_selection": False,
            "actionable": False,
            "execution_connected": False,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-report", action="append", required=True,
                        help="Paired comparison JSON; repeat at least twice")
    parser.add_argument(
        "--allow-baseline-id-variation-with-matching-dataset-hash",
        action="store_true",
        help=(
            "Allow distinct baseline_run_id values only when every report has the same valid "
            "baseline_dataset_sha256; training-run equivalence is not implied"
        ),
    )
    parser.add_argument("--output", required=True, help="Output JSON audit path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_common_heldout_cohort(
        args.comparison_report,
        allow_baseline_id_variation=args.allow_baseline_id_variation_with_matching_dataset_hash,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "errors": report.get("errors", []), "actionable": False}, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

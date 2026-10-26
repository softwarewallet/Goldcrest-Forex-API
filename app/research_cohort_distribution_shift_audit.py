"""Offline cohort-comparability and feature-distribution-shift audit.

Consumes existing manifests and numeric summaries only. It does not load market data,
train models, select models, or connect to broker/execution services.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads

def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


_REQUIRED_COHORT_FIELDS = (
    "symbol",
    "timeframe",
    "feature_schema_sha256",
    "strategy_config_sha256",
)
_SAFETY_FLAGS = {
    "actionable": False,
    "inferential": False,
    "statistical_significance_claimed": False,
    "automatic_model_selection": False,
    "model_training_performed": False,
    "execution_connected": False,
    "automatic_repair_performed": False,
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdefABCDEF" for c in value)
    )


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _distribution_report(
    summaries: Any,
    *,
    psi_warning_threshold: float,
    standardized_mean_warning_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    results: list[dict[str, Any]] = []
    if summaries is None:
        return results, errors, ["distribution_summaries_not_supplied; shift was not assessed"]
    if not isinstance(summaries, list):
        return results, [{"code": "distribution_summaries_must_be_list"}], warnings

    seen: set[str] = set()
    for index, item in enumerate(summaries):
        if not isinstance(item, dict):
            errors.append({"code": "distribution_summary_invalid", "index": index})
            continue
        name = item.get("feature")
        reference = item.get("reference")
        evaluation = item.get("evaluation")
        if not isinstance(name, str) or not name.strip():
            errors.append({"code": "feature_name_invalid", "index": index})
            continue
        if name in seen:
            errors.append({"code": "duplicate_distribution_feature", "feature": name})
            continue
        seen.add(name)
        if not isinstance(reference, dict) or not isinstance(evaluation, dict):
            errors.append({"code": "distribution_stats_missing", "feature": name})
            continue
        stats_ok = True
        for label, stats in (("reference", reference), ("evaluation", evaluation)):
            for key in ("mean", "std"):
                if not _number(stats.get(key)):
                    errors.append({"code": "distribution_stat_invalid", "feature": name, "sample": label, "stat": key})
                    stats_ok = False
            if _number(stats.get("std")) and float(stats["std"]) < 0:
                errors.append({"code": "distribution_std_negative", "feature": name, "sample": label})
                stats_ok = False
            quantiles = stats.get("quantiles")
            if quantiles is not None:
                if not isinstance(quantiles, dict) or any(
                    not isinstance(q, str) or not _number(v)
                    for q, v in quantiles.items()
                ):
                    errors.append({"code": "distribution_quantiles_invalid", "feature": name, "sample": label})
                    stats_ok = False
        if not stats_ok:
            continue
        ref_mean, eval_mean = float(reference["mean"]), float(evaluation["mean"])
        ref_std, eval_std = float(reference["std"]), float(evaluation["std"])
        standardized_mean_shift = abs(eval_mean - ref_mean) / max(ref_std, 1e-12)
        std_ratio = (eval_std / ref_std) if ref_std > 1e-12 else (0.0 if eval_std <= 1e-12 else None)
        result: dict[str, Any] = {
            "feature": name,
            "reference_mean": ref_mean,
            "evaluation_mean": eval_mean,
            "reference_std": ref_std,
            "evaluation_std": eval_std,
            "absolute_standardized_mean_shift": standardized_mean_shift,
            "evaluation_to_reference_std_ratio": std_ratio,
            "population_summary_only": True,
        }

        ref_hist, eval_hist = reference.get("histogram"), evaluation.get("histogram")
        if ref_hist is not None or eval_hist is not None:
            if not isinstance(ref_hist, dict) or not isinstance(eval_hist, dict):
                errors.append({"code": "histogram_pair_required", "feature": name})
            else:
                ref_edges, eval_edges = ref_hist.get("edges"), eval_hist.get("edges")
                ref_counts, eval_counts = ref_hist.get("counts"), eval_hist.get("counts")
                valid_hist = (
                    isinstance(ref_edges, list)
                    and isinstance(eval_edges, list)
                    and ref_edges == eval_edges
                    and len(ref_edges) >= 2
                    and all(_number(x) for x in ref_edges)
                    and all(float(a) < float(b) for a, b in zip(ref_edges, ref_edges[1:]))
                    and isinstance(ref_counts, list)
                    and isinstance(eval_counts, list)
                    and len(ref_counts) == len(eval_counts) == len(ref_edges) - 1
                    and all(_number(x) and float(x) >= 0 for x in ref_counts + eval_counts)
                    and sum(ref_counts) > 0
                    and sum(eval_counts) > 0
                )
                if not valid_hist:
                    errors.append({"code": "histogram_invalid_or_incompatible", "feature": name})
                else:
                    # Add a small smoothing constant, then renormalize to avoid log(0).
                    epsilon = 1e-6
                    r_total = sum(float(x) for x in ref_counts) + epsilon * len(ref_counts)
                    e_total = sum(float(x) for x in eval_counts) + epsilon * len(eval_counts)
                    psi = 0.0
                    for rc, ec in zip(ref_counts, eval_counts):
                        p = (float(rc) + epsilon) / r_total
                        q = (float(ec) + epsilon) / e_total
                        psi += (q - p) * math.log(q / p)
                    result["population_stability_index"] = psi
                    result["histogram_bins"] = len(ref_counts)
                    if psi >= psi_warning_threshold:
                        warnings.append(f"population_stability_index_high:{name}")
        if standardized_mean_shift >= standardized_mean_warning_threshold:
            warnings.append(f"standardized_mean_shift_high:{name}")
        results.append(result)
    return results, errors, warnings


def audit_cohort_distribution_shift(
    manifest: dict[str, Any],
    *,
    manifest_sha256: str,
    psi_warning_threshold: float = 0.2,
    standardized_mean_warning_threshold: float = 1.0,
) -> dict[str, Any]:
    """Check evaluation cohort comparability and summarize supplied feature drift."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest_must_be_object")
    if not _is_hash(manifest_sha256):
        raise ValueError("manifest_sha256_invalid")
    for name, value in (
        ("psi_warning_threshold", psi_warning_threshold),
        ("standardized_mean_warning_threshold", standardized_mean_warning_threshold),
    ):
        if not _number(value) or float(value) < 0:
            raise ValueError(f"{name}_must_be_finite_nonnegative_number")

    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs_must_be_nonempty_list")

    reference_values: dict[str, Any] = {}
    for field in _REQUIRED_COHORT_FIELDS:
        values = {str(run.get(field, "")).strip() for run in runs if isinstance(run, dict)}
        values.discard("")
        if not values:
            errors.append({"code": "required_cohort_field_missing", "field": field})
        elif len(values) > 1:
            errors.append({"code": "cohort_field_mismatch", "field": field, "values": sorted(values)})
        else:
            reference_values[field] = next(iter(values))
    if not _is_hash(reference_values.get("feature_schema_sha256")):
        errors.append({"code": "feature_schema_sha256_invalid"})
    if not _is_hash(reference_values.get("strategy_config_sha256")):
        errors.append({"code": "strategy_config_sha256_invalid"})

    seen_run_ids: set[str] = set()
    seen_seed_fold: set[tuple[int, str]] = set()
    dataset_by_fold: dict[str, set[str]] = {}
    windows_by_fold: dict[str, set[tuple[str, str]]] = {}
    valid_run_count = 0
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            errors.append({"code": "run_record_invalid", "index": index})
            continue
        run_id = run.get("run_id")
        seed = run.get("seed")
        fold = run.get("fold_id")
        dataset_hash = run.get("dataset_sha256")
        start, end = run.get("window_start"), run.get("window_end")
        item_errors: list[str] = []
        if not isinstance(run_id, str) or not run_id.strip():
            item_errors.append("run_id_invalid")
        elif run_id in seen_run_ids:
            item_errors.append("duplicate_run_id")
        else:
            seen_run_ids.add(run_id)
        if not isinstance(seed, int) or isinstance(seed, bool):
            item_errors.append("seed_invalid")
        if not isinstance(fold, (str, int)) or isinstance(fold, bool) or str(fold).strip() == "":
            item_errors.append("fold_id_invalid")
            fold_key = f"invalid_{index}"
        else:
            fold_key = str(fold)
        if isinstance(seed, int) and not isinstance(seed, bool):
            key = (seed, fold_key)
            if key in seen_seed_fold:
                item_errors.append("duplicate_seed_fold")
            seen_seed_fold.add(key)
        if not _is_hash(dataset_hash):
            item_errors.append("dataset_sha256_invalid")
        else:
            dataset_by_fold.setdefault(fold_key, set()).add(dataset_hash.lower())
        start_date, end_date = _date(start), _date(end)
        if start_date is None or end_date is None or start_date > end_date:
            item_errors.append("evaluation_window_invalid")
        else:
            windows_by_fold.setdefault(fold_key, set()).add((start_date.isoformat(), end_date.isoformat()))
        for code in item_errors:
            errors.append({"code": code, "index": index, "run_id": run_id if isinstance(run_id, str) else None})
        if not item_errors:
            valid_run_count += 1

    for fold, hashes in sorted(dataset_by_fold.items()):
        if len(hashes) > 1:
            errors.append({"code": "dataset_fingerprint_mismatch_within_fold", "fold_id": fold, "values": sorted(hashes)})
    for fold, windows in sorted(windows_by_fold.items()):
        if len(windows) > 1:
            errors.append({"code": "evaluation_window_mismatch_within_fold", "fold_id": fold, "windows": sorted(windows)})

    expected_folds = manifest.get("expected_folds")
    observed_folds = {str(run.get("fold_id")) for run in runs if isinstance(run, dict) and run.get("fold_id") is not None}
    if expected_folds is None:
        warnings.append("expected_folds_not_declared; cohort coverage is partial")
    elif (
        not isinstance(expected_folds, list)
        or not expected_folds
        or any(not isinstance(f, (str, int)) or isinstance(f, bool) for f in expected_folds)
        or len({str(f) for f in expected_folds}) != len(expected_folds)
    ):
        errors.append({"code": "expected_folds_invalid"})
    else:
        missing = sorted({str(f) for f in expected_folds} - observed_folds)
        unexpected = sorted(observed_folds - {str(f) for f in expected_folds})
        if missing:
            errors.append({"code": "expected_folds_missing", "fold_ids": missing})
        if unexpected:
            errors.append({"code": "unexpected_folds_present", "fold_ids": unexpected})

    distributions, distribution_errors, distribution_warnings = _distribution_report(
        manifest.get("distribution_summaries"),
        psi_warning_threshold=float(psi_warning_threshold),
        standardized_mean_warning_threshold=float(standardized_mean_warning_threshold),
    )
    errors.extend(distribution_errors)
    warnings.extend(distribution_warnings)
    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "research_cohort_comparability_and_distribution_shift",
        "manifest_sha256": manifest_sha256.lower(),
        "status": status,
        "runs_supplied": len(runs),
        "runs_validated": valid_run_count,
        "cohort_definition": reference_values,
        "observed_folds": sorted(observed_folds),
        "expected_folds": sorted({str(f) for f in expected_folds}) if isinstance(expected_folds, list) else [],
        "dataset_fingerprints_by_fold": {k: sorted(v) for k, v in sorted(dataset_by_fold.items())},
        "distribution_shift": {
            "features_assessed": len(distributions),
            "psi_warning_threshold": float(psi_warning_threshold),
            "standardized_mean_warning_threshold": float(standardized_mean_warning_threshold),
            "feature_results": distributions,
        },
        "errors": errors,
        "warnings": warnings,
        **_SAFETY_FLAGS,
        "scope_limitations": [
            "Uses supplied cohort metadata and distribution summaries; it does not load or verify raw market data.",
            "Distribution statistics and PSI are descriptive diagnostics, not hypothesis tests.",
            "PSI depends on the supplied histogram binning and does not establish predictive degradation.",
            "Passing does not prove leakage-free evaluation, profitability, or deployment readiness.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--psi-warning-threshold", type=float, default=0.2)
    parser.add_argument("--standardized-mean-warning-threshold", type=float, default=1.0)
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
        manifest = _load_json_object(raw, source=str(args.input))
        report = audit_cohort_distribution_shift(
            manifest,
            manifest_sha256=_sha256(raw),
            psi_warning_threshold=args.psi_warning_threshold,
            standardized_mean_warning_threshold=args.standardized_mean_warning_threshold,
        )
        output = args.output or args.input.with_name(args.input.stem + "_cohort_shift_audit.json")
        if output.exists():
            raise ValueError(f"output_already_exists:{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "report_path": str(output.resolve()),
            "runs_validated": report["runs_validated"],
            "features_assessed": report["distribution_shift"]["features_assessed"],
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

"""Recover legacy fold-level eligibility metadata from immutable comparison reports.

The recovery is metadata-only: it never trains models, selects thresholds, or connects
to trading/execution services. Original research artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path("data/research/runs")
DEFAULT_REPORTS = {
    1.0: ROOT / "eurusd-m15-demo-m365-baseline-validation-2026102-seedcap-1h_seed_sweep_report.json",
    1.5: ROOT / "eurusd-m15-demo-m365-bas-cap-sweep-cap-1p5h-rerun-2_seed_sweep_report.json",
    8.0: ROOT / "eurusd-m15-demo-m365-bas-cap-sweep-cap-8h-rerun-2_seed_sweep_report.json",
}
DEFAULT_THRESHOLD_REPORT = ROOT / "eurusd-m15-demo-m365-bas-cap-sweep-rerun-2_threshold_sweep_report.json"
PASS = {"PASS", "PASS_WITH_WARNINGS"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"artifact_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"artifact_root_not_object:{path}")
    return value


def _safe_write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"output_already_exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _safety_check(report: dict[str, Any], label: str) -> None:
    for field in ("actionable", "execution_connected", "automatic_model_selection", "inferential"):
        if field in report and report[field] is not False:
            raise ValueError(f"{label}_unexpected_safety_flag:{field}")


def _fold_rows(comparison: dict[str, Any], label: str) -> tuple[list[dict[str, Any]], str, list[str]]:
    if comparison.get("status") not in PASS:
        raise ValueError(f"{label}_comparison_not_pass")
    _safety_check(comparison, label)
    digest = comparison.get("raw_ohlc_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError(f"{label}_raw_ohlc_hash_invalid")

    experiments = comparison.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"{label}_experiments_missing")
    canonical: list[dict[str, Any]] | None = None
    keys: set[str] = set()

    for exp in experiments:
        if not isinstance(exp, dict):
            raise ValueError(f"{label}_experiment_invalid")
        model, variant = exp.get("model_type"), exp.get("feature_variant")
        if not isinstance(model, str) or not model or not isinstance(variant, str) or not variant:
            raise ValueError(f"{label}_experiment_identity_invalid")
        key = f"{model}/{variant}"
        if key in keys:
            raise ValueError(f"{label}_duplicate_experiment:{key}")
        keys.add(key)

        folds = exp.get("fold_results")
        if not isinstance(folds, list) or not folds:
            raise ValueError(f"{label}_fold_results_missing")
        rows: list[dict[str, Any]] = []
        seen: set[int] = set()
        for fold in folds:
            if not isinstance(fold, dict):
                raise ValueError(f"{label}_fold_invalid")
            idx, count = fold.get("fold_index"), fold.get("eligible_test_rows")
            start, end = fold.get("test_start_utc"), fold.get("test_end_utc")
            if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx in seen:
                raise ValueError(f"{label}_fold_index_invalid")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{label}_eligible_test_rows_invalid")
            if not isinstance(start, str) or not start.strip() or not isinstance(end, str) or not end.strip():
                raise ValueError(f"{label}_fold_window_invalid")
            seen.add(idx)
            rows.append({
                "fold_index": idx,
                "test_start_utc": start,
                "test_end_utc": end,
                "eligible_test_rows": count,
            })
        rows.sort(key=lambda row: row["fold_index"])
        if canonical is None:
            canonical = rows
        elif rows != canonical:
            raise ValueError(f"{label}_experiments_disagree_on_fold_counts_or_windows")

    assert canonical is not None
    return canonical, digest.lower(), sorted(keys)


def _comparison_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def recover_seed_report(source_path: Path, cap: float, output_path: Path) -> dict[str, Any]:
    source = _read_json(source_path)
    _safety_check(source, f"seed_report_{cap:g}h")
    if source.get("status") not in PASS:
        raise ValueError(f"seed_report_{cap:g}h_status_not_pass")
    recorded_cap = source.get("max_label_elapsed_hours")
    if (isinstance(recorded_cap, bool) or not isinstance(recorded_cap, (int, float))
            or not math.isclose(float(recorded_cap), cap, rel_tol=0, abs_tol=1e-9)):
        raise ValueError(f"seed_report_{cap:g}h_cap_mismatch")

    seeds, records = source.get("requested_seeds"), source.get("seed_results")
    if not isinstance(seeds, list) or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError(f"seed_report_{cap:g}h_seed_list_invalid")
    if not isinstance(records, list) or len(records) != len(seeds):
        raise ValueError(f"seed_report_{cap:g}h_seed_records_invalid")
    if source.get("failed_seed_count") != 0 or source.get("successful_seed_count") != len(seeds):
        raise ValueError(f"seed_report_{cap:g}h_incomplete")

    seed_rows: dict[int, list[dict[str, Any]]] = {}
    hashes: set[str] = set()
    windows: set[tuple] = set()
    matrices: set[tuple[str, ...]] = set()

    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"seed_report_{cap:g}h_seed_record_invalid")
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in seeds or seed in seed_rows:
            raise ValueError(f"seed_report_{cap:g}h_seed_identity_invalid")
        if record.get("status") not in PASS:
            raise ValueError(f"seed_report_{cap:g}h_seed_not_pass:{seed}")
        comparison_value = record.get("comparison_report_path")
        if not isinstance(comparison_value, str) or not comparison_value:
            raise ValueError(f"seed_report_{cap:g}h_comparison_path_missing:{seed}")
        comparison_path = _comparison_path(comparison_value)
        comparison = _read_json(comparison_path)
        rows, digest, keys = _fold_rows(comparison, f"cap_{cap:g}h_seed_{seed}")
        if record.get("raw_ohlc_sha256") and record["raw_ohlc_sha256"].lower() != digest:
            raise ValueError(f"cap_{cap:g}h_seed_{seed}_raw_hash_mismatch")
        seed_rows[seed] = rows
        hashes.add(digest)
        windows.add(tuple((r["fold_index"], r["test_start_utc"], r["test_end_utc"]) for r in rows))
        matrices.add(tuple(keys))

    if list(seed_rows) != seeds:
        raise ValueError(f"seed_report_{cap:g}h_seed_order_mismatch")
    if len(hashes) != 1 or len(windows) != 1 or len(matrices) != 1:
        raise ValueError(f"seed_report_{cap:g}h_cross_seed_inconsistency")
    canonical = seed_rows[seeds[0]]
    if any(seed_rows[seed] != canonical for seed in seeds):
        raise ValueError(f"seed_report_{cap:g}h_fold_counts_differ_between_seeds")

    recovered = copy.deepcopy(source)
    recovered["fold_eligibility_counts"] = copy.deepcopy(canonical)
    recovered["raw_ohlc_sha256"] = next(iter(hashes))
    for record in recovered["seed_results"]:
        record["fold_eligibility_counts"] = copy.deepcopy(seed_rows[record["seed"]])
    recovered["fold_eligibility_metadata_recovery"] = {
        "method": "validated_existing_comparison_artifacts",
        "source_seed_sweep_report": str(source_path.resolve()),
        "comparison_artifacts_checked": len(seeds),
        "model_retraining_performed": False,
        "source_artifacts_modified": False,
    }
    _safe_write(output_path, recovered)
    return recovered


def recover_threshold_report(source_path: Path, output_path: Path, recovered_reports: dict[float, tuple[dict[str, Any], Path]]) -> dict[str, Any]:
    source = _read_json(source_path)
    _safety_check(source, "threshold_report")
    report = copy.deepcopy(source)
    cap_records = report.get("cap_results")
    if not isinstance(cap_records, list) or not cap_records:
        raise ValueError("threshold_report_cap_results_missing")

    found: set[float] = set()
    for record in cap_records:
        if not isinstance(record, dict):
            raise ValueError("threshold_report_cap_record_invalid")
        value = record.get("max_label_elapsed_hours")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        cap = next((c for c in recovered_reports if math.isclose(float(value), c, rel_tol=0, abs_tol=1e-9)), None)
        if cap is None:
            continue
        seed_report, path = recovered_reports[cap]
        rows = seed_report["fold_eligibility_counts"]
        record["seed_sweep_report_path"] = str(path.resolve())
        record["fold_eligibility_counts"] = copy.deepcopy(rows)
        record["fold_window_signature"] = [
            [r["fold_index"], r["test_start_utc"], r["test_end_utc"]] for r in rows
        ]
        found.add(cap)

    if found != set(recovered_reports):
        raise ValueError(f"threshold_cap_coverage_mismatch:{sorted(found)}")
    windows = {
        tuple((r["fold_index"], r["test_start_utc"], r["test_end_utc"])
              for r in report["fold_eligibility_counts"])
        for report, _ in recovered_reports.values()
    }
    hashes = {report["raw_ohlc_sha256"] for report, _ in recovered_reports.values()}
    if len(windows) != 1 or len(hashes) != 1:
        raise ValueError("recovered_reports_disagree_across_caps")

    report["fold_eligibility_metadata_recovery"] = {
        "method": "validated_existing_comparison_artifacts",
        "source_threshold_report": str(source_path.resolve()),
        "recovered_seed_sweep_reports": [str(path.resolve()) for _, path in recovered_reports.values()],
        "comparison_artifacts_checked": sum(r["successful_seed_count"] for r, _ in recovered_reports.values()),
        "model_retraining_performed": False,
        "source_artifacts_modified": False,
    }
    _safe_write(output_path, report)
    return report


def run_recovery(threshold_report: Path, output_prefix: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", output_prefix) or ".." in output_prefix:
        raise ValueError("output_prefix_invalid")
    recovered_reports: dict[float, tuple[dict[str, Any], Path]] = {}
    for cap, source in DEFAULT_REPORTS.items():
        output = ROOT / f"{output_prefix}-cap-{format(cap, 'g').replace('.', 'p')}h_seed_sweep_report.json"
        recovered = recover_seed_report(source, cap, output)
        recovered_reports[cap] = (recovered, output)
        print(f"RECOVERED {cap:g}h: {[row['eligible_test_rows'] for row in recovered['fold_eligibility_counts']]}")
        print(f"  {output.resolve()}")

    threshold_output = ROOT / f"{output_prefix}_threshold_sweep_report.json"
    recovered_threshold = recover_threshold_report(threshold_report, threshold_output, recovered_reports)
    print(f"RECOVERED THRESHOLD REPORT: {threshold_output.resolve()}")
    print("Original source artifacts were not overwritten; no model retraining or execution occurred.")
    return recovered_threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-report", type=Path, default=DEFAULT_THRESHOLD_REPORT)
    parser.add_argument("--output-prefix", default="legacy-fold-metadata-recovery-20261025")
    args = parser.parse_args()
    try:
        result = run_recovery(args.threshold_report, args.output_prefix)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2
    print(json.dumps({
        "status": result.get("status"),
        "report_path": str((ROOT / f"{args.output_prefix}_threshold_sweep_report.json").resolve()),
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "inferential": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

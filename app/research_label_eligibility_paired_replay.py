"""Run a controlled uncapped/capped label-eligibility replay pair.

This runner reconstructs settings from an immutable historical manifest, then fits a fresh
uncapped replay and an elapsed-time-capped replay with identical fold geometry, trial budget,
random seed, bootstrap count, and raw OHLC snapshot. It is offline-only and never selects a
model or connects to broker/execution services.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from app.research_label_eligibility_comparison import compare_label_eligibility_runs
from app.research_label_eligibility_variant import (
    _comparison_summary_lines,
    _read_json,
    _variant_run_collisions,
    build_variant_plan,
)
from app.research_pipeline import run_research_pipeline


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_paired_replay_run_ids(
    output_dir: str | Path,
    run_id_prefix: str,
    *,
    symbol: str,
    timeframe: str,
) -> tuple[str, str, bool]:
    """Choose two free immutable run namespaces as a pair, without overwriting artifacts."""
    if (
        not isinstance(run_id_prefix, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,89}", run_id_prefix)
        or ".." in run_id_prefix
    ):
        raise ValueError("run_id_prefix must be a safe filename token of 1-90 characters")

    destination = Path(output_dir)
    for suffix in range(1, 10002):
        run_suffix = "" if suffix == 1 else f"-rerun-{suffix}"
        uncapped_id = f"{run_id_prefix}-uncapped{run_suffix}"
        capped_id = f"{run_id_prefix}-capped{run_suffix}"
        ids = (uncapped_id, capped_id)
        if any(len(run_id) > 128 for run_id in ids):
            raise ValueError("paired replay run ID exceeds the pipeline's 128-character limit")

        collisions: list[Path] = []
        for run_id in ids:
            collisions.extend(
                _variant_run_collisions(
                    destination,
                    run_id,
                    symbol=symbol,
                    timeframe=timeframe,
                )
            )
        for receipt_path in (
            destination / f"{uncapped_id}_paired_replay_receipt.json",
            destination / f"{capped_id}_paired_replay_receipt.json",
        ):
            if receipt_path.exists():
                collisions.append(receipt_path)

        if not collisions:
            return uncapped_id, capped_id, suffix != 1

    raise ValueError("unable_to_find_free_paired_run_ids_after_10000_attempts")


def _run_settings(plan: dict[str, Any], run_id: str, *, cap_hours: float | None) -> dict[str, Any]:
    settings = plan["settings"]
    return {
        "symbol": settings["symbol"],
        "timeframe": settings["timeframe"],
        "horizon_bars": settings["horizon_bars"],
        "class_1_means": settings["class_1_means"],
        "terms_reference": settings["terms_reference"],
        "source_version": settings["source_version"],
        "output_dir": settings["output_dir"],
        "run_id": run_id,
        "retrieved_at_utc": settings["retrieved_at_utc"],
        "ohlc_csv": plan["raw_ohlc_path"],
        "fetch_ctrader": False,
        "initial_train_bars": settings["initial_train_bars"],
        "validation_bars": settings["validation_bars"],
        "test_bars": settings["test_bars"],
        "step_bars": settings["step_bars"],
        "trials_per_fold": settings["trials_per_fold"],
        "random_seed": settings["random_seed"],
        "bootstrap_resamples": settings["bootstrap_resamples"],
        "max_experiments": settings["max_experiments"],
        "include_ablation": settings["include_ablation"],
        "strict_provenance_audit": settings["strict_provenance_audit"],
        "strict_validation": settings["strict_validation"],
        "strict_temporal_calibration_audit": settings["strict_temporal_calibration_audit"],
        "max_label_elapsed_hours": cap_hours,
    }


def run_controlled_paired_replay(
    baseline_manifest_path: str | Path,
    *,
    max_label_elapsed_hours: float = 1.0,
    run_id_prefix: str | None = None,
    random_seed: int | None = None,
    bootstrap_resamples: int | None = None,
) -> dict[str, Any]:
    """Fit a matched uncapped/capped replay pair and produce a paired report."""
    if (
        isinstance(max_label_elapsed_hours, bool)
        or not isinstance(max_label_elapsed_hours, (int, float))
        or not math.isfinite(float(max_label_elapsed_hours))
        or max_label_elapsed_hours <= 0
    ):
        raise ValueError("max_label_elapsed_hours must be finite and positive")

    # This verifies the historical artifacts and reconstructs fold geometry/trial budget.
    historical_plan = build_variant_plan(
        baseline_manifest_path,
        max_label_elapsed_hours=float(max_label_elapsed_hours),
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    historical_settings = historical_plan["settings"]
    cap_token = format(float(max_label_elapsed_hours), "g").replace(".", "p")
    prefix = run_id_prefix or (
        f"{historical_plan['baseline_run_id']}-paired-cap-{cap_token}h"
    )
    uncapped_id, capped_id, auto_versioned = choose_paired_replay_run_ids(
        historical_settings["output_dir"],
        prefix,
        symbol=historical_settings["symbol"],
        timeframe=historical_settings["timeframe"],
    )

    # Rebuild the plan with the selected explicit cap ID; this performs a second strict
    # collision check before either training run is started.
    plan = build_variant_plan(
        baseline_manifest_path,
        max_label_elapsed_hours=float(max_label_elapsed_hours),
        run_id=capped_id,
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    settings = plan["settings"]
    uncapped_manifest_path = Path(settings["output_dir"]) / f"{uncapped_id}_pipeline_manifest.json"
    capped_manifest_path = Path(settings["output_dir"]) / f"{capped_id}_pipeline_manifest.json"
    comparison_path = Path(settings["output_dir"]) / f"{capped_id}_label_eligibility_comparison.json"
    receipt_path = Path(settings["output_dir"]) / f"{capped_id}_paired_replay_receipt.json"

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "scope": "GOLDCREST_CONTROLLED_PAIRED_LABEL_ELIGIBILITY_REPLAY",
        "status": "FAIL",
        "historical_baseline_run_id": plan["baseline_run_id"],
        "uncapped_replay_run_id": uncapped_id,
        "capped_replay_run_id": capped_id,
        "requested_run_id_prefix": prefix,
        "run_ids_auto_versioned": auto_versioned,
        "max_label_elapsed_hours": float(max_label_elapsed_hours),
        "raw_ohlc_sha256": plan["raw_ohlc_sha256"],
        "replay_settings": settings,
        "same_random_seed_and_trial_budget": True,
        "same_fold_geometry": True,
        "same_raw_ohlc_snapshot": True,
        "uncapped_replay_manifest_path": str(uncapped_manifest_path.resolve()),
        "capped_replay_manifest_path": str(capped_manifest_path.resolve()),
        "paired_comparison_report_path": str(comparison_path.resolve()),
        "errors": [],
        "warnings": list(plan["assumptions"]),
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "interpretation": (
            "Fresh controlled offline replay. The uncapped and capped runs share the reconstructed "
            "fold geometry, experiment count, trial budget, random seed, bootstrap count and exact "
            "raw OHLC snapshot. This controls key differences between these two fresh runs but does "
            "not reproduce the historical baseline's unknown random seed or guarantee profitability."
        ),
    }

    try:
        uncapped_result = run_research_pipeline(
            **_run_settings(plan, uncapped_id, cap_hours=None)
        )
        for field, error_name in (
            ("validation_status", "uncapped_replay_validation_failed"),
            ("provenance_audit_status", "uncapped_replay_provenance_audit_failed"),
            ("temporal_calibration_audit_status", "uncapped_replay_temporal_calibration_audit_failed"),
        ):
            if uncapped_result.get(field) == "FAIL":
                receipt["errors"].append(error_name)
        if receipt["errors"]:
            receipt["status"] = "FAIL"
            receipt["uncapped_replay_validation_status"] = uncapped_result.get("validation_status")
            receipt["uncapped_replay_provenance_audit_status"] = uncapped_result.get("provenance_audit_status")
            receipt["uncapped_replay_temporal_calibration_audit_status"] = uncapped_result.get("temporal_calibration_audit_status")
            return receipt

        capped_result = run_research_pipeline(
            **_run_settings(plan, capped_id, cap_hours=float(max_label_elapsed_hours))
        )
        for field, error_name in (
            ("validation_status", "capped_replay_validation_failed"),
            ("provenance_audit_status", "capped_replay_provenance_audit_failed"),
            ("temporal_calibration_audit_status", "capped_replay_temporal_calibration_audit_failed"),
        ):
            if capped_result.get(field) == "FAIL":
                receipt["errors"].append(error_name)

        uncapped_manifest = _read_json(uncapped_manifest_path)
        capped_manifest = _read_json(capped_manifest_path)
        receipt["uncapped_replay_manifest_sha256"] = _sha256(uncapped_manifest_path)
        receipt["capped_replay_manifest_sha256"] = _sha256(capped_manifest_path)
        receipt["raw_ohlc_hashes_match"] = (
            uncapped_manifest.get("raw_ohlc_sha256")
            == capped_manifest.get("raw_ohlc_sha256")
            == plan["raw_ohlc_sha256"]
        )
        if receipt["raw_ohlc_hashes_match"] is not True:
            receipt["errors"].append("paired_replay_raw_ohlc_hash_mismatch")

        comparison = compare_label_eligibility_runs(
            str(uncapped_manifest_path), str(capped_manifest_path)
        )
        comparison_path.write_text(
            json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        receipt["comparison_status"] = comparison.get("status")
        receipt["comparison_errors"] = comparison.get("errors", [])
        receipt["comparison_warnings"] = comparison.get("warnings", [])
        receipt["paired_comparison_report_sha256"] = _sha256(comparison_path)
        if comparison.get("status") == "FAIL":
            receipt["errors"].append("paired_replay_comparison_failed")
        receipt["warnings"].extend(comparison.get("warnings") or [])
        for result in (uncapped_result, capped_result):
            receipt["warnings"].extend(result.get("validation_warnings") or [])
        receipt["uncapped_replay_validation_status"] = uncapped_result.get("validation_status")
        receipt["capped_replay_validation_status"] = capped_result.get("validation_status")
        receipt["uncapped_replay_provenance_audit_status"] = uncapped_result.get("provenance_audit_status")
        receipt["capped_replay_provenance_audit_status"] = capped_result.get("provenance_audit_status")
        receipt["uncapped_replay_temporal_calibration_audit_status"] = uncapped_result.get("temporal_calibration_audit_status")
        receipt["capped_replay_temporal_calibration_audit_status"] = capped_result.get("temporal_calibration_audit_status")
        receipt["status"] = (
            "FAIL" if receipt["errors"]
            else "PASS_WITH_WARNINGS" if receipt["warnings"] or comparison.get("status") == "PASS_WITH_WARNINGS"
            else "PASS"
        )
        receipt["actionable"] = False
        receipt["execution_connected"] = False
        receipt["automatic_model_selection"] = False

        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        receipt["receipt_path"] = str(receipt_path.resolve())
        receipt["comparison_summary_lines"] = _comparison_summary_lines(comparison, comparison_path)
        return receipt
    except Exception as exc:
        receipt["status"] = "FAIL"
        receipt["errors"].append(f"{type(exc).__name__}:{exc}")
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        receipt["receipt_path"] = str(receipt_path.resolve())
        return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fresh uncapped and elapsed-time-capped offline research runs with matched settings, "
            "then compare them on identical eligible held-out timestamps."
        )
    )
    parser.add_argument("--baseline-manifest", required=True, help="Historical all-label baseline manifest used only to reconstruct settings")
    parser.add_argument("--max-label-elapsed-hours", type=float, default=1.0)
    parser.add_argument("--run-id-prefix", help="Optional safe prefix for the paired replay's new artifact IDs")
    parser.add_argument("--random-seed", type=int, help="Shared random seed for both fresh replay runs")
    parser.add_argument("--bootstrap-resamples", type=int, help="Shared bootstrap count for both fresh replay runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_controlled_paired_replay(
            args.baseline_manifest,
            max_label_elapsed_hours=args.max_label_elapsed_hours,
            run_id_prefix=args.run_id_prefix,
            random_seed=args.random_seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    except Exception as exc:
        report = {
            "status": "FAIL",
            "scope": "GOLDCREST_CONTROLLED_PAIRED_LABEL_ELIGIBILITY_REPLAY",
            "errors": [f"{type(exc).__name__}:{exc}"],
            "warnings": [],
            "actionable": False,
            "execution_connected": False,
            "automatic_model_selection": False,
        }

    print(json.dumps({
        "status": report.get("status"),
        "scope": report.get("scope"),
        "historical_baseline_run_id": report.get("historical_baseline_run_id"),
        "uncapped_replay_run_id": report.get("uncapped_replay_run_id"),
        "capped_replay_run_id": report.get("capped_replay_run_id"),
        "comparison_status": report.get("comparison_status"),
        "receipt_path": report.get("receipt_path"),
        "errors": report.get("errors", []),
        "warnings": report.get("warnings", []),
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
    }, indent=2, sort_keys=True))
    for line in report.get("comparison_summary_lines", []):
        print(line)
    return 0 if report.get("status") in {"PASS", "PASS_WITH_WARNINGS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

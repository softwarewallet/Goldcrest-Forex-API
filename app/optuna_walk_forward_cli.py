"""CLI for provenance-bound, Optuna-tuned purged walk-forward research.

Consumes an already-labeled historical CSV. It never fetches data, creates labels,
serves predictions, changes live configuration, or connects to broker/execution APIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from app.dataset_provenance import (
    build_dataset_provenance,
    parse_retrieved_at_utc,
    verify_dataset_provenance,
)
from app.historical_tabular_research import load_historical_dataset
from app.optuna_walk_forward import run_optuna_walk_forward


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline Optuna tuning inside purged chronological walk-forward folds."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--features", required=True, help="Comma-separated numeric feature columns")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--data-source", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--retrieved-at-utc", required=True)
    parser.add_argument("--timestamp-timezone", required=True)
    parser.add_argument("--duplicate-policy", required=True)
    parser.add_argument("--adjustment-policy", required=True)
    parser.add_argument("--license-or-terms-reference", required=True)
    parser.add_argument("--initial-train-bars", type=int, required=True)
    parser.add_argument("--validation-bars", type=int, required=True)
    parser.add_argument("--test-bars", type=int, required=True)
    parser.add_argument("--step-bars", type=int, required=True)
    parser.add_argument("--horizon-bars", type=int, required=True)
    parser.add_argument("--model-type", choices=("xgboost", "lightgbm"), default="xgboost")
    parser.add_argument("--trials-per-fold", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20261021)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    feature_names = tuple(part.strip() for part in args.features.split(",") if part.strip())
    bars, examples, dataset_sha256 = load_historical_dataset(args.csv, feature_names)
    symbol, timeframe = bars[0].symbol, bars[0].timeframe

    provenance = build_dataset_provenance(
        source_name=args.data_source,
        source_version=args.dataset_version,
        snapshot_sha256=dataset_sha256,
        retrieved_at=parse_retrieved_at_utc(args.retrieved_at_utc),
        timestamp_timezone=args.timestamp_timezone,
        duplicate_policy=args.duplicate_policy,
        adjustment_policy=args.adjustment_policy,
        license_or_terms_reference=args.license_or_terms_reference,
        instruments=[symbol],
        timeframe=timeframe,
        row_count=len(bars),
    )
    provenance_verification = verify_dataset_provenance(
        provenance,
        actual_snapshot_sha256=dataset_sha256,
        actual_row_count=len(bars),
        actual_instruments=[symbol],
        actual_timeframe=timeframe,
    )

    result = run_optuna_walk_forward(
        bars,
        examples,
        feature_names=feature_names,
        run_id=args.run_id,
        dataset_sha256=dataset_sha256,
        model_type=args.model_type,
        initial_train_bars=args.initial_train_bars,
        validation_bars=args.validation_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        horizon_bars=args.horizon_bars,
        n_trials_per_fold=args.trials_per_fold,
        random_seed=args.random_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    payload = result.to_dict()
    payload["dataset_provenance"] = provenance.to_dict()
    payload["dataset_provenance_verification"] = provenance_verification
    payload["feature_timestamp_rule"] = (
        "Feature values must have been available at their observation timestamp; "
        "the CSV producer must verify point-in-time feature construction."
    )
    payload["selection_protocol"] = (
        "Independent Optuna search per fold using that fold's train/validation partitions; "
        "refit on labels matured by test start; evaluate the untouched test partition once."
    )
    payload["actionable"] = False
    payload["test_metrics_used_for_tuning"] = False
    payload["report_warning"] = (
        "Offline predictive research only. Metrics are not evidence of profitability; "
        "do not connect this report to broker execution."
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["report_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "run_id": args.run_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "model_type": args.model_type,
        "dataset_sha256": dataset_sha256,
        "fold_count": result.fold_count,
        "report_path": str(destination),
        "report_payload_sha256": payload["report_payload_sha256"],
        "actionable": False,
        "test_metrics_used_for_tuning": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

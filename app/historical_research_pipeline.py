"""Orchestrate the offline raw-OHLC -> causal-features -> matured-labels -> evaluation pipeline.

This is a convenience wrapper around the independently auditable preparation and
walk-forward tools. It does not fetch market data, serve predictions, or trade.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.historical_feature_preparation import FEATURE_COLUMNS, prepare_features
from app.historical_label_preparation import label_historical_features
from app.historical_tabular_research import main as run_research


def run_historical_research_pipeline(
    input_csv: str | Path,
    *,
    features_csv: str | Path,
    labeled_csv: str | Path,
    feature_report: str | Path,
    label_report: str | Path,
    report: str | Path,
    run_id: str,
    data_source: str,
    dataset_version: str,
    horizon_bars: int,
    neutral_band_bps: float,
    neutral_policy: str,
    initial_train_bars: int,
    validation_bars: int,
    test_bars: int,
    step_bars: int,
    block_length: int,
    spread_cost_notes: str,
    slippage_and_commission_notes: str,
    bootstrap_resamples: int = 5000,
    ledger: str | Path = "data/research/prediction-ledger.sqlite",
) -> dict[str, object]:
    """Run each explicit offline stage and return a provenance-linked summary."""
    raw = Path(input_csv).resolve()
    features_path = Path(features_csv).resolve()
    labeled_path = Path(labeled_csv).resolve()
    feature_report_path = Path(feature_report).resolve()
    label_report_path = Path(label_report).resolve()
    report_path = Path(report).resolve()
    evaluation_report_path = report_path.with_name(report_path.stem + ".evaluation.json")
    ledger_path = Path(ledger).resolve()

    outputs = [
        features_path, labeled_path, feature_report_path, label_report_path,
        report_path, evaluation_report_path, ledger_path,
    ]
    if len(set(outputs)) != len(outputs):
        raise ValueError("all output/report/ledger paths must be distinct")
    if raw in outputs:
        raise ValueError("input CSV must not collide with any output/report/ledger path")
    if not run_id.strip() or not data_source.strip() or not dataset_version.strip():
        raise ValueError("run_id, data_source, and dataset_version must be non-empty")
    if initial_train_bars < 1 or validation_bars < 1 or test_bars < 1 or step_bars < 1:
        raise ValueError("train, validation, test, and step sizes must all be positive")
    if block_length < 1 or bootstrap_resamples < 1:
        raise ValueError("block_length and bootstrap_resamples must be positive")

    feature_audit = prepare_features(raw, features_path)
    feature_report_path.parent.mkdir(parents=True, exist_ok=True)
    feature_report_path.write_text(
        json.dumps(feature_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    label_audit = label_historical_features(
        features_path, labeled_path, horizon_bars=horizon_bars,
        neutral_band_bps=neutral_band_bps, neutral_policy=neutral_policy,
    )
    label_report_path.parent.mkdir(parents=True, exist_ok=True)
    label_report_path.write_text(
        json.dumps(label_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    feature_names = ",".join(FEATURE_COLUMNS)
    research_args = [
        "--csv", str(labeled_path),
        "--features", feature_names,
        "--run-id", run_id,
        "--data-source", data_source,
        "--dataset-version", dataset_version,
        "--label-definition",
        f"close[t+{horizon_bars}] vs close[t], neutral_band_bps={neutral_band_bps}, neutral_policy={neutral_policy}; class 1 means UP",
        "--class-1-means", "UP",
        "--initial-train-bars", str(initial_train_bars),
        "--validation-bars", str(validation_bars),
        "--test-bars", str(test_bars),
        "--step-bars", str(step_bars),
        "--horizon-bars", str(horizon_bars),
        "--block-length", str(block_length),
        "--bootstrap-resamples", str(bootstrap_resamples),
        "--ledger", str(ledger_path),
        "--report", str(evaluation_report_path),
        "--spread-cost-notes", spread_cost_notes,
        "--slippage-and-commission-notes", slippage_and_commission_notes,
    ]
    result_code = run_research(research_args)
    if result_code != 0:
        raise RuntimeError(f"historical evaluation CLI failed with exit code {result_code}")
    evaluation = json.loads(evaluation_report_path.read_text(encoding="utf-8"))

    combined = {
        "pipeline": "raw_ohlc_to_walk_forward_research",
        "raw_input_path": str(raw),
        "raw_input_sha256": feature_audit["input_sha256"],
        "feature_preparation": feature_audit,
        "label_preparation": label_audit,
        "evaluation": evaluation,
        "artifacts": {
            "features_csv": str(features_path),
            "labeled_csv": str(labeled_path),
            "feature_report": str(feature_report_path),
            "label_report": str(label_report_path),
            "evaluation_report": str(evaluation_report_path),
            "prediction_ledger": str(ledger_path),
            "combined_report": str(report_path),
        },
        "scope": "OFFLINE_RESEARCH_ONLY",
        "actionable": False,
        "limitations": [
            "Requires genuine provider-exported historical OHLC; synthetic data is not evidence of real-world performance.",
            "Labels represent close-to-close direction, not cost-adjusted trade P&L.",
            "Cost notes are provenance only; the evaluation is not a strategy backtest.",
            "Feature causality is defined by this preparation utility; external data provenance must still be verified.",
            "No model is selected or promoted automatically and no output authorizes trading.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return combined


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline raw OHLC -> causal features -> explicit labels -> walk-forward model evaluation."
    )
    parser.add_argument("--input-csv", required=True, help="Genuine provider-exported raw OHLC CSV")
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--labeled-csv", required=True)
    parser.add_argument("--feature-report", required=True)
    parser.add_argument("--label-report", required=True)
    parser.add_argument("--report", required=True, help="Combined JSON pipeline report")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-source", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--horizon-bars", required=True, type=int)
    parser.add_argument("--neutral-band-bps", required=True, type=float)
    parser.add_argument("--neutral-policy", required=True, choices=("up", "down", "error"))
    parser.add_argument("--initial-train-bars", required=True, type=int)
    parser.add_argument("--validation-bars", required=True, type=int)
    parser.add_argument("--test-bars", required=True, type=int)
    parser.add_argument("--step-bars", required=True, type=int)
    parser.add_argument("--block-length", required=True, type=int)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--ledger", default="data/research/prediction-ledger.sqlite")
    parser.add_argument("--spread-cost-notes", required=True)
    parser.add_argument("--slippage-and-commission-notes", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_historical_research_pipeline(
        args.input_csv, features_csv=args.features_csv, labeled_csv=args.labeled_csv,
        feature_report=args.feature_report, label_report=args.label_report, report=args.report,
        run_id=args.run_id, data_source=args.data_source, dataset_version=args.dataset_version,
        horizon_bars=args.horizon_bars, neutral_band_bps=args.neutral_band_bps,
        neutral_policy=args.neutral_policy, initial_train_bars=args.initial_train_bars,
        validation_bars=args.validation_bars, test_bars=args.test_bars, step_bars=args.step_bars,
        block_length=args.block_length, spread_cost_notes=args.spread_cost_notes,
        slippage_and_commission_notes=args.slippage_and_commission_notes,
        bootstrap_resamples=args.bootstrap_resamples, ledger=args.ledger,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

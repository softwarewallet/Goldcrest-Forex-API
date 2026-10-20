"""CSV-driven, offline XGBoost/LightGBM walk-forward research CLI.

The input dataset must contain point-in-time features and externally defined, matured
binary labels. This tool never creates labels, fetches market data, serves predictions,
or authorizes trading.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Sequence

from app.dataset_provenance import build_dataset_provenance, parse_retrieved_at_utc, verify_dataset_provenance
from app.evaluation import ForexBar, validate_bars
from app.model_walk_forward import TimedBinaryExample
from app.models.tabular import feature_schema_fingerprint
from app.offline_experiment_runner import run_offline_experiment
from app.prediction_ledger import PredictionLedger
from app.research_model_stack import build_tabular_model_factories
from app.walk_forward import generate_walk_forward_folds
from app.walk_forward_audit import audit_walk_forward_plan


REQUIRED_COLUMNS = {
    "symbol", "timeframe", "timestamp", "open", "high", "low", "close",
    "label", "label_available_at",
}


def parse_timestamp(value: str, *, field: str) -> datetime:
    """Parse an ISO-8601 timestamp and reject timezone-naive values."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset")
    return parsed


def load_historical_dataset(
    csv_path: str | Path,
    feature_names: Sequence[str],
) -> tuple[list[ForexBar], list[TimedBinaryExample], str]:
    """Load a strict CSV without sorting, imputing, dropping, or generating labels."""
    path = Path(csv_path)
    raw_bytes = path.read_bytes()
    dataset_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    names = tuple(feature_names)
    feature_schema_fingerprint(names)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = (REQUIRED_COLUMNS | set(names)) - columns
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("CSV contains no data rows")

    bars: list[ForexBar] = []
    examples: list[TimedBinaryExample] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            symbol = row["symbol"].strip().upper()
            timeframe = row["timeframe"].strip().upper()
            timestamp = parse_timestamp(row["timestamp"], field=f"line {line_number} timestamp")
            label_available_at = parse_timestamp(
                row["label_available_at"], field=f"line {line_number} label_available_at"
            )
            label_text = row["label"].strip()
            if label_text not in {"0", "1"}:
                raise ValueError("label must be exactly 0 or 1")
            label = int(label_text)

            numeric = {}
            for field in ("open", "high", "low", "close"):
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError(f"{field} must be finite")
                numeric[field] = value
            features: dict[str, float] = {}
            for name in names:
                value = float(row[name])
                if not math.isfinite(value):
                    raise ValueError(f"feature {name!r} must be finite")
                features[name] = value
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid CSV row at line {line_number}: {exc}") from exc

        bars.append(ForexBar(
            symbol=symbol, timeframe=timeframe, timestamp=timestamp,
            open=numeric["open"], high=numeric["high"],
            low=numeric["low"], close=numeric["close"],
        ))
        examples.append(TimedBinaryExample(
            timestamp=timestamp, features=features, label=label,
            label_available_at=label_available_at,
        ))

    quality = validate_bars(bars)
    if quality.invalid_rows or quality.duplicate_timestamps or quality.non_monotonic_timestamps:
        raise ValueError(
            "CSV failed OHLC/order checks: "
            f"invalid_rows={quality.invalid_rows}, "
            f"duplicates={quality.duplicate_timestamps}, "
            f"non_monotonic={quality.non_monotonic_timestamps}"
        )
    if len(quality.symbols) != 1 or len(quality.timeframes) != 1:
        raise ValueError("one CSV run must contain exactly one symbol and timeframe")
    if [bar.timestamp for bar in bars] != sorted(bar.timestamp for bar in bars):
        raise ValueError("CSV rows must already be chronological; automatic sorting is forbidden")
    for example in examples:
        if example.label_available_at < example.timestamp:
            raise ValueError("label_available_at cannot precede its feature timestamp")

    return bars, examples, dataset_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run research-only XGBoost and LightGBM walk-forward evaluation on CSV data."
    )
    parser.add_argument("--csv", required=True, help="CSV containing OHLC, point-in-time features and explicit labels")
    parser.add_argument("--features", required=True, help="Comma-separated numeric feature column names")
    parser.add_argument("--run-id", required=True, help="Unique immutable run ID; use a new ID for every run")
    parser.add_argument("--data-source", required=True, help="Origin of historical data")
    parser.add_argument("--dataset-version", required=True, help="Provider/export version or dataset revision")
    parser.add_argument("--retrieved-at-utc", required=True, help="Declared snapshot retrieval time in UTC ISO-8601")
    parser.add_argument("--timestamp-timezone", required=True, help="Timezone used by the source/export timestamps")
    parser.add_argument("--duplicate-policy", required=True, help="Explicit duplicate timestamp policy")
    parser.add_argument("--adjustment-policy", required=True, help="Corporate-action/price-adjustment policy or explicit none")
    parser.add_argument("--license-or-terms-reference", required=True, help="Data license or terms reference; do not invent")
    parser.add_argument("--label-definition", required=True, help="Exact definition of the externally prepared binary label")
    parser.add_argument("--class-1-means", required=True, choices=("UP", "DOWN"), help="Explicit semantic meaning of label 1")
    parser.add_argument("--initial-train-bars", type=int, required=True)
    parser.add_argument("--validation-bars", type=int, required=True)
    parser.add_argument("--test-bars", type=int, required=True)
    parser.add_argument("--step-bars", type=int, required=True)
    parser.add_argument("--horizon-bars", type=int, required=True, help="Label horizon in bars; also used for purging")
    parser.add_argument("--block-length", type=int, required=True, help="Block length for paired OOS uncertainty analysis")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--ledger", default="data/research/prediction-ledger.sqlite")
    parser.add_argument("--report", required=True, help="Output JSON report path")
    parser.add_argument("--spread-cost-notes", required=True, help="Document instrument-specific spread/cost assumptions; metrics remain predictive, not P&L")
    parser.add_argument("--slippage-and-commission-notes", required=True, help="Document slippage/commission assumptions and limitations")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    feature_names = tuple(part.strip() for part in args.features.split(",") if part.strip())
    feature_schema_fingerprint(feature_names)
    bars, examples, dataset_sha256 = load_historical_dataset(args.csv, feature_names)

    symbol, timeframe = bars[0].symbol, bars[0].timeframe
    provenance_record = build_dataset_provenance(
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
        provenance_record,
        actual_snapshot_sha256=dataset_sha256,
        actual_row_count=len(bars),
        actual_instruments=[symbol],
        actual_timeframe=timeframe,
    )
    plan = generate_walk_forward_folds(
        bars,
        initial_train_bars=args.initial_train_bars,
        validation_bars=args.validation_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        horizon_bars=args.horizon_bars,
    )
    temporal_integrity_audit = audit_walk_forward_plan(plan, examples)
    factories = build_tabular_model_factories(feature_names, model_version="candidate-v1")
    manifest = {
        "run_id": args.run_id,
        "dataset_id": args.dataset_version,
        "dataset_sha256": dataset_sha256,
        "data_source": args.data_source,
        "dataset_provenance": provenance_record.to_dict(),
        "dataset_provenance_verification": provenance_verification,
        "symbol": symbol,
        "timeframe": timeframe,
        "feature_names": list(feature_names),
        "feature_schema_sha256": feature_schema_fingerprint(feature_names),
        "temporal_integrity_audit": temporal_integrity_audit,
        "label_definition": args.label_definition,
        "class_1_means": args.class_1_means,
        "horizon_bars": args.horizon_bars,
        "scope": "OFFLINE_RESEARCH_ONLY",
        "actionable": False,
    }
    evaluation_contract = {
        "data_source": args.data_source,
        "dataset_version": args.dataset_version,
        "dataset_sha256": dataset_sha256,
        "retrieved_at_utc": provenance_record.retrieved_at,
        "timestamp_timezone": provenance_record.timestamp_timezone,
        "duplicate_policy": provenance_record.duplicate_policy,
        "adjustment_policy": provenance_record.adjustment_policy,
        "license_or_terms_reference": provenance_record.license_or_terms_reference,
        "dataset_provenance": provenance_record.to_dict(),
        "dataset_provenance_verification": provenance_verification,
        "symbol": symbol,
        "timeframe": timeframe,
        "feature_names": list(feature_names),
        "feature_timestamp_rule": "Each row's features must be available no later than timestamp; verified by dataset producer, not inferable from CSV alone.",
        "label_definition": args.label_definition,
        "class_1_means": args.class_1_means,
        "label_maturity_rule": "label_available_at must be timestamped; training examples are used only when the label has matured by the fold's training cutoff.",
        "horizon_bars": args.horizon_bars,
        "fold_rule": "Expanding-window chronological train/validation/test with horizon-sized boundary purging.",
        "spread_cost_notes": args.spread_cost_notes,
        "slippage_and_commission_notes": args.slippage_and_commission_notes,
        "metric_scope": "Predictive probability metrics only (Brier score, log loss, fixed-0.5 accuracy). Cost notes are provenance only; this is not a strategy/P&L backtest.",
        "model_selection_rule": "No automatic model selection; do not tune against final test metrics.",
        "actionable": False,
    }
    result = run_offline_experiment(
        plan,
        examples,
        feature_names=feature_names,
        manifest=manifest,
        model_factories=factories,
        ledger=PredictionLedger(args.ledger),
        evaluation_contract=evaluation_contract,
        exclusions={"invalid_rows": 0, "duplicate_timestamps": 0, "non_monotonic_timestamps": 0},
        block_length=args.block_length,
        bootstrap_resamples=args.bootstrap_resamples,
        report_path=args.report,
    )
    print(json.dumps({
        "run_id": result.run_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "dataset_sha256": dataset_sha256,
        "fold_count": len(plan.folds),
        "model_names": sorted(factories),
        "prediction_rows_inserted": result.prediction_rows_inserted,
        "ledger_path": result.ledger_path,
        "report_path": result.report_path,
        "actionable": False,
        "warning": "Predictive metrics are not profitability evidence. Review leakage, label semantics and data provenance before interpreting results.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

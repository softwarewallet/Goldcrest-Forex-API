"""Explicit, offline-first orchestration for the Goldcrest research pipeline.

Optional cTrader retrieval is DEMO-only and occurs only with --fetch-ctrader. All model
experiments and reports remain research-only and are never connected to execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from app.ctrader_history_ingest import download_ctrader_history
from app.historical_tabular_research import load_historical_dataset
from app.research_dataset_builder import prepare_research_dataset
from app.research_experiment_matrix import run_experiment_matrix, write_experiment_matrix_report
from app.research_provenance_auditor import audit_experiment_matrix_file
from app.research_data_quality import audit_ohlc_csv_file
from app.research_temporal_leakage_auditor import audit_temporal_integrity_file


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _feature_names_from_dataset(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
    required = {"symbol", "timeframe", "timestamp", "open", "high", "low", "close", "label", "label_available_at"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"prepared dataset is missing required columns: {sorted(missing)}")
    features = [name for name in columns if name not in required]
    if not features:
        raise ValueError("prepared dataset contains no feature columns")
    return features


def run_research_pipeline(
    *,
    symbol: str,
    timeframe: str,
    horizon_bars: int,
    class_1_means: str,
    terms_reference: str,
    source_version: str,
    output_dir: str | Path,
    run_id: str,
    retrieved_at_utc: str | None = None,
    ohlc_csv: str | Path | None = None,
    fetch_ctrader: bool = False,
    lookback_days: int = 365,
    initial_train_bars: int,
    validation_bars: int,
    test_bars: int,
    step_bars: int,
    trials_per_fold: int = 5,
    random_seed: int = 20261021,
    bootstrap_resamples: int = 500,
    max_experiments: int = 20,
    include_ablation: bool = True,
    strict_provenance_audit: bool = False,
    strict_temporal_audit: bool = False,
) -> dict[str, Any]:
    """Run ingestion (optional), dataset preparation, and fixed-fold model comparison."""
    if fetch_ctrader == (ohlc_csv is not None):
        raise ValueError("provide exactly one of ohlc_csv or fetch_ctrader=True")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if not terms_reference.strip():
        raise ValueError("terms_reference must be the actual applicable provider terms")
    instrument = symbol.strip().upper()
    period = timeframe.strip().upper()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if fetch_ctrader:
        raw_csv = destination / f"{instrument}_{period}_ctrader_ohlc.csv"
        ingestion = download_ctrader_history(
            symbol=instrument,
            timeframe=period,
            lookback_days=lookback_days,
            output_csv=raw_csv,
        )
        source_name = "cTrader Open API"
        retrieved_at_utc = ingestion.get("retrieved_at_utc") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        actual_source_version = source_version or "ctrader-open-api==0.9.2"
        raw_sha256 = _sha256(raw_csv)
    else:
        raw_csv = Path(ohlc_csv)  # type: ignore[arg-type]
        if not raw_csv.is_file():
            raise ValueError(f"OHLC CSV does not exist: {raw_csv}")
        if not retrieved_at_utc:
            raise ValueError("retrieved_at_utc is required when using an existing OHLC CSV")
        source_name = "Operator-supplied canonical OHLC CSV"
        actual_source_version = source_version
        raw_sha256 = _sha256(raw_csv)
        ingestion = {
            "scope": "LOCAL_OHLC_INPUT",
            "rows": None,
            "snapshot_sha256": raw_sha256,
            "host": "not_applicable",
            "actionable": False,
        }

    data_quality_path = destination / f"{run_id}_data_quality_audit.json"
    data_quality = audit_ohlc_csv_file(
        raw_csv, data_quality_path, expected_symbol=instrument, expected_timeframe=period
    )
    if data_quality["status"] == "FAIL":
        raise ValueError(
            f"historical OHLC quality audit failed; inspect {data_quality_path}: "
            f"{data_quality['errors']}"
        )

    dataset_csv = destination / f"{instrument}_{period}_research_dataset.csv"
    dataset_manifest_path = destination / f"{instrument}_{period}_research_dataset.manifest.json"
    dataset_manifest = prepare_research_dataset(
        raw_csv,
        dataset_csv,
        dataset_manifest_path,
        symbol=instrument,
        timeframe=period,
        horizon_bars=horizon_bars,
        class_1_means=class_1_means,
        data_source=source_name,
        source_version=actual_source_version,
        retrieved_at_utc=retrieved_at_utc,
        terms_reference=terms_reference,
    )
    feature_names = _feature_names_from_dataset(dataset_csv)
    bars, examples, dataset_sha256 = load_historical_dataset(dataset_csv, feature_names)
    report = run_experiment_matrix(
        bars,
        examples,
        feature_names=feature_names,
        run_id=run_id,
        dataset_sha256=dataset_sha256,
        model_types=("xgboost", "lightgbm"),
        initial_train_bars=initial_train_bars,
        validation_bars=validation_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        horizon_bars=horizon_bars,
        n_trials_per_fold=trials_per_fold,
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
        include_ablation=include_ablation,
        max_experiments=max_experiments,
    )
    matrix_report_path = destination / f"{run_id}_experiment_matrix.json"
    write_experiment_matrix_report(report, matrix_report_path)
    provenance_audit_path = destination / f"{run_id}_provenance_audit.json"
    provenance_audit = audit_experiment_matrix_file(matrix_report_path, provenance_audit_path)
    provenance_audit_sha256 = _sha256(provenance_audit_path)
    temporal_audit_path = destination / f"{run_id}_temporal_audit.json"
    temporal_audit = audit_temporal_integrity_file(matrix_report_path, temporal_audit_path)
    temporal_audit_sha256 = _sha256(temporal_audit_path)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "scope": "GOLDCREST_OFFLINE_RESEARCH_PIPELINE",
        "symbol": instrument,
        "timeframe": period,
        "source": source_name,
        "source_version": actual_source_version,
        "retrieved_at_utc": retrieved_at_utc,
        "raw_ohlc_path": str(raw_csv),
        "raw_ohlc_sha256": raw_sha256,
        "data_quality_audit_path": str(data_quality_path),
        "data_quality_audit_sha256": _sha256(data_quality_path),
        "data_quality_audit_status": data_quality["status"],
        "data_quality_observed_gap_count": data_quality["observed_gap_count"],
        "data_quality_warnings": data_quality["warnings"],
        "dataset_path": str(dataset_csv),
        "dataset_sha256": _sha256(dataset_csv),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "fold_count": report.fold_count,
        "experiment_count": len(report.experiments),
        "experiment_matrix_report": str(matrix_report_path),
        "experiment_matrix_report_sha256": _sha256(matrix_report_path),
        "provenance_audit_path": str(provenance_audit_path),
        "provenance_audit_sha256": provenance_audit_sha256,
        "provenance_audit_status": provenance_audit["status"],
        "temporal_audit_path": str(temporal_audit_path),
        "temporal_audit_sha256": temporal_audit_sha256,
        "temporal_audit_status": temporal_audit["status"],
        "strict_temporal_audit": strict_temporal_audit,
        "strict_provenance_audit": strict_provenance_audit,
        "ingestion": ingestion,
        "label_horizon_bars": horizon_bars,
        "class_1_means": class_1_means,
        "test_metrics_used_for_selection": False,
        "automatic_winner_selection": False,
        "actionable": False,
        "execution_connected": False,
        "warning": "Offline research only. No model is promoted to live trading and no execution API is called.",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["pipeline_manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest_path = destination / f"{run_id}_pipeline_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Goldcrest's explicit offline research pipeline.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ohlc-csv", help="Existing canonical OHLC CSV; no network access")
    source.add_argument("--fetch-ctrader", action="store_true", help="Explicitly download from cTrader DEMO")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True, choices=("M1", "M5", "M15", "M30", "H1", "H4", "D1"))
    parser.add_argument("--horizon-bars", type=int, required=True)
    parser.add_argument("--class-1-means", choices=("UP", "DOWN"), required=True)
    parser.add_argument("--retrieved-at-utc", help="Required for --ohlc-csv; ISO-8601 timestamp")
    parser.add_argument("--terms-reference", required=True, help="Actual applicable data-provider terms reference")
    parser.add_argument("--source-version", required=True, help="Provider/export version; never invent provenance")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--output-dir", default="data/research/runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--initial-train-bars", type=int, required=True)
    parser.add_argument("--validation-bars", type=int, required=True)
    parser.add_argument("--test-bars", type=int, required=True)
    parser.add_argument("--step-bars", type=int, required=True)
    parser.add_argument("--trials-per-fold", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20261021)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--max-experiments", type=int, default=20)
    parser.add_argument("--no-ablation", action="store_true", help="Compare only the full causal feature set")
    parser.add_argument("--strict-provenance-audit", action="store_true", help="Return a nonzero exit code if the structural provenance audit fails")
    parser.add_argument("--strict-temporal-audit", action="store_true", help="Return exit code 4 if temporal-boundary audit fails")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_research_pipeline(
            symbol=args.symbol,
            timeframe=args.timeframe,
            horizon_bars=args.horizon_bars,
            class_1_means=args.class_1_means,
            terms_reference=args.terms_reference,
            source_version=args.source_version,
            output_dir=args.output_dir,
            run_id=args.run_id,
            retrieved_at_utc=args.retrieved_at_utc,
            ohlc_csv=args.ohlc_csv,
            fetch_ctrader=args.fetch_ctrader,
            lookback_days=args.lookback_days,
            initial_train_bars=args.initial_train_bars,
            validation_bars=args.validation_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
            trials_per_fold=args.trials_per_fold,
            random_seed=args.random_seed,
            bootstrap_resamples=args.bootstrap_resamples,
            max_experiments=args.max_experiments,
            include_ablation=not args.no_ablation,
            strict_provenance_audit=args.strict_provenance_audit,
            strict_temporal_audit=args.strict_temporal_audit,
        )
    except Exception as exc:
        # Errors are concise and do not print credential-bearing objects.
        print(f"research pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    if args.strict_provenance_audit and result.get("provenance_audit_status") == "FAIL":
        print("research pipeline completed, but strict provenance audit failed; inspect the audit receipt", file=sys.stderr)
        return 3
    if args.strict_temporal_audit and result.get("temporal_audit_status") == "FAIL":
        print("research pipeline completed, but strict temporal audit failed; inspect the temporal audit receipt", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

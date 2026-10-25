"""Research-only point-in-time feature integrity audit for causal OHLC features.

Accepts a JSON object with a "bars" array or a bare array of OHLC records. This audit
checks the repository's causal feature builder, reproducibility, prefix invariance,
future perturbation invariance, and chronological split isolation. It does not certify
downstream model pipelines, label construction, session calendars, or execution behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from app.causal_ta_features import build_causal_ohlc_features
from app.evaluation import ForexBar, chronological_split


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bars(path: Path) -> tuple[list[ForexBar], str]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("input_json_invalid") from exc
    records = payload.get("bars") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("input_bars_missing")
    bars: list[ForexBar] = []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"bar_record_invalid:{index}")
        try:
            stamp = row["timestamp"]
            if not isinstance(stamp, str):
                raise TypeError
            timestamp = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            bars.append(ForexBar(
                symbol=str(row["symbol"]),
                timeframe=str(row["timeframe"]),
                timestamp=timestamp,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            ))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"bar_record_schema_invalid:{index}") from exc
    return bars, _sha256(raw)


def _batch_rows_equal(left: Any, right: Any, end: int) -> bool:
    return (
        left.timestamps_utc[:end] == right.timestamps_utc[:end]
        and left.feature_names == right.feature_names
        and left.rows[:end] == right.rows[:end]
    )


def _sample_cutoffs(row_count: int, sample_count: int) -> list[int]:
    if row_count < 3:
        return []
    count = min(max(1, sample_count), row_count - 1)
    values = {max(1, min(row_count - 1, round(i * (row_count - 1) / (count + 1))))
              for i in range(1, count + 1)}
    values.update({min(row_count - 1, 20), min(row_count - 1, 35), row_count // 2})
    return sorted(value for value in values if 1 <= value < row_count)


def audit_feature_temporal_integrity(
    bars: list[ForexBar],
    *,
    source_data_sha256: str,
    feature_source_path: Path | None = None,
    horizon_bars: int = 1,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    sample_count: int = 24,
) -> dict[str, Any]:
    """Audit point-in-time invariance and chronological partition isolation."""
    errors: list[str] = []
    warnings = [
        "downstream feature transformations and model preprocessing are not inspected",
        "timestamp gaps do not establish missing candles or market-session expectations",
        "audit is deterministic and descriptive; it does not establish predictive validity",
    ]
    if not isinstance(source_data_sha256, str) or len(source_data_sha256) != 64:
        raise ValueError("source_data_sha256_invalid")
    if sample_count < 1:
        raise ValueError("sample_count_must_be_positive")

    full = build_causal_ohlc_features(bars)
    cutoffs = _sample_cutoffs(len(bars), sample_count)
    prefix_checks: list[dict[str, Any]] = []
    perturbation_checks: list[dict[str, Any]] = []

    # Prefix invariance checks the same historical prefix both alone and with future rows.
    for cutoff in cutoffs:
        prefix = build_causal_ohlc_features(bars[:cutoff])
        matches = _batch_rows_equal(prefix, full, cutoff)
        prefix_checks.append({"cutoff_row_exclusive": cutoff, "prefix_matches_full_history": matches})
        if not matches:
            errors.append(f"prefix_invariance_failed:{cutoff}")

        # Multiply all OHLC prices in the future suffix by a constant, preserving candle
        # geometry while materially changing future prices. Historical features must not move.
        factor = 1.271828
        altered = list(bars[:cutoff])
        for bar in bars[cutoff:]:
            altered.append(ForexBar(
                symbol=bar.symbol,
                timeframe=bar.timeframe,
                timestamp=bar.timestamp,
                open=bar.open * factor,
                high=bar.high * factor,
                low=bar.low * factor,
                close=bar.close * factor,
            ))
        perturbed_batch = build_causal_ohlc_features(altered)
        matches = _batch_rows_equal(full, perturbed_batch, cutoff)
        perturbation_checks.append({
            "cutoff_row_exclusive": cutoff,
            "future_suffix_ohlc_scaled": True,
            "historical_features_unchanged": matches,
        })
        if not matches:
            errors.append(f"future_perturbation_invariance_failed:{cutoff}")

    # The repository's chronological split purges horizon rows at each partition boundary.
    split = chronological_split(
        bars,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        horizon_bars=horizon_bars,
    )
    train_end = split.train[-1].timestamp
    validation_start = split.validation[0].timestamp
    validation_end = split.validation[-1].timestamp
    test_start = split.test[0].timestamp
    ordered = train_end < validation_start < validation_end < test_start
    if not ordered:
        errors.append("chronological_partition_order_failed")
    partitions = {
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
        "purged_rows": split.purged_rows,
        "train_end_utc": train_end.isoformat(),
        "validation_start_utc": validation_start.isoformat(),
        "validation_end_utc": validation_end.isoformat(),
        "test_start_utc": test_start.isoformat(),
        "strictly_ordered_non_overlapping": ordered,
        "horizon_bars_purged_at_boundaries": horizon_bars,
    }

    feature_source_sha256 = None
    if feature_source_path is not None:
        try:
            feature_source_sha256 = _sha256(feature_source_path.read_bytes())
        except OSError as exc:
            raise ValueError(f"feature_source_unreadable:{feature_source_path}") from exc

    return {
        "schema_version": 1,
        "audit_name": "feature_temporal_integrity",
        "status": "FAIL" if errors else "PASS_WITH_WARNINGS",
        "errors": errors,
        "warnings": warnings,
        "dataset_rows": len(bars),
        "dataset_sha256": source_data_sha256,
        "feature_source_sha256": feature_source_sha256,
        "feature_source_path": str(feature_source_path) if feature_source_path else None,
        "feature_names": list(full.feature_names),
        "indicator_backend": full.indicator_backend,
        "prefix_invariance": {
            "checks": len(prefix_checks),
            "all_passed": bool(prefix_checks) and all(item["prefix_matches_full_history"] for item in prefix_checks),
            "results": prefix_checks,
        },
        "future_perturbation_invariance": {
            "checks": len(perturbation_checks),
            "all_passed": bool(perturbation_checks) and all(item["historical_features_unchanged"] for item in perturbation_checks),
            "results": perturbation_checks,
        },
        "chronological_partition_isolation": partitions,
        "actionable": False,
        "automatic_model_selection": False,
        "inferential": False,
        "execution_connected": False,
        "model_training_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, type=Path,
                        help="JSON array of OHLC bars, or object containing a bars array")
    parser.add_argument("--output", type=Path,
                        help="Output JSON path; defaults beside input with an audit suffix")
    parser.add_argument("--horizon-bars", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--sample-count", type=int, default=24,
                        help="Number of deterministic cutoffs, plus fixed boundary samples")
    args = parser.parse_args()
    output = args.output or args.input_json.with_name(args.input_json.stem + "_feature_temporal_integrity_audit.json")
    try:
        bars, dataset_digest = _read_bars(args.input_json)
        report = audit_feature_temporal_integrity(
            bars,
            source_data_sha256=dataset_digest,
            feature_source_path=Path(__file__).with_name("causal_ta_features.py"),
            horizon_bars=args.horizon_bars,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            sample_count=args.sample_count,
        )
        if output.exists():
            raise ValueError(f"output_already_exists:{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "report_path": str(output.resolve()),
            "dataset_rows": report["dataset_rows"],
            "prefix_invariance_passed": report["prefix_invariance"]["all_passed"],
            "future_perturbation_invariance_passed": report["future_perturbation_invariance"]["all_passed"],
            "chronological_partition_isolated": report["chronological_partition_isolation"]["strictly_ordered_non_overlapping"],
            "actionable": False,
            "execution_connected": False,
        }, indent=2))
        return 0 if report["status"] != "FAIL" else 1
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

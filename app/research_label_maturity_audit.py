"""Independent audit of forward-label maturity against the immutable OHLC timeline.

This offline audit verifies each prepared label against its exact row-index forward target.
It does not repair data, infer market sessions, train models, or authorize execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(value: object, *, field: str, line: int) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line_{line}_{field}_missing")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"line_{line}_{field}_invalid_iso8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"line_{line}_{field}_timezone_missing")
    return parsed.astimezone(timezone.utc)


def _read_ohlc(path: Path, *, symbol: str, timeframe: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "open", "high", "low", "close"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"raw_ohlc_missing_columns:{','.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            if row.get("symbol") and row["symbol"].strip().upper() != symbol:
                raise ValueError(f"raw_ohlc_symbol_mismatch_line_{line}")
            if row.get("timeframe") and row["timeframe"].strip().upper() != timeframe:
                raise ValueError(f"raw_ohlc_timeframe_mismatch_line_{line}")
            timestamp = _utc(row.get("timestamp"), field="timestamp", line=line)
            try:
                prices = {name: float(row[name]) for name in ("open", "high", "low", "close")}
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"raw_ohlc_price_invalid_line_{line}") from exc
            if any(not math.isfinite(value) or value <= 0 for value in prices.values()):
                raise ValueError(f"raw_ohlc_price_nonfinite_or_nonpositive_line_{line}")
            if prices["low"] > min(prices["open"], prices["close"]) or prices["high"] < max(prices["open"], prices["close"]) or prices["high"] < prices["low"]:
                raise ValueError(f"raw_ohlc_price_relationship_invalid_line_{line}")
            if rows and timestamp <= rows[-1]["timestamp"]:
                raise ValueError(f"raw_ohlc_timestamps_not_strictly_increasing_line_{line}")
            rows.append({"timestamp": timestamp, **prices})
    if not rows:
        raise ValueError("raw_ohlc_has_no_rows")
    return rows


def audit_label_maturity(
    raw_ohlc_path: str | Path,
    dataset_path: str | Path,
    *,
    symbol: str,
    timeframe: str,
    horizon_bars: int,
    class_1_means: str,
    max_label_elapsed_hours: float | None = None,
) -> dict[str, Any]:
    """Verify label, maturity timestamp, and optional eligibility mask against raw OHLC."""
    raw_path = Path(raw_ohlc_path).expanduser().resolve()
    data_path = Path(dataset_path).expanduser().resolve()
    instrument = symbol.strip().upper()
    period = timeframe.strip().upper()
    meaning = class_1_means.strip().upper()
    errors: list[str] = []

    if not instrument:
        errors.append("symbol_invalid")
    if period not in _TIMEFRAME_MINUTES:
        errors.append("timeframe_unsupported")
    if isinstance(horizon_bars, bool) or not isinstance(horizon_bars, int) or horizon_bars < 1:
        errors.append("horizon_bars_invalid")
    if meaning not in {"UP", "DOWN"}:
        errors.append("class_1_means_invalid")
    if max_label_elapsed_hours is not None and (
        isinstance(max_label_elapsed_hours, bool)
        or not isinstance(max_label_elapsed_hours, (int, float))
        or not math.isfinite(float(max_label_elapsed_hours))
        or float(max_label_elapsed_hours) <= 0
    ):
        errors.append("max_label_elapsed_hours_invalid")
    if errors:
        return _report(raw_path, data_path, errors, 0, 0, 0, None, None)

    try:
        raw_rows = _read_ohlc(raw_path, symbol=instrument, timeframe=period)
        raw_hash = _sha256(raw_path)
        dataset_hash = _sha256(data_path)
    except (OSError, ValueError) as exc:
        return _report(raw_path, data_path, [f"source_unreadable_or_invalid:{exc}"], 0, 0, 0, None, None)

    raw_index = {row["timestamp"]: index for index, row in enumerate(raw_rows)}
    checked = 0
    eligible_count = 0
    ineligible_count = 0
    seen: set[datetime] = set()
    try:
        with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"timestamp", "open", "high", "low", "close", "label", "label_available_at"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"dataset_missing_columns:{','.join(sorted(missing))}")
            fields = set(reader.fieldnames or ())
            has_eligibility = {"label_elapsed_hours", "label_eligible"}.issubset(fields)
            if max_label_elapsed_hours is not None and not has_eligibility:
                raise ValueError("dataset_eligibility_columns_missing_for_configured_cap")
            previous_dataset_timestamp: datetime | None = None
            for line, row in enumerate(reader, start=2):
                if row.get("symbol") and row["symbol"].strip().upper() != instrument:
                    errors.append(f"dataset_symbol_mismatch_line_{line}")
                if row.get("timeframe") and row["timeframe"].strip().upper() != period:
                    errors.append(f"dataset_timeframe_mismatch_line_{line}")
                start = _utc(row.get("timestamp"), field="timestamp", line=line)
                target = _utc(row.get("label_available_at"), field="label_available_at", line=line)
                if start in seen:
                    errors.append(f"dataset_duplicate_timestamp_line_{line}")
                if previous_dataset_timestamp is not None and start <= previous_dataset_timestamp:
                    errors.append(f"dataset_timestamps_not_strictly_chronological_line_{line}")
                previous_dataset_timestamp = start
                seen.add(start)
                if start not in raw_index:
                    errors.append(f"dataset_start_not_in_raw_timeline_line_{line}")
                    continue
                start_index = raw_index[start]
                target_index = start_index + horizon_bars
                if target_index >= len(raw_rows):
                    errors.append(f"label_target_beyond_raw_timeline_line_{line}")
                    continue
                expected_target = raw_rows[target_index]["timestamp"]
                if target != expected_target:
                    errors.append(f"label_maturity_timestamp_mismatch_line_{line}")
                try:
                    current_close = float(row["close"])
                    label = int(row["label"])
                except (TypeError, ValueError):
                    errors.append(f"dataset_close_or_label_invalid_line_{line}")
                    continue
                if label not in (0, 1):
                    errors.append(f"dataset_label_not_binary_line_{line}")
                raw_start = raw_rows[start_index]
                for price_field in ("open", "high", "low", "close"):
                    try:
                        dataset_price = float(row[price_field])
                    except (TypeError, ValueError, KeyError):
                        errors.append(f"dataset_{price_field}_invalid_line_{line}")
                        continue
                    raw_price = raw_start[price_field]
                    if not math.isfinite(dataset_price) or not math.isclose(
                        dataset_price, raw_price, rel_tol=0.0, abs_tol=1e-10
                    ):
                        errors.append(f"dataset_{price_field}_mismatch_raw_ohlc_line_{line}")
                raw_close = raw_start["close"]
                target_close = raw_rows[target_index]["close"]
                expected_label = int(target_close > raw_close) if meaning == "UP" else int(target_close < raw_close)
                if label != expected_label:
                    errors.append(f"dataset_label_value_mismatch_line_{line}")
                elapsed_hours = (expected_target - start).total_seconds() / 3600.0
                if "label_elapsed_hours" in fields and row.get("label_elapsed_hours", "").strip():
                    try:
                        recorded_elapsed = float(row["label_elapsed_hours"])
                    except ValueError:
                        errors.append(f"dataset_label_elapsed_hours_invalid_line_{line}")
                    else:
                        if not math.isfinite(recorded_elapsed) or abs(recorded_elapsed - elapsed_hours) > 1.1e-6:
                            errors.append(f"dataset_label_elapsed_hours_mismatch_line_{line}")
                if max_label_elapsed_hours is not None:
                    expected_eligible = elapsed_hours <= float(max_label_elapsed_hours) + 1e-9
                    recorded = row.get("label_eligible", "").strip()
                    if recorded not in {"0", "1"} or (recorded == "1") != expected_eligible:
                        errors.append(f"dataset_label_eligibility_mask_mismatch_line_{line}")
                    eligible_count += int(expected_eligible)
                    ineligible_count += int(not expected_eligible)
                checked += 1
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"dataset_unreadable_or_invalid:{exc}")

    return _report(
        raw_path, data_path, list(dict.fromkeys(errors)), checked, eligible_count,
        ineligible_count, raw_hash, dataset_hash,
    )


def _report(
    raw_path: Path,
    data_path: Path,
    errors: list[str],
    checked: int,
    eligible: int,
    ineligible: int,
    raw_hash: str | None,
    dataset_hash: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "GOLDCREST_FORWARD_LABEL_MATURITY_AUDIT",
        "status": "FAIL" if errors else "PASS",
        "raw_ohlc_path": str(raw_path),
        "raw_ohlc_sha256": raw_hash,
        "dataset_path": str(data_path),
        "dataset_sha256": dataset_hash,
        "checked_label_rows": checked,
        "eligible_label_rows": eligible,
        "ineligible_label_rows": ineligible,
        "errors": errors,
        "warnings": [],
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "interpretation": "Checks labels against the row-index forward close and actual maturity timestamp; does not infer or impute market-session gaps.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit forward-label maturity against raw OHLC.")
    parser.add_argument("--raw-ohlc", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True, choices=tuple(_TIMEFRAME_MINUTES))
    parser.add_argument("--horizon-bars", type=int, required=True)
    parser.add_argument("--class-1-means", choices=("UP", "DOWN"), required=True)
    parser.add_argument("--max-label-elapsed-hours", type=float)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_label_maturity(
        args.raw_ohlc, args.dataset, symbol=args.symbol, timeframe=args.timeframe,
        horizon_bars=args.horizon_bars, class_1_means=args.class_1_means,
        max_label_elapsed_hours=args.max_label_elapsed_hours,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

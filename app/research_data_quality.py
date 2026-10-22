"""Offline OHLC CSV quality audit for historical research inputs.

This auditor never sorts, repairs, imputes, downloads, or trades. It reports structural
issues and observed timestamp gaps; gaps are not automatically called missing candles
because market closures and provider session calendars may explain them.
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

_TIMEFRAME_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}
_REQUIRED = {"timestamp", "open", "high", "low", "close"}


def _timestamp(raw: str) -> datetime:
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def audit_ohlc_csv(
    source: str | Path,
    *,
    expected_symbol: str | None = None,
    expected_timeframe: str | None = None,
) -> dict[str, Any]:
    """Audit a CSV without modifying it; return deterministic findings and source hash."""
    path = Path(source)
    raw_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    errors: list[str] = []
    warnings: list[str] = []
    malformed_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    duplicate_timestamps = 0
    non_monotonic_timestamps = 0
    invalid_ohlc_rows = 0
    symbol_mismatches = 0
    timeframe_mismatches = 0
    observed_gaps: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED - fields)
        if missing:
            return {
                "schema_version": 1, "status": "FAIL",
                "errors": [f"missing_required_columns:{','.join(missing)}"],
                "warnings": [], "source_path": str(path),
                "source_sha256": source_sha256, "row_count": 0,
                "actionable": False,
            }
        has_symbol = "symbol" in fields
        has_timeframe = "timeframe" in fields
        seen: set[datetime] = set()
        previous: datetime | None = None
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = _timestamp(row.get("timestamp") or "")
                prices = {name: float(row.get(name) or "") for name in ("open", "high", "low", "close")}
                if any(not math.isfinite(value) or value <= 0 for value in prices.values()):
                    raise ValueError("OHLC values must be finite and positive")
                if (prices["high"] < max(prices["open"], prices["close"], prices["low"])
                        or prices["low"] > min(prices["open"], prices["close"], prices["high"])):
                    raise ValueError("invalid OHLC high/low relationships")
            except (TypeError, ValueError, OverflowError) as exc:
                invalid_ohlc_rows += 1
                malformed_rows.append({"line": line_number, "reason": str(exc)})
                continue

            symbol = (row.get("symbol") or "").strip().upper() if has_symbol else None
            timeframe = (row.get("timeframe") or "").strip().upper() if has_timeframe else None
            if expected_symbol and has_symbol and symbol != expected_symbol.strip().upper():
                symbol_mismatches += 1
            if expected_timeframe and has_timeframe and timeframe != expected_timeframe.strip().upper():
                timeframe_mismatches += 1

            if timestamp in seen:
                duplicate_timestamps += 1
            seen.add(timestamp)
            if previous is not None:
                if timestamp < previous:
                    non_monotonic_timestamps += 1
                elif timestamp > previous:
                    period = timeframe or expected_timeframe
                    interval = _TIMEFRAME_SECONDS.get(period or "")
                    delta_seconds = (timestamp - previous).total_seconds()
                    if interval and delta_seconds > interval * 1.5:
                        observed_gaps.append({
                            "previous_timestamp_utc": previous.isoformat().replace("+00:00", "Z"),
                            "next_timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
                            "elapsed_seconds": int(delta_seconds),
                            "nominal_interval_seconds": interval,
                            "nominal_intervals_spanned": round(delta_seconds / interval, 3),
                        })
            previous = timestamp
            parsed_rows.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "timeframe": timeframe,
            })

    if not parsed_rows:
        errors.append("no_valid_rows")
    if invalid_ohlc_rows:
        errors.append("invalid_ohlc_rows_present")
    if duplicate_timestamps:
        errors.append("duplicate_timestamps_present")
    if non_monotonic_timestamps:
        errors.append("non_monotonic_timestamps_present")
    if symbol_mismatches:
        errors.append("symbol_mismatches_present")
    if timeframe_mismatches:
        errors.append("timeframe_mismatches_present")
    if observed_gaps:
        warnings.append("observed_timestamp_gaps_require_session_calendar_review")
    if not has_symbol:
        warnings.append("symbol_column_absent; symbol identity cannot be checked from rows")
    if not has_timeframe:
        warnings.append("timeframe_column_absent; timeframe identity cannot be checked from rows")
    status = "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS"
    valid_timestamps = [row["timestamp"] for row in parsed_rows]
    return {
        "schema_version": 1,
        "status": status,
        "source_path": str(path),
        "source_sha256": source_sha256,
        "row_count": len(parsed_rows) + invalid_ohlc_rows,
        "valid_row_count": len(parsed_rows),
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "malformed_rows": malformed_rows[:100],
        "duplicate_timestamps": duplicate_timestamps,
        "non_monotonic_timestamps": non_monotonic_timestamps,
        "symbol_mismatches": symbol_mismatches,
        "timeframe_mismatches": timeframe_mismatches,
        "observed_gap_count": len(observed_gaps),
        "observed_gaps": observed_gaps[:100],
        "first_timestamp_utc": min(valid_timestamps).isoformat().replace("+00:00", "Z") if valid_timestamps else None,
        "last_timestamp_utc": max(valid_timestamps).isoformat().replace("+00:00", "Z") if valid_timestamps else None,
        "expected_symbol": expected_symbol.strip().upper() if expected_symbol else None,
        "expected_timeframe": expected_timeframe.strip().upper() if expected_timeframe else None,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "actionable": False,
        "interpretation": (
            "Structural CSV audit only. Observed gaps are not automatically missing candles; "
            "review instrument sessions, holidays, provider policy, and market closures. "
            "This report does not certify data rights, accuracy, or model profitability."
        ),
    }


def audit_ohlc_csv_file(
    source: str | Path,
    destination: str | Path,
    *,
    expected_symbol: str | None = None,
    expected_timeframe: str | None = None,
) -> dict[str, Any]:
    """Write a deterministic JSON quality receipt for an OHLC CSV."""
    result = audit_ohlc_csv(
        source, expected_symbol=expected_symbol, expected_timeframe=expected_timeframe
    )
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    result["receipt_path"] = str(target)
    result["receipt_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a historical OHLC CSV without changing it.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--symbol")
    parser.add_argument("--timeframe", choices=tuple(_TIMEFRAME_SECONDS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_ohlc_csv_file(
        args.input_csv, args.receipt,
        expected_symbol=args.symbol, expected_timeframe=args.timeframe,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())

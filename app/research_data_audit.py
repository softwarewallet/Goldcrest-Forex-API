"""Audit a downloaded cTrader OHLC snapshot before research dataset preparation.

The audit is local-only: it does not authenticate, connect to a broker, alter CSV rows,
impute data, or promote the data to actionable use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

_TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
_REQUIRED_COLUMNS = {"symbol", "timeframe", "timestamp", "open", "high", "low", "close"}


def _parse_utc(value: str, *, field: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_weekend_closure_like_gap(left: datetime, right: datetime) -> bool:
    """Conservative UTC heuristic; labels a likely FX weekend closure, not a guarantee.

    Candidate gaps must start late Friday and resume Sunday evening or early Monday.
    Other long gaps remain reviewable as non-weekend gaps instead of being silently ignored.
    """
    gap_hours = (right - left).total_seconds() / 3600
    resumes_near_open = (
        (right.weekday() == 6 and right.hour >= 16)
        or (right.weekday() == 0 and right.hour <= 6)
    )
    return (
        left.weekday() == 4
        and left.hour >= 16
        and resumes_near_open
        and 24 <= gap_hours <= 72
    )


def _fixed_date_calendar_context(left: datetime, right: datetime) -> list[str]:
    """Annotate fixed-date calendar context without asserting market closure.

    The tags are informational only. They do not reclassify a gap, suppress a warning,
    or replace an instrument/provider-specific market-session calendar.
    """
    known_dates = {
        (1, 1): "new_years_day",
        (12, 25): "christmas_day",
    }
    contexts: list[str] = []
    current = left.date()
    end = right.date()
    while current <= end:
        context = known_dates.get((current.month, current.day))
        if context and context not in contexts:
            contexts.append(context)
        current += timedelta(days=1)
    return contexts


def audit_raw_ohlc_file(
    csv_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    expected_symbol: str | None = None,
    expected_timeframe: str | None = None,
    require_manifest: bool = True,
) -> dict[str, Any]:
    """Validate snapshot integrity, OHLC relationships, chronology, and coverage metadata."""
    path = Path(csv_path)
    manifest_file = Path(manifest_path) if manifest_path else Path(str(path) + ".manifest.json")
    errors: list[str] = []
    warnings: list[str] = []
    manifest: dict[str, Any] = {}

    if not path.is_file():
        return {
            "status": "FAIL", "scope": "CTRADER_RAW_OHLC_INTEGRITY_AUDIT",
            "csv_path": str(path), "manifest_path": str(manifest_file),
            "errors": ["csv_file_not_found"], "warnings": [],
            "actionable": False, "execution_connected": False,
        }

    try:
        raw_digest = _sha256(path)
    except OSError:
        return {
            "status": "FAIL", "scope": "CTRADER_RAW_OHLC_INTEGRITY_AUDIT",
            "csv_path": str(path), "manifest_path": str(manifest_file),
            "errors": ["csv_file_unreadable"], "warnings": [],
            "actionable": False, "execution_connected": False,
        }

    if not manifest_file.is_file():
        if require_manifest:
            errors.append("snapshot_manifest_not_found")
        else:
            warnings.append("snapshot_manifest_missing_provenance_hash_unverified")
    else:
        try:
            loaded = json.loads(manifest_file.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                errors.append("snapshot_manifest_not_object")
            else:
                manifest = loaded
        except (OSError, json.JSONDecodeError):
            errors.append("snapshot_manifest_unreadable")

    is_ctrader_manifest = bool(manifest) and manifest.get("scope") == "CTRADER_OPEN_API_HISTORICAL_OHLC_NORMALIZATION"
    if manifest:
        if not is_ctrader_manifest:
            if require_manifest:
                errors.append("unexpected_manifest_scope")
            else:
                warnings.append("non_ctrader_manifest_metadata_unverified")
        else:
            if manifest.get("host") != "demo":
                errors.append("snapshot_host_is_not_demo")
            if manifest.get("actionable") is not False:
                errors.append("snapshot_not_explicitly_non_actionable")
            manifest_digest = manifest.get("snapshot_sha256")
            if not isinstance(manifest_digest, str) or manifest_digest != raw_digest:
                errors.append("snapshot_sha256_mismatch")
            page_info = manifest.get("pagination")
            if not isinstance(page_info, dict):
                warnings.append("pagination_metadata_missing_legacy_snapshot")
            elif page_info.get("complete") is not True:
                errors.append("pagination_not_marked_complete")
            if expected_symbol and str(manifest.get("symbol", "")).upper() != expected_symbol.strip().upper():
                errors.append("manifest_symbol_mismatch")
            if expected_timeframe and str(manifest.get("timeframe", "")).upper() != expected_timeframe.strip().upper():
                errors.append("manifest_timeframe_mismatch")

    rows_count = 0
    timestamps: list[datetime] = []
    intervals_seconds: list[float] = []
    symbols: set[str] = set()
    timeframes: set[str] = set()
    invalid_ohlc_rows = 0
    nonfinite_rows = 0
    unaligned_rows = 0
    first_bad_ohlc_line: int | None = None
    first_bad_timestamp_line: int | None = None

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = _REQUIRED_COLUMNS - columns
            if missing:
                errors.append("csv_missing_columns:" + ",".join(sorted(missing)))
                reader = None
            if reader is not None:
                for line_number, row in enumerate(reader, start=2):
                    rows_count += 1
                    symbol = str(row.get("symbol", "")).strip().upper()
                    timeframe = str(row.get("timeframe", "")).strip().upper()
                    symbols.add(symbol)
                    timeframes.add(timeframe)
                    try:
                        ts = _parse_utc(str(row["timestamp"]), field=f"line {line_number} timestamp")
                        if timestamps and ts <= timestamps[-1]:
                            if first_bad_timestamp_line is None:
                                first_bad_timestamp_line = line_number
                        timestamps.append(ts)
                    except (KeyError, ValueError, TypeError):
                        errors.append(f"invalid_timestamp_line_{line_number}")
                        continue

                    try:
                        values = {name: float(row[name]) for name in ("open", "high", "low", "close")}
                        if any(not math.isfinite(value) for value in values.values()):
                            nonfinite_rows += 1
                        elif (
                            any(value <= 0 for value in values.values())
                            or values["low"] > min(values["open"], values["close"])
                            or values["high"] < max(values["open"], values["close"])
                            or values["high"] < values["low"]
                        ):
                            invalid_ohlc_rows += 1
                    except (KeyError, ValueError, TypeError):
                        invalid_ohlc_rows += 1

                    if timeframe in _TIMEFRAME_MINUTES and ts.timestamp() % (_TIMEFRAME_MINUTES[timeframe] * 60):
                        unaligned_rows += 1

    except (OSError, csv.Error, UnicodeError):
        errors.append("csv_parse_failed")

    if rows_count == 0:
        errors.append("csv_contains_no_data_rows")
    if len(symbols) > 1:
        errors.append("multiple_symbols_in_csv")
    if len(timeframes) > 1:
        errors.append("multiple_timeframes_in_csv")
    if symbols and "" in symbols:
        errors.append("blank_symbol_in_csv")
    if timeframes and ("" in timeframes or any(frame not in _TIMEFRAME_MINUTES for frame in timeframes)):
        errors.append("unsupported_or_blank_timeframe_in_csv")
    if expected_symbol and symbols and symbols != {expected_symbol.strip().upper()}:
        errors.append("csv_symbol_mismatch")
    if expected_timeframe and timeframes and timeframes != {expected_timeframe.strip().upper()}:
        errors.append("csv_timeframe_mismatch")
    if is_ctrader_manifest and len(symbols) == 1 and str(manifest.get("symbol", "")).upper() != next(iter(symbols)):
        errors.append("manifest_csv_symbol_mismatch")
    if is_ctrader_manifest and len(timeframes) == 1 and str(manifest.get("timeframe", "")).upper() != next(iter(timeframes)):
        errors.append("manifest_csv_timeframe_mismatch")
    if first_bad_timestamp_line is not None:
        errors.append(f"timestamps_not_strictly_increasing_line_{first_bad_timestamp_line}")
    if invalid_ohlc_rows:
        errors.append(f"invalid_ohlc_rows:{invalid_ohlc_rows}")
    if nonfinite_rows:
        errors.append(f"nonfinite_ohlc_rows:{nonfinite_rows}")
    if unaligned_rows:
        errors.append(f"timestamps_not_aligned_to_timeframe:{unaligned_rows}")

    if timestamps and len(timestamps) == rows_count:
        if is_ctrader_manifest and timestamps[0].isoformat().replace("+00:00", "Z") != manifest.get("first_timestamp_utc"):
            errors.append("manifest_first_timestamp_mismatch")
        if is_ctrader_manifest and timestamps[-1].isoformat().replace("+00:00", "Z") != manifest.get("last_timestamp_utc"):
            errors.append("manifest_last_timestamp_mismatch")
        if is_ctrader_manifest and manifest.get("rows") != rows_count:
            errors.append("manifest_row_count_mismatch")
        intervals_seconds = [
            (right - left).total_seconds() for left, right in zip(timestamps, timestamps[1:])
        ]
        frame = next(iter(timeframes), "")
        frame_seconds = _TIMEFRAME_MINUTES.get(frame, 0) * 60
        long_gaps: list[dict[str, Any]] = []
        for left, right in zip(timestamps, timestamps[1:]):
            gap_seconds = (right - left).total_seconds()
            if gap_seconds <= 8 * 3600:
                continue
            weekend_like = _is_weekend_closure_like_gap(left, right)
            long_gaps.append({
                "from_utc": left.isoformat().replace("+00:00", "Z"),
                "to_utc": right.isoformat().replace("+00:00", "Z"),
                "gap_hours": round(gap_seconds / 3600, 3),
                "gap_category": "weekend_closure_like" if weekend_like else "non_weekend_long_gap",
                "calendar_context": _fixed_date_calendar_context(left, right),
            })
        weekend_like_gaps = [gap for gap in long_gaps if gap["gap_category"] == "weekend_closure_like"]
        non_weekend_long_gaps = [gap for gap in long_gaps if gap["gap_category"] == "non_weekend_long_gap"]
        calendar_context_gaps = [gap for gap in long_gaps if gap["calendar_context"]]
        if unaligned_rows == 0 and len(timeframes) == 1 and frame_seconds == 0:
            warnings.append("unknown_timeframe_candle_alignment_not_checked")
        if non_weekend_long_gaps:
            warnings.append("non_weekend_long_gaps_detected_review_data_coverage")
    else:
        long_gaps = []
        weekend_like_gaps = []
        non_weekend_long_gaps = []
        calendar_context_gaps = []

    if not manifest:
        pagination_summary = None
    else:
        page_info = manifest.get("pagination")
        pagination_summary = {
            "complete": page_info.get("complete") if isinstance(page_info, dict) else None,
            "window_count": page_info.get("window_count") if isinstance(page_info, dict) else None,
            "pages_requested": page_info.get("pages_requested") if isinstance(page_info, dict) else None,
            "duplicate_bars_removed": page_info.get("duplicate_bars_removed") if isinstance(page_info, dict) else None,
            "has_more_flag_available": page_info.get("has_more_flag_available") if isinstance(page_info, dict) else None,
        }

    count = len(timestamps)
    status = "FAIL" if errors else "PASS_WITH_WARNINGS" if warnings else "PASS"
    return {
        "schema_version": 1,
        "scope": "CTRADER_RAW_OHLC_INTEGRITY_AUDIT",
        "status": status,
        "csv_path": str(path.resolve()),
        "manifest_path": str(manifest_file.resolve()),
        "snapshot_sha256": raw_digest,
        "manifest_snapshot_sha256": manifest.get("snapshot_sha256"),
        "row_count": rows_count,
        "symbol": next(iter(symbols)) if len(symbols) == 1 else None,
        "timeframe": next(iter(timeframes)) if len(timeframes) == 1 else None,
        "first_timestamp_utc": timestamps[0].isoformat().replace("+00:00", "Z") if count else None,
        "last_timestamp_utc": timestamps[-1].isoformat().replace("+00:00", "Z") if count else None,
        "time_span_days": round((timestamps[-1] - timestamps[0]).total_seconds() / 86400, 4) if count > 1 else 0,
        "interval_seconds": {
            "median": statistics.median(intervals_seconds) if intervals_seconds else None,
            "minimum": min(intervals_seconds) if intervals_seconds else None,
            "maximum": max(intervals_seconds) if intervals_seconds else None,
        },
        "large_gap_count_over_8_hours": len(long_gaps),
        "large_gap_examples": long_gaps[:10],
        "weekend_closure_like_gap_count_over_8_hours": len(weekend_like_gaps),
        "non_weekend_long_gap_count_over_8_hours": len(non_weekend_long_gaps),
        "non_weekend_long_gap_examples": non_weekend_long_gaps[:10],
        "holiday_context_gap_count_over_8_hours": len(calendar_context_gaps),
        "holiday_context_gap_examples": calendar_context_gaps[:10],
        "gap_classification": {
            "method": "conservative_utc_weekend_window_heuristic",
            "weekend_closure_like": "Gap starts Friday at or after 16:00 UTC and resumes Sunday at/after 16:00 UTC or Monday by 06:00 UTC, lasting 24-72 hours. Heuristic only; not proof of an expected closure.",
            "non_weekend_long_gap": "Any gap over 8 hours not matching the weekend-like heuristic; review market calendar and data coverage.",
            "fixed_date_calendar_context": "Informational UTC-date tags for Christmas Day and New Year's Day only. These tags do not establish expected market closure, alter gap_category, or suppress the non-weekend long-gap warning.",
            "recognized_fixed_date_contexts": {
                "christmas_day": "December 25 UTC calendar date",
                "new_years_day": "January 1 UTC calendar date",
            },
        },
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "nonfinite_ohlc_rows": nonfinite_rows,
        "unaligned_timestamp_rows": unaligned_rows,
        "pagination": pagination_summary,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "interpretation": (
            "Local snapshot integrity and OHLC structure only. Weekend-like gap classification is a "
            "conservative timestamp heuristic, not proof of expected market closure. Fixed-date holiday "
            "context tags are informational only; review all non-weekend long gaps, instrument-specific "
            "sessions and provider coverage before model evaluation."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a local OHLC CSV and, when required, its cTrader DEMO integrity manifest.")
    parser.add_argument("--csv", required=True, help="Downloaded CSV to audit")
    parser.add_argument("--manifest", help="Manifest path; defaults to <csv>.manifest.json")
    parser.add_argument("--symbol", help="Optional expected symbol, e.g. EURUSD")
    parser.add_argument("--timeframe", choices=tuple(_TIMEFRAME_MINUTES), help="Optional expected timeframe")
    parser.add_argument("--output", help="Optional JSON report output path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_raw_ohlc_file(
        args.csv, manifest_path=args.manifest,
        expected_symbol=args.symbol, expected_timeframe=args.timeframe,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        report["report_path"] = str(output.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"Raw OHLC audit: {report['status']}")
        print(f"Rows: {report.get('row_count', 0)} | Symbol: {report.get('symbol')} | Timeframe: {report.get('timeframe')}")
        print(f"SHA-256: {report.get('snapshot_sha256')}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        if report.get("report_path"):
            print(f"Report: {report['report_path']}")
        print("Actionable: false")
    return 3 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

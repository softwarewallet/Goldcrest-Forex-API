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
from typing import Any

from app.research_strict_json import strict_json_loads, Sequence

def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


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

def _gap_pattern_diagnostics(gaps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize long-gap duration and UTC boundary patterns without reclassifying them."""
    duration_buckets = {
        "greater_than_8_to_12_hours": 0,
        "greater_than_12_to_24_hours": 0,
        "greater_than_24_to_48_hours": 0,
        "greater_than_48_to_72_hours": 0,
        "greater_than_72_hours": 0,
    }
    category_counts: dict[str, int] = {}
    start_windows: dict[tuple[str, int], int] = {}
    end_windows: dict[tuple[str, int], int] = {}
    boundary_patterns: dict[tuple[str, int, str, int], int] = {}
    unparsed_boundary_count = 0

    for gap in gaps:
        hours = float(gap["gap_hours"])
        if hours <= 12:
            duration_bucket = "greater_than_8_to_12_hours"
        elif hours <= 24:
            duration_bucket = "greater_than_12_to_24_hours"
        elif hours <= 48:
            duration_bucket = "greater_than_24_to_48_hours"
        elif hours <= 72:
            duration_bucket = "greater_than_48_to_72_hours"
        else:
            duration_bucket = "greater_than_72_hours"
        duration_buckets[duration_bucket] += 1

        category = str(gap.get("gap_category", "unknown"))
        category_counts[category] = category_counts.get(category, 0) + 1

        try:
            left = _parse_utc(str(gap["from_utc"]), field="gap from_utc")
            right = _parse_utc(str(gap["to_utc"]), field="gap to_utc")
        except (KeyError, TypeError, ValueError):
            unparsed_boundary_count += 1
            continue

        start_key = (left.strftime("%A").upper(), left.hour)
        end_key = (right.strftime("%A").upper(), right.hour)
        start_windows[start_key] = start_windows.get(start_key, 0) + 1
        end_windows[end_key] = end_windows.get(end_key, 0) + 1
        pattern_key = (*start_key, *end_key)
        boundary_patterns[pattern_key] = boundary_patterns.get(pattern_key, 0) + 1

    def window_rows(counts: dict[tuple[str, int], int]) -> list[dict[str, Any]]:
        return [
            {"weekday_utc": weekday, "hour_utc": hour, "count": count}
            for (weekday, hour), count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    repeated_patterns = [
        {
            "from_weekday_utc": from_weekday,
            "from_hour_utc": from_hour,
            "to_weekday_utc": to_weekday,
            "to_hour_utc": to_hour,
            "count": count,
        }
        for (from_weekday, from_hour, to_weekday, to_hour), count in sorted(
            boundary_patterns.items(), key=lambda item: (-item[1], item[0])
        )
        if count > 1
    ]

    return {
        "threshold_hours_exclusive": 8,
        "gap_count": len(gaps),
        "category_counts": dict(sorted(category_counts.items())),
        "duration_buckets": duration_buckets,
        "start_weekday_hour_utc_counts": window_rows(start_windows),
        "end_weekday_hour_utc_counts": window_rows(end_windows),
        "repeated_boundary_patterns": repeated_patterns,
        "unparsed_boundary_count": unparsed_boundary_count,
        "interpretation": (
            "Descriptive timestamp-pattern summaries only. Repeated windows do not establish an expected "
            "market closure or prove data loss. The audit does not suppress warnings, alter gap categories, "
            "repair candles, or replace an instrument/provider-specific session calendar."
        ),
    }


def _within_session_gap_diagnostics(
    timestamps: Sequence[datetime],
    frame_seconds: int,
    *,
    max_gap_seconds: int = 8 * 3600,
) -> dict[str, Any]:
    """Describe timestamp intervals larger than one timeframe and no longer than eight hours.

    This is a timestamp-spacing report, not an estimate of missing tradable candles:
    instrument sessions, holidays, provider coverage and expected market closures are not inferred.
    """
    bucket_names = (
        "greater_than_1_to_2_frames",
        "greater_than_2_to_4_frames",
        "greater_than_4_to_8_frames",
        "greater_than_8_to_16_frames",
        "greater_than_16_frames_to_8_hours",
    )
    buckets = {name: 0 for name in bucket_names}
    if frame_seconds <= 0:
        return {
            "status": "UNAVAILABLE_TIMEFRAME_DURATION",
            "expected_frame_seconds": None,
            "upper_bound_seconds_inclusive": max_gap_seconds,
            "gap_count": None,
            "frame_multiple_buckets": buckets,
            "start_weekday_hour_utc_counts": [],
            "end_weekday_hour_utc_counts": [],
            "repeated_boundary_patterns": [],
            "largest_gap_examples": [],
            "interpretation": (
                "No timeframe-duration diagnostic was computed because the CSV timeframe is unsupported "
                "or unavailable. No market-closure or missing-candle conclusion is inferred."
            ),
        }
    if len(timestamps) < 2:
        return {
            "status": "INSUFFICIENT_INTERVALS",
            "expected_frame_seconds": frame_seconds,
            "upper_bound_seconds_inclusive": max_gap_seconds,
            "gap_count": 0,
            "frame_multiple_buckets": buckets,
            "start_weekday_hour_utc_counts": [],
            "end_weekday_hour_utc_counts": [],
            "repeated_boundary_patterns": [],
            "largest_gap_examples": [],
            "interpretation": (
                "At least two valid timestamps are needed to describe spacing. No market-closure or "
                "missing-candle conclusion is inferred."
            ),
        }

    intervals = [(right - left).total_seconds() for left, right in zip(timestamps, timestamps[1:])]
    if any(seconds <= 0 for seconds in intervals):
        return {
            "status": "UNAVAILABLE_NON_MONOTONIC_TIMESTAMPS",
            "expected_frame_seconds": frame_seconds,
            "upper_bound_seconds_inclusive": max_gap_seconds,
            "gap_count": None,
            "frame_multiple_buckets": buckets,
            "start_weekday_hour_utc_counts": [],
            "end_weekday_hour_utc_counts": [],
            "repeated_boundary_patterns": [],
            "largest_gap_examples": [],
            "interpretation": (
                "Timestamp intervals are not strictly increasing, so spacing-pattern summaries were not "
                "computed. Correct chronology must be reviewed before interpreting intervals."
            ),
        }

    gaps: list[dict[str, Any]] = []
    start_windows: dict[tuple[str, int], int] = {}
    end_windows: dict[tuple[str, int], int] = {}
    patterns: dict[tuple[str, int, str, int], int] = {}

    for left, right, seconds in zip(timestamps, timestamps[1:], intervals):
        if seconds <= frame_seconds or seconds > max_gap_seconds:
            continue
        multiple = seconds / frame_seconds
        if multiple <= 2:
            bucket = bucket_names[0]
        elif multiple <= 4:
            bucket = bucket_names[1]
        elif multiple <= 8:
            bucket = bucket_names[2]
        elif multiple <= 16:
            bucket = bucket_names[3]
        else:
            bucket = bucket_names[4]
        buckets[bucket] += 1

        start_key = (left.strftime("%A").upper(), left.hour)
        end_key = (right.strftime("%A").upper(), right.hour)
        start_windows[start_key] = start_windows.get(start_key, 0) + 1
        end_windows[end_key] = end_windows.get(end_key, 0) + 1
        pattern_key = (*start_key, *end_key)
        patterns[pattern_key] = patterns.get(pattern_key, 0) + 1
        gaps.append({
            "from_utc": left.isoformat().replace("+00:00", "Z"),
            "to_utc": right.isoformat().replace("+00:00", "Z"),
            "interval_minutes": round(seconds / 60, 3),
            "elapsed_timeframe_equivalents": round(multiple, 4),
            "aligned_to_timeframe_multiple": math.isclose(
                seconds / frame_seconds, round(multiple), rel_tol=0.0, abs_tol=1e-9
            ),
        })

    def window_rows(counts: dict[tuple[str, int], int]) -> list[dict[str, Any]]:
        return [
            {"weekday_utc": weekday, "hour_utc": hour, "count": count}
            for (weekday, hour), count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    repeated_patterns = [
        {
            "from_weekday_utc": from_weekday,
            "from_hour_utc": from_hour,
            "to_weekday_utc": to_weekday,
            "to_hour_utc": to_hour,
            "count": count,
        }
        for (from_weekday, from_hour, to_weekday, to_hour), count in sorted(
            patterns.items(), key=lambda item: (-item[1], item[0])
        )
        if count > 1
    ]
    largest_examples = sorted(
        gaps, key=lambda gap: (-gap["interval_minutes"], gap["from_utc"])
    )[:10]
    return {
        "status": "AVAILABLE",
        "expected_frame_seconds": frame_seconds,
        "upper_bound_seconds_inclusive": max_gap_seconds,
        "gap_count": len(gaps),
        "frame_multiple_buckets": buckets,
        "start_weekday_hour_utc_counts": window_rows(start_windows),
        "end_weekday_hour_utc_counts": window_rows(end_windows),
        "repeated_boundary_patterns": repeated_patterns,
        "largest_gap_examples": largest_examples,
        "interpretation": (
            "Descriptive timestamp-spacing summaries for intervals longer than one nominal timeframe and "
            "no longer than eight hours. These are not confirmed missing candles: session breaks, holidays, "
            "provider coverage and instrument-specific trading hours are not inferred. Repeated boundaries "
            "are clues for review only and do not suppress any audit warning."
        ),
    }


def _timestamp_coverage_profile(
    timestamps: Sequence[datetime],
    *,
    unobserved_date_example_limit: int = 50,
) -> dict[str, Any]:
    """Summarize observed candle rows by UTC calendar date and hour, without inferring completeness."""
    if (
        isinstance(unobserved_date_example_limit, bool)
        or not isinstance(unobserved_date_example_limit, int)
        or unobserved_date_example_limit < 0
    ):
        raise ValueError("unobserved_date_example_limit must be a non-negative integer")

    unavailable = {
        "status": "NO_VALID_TIMESTAMPS",
        "timestamp_count": 0,
        "first_timestamp_utc": None,
        "last_timestamp_utc": None,
        "calendar_span_days_inclusive": 0,
        "observed_utc_dates_count": 0,
        "unobserved_utc_dates_within_span_count": 0,
        "unobserved_utc_date_examples": [],
        "daily_row_count_summary": {
            "minimum": None, "median": None, "mean": None, "maximum": None,
        },
        "daily_row_counts": [],
        "hour_of_day_utc_counts": [
            {"hour_utc": hour, "count": 0} for hour in range(24)
        ],
        "interpretation": (
            "No valid timestamps were available for the coverage profile. This profile does not infer "
            "expected candle counts or market-session completeness."
        ),
    }
    if not timestamps:
        return unavailable

    normalized = [item.astimezone(timezone.utc) for item in timestamps]
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        result = dict(unavailable)
        result["status"] = "UNAVAILABLE_NON_MONOTONIC_TIMESTAMPS"
        result["timestamp_count"] = len(normalized)
        result["interpretation"] = (
            "UTC date coverage was not summarized because timestamps are not strictly increasing. "
            "Resolve chronology before interpreting coverage patterns; no candle completeness is inferred."
        )
        return result

    daily: dict[Any, list[datetime]] = {}
    hour_counts = {hour: 0 for hour in range(24)}
    for timestamp in normalized:
        daily.setdefault(timestamp.date(), []).append(timestamp)
        hour_counts[timestamp.hour] += 1

    first_date, last_date = normalized[0].date(), normalized[-1].date()
    span_days = (last_date - first_date).days + 1
    unobserved_dates: list[str] = []
    current_date = first_date
    while current_date <= last_date:
        if current_date not in daily and len(unobserved_dates) < unobserved_date_example_limit:
            unobserved_dates.append(current_date.isoformat())
        current_date += timedelta(days=1)

    row_counts = [len(items) for _, items in sorted(daily.items())]
    daily_rows = [
        {
            "date_utc": day.isoformat(),
            "row_count": len(items),
            "first_timestamp_utc": items[0].isoformat().replace("+00:00", "Z"),
            "last_timestamp_utc": items[-1].isoformat().replace("+00:00", "Z"),
        }
        for day, items in sorted(daily.items())
    ]
    return {
        "status": "AVAILABLE",
        "timestamp_count": len(normalized),
        "first_timestamp_utc": normalized[0].isoformat().replace("+00:00", "Z"),
        "last_timestamp_utc": normalized[-1].isoformat().replace("+00:00", "Z"),
        "calendar_span_days_inclusive": span_days,
        "observed_utc_dates_count": len(daily),
        "unobserved_utc_dates_within_span_count": span_days - len(daily),
        "unobserved_utc_date_examples": unobserved_dates,
        "daily_row_count_summary": {
            "minimum": min(row_counts),
            "median": statistics.median(row_counts),
            "mean": round(statistics.mean(row_counts), 4),
            "maximum": max(row_counts),
        },
        "daily_row_counts": daily_rows,
        "hour_of_day_utc_counts": [
            {"hour_utc": hour, "count": hour_counts[hour]} for hour in range(24)
        ],
        "interpretation": (
            "Descriptive UTC calendar-date and hour-of-day coverage only. Unobserved dates may reflect "
            "weekends, instrument sessions, holidays, provider coverage, or collection choices; they are not "
            "automatically classified as missing-data incidents. Daily counts are not compared with an "
            "assumed expected candle count. No market calendar, data repair, model selection, or execution "
            "decision is applied."
        ),
    }


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
            loaded = _load_json_object(manifest_file.read_bytes(), source=str(manifest_file))
            if not isinstance(loaded, dict):
                errors.append("snapshot_manifest_not_object")
            else:
                manifest = loaded
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
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
        frame = next(iter(timeframes), "") if len(timeframes) == 1 else ""
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
        within_session_gap_diagnostics = _within_session_gap_diagnostics(timestamps, frame_seconds)
        timestamp_coverage_profile = _timestamp_coverage_profile(timestamps)
        if unaligned_rows == 0 and len(timeframes) == 1 and frame_seconds == 0:
            warnings.append("unknown_timeframe_candle_alignment_not_checked")
        if non_weekend_long_gaps:
            warnings.append("non_weekend_long_gaps_detected_review_data_coverage")
    else:
        long_gaps = []
        weekend_like_gaps = []
        non_weekend_long_gaps = []
        calendar_context_gaps = []
        within_session_gap_diagnostics = _within_session_gap_diagnostics([], 0)
        timestamp_coverage_profile = _timestamp_coverage_profile([])

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
        "gap_diagnostics_over_8_hours": _gap_pattern_diagnostics(long_gaps),
        "gap_diagnostics_between_one_frame_and_8_hours": within_session_gap_diagnostics,
        "timestamp_coverage_profile": timestamp_coverage_profile,
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




def _load_pipeline_manifest(manifest_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a verified research-run manifest and resolve its exact raw OHLC input."""
    path = Path(manifest_path)
    try:
        loaded = _load_json_object(path.read_bytes(), source=str(path))
    except FileNotFoundError as exc:
        raise ValueError("pipeline_manifest_not_found") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("pipeline_manifest_unreadable") from exc

    if not isinstance(loaded, dict):
        raise ValueError("pipeline_manifest_not_object")
    if loaded.get("scope") != "GOLDCREST_OFFLINE_RESEARCH_PIPELINE":
        raise ValueError("unexpected_pipeline_manifest_scope")
    if loaded.get("actionable") is not False or loaded.get("execution_connected") is not False:
        raise ValueError("pipeline_manifest_must_be_non_actionable_and_offline")

    declared_hash = loaded.get("pipeline_manifest_sha256")
    payload = dict(loaded)
    payload.pop("pipeline_manifest_sha256", None)
    try:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        actual_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError("pipeline_manifest_not_canonical_json") from exc
    if not isinstance(declared_hash, str) or declared_hash != actual_hash:
        raise ValueError("pipeline_manifest_sha256_mismatch")

    raw_path_value = loaded.get("raw_ohlc_path")
    if not isinstance(raw_path_value, str) or not raw_path_value.strip():
        raise ValueError("pipeline_manifest_raw_ohlc_path_missing")
    csv_path = Path(raw_path_value).expanduser()
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()

    return loaded, csv_path


def _pipeline_manifest_failure(path: str | Path, error: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "CTRADER_RAW_OHLC_INTEGRITY_AUDIT",
        "status": "FAIL",
        "csv_path": None,
        "manifest_path": None,
        "pipeline_manifest_path": str(Path(path)),
        "errors": [error],
        "warnings": [],
        "actionable": False,
        "execution_connected": False,
        "automatic_model_selection": False,
        "interpretation": (
            "The pipeline manifest could not be safely used to resolve the exact raw OHLC input. "
            "No model selection or execution decision was made."
        ),
    }



def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a local OHLC CSV and optional cTrader DEMO snapshot manifest. "
            "Use --pipeline-manifest to resolve the exact raw CSV recorded by a completed research run."
        )
    )
    parser.add_argument("--csv", help="CSV to audit; may be omitted when --pipeline-manifest is supplied")
    parser.add_argument("--manifest", help="cTrader snapshot manifest; defaults to <csv>.manifest.json")
    parser.add_argument(
        "--pipeline-manifest",
        help="Optional completed research pipeline manifest; verifies its fingerprint and recorded raw CSV hash",
    )
    parser.add_argument("--symbol", help="Optional expected symbol; defaults to the pipeline manifest symbol")
    parser.add_argument("--timeframe", choices=tuple(_TIMEFRAME_MINUTES), help="Optional expected timeframe; defaults to the pipeline manifest timeframe")
    parser.add_argument("--output", help="Optional JSON report output path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    pipeline_manifest: dict[str, Any] | None = None
    csv_path = args.csv

    if args.pipeline_manifest:
        try:
            pipeline_manifest, recorded_csv_path = _load_pipeline_manifest(args.pipeline_manifest)
            if csv_path is None:
                csv_path = str(recorded_csv_path)
        except ValueError as exc:
            report = _pipeline_manifest_failure(args.pipeline_manifest, str(exc))
        else:
            expected_symbol = args.symbol or str(pipeline_manifest.get("symbol") or "") or None
            expected_timeframe = args.timeframe or str(pipeline_manifest.get("timeframe") or "") or None
            report = audit_raw_ohlc_file(
                csv_path,
                manifest_path=args.manifest,
                expected_symbol=expected_symbol,
                expected_timeframe=expected_timeframe,
                require_manifest=(
                    pipeline_manifest.get("source") != "Operator-supplied canonical OHLC CSV"
                ),
            )
            expected_raw_hash = pipeline_manifest.get("raw_ohlc_sha256")
            actual_raw_hash = report.get("snapshot_sha256")
            pipeline_hash_verified = (
                isinstance(expected_raw_hash, str)
                and bool(actual_raw_hash)
                and expected_raw_hash == actual_raw_hash
            )
            if not isinstance(expected_raw_hash, str) or not expected_raw_hash:
                report["errors"].append("pipeline_manifest_raw_ohlc_sha256_missing")
            elif actual_raw_hash and expected_raw_hash != actual_raw_hash:
                report["errors"].append("pipeline_manifest_raw_ohlc_sha256_mismatch")
            report["errors"] = sorted(set(report.get("errors", [])))
            report["status"] = (
                "FAIL" if report["errors"]
                else "PASS_WITH_WARNINGS" if report.get("warnings")
                else "PASS"
            )
            report["pipeline_manifest_path"] = str(Path(args.pipeline_manifest).resolve())
            report["pipeline_manifest_sha256_verified"] = True
            report["pipeline_manifest_raw_ohlc_sha256_verified"] = pipeline_hash_verified
    else:
        if csv_path is None:
            _parser().error("provide --csv or --pipeline-manifest")
        report = audit_raw_ohlc_file(
            csv_path, manifest_path=args.manifest,
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
        if report.get("pipeline_manifest_path"):
            print(f"Pipeline manifest: {report['pipeline_manifest_path']}")
        if report.get("report_path"):
            print(f"Report: {report['report_path']}")
        print("Actionable: false")
    return 3 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

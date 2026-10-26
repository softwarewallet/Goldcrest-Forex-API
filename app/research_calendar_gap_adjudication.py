"""Evidence-bound adjudication of OHLC timestamp gaps against an explicit market calendar.

This tool is offline and read-only. It never repairs candles or changes the raw-data audit.
A gap is adjudicated as a documented closure only when a hash-verified source-evidence
record declares a matching instrument/provider/timeframe and a closed interval fully covers
the missing-bar interval. A calendar assertion is not independently proof of authority.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.research_strict_json import strict_json_loads

_TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_PLACEHOLDER = re.compile(r"<[^>]+>|\b(?:TODO|TBD|PLACEHOLDER|EXAMPLE\.COM|REPLACE THIS)\b", re.I)


def _utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}_missing_or_invalid")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field}_must_be_timezone_aware")
    return parsed.astimezone(timezone.utc)


def _read_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return value, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_status(errors: list[str], unresolved: int, gap_count: int) -> str:
    if errors:
        return "FAIL"
    if unresolved:
        return "PASS_WITH_WARNINGS"
    return "PASS"


def adjudicate_market_calendar_gaps(
    ohlc_csv: str | Path,
    snapshot_manifest: str | Path,
    calendar_evidence: str | Path,
    source_evidence: str | Path,
    *,
    threshold_hours: float = 8.0,
) -> dict[str, Any]:
    csv_path = Path(ohlc_csv).expanduser().resolve()
    manifest_path = Path(snapshot_manifest).expanduser().resolve()
    calendar_path = Path(calendar_evidence).expanduser().resolve()
    source_path = Path(source_evidence).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, str] = {}

    try:
        manifest, _ = _read_object(manifest_path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return _report("FAIL", [f"snapshot_manifest_unreadable:{type(exc).__name__}"], [], {}, [], None)

    try:
        evidence, evidence_bytes = _read_object(calendar_path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return _report("FAIL", [f"calendar_evidence_unreadable:{type(exc).__name__}"], [], {}, [], None)

    try:
        csv_bytes = csv_path.read_bytes()
        checks["raw_snapshot_sha256"] = "PASS" if _sha256(csv_bytes) == manifest.get("snapshot_sha256") else "FAIL"
        if checks["raw_snapshot_sha256"] == "FAIL":
            errors.append("raw_snapshot_sha256_mismatch")
    except OSError:
        return _report("FAIL", ["raw_snapshot_unreadable"], [], {}, [], None)

    if evidence.get("schema_version") != 1 or evidence.get("evidence_type") != "market_calendar_gap_adjudication":
        errors.append("calendar_evidence_schema_or_type_invalid")
    for field in ("provider", "symbol", "timeframe", "authority", "source_reference", "retrieved_at_utc"):
        value = evidence.get(field)
        if not isinstance(value, str) or not value.strip() or _PLACEHOLDER.search(value):
            errors.append(f"calendar_evidence_{field}_missing_or_placeholder")
    try:
        _utc(evidence.get("retrieved_at_utc"), "retrieved_at_utc")
    except (ValueError, TypeError):
        errors.append("calendar_evidence_retrieved_at_utc_invalid")

    symbol = manifest.get("symbol")
    timeframe = manifest.get("timeframe")
    provider = manifest.get("provider", manifest.get("source", ""))
    if evidence.get("symbol") != symbol:
        errors.append("calendar_evidence_symbol_mismatch")
    if evidence.get("timeframe") != timeframe:
        errors.append("calendar_evidence_timeframe_mismatch")
    if not isinstance(provider, str) or not provider.strip():
        warnings.append("snapshot_manifest_provider_not_recorded; compare against documented provider evidence manually")
    elif evidence.get("provider") != provider:
        errors.append("calendar_evidence_provider_mismatch")

    try:
        declared_hash = evidence.get("source_evidence_sha256")
        source_bytes = source_path.read_bytes()
        actual_source_hash = _sha256(source_bytes)
        if not isinstance(declared_hash, str) or not _SHA256.fullmatch(declared_hash):
            errors.append("calendar_source_evidence_sha256_invalid")
        elif declared_hash.lower() != actual_source_hash:
            errors.append("calendar_source_evidence_sha256_mismatch")
        else:
            checks["calendar_source_evidence_hash"] = "PASS"
    except OSError:
        errors.append("calendar_source_evidence_missing_or_unreadable")

    try:
        coverage_start = _utc(evidence.get("coverage_start_utc"), "coverage_start_utc")
        coverage_end = _utc(evidence.get("coverage_end_utc"), "coverage_end_utc")
        if coverage_end <= coverage_start:
            errors.append("calendar_evidence_coverage_interval_invalid")
    except (ValueError, TypeError):
        coverage_start = coverage_end = None
        errors.append("calendar_evidence_coverage_interval_invalid")

    if not isinstance(evidence.get("closed_intervals"), list):
        errors.append("calendar_closed_intervals_must_be_array")
        intervals: list[dict[str, Any]] = []
    else:
        intervals = evidence["closed_intervals"]

    normalized_intervals: list[tuple[datetime, datetime, str]] = []
    for index, interval in enumerate(intervals):
        if not isinstance(interval, dict):
            errors.append(f"calendar_closed_interval_{index}_must_be_object")
            continue
        try:
            start = _utc(interval.get("start_utc"), f"closed_interval_{index}_start_utc")
            end = _utc(interval.get("end_utc"), f"closed_interval_{index}_end_utc")
            reason = interval.get("reason")
            locator = interval.get("source_locator")
            if end <= start:
                raise ValueError("end_must_follow_start")
            if not isinstance(reason, str) or not reason.strip() or _PLACEHOLDER.search(reason):
                raise ValueError("reason_missing_or_placeholder")
            if not isinstance(locator, str) or not locator.strip() or _PLACEHOLDER.search(locator):
                raise ValueError("source_locator_missing_or_placeholder")
            normalized_intervals.append((start, end, reason.strip()))
        except (ValueError, TypeError) as exc:
            errors.append(f"calendar_closed_interval_{index}_invalid:{str(exc)}")

    frame_minutes = _TIMEFRAME_MINUTES.get(str(timeframe))
    if frame_minutes is None:
        errors.append("snapshot_timeframe_unsupported")
        frame = timedelta(minutes=15)
    else:
        frame = timedelta(minutes=frame_minutes)

    rows: list[tuple[datetime, int]] = []
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"timestamp", "symbol", "timeframe"}.issubset(reader.fieldnames):
                errors.append("ohlc_required_columns_missing")
            for line_number, row in enumerate(reader, start=2):
                try:
                    stamp = _utc(row.get("timestamp"), f"line_{line_number}_timestamp")
                except (ValueError, TypeError):
                    errors.append(f"ohlc_timestamp_invalid_line_{line_number}")
                    continue
                if row.get("symbol") != symbol:
                    errors.append(f"ohlc_symbol_mismatch_line_{line_number}")
                if row.get("timeframe") != timeframe:
                    errors.append(f"ohlc_timeframe_mismatch_line_{line_number}")
                rows.append((stamp, line_number))
    except OSError:
        errors.append("ohlc_csv_unreadable")

    if len(rows) != manifest.get("rows"):
        errors.append("ohlc_row_count_manifest_mismatch")
    if rows:
        if rows[0][0] != _utc(manifest.get("first_timestamp_utc"), "first_timestamp_utc"):
            errors.append("ohlc_first_timestamp_manifest_mismatch")
        if rows[-1][0] != _utc(manifest.get("last_timestamp_utc"), "last_timestamp_utc"):
            errors.append("ohlc_last_timestamp_manifest_mismatch")
    for previous, current in zip(rows, rows[1:]):
        if current[0] <= previous[0]:
            errors.append(f"ohlc_timestamps_not_strictly_increasing_lines_{previous[1]}_{current[1]}")

    gaps: list[dict[str, Any]] = []
    unresolved = 0
    for (left, left_line), (right, right_line) in zip(rows, rows[1:]):
        gap_hours = (right - left).total_seconds() / 3600
        if gap_hours <= threshold_hours:
            continue
        missing_start = left + frame
        missing_end = right
        matched = None
        if coverage_start is not None and coverage_end is not None and coverage_start <= missing_start and coverage_end >= missing_end:
            for start, end, reason in normalized_intervals:
                if start <= missing_start and end >= missing_end:
                    matched = {"reason": reason, "interval_start_utc": start.isoformat().replace("+00:00", "Z"),
                               "interval_end_utc": end.isoformat().replace("+00:00", "Z")}
                    break
        status = "ADJUDICATED_DOCUMENTED_CLOSURE" if matched else "UNRESOLVED"
        if not matched:
            unresolved += 1
        gaps.append({
            "from_utc": left.isoformat().replace("+00:00", "Z"),
            "to_utc": right.isoformat().replace("+00:00", "Z"),
            "missing_interval_start_utc": missing_start.isoformat().replace("+00:00", "Z"),
            "missing_interval_end_utc": missing_end.isoformat().replace("+00:00", "Z"),
            "gap_hours": round(gap_hours, 6),
            "source_lines": [left_line, right_line],
            "adjudication": status,
            "evidence_match": matched,
        })

    if unresolved:
        warnings.append("unresolved_timestamp_gaps_remain")
    if not gaps:
        warnings.append("no_gaps_exceeded_threshold; this is not proof of full market-calendar coverage")
    if errors:
        checks["evidence_integrity"] = "FAIL"
    else:
        checks["evidence_integrity"] = "PASS"
    status = _safe_status(errors, unresolved, len(gaps))
    return _report(status, errors, warnings, checks, gaps, {
        "provider": evidence.get("provider"), "symbol": symbol, "timeframe": timeframe,
        "authority": evidence.get("authority"), "source_reference": evidence.get("source_reference"),
        "source_evidence_sha256": evidence.get("source_evidence_sha256"),
        "calendar_evidence_sha256": _sha256(evidence_bytes),
        "coverage_start_utc": evidence.get("coverage_start_utc"),
        "coverage_end_utc": evidence.get("coverage_end_utc"),
        "threshold_hours_exclusive": threshold_hours,
        "intervals_declared": len(intervals),
        "intervals_validated": len(normalized_intervals),
        "raw_snapshot_sha256": _sha256(csv_bytes),
    })


def _report(status: str, errors: list[str], warnings: list[str], checks: dict[str, str],
            gaps: list[dict[str, Any]], metadata: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit_name": "authoritative_market_calendar_gap_adjudication",
        "status": status,
        "adjudication_complete": status == "PASS" and not errors and all(
            item.get("adjudication") == "ADJUDICATED_DOCUMENTED_CLOSURE" for item in gaps
        ),
        "metadata": metadata or {},
        "checks": checks,
        "gap_count_over_threshold": len(gaps),
        "adjudicated_gap_count": sum(item.get("adjudication") == "ADJUDICATED_DOCUMENTED_CLOSURE" for item in gaps),
        "unresolved_gap_count": sum(item.get("adjudication") == "UNRESOLVED" for item in gaps),
        "gaps": gaps,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "actionable": False,
        "execution_connected": False,
        "automatic_repair_performed": False,
        "automatic_warning_suppression_performed": False,
        "model_qualified": False,
        "deployment_authorized": False,
        "scope_limitations": [
            "This report validates the supplied evidence record, hashes, scope, and explicit interval coverage; it does not independently authenticate the authority or legal status of the source.",
            "Only missing-bar intervals fully covered by an explicitly declared closed interval are adjudicated; partial overlap is unresolved.",
            "Raw OHLC files and their original audits are never modified. No candles are synthesized, repaired, or imputed.",
            "A PASS is evidence-conditional and is not a data-quality guarantee, model qualification, profitability finding, or deployment approval.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adjudicate raw OHLC timestamp gaps using hash-verified explicit calendar evidence.")
    parser.add_argument("--ohlc-csv", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--calendar-evidence", required=True, type=Path)
    parser.add_argument("--source-evidence", required=True, type=Path, help="Archived source document/file whose SHA-256 is recorded in calendar evidence.")
    parser.add_argument("--threshold-hours", type=float, default=8.0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    inputs = {p.expanduser().resolve() for p in (args.ohlc_csv, args.snapshot_manifest, args.calendar_evidence, args.source_evidence)}
    if output in inputs or output.exists():
        print("output_must_not_alias_input_or_overwrite_existing_file", file=sys.stderr)
        return 3
    if args.threshold_hours <= 0:
        print("threshold_hours_must_be_positive", file=sys.stderr)
        return 3
    report = adjudicate_market_calendar_gaps(args.ohlc_csv, args.snapshot_manifest, args.calendar_evidence,
                                             args.source_evidence, threshold_hours=args.threshold_hours)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except OSError as exc:
        print(f"report_write_failed:{type(exc).__name__}", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Market-calendar gap adjudication: {report['status']}")
        print(f"Adjudicated: {report['adjudicated_gap_count']} / {report['gap_count_over_threshold']}")
        print(f"Unresolved: {report['unresolved_gap_count']}")
        for item in report["errors"]:
            print(f"ERROR: {item}")
        for item in report["warnings"]:
            print(f"WARNING: {item}")
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())

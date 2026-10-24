"""Map individual raw OHLC timestamp gaps to the labels whose forward windows cross them.

This is a report-only integrity audit. It uses immutable hashed input artifacts and
does not infer that a gap is missing data or an expected market closure.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.research_label_duration_gap_attribution import (
    _ATTRIBUTION_SCOPE,
    _PASS,
    _PIPELINE_SCOPE,
    _TIMEFRAME_MINUTES,
    _load_dataset,
    _load_raw_timestamps,
    _read_json,
    _resolve_artifact,
    _sha256,
)

_EVENT_SCOPE = "GOLDCREST_LABEL_GAP_EVENT_PROFILE"
_EPSILON_HOURS = 1e-9
_EPSILON_SECONDS = 1e-6


def _profile_caps(attribution: Mapping[str, Any]) -> list[float]:
    raw = attribution.get("caps_hours")
    if not isinstance(raw, list) or not raw:
        raise ValueError("attribution_caps_missing")
    caps: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("attribution_cap_invalid")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("attribution_cap_invalid")
        caps.append(number)
    if len(caps) != len(set(caps)):
        raise ValueError("attribution_caps_not_unique")
    return sorted(caps)


def _duration_bucket(hours: float, caps: Sequence[float]) -> str:
    lower: float | None = None
    for cap in caps:
        if hours <= cap + _EPSILON_HOURS:
            return f"<= {cap:g}" if lower is None else f"({lower:g}, {cap:g}]"
        lower = cap
    return f"> {caps[-1]:g}"


def _gap_class(excess_hours: float) -> str:
    if excess_hours <= 1.0 + _EPSILON_HOURS:
        return "excess_up_to_1h"
    if excess_hours <= 8.0 + _EPSILON_HOURS:
        return "excess_over_1_to_8h"
    return "excess_over_8h"


def _weekend_like(left: datetime, right: datetime, gap_hours: float) -> bool:
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


def _calendar_context(left: datetime, right: datetime) -> list[str]:
    found: list[str] = []
    day = left.date()
    last = right.date()
    while day <= last:
        if (day.month, day.day) == (12, 25) and "christmas_day_utc_date" not in found:
            found.append("christmas_day_utc_date")
        if (day.month, day.day) == (1, 1) and "new_years_day_utc_date" not in found:
            found.append("new_years_day_utc_date")
        day += timedelta(days=1)
    return found


def _free_output_path(manifest_path: Path, run_id: str) -> Path:
    base = manifest_path.parent / f"{run_id}_label_gap_event_profile.json"
    if not base.exists():
        return base
    for revision in range(2, 10002):
        candidate = manifest_path.parent / f"{run_id}-rerun-{revision}_label_gap_event_profile.json"
        if not candidate.exists():
            return candidate
    raise ValueError("unable_to_find_free_gap_event_profile_path")


def audit_label_gap_event_profile(
    baseline_manifest_path: str | Path,
    duration_gap_attribution_report_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(baseline_manifest_path).expanduser().resolve()
    baseline = _read_json(manifest_path)
    if baseline.get("scope") != _PIPELINE_SCOPE:
        raise ValueError("input_is_not_an_offline_research_pipeline_manifest")
    if baseline.get("label_eligibility_mode", "all_labels_included") != "all_labels_included":
        raise ValueError("baseline_must_be_uncapped")
    if baseline.get("max_label_elapsed_hours") is not None:
        raise ValueError("baseline_must_have_no_elapsed_time_cap")
    for flag in ("actionable", "execution_connected", "automatic_winner_selection"):
        if baseline.get(flag) is not False:
            raise ValueError(f"baseline_research_only_flag_invalid:{flag}")

    attribution_path = Path(duration_gap_attribution_report_path).expanduser().resolve()
    attribution = _read_json(attribution_path)
    if attribution.get("scope") != _ATTRIBUTION_SCOPE:
        raise ValueError("input_is_not_a_label_duration_gap_attribution_report")
    if attribution.get("status") not in _PASS:
        raise ValueError("duration_gap_attribution_status_not_pass")
    for flag in ("inferential", "automatic_model_selection", "actionable", "execution_connected"):
        if attribution.get(flag) is not False:
            raise ValueError(f"duration_gap_attribution_research_only_flag_invalid:{flag}")

    raw_path = _resolve_artifact(manifest_path, baseline.get("raw_ohlc_path"), field="raw_ohlc_path")
    dataset_path = _resolve_artifact(manifest_path, baseline.get("dataset_path"), field="dataset_path")
    raw_hash = baseline.get("raw_ohlc_sha256")
    dataset_hash = baseline.get("dataset_sha256")
    if not isinstance(raw_hash, str) or _sha256(raw_path) != raw_hash:
        raise ValueError("baseline_raw_ohlc_file_hash_mismatch")
    if not isinstance(dataset_hash, str) or _sha256(dataset_path) != dataset_hash:
        raise ValueError("baseline_dataset_file_hash_mismatch")
    if attribution.get("baseline_raw_ohlc_sha256") != raw_hash:
        raise ValueError("baseline_attribution_raw_ohlc_hash_mismatch")
    if attribution.get("baseline_dataset_sha256") != dataset_hash:
        raise ValueError("baseline_attribution_dataset_hash_mismatch")
    if attribution.get("baseline_manifest_path") and Path(str(attribution["baseline_manifest_path"])).resolve() != manifest_path:
        raise ValueError("duration_gap_attribution_baseline_manifest_path_mismatch")
    if attribution.get("baseline_raw_ohlc_path") and Path(str(attribution["baseline_raw_ohlc_path"])).resolve() != raw_path:
        raise ValueError("duration_gap_attribution_raw_path_mismatch")
    if attribution.get("baseline_dataset_path") and Path(str(attribution["baseline_dataset_path"])).resolve() != dataset_path:
        raise ValueError("duration_gap_attribution_dataset_path_mismatch")

    timeframe = str(baseline.get("timeframe", "")).upper()
    if timeframe not in _TIMEFRAME_MINUTES:
        raise ValueError(f"baseline_timeframe_unsupported:{timeframe or 'missing'}")
    frame_seconds = _TIMEFRAME_MINUTES[timeframe] * 60
    horizon = baseline.get("label_horizon_bars")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("baseline_label_horizon_bars_invalid")
    caps = _profile_caps(attribution)

    raw_timestamps, raw_errors = _load_raw_timestamps(raw_path)
    data_rows = _load_dataset(dataset_path, timeframe)
    errors = list(raw_errors)
    warnings = list(attribution.get("warnings") or [])
    if any(right <= left for left, right in zip(raw_timestamps, raw_timestamps[1:])):
        errors.append("raw_timestamps_not_strictly_increasing")
    if len(set(raw_timestamps)) != len(raw_timestamps):
        errors.append("raw_timestamps_not_unique")

    raw_index: dict[datetime, int] = {}
    for idx, timestamp in enumerate(raw_timestamps):
        raw_index.setdefault(timestamp, idx)

    labels_by_start_index: dict[int, dict[str, Any]] = {}
    valid_label_rows: list[dict[str, Any]] = []
    missing_start = 0
    missing_target = 0
    horizon_mismatches = 0
    for row_number, (start, target) in enumerate(data_rows, start=2):
        if start not in raw_index:
            missing_start += 1
            errors.append(f"dataset_start_timestamp_missing_from_raw_ohlc_line_{row_number}")
            continue
        if target not in raw_index:
            missing_target += 1
            errors.append(f"dataset_label_maturity_timestamp_missing_from_raw_ohlc_line_{row_number}")
            continue
        start_index, target_index = raw_index[start], raw_index[target]
        if target_index - start_index != horizon:
            horizon_mismatches += 1
            errors.append(f"dataset_label_horizon_index_mismatch_line_{row_number}:{target_index - start_index}!={horizon}")
            continue
        elapsed_hours = (target - start).total_seconds() / 3600.0
        row = {
            "dataset_row_number": row_number,
            "start": start,
            "target": target,
            "start_index": start_index,
            "target_index": target_index,
            "elapsed_hours": elapsed_hours,
            "duration_bucket": _duration_bucket(elapsed_hours, caps),
        }
        valid_label_rows.append(row)
        labels_by_start_index[start_index] = row

    frame_hours = frame_seconds / 3600.0
    event_records: list[dict[str, Any]] = []
    row_event_counts: dict[int, int] = {row["dataset_row_number"]: 0 for row in valid_label_rows}
    row_event_buckets: dict[int, set[str]] = {row["dataset_row_number"]: set() for row in valid_label_rows}
    associations = 0

    for right_index in range(1, len(raw_timestamps)):
        left, right = raw_timestamps[right_index - 1], raw_timestamps[right_index]
        interval_seconds = (right - left).total_seconds()
        if interval_seconds <= frame_seconds + _EPSILON_SECONDS:
            continue
        gap_hours = interval_seconds / 3600.0
        excess_hours = gap_hours - frame_hours
        affected: list[dict[str, Any]] = []
        for start_index in range(max(0, right_index - horizon), right_index):
            label = labels_by_start_index.get(start_index)
            if label is None or label["target_index"] < right_index:
                continue
            affected.append(label)
            row_number = label["dataset_row_number"]
            row_event_counts[row_number] += 1
            row_event_buckets[row_number].add(label["duration_bucket"])
            associations += 1
        is_long = gap_hours > 8.0
        weekend_like = _weekend_like(left, right, gap_hours) if is_long else None
        context = _calendar_context(left, right)
        event_records.append({
            "gap_event_id": f"gap-{right_index:06d}",
            "left_raw_row_index": right_index - 1,
            "right_raw_row_index": right_index,
            "from_utc": left.isoformat().replace("+00:00", "Z"),
            "to_utc": right.isoformat().replace("+00:00", "Z"),
            "interval_minutes": round(interval_seconds / 60.0, 6),
            "interval_hours": round(gap_hours, 8),
            "excess_over_nominal_frame_hours": round(excess_hours, 8),
            "gap_excess_class": _gap_class(excess_hours),
            "over_8h_gap": is_long,
            "long_gap_weekend_closure_like_heuristic": weekend_like,
            "fixed_date_calendar_context": context,
            "affected_label_count": len(affected),
            "affected_label_rows": [item["dataset_row_number"] for item in affected],
            "first_affected_label_timestamp_utc": (
                min((item["start"] for item in affected)).isoformat().replace("+00:00", "Z")
                if affected else None
            ),
            "last_affected_label_timestamp_utc": (
                max((item["start"] for item in affected)).isoformat().replace("+00:00", "Z")
                if affected else None
            ),
            "affected_duration_bucket_counts": {
                name: sum(item["duration_bucket"] == name for item in affected)
                for name in [f"<= {caps[0]:g}"]
                + [f"({caps[i - 1]:g}, {caps[i]:g}]" for i in range(1, len(caps))]
                + [f"> {caps[-1]:g}"]
            },
        })

    rows_with_any_gap = sum(count > 0 for count in row_event_counts.values())
    rows_with_multiple_gaps = sum(count > 1 for count in row_event_counts.values())
    max_gaps_per_label = max(row_event_counts.values(), default=0)
    counts_by_class: dict[str, int] = {
        "excess_up_to_1h": 0,
        "excess_over_1_to_8h": 0,
        "excess_over_8h": 0,
    }
    for event in event_records:
        counts_by_class[event["gap_excess_class"]] += 1
    labels_affected_by_class: dict[str, int] = {}
    for name in counts_by_class:
        affected_rows = {
            row_number
            for event in event_records if event["gap_excess_class"] == name
            for row_number in event["affected_label_rows"]
        }
        labels_affected_by_class[name] = len(affected_rows)

    cross_checks: list[dict[str, Any]] = []
    for field, computed in (
        ("rows_with_any_interbar_gap", rows_with_any_gap),
        ("rows_with_elapsed_duration_above_nominal", rows_with_any_gap),
        ("rows_with_duration_gap_excess_reconciled", len(valid_label_rows)),
    ):
        reported = attribution.get(field)
        matches = isinstance(reported, int) and not isinstance(reported, bool) and reported == computed
        cross_checks.append({
            "field": field,
            "computed_value": computed,
            "reported_value": reported,
            "matches": matches,
        })
        if not matches:
            errors.append(f"duration_gap_attribution_cross_check_mismatch:{field}:{computed}!={reported}")

    if missing_start or missing_target or horizon_mismatches:
        pass
    if len(valid_label_rows) != len(data_rows):
        errors.append(f"valid_label_row_count_mismatch:{len(valid_label_rows)}!={len(data_rows)}")
    if associations != sum(len(event["affected_label_rows"]) for event in event_records):
        errors.append("gap_label_association_total_mismatch")
    if rows_with_any_gap != int(attribution.get("rows_with_any_interbar_gap", -1)):
        errors.append("unique_gap_exposed_label_count_mismatch")
    if rows_with_multiple_gaps:
        warnings.append("some_label_windows_cross_multiple_observed_gaps")
    if any(event["affected_label_count"] != horizon for event in event_records):
        warnings.append("some_gap_events_have_fewer_or_more_than_horizon_count_affected_labels_due_to_dataset_edges_or_overlapping_gaps")
    warnings.append(
        "Gap-event and weekend-like classifications are timestamp-based diagnostics only; they do not prove "
        "missing candles, data-provider failure, or an expected market-session closure."
    )
    if baseline.get("raw_ohlc_audit_status") == "PASS_WITH_WARNINGS":
        warnings.append("raw_ohlc_audit_passed_with_warnings")
    warnings = list(dict.fromkeys(str(item) for item in warnings))
    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")

    duration_bucket_counts: dict[str, int] = {}
    for row in valid_label_rows:
        duration_bucket_counts[row["duration_bucket"]] = duration_bucket_counts.get(row["duration_bucket"], 0) + 1
    event_count_by_affected_labels: dict[str, int] = {}
    for event in event_records:
        key = str(event["affected_label_count"])
        event_count_by_affected_labels[key] = event_count_by_affected_labels.get(key, 0) + 1

    run_id = str(baseline.get("run_id") or manifest_path.stem.removesuffix("_pipeline_manifest"))
    destination = Path(output_path).expanduser().resolve() if output_path else _free_output_path(manifest_path, run_id)
    if destination in {manifest_path, dataset_path, raw_path, attribution_path}:
        raise ValueError("output_path_must_not_overwrite_an_input_artifact")
    if destination.exists():
        raise ValueError("output_path_already_exists_refusing_to_overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_version": 1,
        "scope": _EVENT_SCOPE,
        "status": status,
        "baseline_manifest_path": str(manifest_path),
        "baseline_dataset_path": str(dataset_path),
        "baseline_dataset_sha256": dataset_hash,
        "baseline_raw_ohlc_path": str(raw_path),
        "baseline_raw_ohlc_sha256": raw_hash,
        "duration_gap_attribution_report_path": str(attribution_path),
        "duration_gap_attribution_report_sha256": _sha256(attribution_path),
        "symbol": baseline.get("symbol"),
        "timeframe": timeframe,
        "timeframe_minutes": _TIMEFRAME_MINUTES[timeframe],
        "label_horizon_bars": horizon,
        "nominal_label_horizon_hours": horizon * frame_hours,
        "dataset_rows": len(data_rows),
        "valid_label_rows": len(valid_label_rows),
        "raw_non_nominal_gap_event_count": len(event_records),
        "gap_event_counts_by_excess_class": counts_by_class,
        "unique_label_rows_crossing_any_gap": rows_with_any_gap,
        "label_gap_association_count": associations,
        "label_rows_crossing_multiple_gap_events": rows_with_multiple_gaps,
        "maximum_gap_events_crossed_by_one_label": max_gaps_per_label,
        "event_count_by_affected_label_count": dict(sorted(event_count_by_affected_labels.items(), key=lambda item: int(item[0]))),
        "labels_affected_by_gap_class": labels_affected_by_class,
        "duration_bucket_counts_recomputed": duration_bucket_counts,
        "duration_gap_attribution_cross_checks": cross_checks,
        "missing_start_timestamps": missing_start,
        "missing_label_maturity_timestamps": missing_target,
        "row_index_horizon_mismatches": horizon_mismatches,
        "gap_events": event_records,
        "errors": list(dict.fromkeys(errors)),
        "warnings": warnings,
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
        "data_rewritten": False,
        "models_trained": False,
        "interpretation": (
            "Each event identifies one consecutive raw OHLC timestamp interval longer than the nominal "
            "timeframe and lists dataset label rows whose configured row-index forward window crosses it. "
            "Overlapping events may affect the same label more than once. Weekend-like and fixed-date tags "
            "are heuristics or informational context, not proof of expected market closures. This report "
            "does not repair data, choose a cap, retrain models, or connect to execution."
        ),
        "report_path": str(destination),
    }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile individual timestamp-gap events and their label-window exposures."
    )
    parser.add_argument("--baseline-manifest", required=True, help="Immutable uncapped pipeline manifest")
    parser.add_argument("--duration-gap-attribution-report", required=True, help="Completed duration-gap attribution report")
    parser.add_argument("--output-path", help="Optional immutable report destination")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_label_gap_event_profile(
            args.baseline_manifest,
            args.duration_gap_attribution_report,
            output_path=args.output_path,
        )
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL",
            "scope": _EVENT_SCOPE,
            "errors": [f"{type(exc).__name__}:{exc}"],
            "inferential": False,
            "automatic_model_selection": False,
            "actionable": False,
            "execution_connected": False,
        }, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": report["status"],
        "report_path": report["report_path"],
        "dataset_rows": report["dataset_rows"],
        "raw_non_nominal_gap_event_count": report["raw_non_nominal_gap_event_count"],
        "gap_event_counts_by_excess_class": report["gap_event_counts_by_excess_class"],
        "unique_label_rows_crossing_any_gap": report["unique_label_rows_crossing_any_gap"],
        "label_gap_association_count": report["label_gap_association_count"],
        "label_rows_crossing_multiple_gap_events": report["label_rows_crossing_multiple_gap_events"],
        "maximum_gap_events_crossed_by_one_label": report["maximum_gap_events_crossed_by_one_label"],
        "event_count_by_affected_label_count": report["event_count_by_affected_label_count"],
        "labels_affected_by_gap_class": report["labels_affected_by_gap_class"],
        "duration_bucket_counts_recomputed": report["duration_bucket_counts_recomputed"],
        "duration_gap_attribution_cross_checks": report["duration_gap_attribution_cross_checks"],
        "missing_start_timestamps": report["missing_start_timestamps"],
        "missing_label_maturity_timestamps": report["missing_label_maturity_timestamps"],
        "row_index_horizon_mismatches": report["row_index_horizon_mismatches"],
        "errors": report["errors"],
        "warnings": report["warnings"],
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
    }, indent=2, sort_keys=True))
    return 0 if report["status"] in _PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())

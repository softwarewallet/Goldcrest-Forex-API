"""Attribute long label-maturity durations to observed OHLC timestamp gaps.

This is a report-only offline audit. It checks that each label's start and maturity
timestamps are present in the immutable raw timeline, verifies the configured
row-index horizon, and decomposes elapsed duration into nominal bar time plus
observed inter-bar excess time. It does not infer missing candles or session rules.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_PIPELINE_SCOPE = "GOLDCREST_OFFLINE_RESEARCH_PIPELINE"
_PROFILE_SCOPE = "GOLDCREST_LABEL_ELAPSED_DURATION_CAP_BOUNDARY_PROFILE"
_ATTRIBUTION_SCOPE = "GOLDCREST_LABEL_DURATION_GAP_ATTRIBUTION_AUDIT"
_PASS = {"PASS", "PASS_WITH_WARNINGS"}
_TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
_EPSILON_SECONDS = 1e-6


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"json_unreadable:{resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"json_not_object:{resolved}")
    return payload


def _parse_utc(value: object, *, field: str, row_number: int) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row_{row_number}_{field}_missing")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"row_{row_number}_{field}_not_iso8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"row_{row_number}_{field}_timezone_missing")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"artifact_unreadable:{path}") from exc


def _resolve_artifact(manifest_path: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"baseline_manifest_{field}_missing")
    supplied = Path(value).expanduser()
    candidates = [supplied] if supplied.is_absolute() else [manifest_path.parent / supplied, supplied]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise ValueError(f"baseline_artifact_not_found:{field}:{value}")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label}_must_be_positive_integer")
    return value


def _caps(profile: Mapping[str, Any]) -> list[float]:
    raw = profile.get("caps_hours")
    if not isinstance(raw, list) or not raw:
        raise ValueError("duration_profile_caps_missing")
    result: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("duration_profile_cap_invalid")
        number = float(item)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("duration_profile_cap_invalid")
        result.append(number)
    if len(result) != len(set(result)):
        raise ValueError("duration_profile_caps_not_unique")
    return sorted(result)


def _load_raw_timestamps(path: Path) -> tuple[list[datetime], list[str]]:
    timestamps: list[datetime] = []
    errors: list[str] = []
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"raw_ohlc_unreadable:{path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "open", "high", "low", "close"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"raw_ohlc_missing_columns:{','.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            timestamp = _parse_utc(row.get("timestamp"), field="raw_timestamp", row_number=line)
            try:
                values = {name: float(row[name]) for name in ("open", "high", "low", "close")}
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"raw_ohlc_invalid_prices_line_{line}") from exc
            if (
                any(not math.isfinite(value) or value <= 0 for value in values.values())
                or values["low"] > min(values["open"], values["close"])
                or values["high"] < max(values["open"], values["close"])
                or values["high"] < values["low"]
            ):
                errors.append(f"raw_ohlc_invalid_prices_line_{line}")
            if timestamps and timestamp <= timestamps[-1]:
                errors.append(f"raw_ohlc_timestamps_not_strictly_increasing_line_{line}")
            timestamps.append(timestamp)
    if not timestamps:
        raise ValueError("raw_ohlc_has_no_rows")
    return timestamps, list(dict.fromkeys(errors))


def _load_dataset(path: Path, expected_timeframe: str) -> list[tuple[datetime, datetime]]:
    result: list[tuple[datetime, datetime]] = []
    previous: datetime | None = None
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"dataset_unreadable:{path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "label_available_at"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"dataset_missing_columns:{','.join(sorted(missing))}")
        if "timeframe" in (reader.fieldnames or []):
            for line, row in enumerate(reader, start=2):
                timeframe = str(row.get("timeframe", "")).strip().upper()
                if timeframe and timeframe != expected_timeframe:
                    raise ValueError(f"dataset_timeframe_mismatch_line_{line}:{timeframe}!={expected_timeframe}")
                timestamp = _parse_utc(row.get("timestamp"), field="timestamp", row_number=line)
                available = _parse_utc(row.get("label_available_at"), field="label_available_at", row_number=line)
                if previous is not None and timestamp <= previous:
                    raise ValueError("dataset_timestamps_not_strictly_chronological_and_unique")
                if available < timestamp:
                    raise ValueError(f"dataset_label_maturity_precedes_timestamp_line_{line}")
                previous = timestamp
                result.append((timestamp, available))
        else:
            for line, row in enumerate(reader, start=2):
                timestamp = _parse_utc(row.get("timestamp"), field="timestamp", row_number=line)
                available = _parse_utc(row.get("label_available_at"), field="label_available_at", row_number=line)
                if previous is not None and timestamp <= previous:
                    raise ValueError("dataset_timestamps_not_strictly_chronological_and_unique")
                if available < timestamp:
                    raise ValueError(f"dataset_label_maturity_precedes_timestamp_line_{line}")
                previous = timestamp
                result.append((timestamp, available))
    if not result:
        raise ValueError("dataset_has_no_rows")
    return result


def _duration_bucket(hours: float, caps: Sequence[float]) -> str:
    lower: float | None = None
    for cap in caps:
        if hours <= cap + 1e-9:
            return f"<= {cap:g}" if lower is None else f"({lower:g}, {cap:g}]"
        lower = cap
    return f"> {caps[-1]:g}"


def _new_bucket(bucket_name: str) -> dict[str, Any]:
    return {
        "duration_bucket": bucket_name,
        "row_count": 0,
        "rows_with_any_interbar_gap": 0,
        "rows_with_elapsed_duration_above_nominal": 0,
        "rows_with_duration_gap_excess_reconciled": 0,
        "elapsed_hours_total": 0.0,
        "nominal_bar_hours_total": 0.0,
        "duration_excess_hours_total": 0.0,
        "observed_gap_excess_hours_total": 0.0,
        "maximum_label_elapsed_hours": 0.0,
        "maximum_single_interbar_gap_hours": 0.0,
    }


def _free_output_path(manifest_path: Path, run_id: str) -> Path:
    base = manifest_path.parent / f"{run_id}_label_duration_gap_attribution.json"
    if not base.exists():
        return base
    for revision in range(2, 10002):
        candidate = manifest_path.parent / f"{run_id}-rerun-{revision}_label_duration_gap_attribution.json"
        if not candidate.exists():
            return candidate
    raise ValueError("unable_to_find_free_gap_attribution_report_path")


def audit_label_duration_gap_attribution(
    baseline_manifest_path: str | Path,
    duration_profile_report_path: str | Path,
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

    profile_path = Path(duration_profile_report_path).expanduser().resolve()
    profile = _read_json(profile_path)
    if profile.get("scope") != _PROFILE_SCOPE:
        raise ValueError("input_is_not_a_label_duration_boundary_profile")
    if profile.get("status") not in _PASS:
        raise ValueError("duration_profile_status_not_pass")
    for flag in ("inferential", "automatic_model_selection", "actionable", "execution_connected"):
        if profile.get(flag) is not False:
            raise ValueError(f"duration_profile_research_only_flag_invalid:{flag}")

    raw_hash = baseline.get("raw_ohlc_sha256")
    if not isinstance(raw_hash, str) or not raw_hash or raw_hash != profile.get("baseline_raw_ohlc_sha256"):
        raise ValueError("baseline_and_duration_profile_raw_hash_mismatch")
    dataset_path = _resolve_artifact(manifest_path, baseline.get("dataset_path"), field="dataset_path")
    raw_path = _resolve_artifact(manifest_path, baseline.get("raw_ohlc_path"), field="raw_ohlc_path")
    dataset_hash = _sha256(dataset_path)
    if dataset_hash != baseline.get("dataset_sha256") or dataset_hash != profile.get("baseline_dataset_sha256"):
        raise ValueError("baseline_or_profile_dataset_hash_mismatch")
    if _sha256(raw_path) != raw_hash:
        raise ValueError("baseline_raw_ohlc_file_hash_mismatch")
    if profile.get("baseline_dataset_path") and Path(str(profile["baseline_dataset_path"])).resolve() != dataset_path:
        raise ValueError("duration_profile_dataset_path_mismatch")
    if profile.get("baseline_manifest_path") and Path(str(profile["baseline_manifest_path"])).resolve() != manifest_path:
        raise ValueError("duration_profile_baseline_manifest_path_mismatch")

    timeframe = str(baseline.get("timeframe", "")).upper()
    if timeframe not in _TIMEFRAME_MINUTES:
        raise ValueError(f"baseline_timeframe_unsupported:{timeframe or 'missing'}")
    frame_seconds = _TIMEFRAME_MINUTES[timeframe] * 60
    horizon_bars = _positive_int(baseline.get("label_horizon_bars"), "baseline_label_horizon_bars")
    caps = _caps(profile)

    raw_timestamps, raw_errors = _load_raw_timestamps(raw_path)
    data_rows = _load_dataset(dataset_path, timeframe)
    errors = list(raw_errors)
    raw_index: dict[datetime, int] = {}
    for index, timestamp in enumerate(raw_timestamps):
        if timestamp in raw_index:
            errors.append("raw_ohlc_duplicate_timestamp")
        else:
            raw_index[timestamp] = index
    if len(raw_timestamps) < horizon_bars + 1:
        raise ValueError("raw_ohlc_has_fewer_rows_than_label_horizon")

    bucket_names = [_duration_bucket(0.0, caps)]
    lower = caps[0]
    bucket_names.extend(f"({caps[i - 1]:g}, {caps[i]:g}]" for i in range(1, len(caps)))
    bucket_names.append(f"> {caps[-1]:g}")
    bucket_summaries = {name: _new_bucket(name) for name in bucket_names}

    row_gap_count = 0
    row_duration_above_nominal = 0
    reconciled_rows = 0
    missing_start_count = 0
    missing_target_count = 0
    horizon_mismatch_count = 0
    maximum_absolute_reconciliation_error_hours = 0.0
    interval_gap_counts = {"none": 0, "excess_up_to_1h": 0, "excess_over_1_to_8h": 0, "excess_over_8h": 0}
    total_elapsed_hours = 0.0
    total_nominal_hours = 0.0
    total_duration_excess_hours = 0.0
    total_observed_gap_excess_hours = 0.0
    largest_gap_examples: list[dict[str, Any]] = []

    for row_number, (start, target) in enumerate(data_rows, start=2):
        if start not in raw_index:
            missing_start_count += 1
            errors.append(f"dataset_start_timestamp_missing_from_raw_ohlc_line_{row_number}")
            continue
        if target not in raw_index:
            missing_target_count += 1
            errors.append(f"dataset_label_available_at_missing_from_raw_ohlc_line_{row_number}")
            continue
        start_index, target_index = raw_index[start], raw_index[target]
        index_delta = target_index - start_index
        if index_delta != horizon_bars:
            horizon_mismatch_count += 1
            errors.append(
                f"dataset_label_horizon_index_mismatch_line_{row_number}:{index_delta}!={horizon_bars}"
            )
            continue

        elapsed_hours = (target - start).total_seconds() / 3600.0
        nominal_hours = horizon_bars * frame_seconds / 3600.0
        observed_gap_excess_hours = 0.0
        interbar_gap_count = 0
        maximum_gap_excess_hours = 0.0
        for interval_index in range(start_index + 1, target_index + 1):
            interval_seconds = (raw_timestamps[interval_index] - raw_timestamps[interval_index - 1]).total_seconds()
            if interval_seconds < frame_seconds - _EPSILON_SECONDS:
                errors.append(f"raw_ohlc_interval_shorter_than_timeframe:{interval_index}")
            excess = max(0.0, interval_seconds - frame_seconds)
            if excess > _EPSILON_SECONDS:
                interbar_gap_count += 1
                excess_hours = excess / 3600.0
                observed_gap_excess_hours += excess_hours
                maximum_gap_excess_hours = max(maximum_gap_excess_hours, excess_hours)

        duration_excess_hours = elapsed_hours - nominal_hours
        reconciliation_error = abs(duration_excess_hours - observed_gap_excess_hours)
        maximum_absolute_reconciliation_error_hours = max(
            maximum_absolute_reconciliation_error_hours, reconciliation_error
        )
        reconciled = reconciliation_error <= 1e-8
        if not reconciled:
            errors.append(
                f"duration_gap_excess_mismatch_line_{row_number}:"
                f"{duration_excess_hours:.12f}!={observed_gap_excess_hours:.12f}"
            )
        else:
            reconciled_rows += 1
        if duration_excess_hours < -1e-9:
            errors.append(f"label_elapsed_shorter_than_nominal_line_{row_number}")

        bucket_name = _duration_bucket(elapsed_hours, caps)
        bucket = bucket_summaries[bucket_name]
        bucket["row_count"] += 1
        bucket["rows_with_any_interbar_gap"] += int(interbar_gap_count > 0)
        bucket["rows_with_elapsed_duration_above_nominal"] += int(duration_excess_hours > 1e-9)
        bucket["rows_with_duration_gap_excess_reconciled"] += int(reconciled)
        bucket["elapsed_hours_total"] += elapsed_hours
        bucket["nominal_bar_hours_total"] += nominal_hours
        bucket["duration_excess_hours_total"] += max(0.0, duration_excess_hours)
        bucket["observed_gap_excess_hours_total"] += observed_gap_excess_hours
        bucket["maximum_label_elapsed_hours"] = max(bucket["maximum_label_elapsed_hours"], elapsed_hours)
        bucket["maximum_single_interbar_gap_hours"] = max(
            bucket["maximum_single_interbar_gap_hours"], maximum_gap_excess_hours
        )

        total_elapsed_hours += elapsed_hours
        total_nominal_hours += nominal_hours
        total_duration_excess_hours += max(0.0, duration_excess_hours)
        total_observed_gap_excess_hours += observed_gap_excess_hours
        row_gap_count += int(interbar_gap_count > 0)
        row_duration_above_nominal += int(duration_excess_hours > 1e-9)
        if interbar_gap_count == 0:
            interval_gap_counts["none"] += 1
        elif maximum_gap_excess_hours <= 1.0 + 1e-9:
            interval_gap_counts["excess_up_to_1h"] += 1
        elif maximum_gap_excess_hours <= 8.0 + 1e-9:
            interval_gap_counts["excess_over_1_to_8h"] += 1
        else:
            interval_gap_counts["excess_over_8h"] += 1
        if maximum_gap_excess_hours > 0:
            largest_gap_examples.append({
                "dataset_row_number": row_number,
                "start_timestamp_utc": start.isoformat().replace("+00:00", "Z"),
                "label_available_at_utc": target.isoformat().replace("+00:00", "Z"),
                "label_elapsed_hours": round(elapsed_hours, 8),
                "maximum_single_interbar_gap_excess_hours": round(maximum_gap_excess_hours, 8),
                "interbar_gap_count": interbar_gap_count,
                "duration_bucket": bucket_name,
            })

    processed_rows = sum(item["row_count"] for item in bucket_summaries.values())
    expected_rows = profile.get("duration_statistics", {}).get("observation_count")
    if processed_rows != len(data_rows):
        errors.append(f"processed_dataset_rows_mismatch:{processed_rows}!={len(data_rows)}")
    if isinstance(expected_rows, int) and expected_rows != len(data_rows):
        errors.append(f"profile_observation_count_mismatch:{expected_rows}!={len(data_rows)}")
    if row_gap_count != row_duration_above_nominal:
        errors.append(
            f"rows_with_gaps_vs_above_nominal_mismatch:{row_gap_count}!={row_duration_above_nominal}"
        )
    total_reconciliation_error = abs(total_duration_excess_hours - total_observed_gap_excess_hours)
    if total_reconciliation_error > 1e-7:
        errors.append(
            "aggregate_duration_gap_excess_mismatch:"
            f"{total_duration_excess_hours:.10f}!={total_observed_gap_excess_hours:.10f}"
        )

    for item in bucket_summaries.values():
        for field in (
            "elapsed_hours_total", "nominal_bar_hours_total", "duration_excess_hours_total",
            "observed_gap_excess_hours_total", "maximum_label_elapsed_hours",
            "maximum_single_interbar_gap_hours",
        ):
            item[field] = round(float(item[field]), 8)
        item["row_fraction_of_dataset"] = round(item["row_count"] / len(data_rows), 10)

    warnings = list(profile.get("warnings") or [])
    warnings.append(
        "Gap attribution is descriptive: observed timestamp spacing does not establish missing candles, "
        "expected trading sessions, or a data-provider failure."
    )
    if baseline.get("raw_ohlc_audit_status") == "PASS_WITH_WARNINGS":
        warnings.append("raw_ohlc_audit_passed_with_warnings")
    warnings = list(dict.fromkeys(str(item) for item in warnings))
    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")

    run_id = str(baseline.get("run_id") or manifest_path.stem.removesuffix("_pipeline_manifest"))
    destination = Path(output_path).expanduser().resolve() if output_path else _free_output_path(manifest_path, run_id)
    if destination in {manifest_path, dataset_path, raw_path, profile_path}:
        raise ValueError("output_path_must_not_overwrite_an_input_artifact")
    if destination.exists():
        raise ValueError("output_path_already_exists_refusing_to_overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_version": 1,
        "scope": _ATTRIBUTION_SCOPE,
        "status": status,
        "baseline_manifest_path": str(manifest_path),
        "baseline_dataset_path": str(dataset_path),
        "baseline_dataset_sha256": dataset_hash,
        "baseline_raw_ohlc_path": str(raw_path),
        "baseline_raw_ohlc_sha256": raw_hash,
        "duration_profile_report_path": str(profile_path),
        "duration_profile_report_sha256": _sha256(profile_path),
        "symbol": baseline.get("symbol"),
        "timeframe": timeframe,
        "timeframe_minutes": _TIMEFRAME_MINUTES[timeframe],
        "label_horizon_bars": horizon_bars,
        "nominal_label_horizon_hours": horizon_bars * frame_seconds / 3600.0,
        "caps_hours": caps,
        "dataset_rows": len(data_rows),
        "processed_rows": processed_rows,
        "row_index_horizon_mismatches": horizon_mismatch_count,
        "missing_start_timestamps": missing_start_count,
        "missing_label_maturity_timestamps": missing_target_count,
        "rows_with_any_interbar_gap": row_gap_count,
        "rows_with_elapsed_duration_above_nominal": row_duration_above_nominal,
        "rows_with_duration_gap_excess_reconciled": reconciled_rows,
        "maximum_absolute_reconciliation_error_hours": round(maximum_absolute_reconciliation_error_hours, 12),
        "aggregate": {
            "elapsed_hours_total": round(total_elapsed_hours, 8),
            "nominal_bar_hours_total": round(total_nominal_hours, 8),
            "duration_excess_hours_total": round(total_duration_excess_hours, 8),
            "observed_gap_excess_hours_total": round(total_observed_gap_excess_hours, 8),
            "absolute_excess_reconciliation_error_hours": round(total_reconciliation_error, 12),
        },
        "duration_bucket_attribution": list(bucket_summaries.values()),
        "gap_excess_class_row_counts": interval_gap_counts,
        "largest_gap_exposure_examples": sorted(
            largest_gap_examples,
            key=lambda item: (
                -item["maximum_single_interbar_gap_excess_hours"],
                item["start_timestamp_utc"],
            ),
        )[:20],
        "errors": list(dict.fromkeys(errors)),
        "warnings": warnings,
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
        "data_rewritten": False,
        "models_trained": False,
        "interpretation": (
            "For each labelled dataset row, label maturity must correspond to exactly the configured "
            "row-index horizon in the hashed raw OHLC timeline. The excess of elapsed UTC duration over "
            "nominal horizon duration is reconciled to the sum of observed inter-bar time above one "
            "nominal timeframe. This is a timestamp arithmetic audit, not a claim that any interval is "
            "a missing candle, should have traded, or follows a known market-session calendar."
        ),
        "report_path": str(destination),
    }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attribute label elapsed-time extensions to observed OHLC timestamp gaps without retraining."
    )
    parser.add_argument("--baseline-manifest", required=True, help="Immutable uncapped pipeline manifest")
    parser.add_argument("--duration-profile-report", required=True, help="Completed label-duration cap boundary profile")
    parser.add_argument("--output-path", help="Optional immutable report destination")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_label_duration_gap_attribution(
            args.baseline_manifest,
            args.duration_profile_report,
            output_path=args.output_path,
        )
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL",
            "scope": _ATTRIBUTION_SCOPE,
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
        "processed_rows": report["processed_rows"],
        "timeframe": report["timeframe"],
        "label_horizon_bars": report["label_horizon_bars"],
        "nominal_label_horizon_hours": report["nominal_label_horizon_hours"],
        "duration_bucket_attribution": report["duration_bucket_attribution"],
        "aggregate": report["aggregate"],
        "gap_excess_class_row_counts": report["gap_excess_class_row_counts"],
        "row_index_horizon_mismatches": report["row_index_horizon_mismatches"],
        "rows_with_any_interbar_gap": report["rows_with_any_interbar_gap"],
        "rows_with_elapsed_duration_above_nominal": report["rows_with_elapsed_duration_above_nominal"],
        "rows_with_duration_gap_excess_reconciled": report["rows_with_duration_gap_excess_reconciled"],
        "maximum_absolute_reconciliation_error_hours": report["maximum_absolute_reconciliation_error_hours"],
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

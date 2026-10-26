"""Profile exact label-maturity durations around predeclared eligibility caps.

This report-only tool recomputes duration buckets from an existing uncapped dataset and
cross-checks eligible counts against the completed threshold-eligibility audit. It never
rebuilds the dataset, trains models, selects a cap, or connects to execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from app.research_strict_json import strict_json_loads
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_PIPELINE_SCOPE = "GOLDCREST_OFFLINE_RESEARCH_PIPELINE"
_AUDIT_SCOPE = "GOLDCREST_LABEL_ELIGIBILITY_THRESHOLD_PROFILE_AUDIT"
_PROFILE_SCOPE = "GOLDCREST_LABEL_ELAPSED_DURATION_CAP_BOUNDARY_PROFILE"
_PASS = {"PASS", "PASS_WITH_WARNINGS"}


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = strict_json_loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"json_unreadable:{resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"json_not_object:{resolved}")
    return payload


def _parse_utc(value: object, *, field: str, row_number: int) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dataset_row_{row_number}_{field}_missing")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"dataset_row_{row_number}_{field}_not_iso8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"dataset_row_{row_number}_{field}_timezone_missing")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _free_output_path(manifest_path: Path, run_id: str) -> Path:
    base = manifest_path.parent / f"{run_id}_label_duration_cap_boundary_profile.json"
    if not base.exists():
        return base
    for revision in range(2, 10002):
        candidate = manifest_path.parent / f"{run_id}-rerun-{revision}_label_duration_cap_boundary_profile.json"
        if not candidate.exists():
            return candidate
    raise ValueError("unable_to_find_free_label_duration_profile_path")


def _positive_caps(audit: Mapping[str, Any]) -> list[float]:
    raw_caps = audit.get("caps_audited_hours")
    if not isinstance(raw_caps, list) or not raw_caps:
        raise ValueError("threshold_audit_caps_missing")
    caps: list[float] = []
    for value in raw_caps:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("threshold_audit_cap_invalid")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("threshold_audit_cap_invalid")
        caps.append(number)
    if len(caps) != len(set(caps)):
        raise ValueError("threshold_audit_caps_not_unique")
    return sorted(caps)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}_must_be_non_negative_integer")
    return value


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile_requires_observations")
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _load_elapsed_hours(dataset_path: Path) -> tuple[list[float], int]:
    elapsed_values: list[float] = []
    previous_timestamp: datetime | None = None
    seen_timestamps: set[datetime] = set()
    try:
        handle = dataset_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"dataset_unreadable:{dataset_path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "label_available_at"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"baseline_dataset_missing_columns:{','.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            timestamp = _parse_utc(row.get("timestamp"), field="timestamp", row_number=row_number)
            label_available_at = _parse_utc(
                row.get("label_available_at"), field="label_available_at", row_number=row_number
            )
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError("baseline_dataset_timestamps_not_strictly_chronological")
            if timestamp in seen_timestamps:
                raise ValueError("baseline_dataset_duplicate_timestamp")
            seen_timestamps.add(timestamp)
            previous_timestamp = timestamp
            elapsed = (label_available_at - timestamp).total_seconds() / 3600.0
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError(f"dataset_row_{row_number}_label_elapsed_hours_invalid")
            elapsed_values.append(elapsed)
    if not elapsed_values:
        raise ValueError("baseline_dataset_has_no_labelled_rows")
    return elapsed_values, len(elapsed_values)


def _count_in_interval(values: Sequence[float], lower_exclusive: float | None, upper_inclusive: float | None) -> int:
    return sum(
        (lower_exclusive is None or value > lower_exclusive + 1e-9)
        and (upper_inclusive is None or value <= upper_inclusive + 1e-9)
        for value in values
    )


def audit_duration_cap_boundaries(
    baseline_manifest_path: str | Path,
    threshold_audit_report_path: str | Path,
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

    audit_path = Path(threshold_audit_report_path).expanduser().resolve()
    audit = _read_json(audit_path)
    if audit.get("scope") != _AUDIT_SCOPE:
        raise ValueError("input_is_not_a_threshold_eligibility_profile_audit")
    if audit.get("status") not in _PASS:
        raise ValueError("threshold_eligibility_audit_status_not_pass")
    for flag in ("actionable", "execution_connected", "automatic_model_selection", "inferential"):
        if audit.get(flag) is not False:
            raise ValueError(f"threshold_eligibility_audit_research_only_flag_invalid:{flag}")

    raw_hash = baseline.get("raw_ohlc_sha256")
    if not isinstance(raw_hash, str) or not raw_hash or raw_hash != audit.get("raw_ohlc_sha256"):
        raise ValueError("baseline_and_threshold_audit_raw_ohlc_hash_mismatch")
    raw_path = _resolve_artifact(manifest_path, baseline.get("raw_ohlc_path"), field="raw_ohlc_path")
    if _sha256(raw_path) != raw_hash:
        raise ValueError("baseline_raw_ohlc_file_hash_mismatch")

    dataset_path = _resolve_artifact(manifest_path, baseline.get("dataset_path"), field="dataset_path")
    dataset_hash = _sha256(dataset_path)
    expected_dataset_hash = baseline.get("dataset_sha256")
    if not isinstance(expected_dataset_hash, str) or dataset_hash != expected_dataset_hash:
        raise ValueError("baseline_dataset_file_hash_mismatch")

    elapsed_values, row_count = _load_elapsed_hours(dataset_path)
    expected_rows = baseline.get("dataset_rows")
    if isinstance(expected_rows, int) and not isinstance(expected_rows, bool) and expected_rows != row_count:
        raise ValueError("baseline_dataset_row_count_mismatch")

    caps = _positive_caps(audit)
    profile_records = audit.get("cap_profiles")
    if not isinstance(profile_records, list):
        raise ValueError("threshold_audit_cap_profiles_missing")
    by_cap: dict[float, Mapping[str, Any]] = {}
    for item in profile_records:
        if not isinstance(item, dict):
            raise ValueError("threshold_audit_cap_profile_invalid")
        cap = item.get("cap_hours")
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(float(cap)):
            raise ValueError("threshold_audit_cap_profile_cap_invalid")
        cap_value = float(cap)
        if cap_value in by_cap:
            raise ValueError("threshold_audit_cap_profile_duplicate")
        by_cap[cap_value] = item
    if set(by_cap) != set(caps):
        raise ValueError("threshold_audit_cap_profiles_do_not_match_cap_list")

    errors: list[str] = []
    warnings = list(audit.get("warnings") or [])
    cap_checks: list[dict[str, Any]] = []
    for cap in caps:
        actual_eligible = _count_in_interval(elapsed_values, None, cap)
        actual_ineligible = row_count - actual_eligible
        recorded = by_cap[cap]
        recorded_rows = _nonnegative_int(recorded.get("dataset_rows"), f"cap_{cap:g}_dataset_rows")
        recorded_eligible = _nonnegative_int(recorded.get("eligible_rows"), f"cap_{cap:g}_eligible_rows")
        recorded_ineligible = _nonnegative_int(recorded.get("ineligible_rows"), f"cap_{cap:g}_ineligible_rows")
        if recorded_rows != row_count:
            errors.append(f"cap_{cap:g}h_dataset_row_count_mismatch:{recorded_rows}!={row_count}")
        if recorded_eligible != actual_eligible:
            errors.append(f"cap_{cap:g}h_eligible_count_mismatch:{recorded_eligible}!={actual_eligible}")
        if recorded_ineligible != actual_ineligible:
            errors.append(f"cap_{cap:g}h_ineligible_count_mismatch:{recorded_ineligible}!={actual_ineligible}")
        cap_checks.append({
            "cap_hours": cap,
            "dataset_rows": row_count,
            "computed_eligible_rows": actual_eligible,
            "reported_eligible_rows": recorded_eligible,
            "computed_ineligible_rows": actual_ineligible,
            "reported_ineligible_rows": recorded_ineligible,
            "eligible_fraction": round(actual_eligible / row_count, 10),
            "counts_match_existing_audit": (
                recorded_rows == row_count
                and recorded_eligible == actual_eligible
                and recorded_ineligible == actual_ineligible
            ),
        })

    duration_buckets: list[dict[str, Any]] = []
    lower: float | None = None
    for cap in caps:
        count = _count_in_interval(elapsed_values, lower, cap)
        duration_buckets.append({
            "interval_hours": f"<= {cap:g}" if lower is None else f"({lower:g}, {cap:g}]",
            "lower_exclusive_hours": lower,
            "upper_inclusive_hours": cap,
            "row_count": count,
            "fraction_of_dataset": round(count / row_count, 10),
        })
        lower = cap
    tail_count = _count_in_interval(elapsed_values, lower, None)
    duration_buckets.append({
        "interval_hours": f"> {lower:g}",
        "lower_exclusive_hours": lower,
        "upper_inclusive_hours": None,
        "row_count": tail_count,
        "fraction_of_dataset": round(tail_count / row_count, 10),
    })
    bucket_sum = sum(item["row_count"] for item in duration_buckets)
    if bucket_sum != row_count:
        errors.append(f"duration_bucket_total_mismatch:{bucket_sum}!={row_count}")

    sorted_values = sorted(elapsed_values)
    duration_stats = {
        "observation_count": row_count,
        "minimum_hours": round(sorted_values[0], 8),
        "p50_hours": round(_percentile(sorted_values, 50), 8),
        "p90_hours": round(_percentile(sorted_values, 90), 8),
        "p95_hours": round(_percentile(sorted_values, 95), 8),
        "p99_hours": round(_percentile(sorted_values, 99), 8),
        "maximum_hours": round(sorted_values[-1], 8),
    }

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    run_id = str(baseline.get("run_id") or manifest_path.stem.removesuffix("_pipeline_manifest"))
    destination = Path(output_path).expanduser().resolve() if output_path else _free_output_path(manifest_path, run_id)
    if destination == manifest_path or destination == dataset_path or destination == audit_path:
        raise ValueError("output_path_must_not_overwrite_an_input_artifact")
    if destination.exists():
        raise ValueError("output_path_already_exists_refusing_to_overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "scope": _PROFILE_SCOPE,
        "status": status,
        "baseline_manifest_path": str(manifest_path),
        "baseline_dataset_path": str(dataset_path),
        "baseline_dataset_sha256": dataset_hash,
        "baseline_raw_ohlc_sha256": raw_hash,
        "threshold_eligibility_audit_path": str(audit_path),
        "threshold_eligibility_audit_sha256": _sha256(audit_path),
        "caps_hours": caps,
        "duration_statistics": duration_stats,
        "duration_buckets": duration_buckets,
        "cap_count_cross_checks": cap_checks,
        "counts_reconcile_to_dataset": bucket_sum == row_count,
        "warnings": list(dict.fromkeys(str(item) for item in warnings)),
        "errors": errors,
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "Durations are recomputed from timestamp to label_available_at on the exact uncapped "
            "dataset whose SHA-256 is recorded in the baseline manifest. Buckets are (lower, upper] "
            "to match the inclusive elapsed-hours eligibility rule. Existing threshold-audit counts "
            "are independently cross-checked; no training or threshold selection occurs."
        ),
        "report_path": str(destination),
    }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile label maturity durations around completed eligibility caps without retraining."
    )
    parser.add_argument("--baseline-manifest", required=True, help="Uncapped pipeline manifest")
    parser.add_argument("--threshold-audit-report", required=True, help="Completed threshold eligibility audit report")
    parser.add_argument("--output-path", help="Optional immutable report destination")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_duration_cap_boundaries(
            args.baseline_manifest,
            args.threshold_audit_report,
            output_path=args.output_path,
        )
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL",
            "scope": _PROFILE_SCOPE,
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
        "dataset_rows": report["duration_statistics"]["observation_count"],
        "duration_statistics": report["duration_statistics"],
        "duration_buckets": report["duration_buckets"],
        "cap_count_cross_checks": report["cap_count_cross_checks"],
        "counts_reconcile_to_dataset": report["counts_reconcile_to_dataset"],
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

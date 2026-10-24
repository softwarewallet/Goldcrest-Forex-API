"""Audit exact row-mask equivalence between elapsed-time caps and observed OHLC gaps."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.research_label_duration_gap_attribution import (
    _ATTRIBUTION_SCOPE, _PASS, _PIPELINE_SCOPE, _TIMEFRAME_MINUTES,
    _load_dataset, _load_raw_timestamps, _read_json, _resolve_artifact, _sha256,
)
from app.research_label_eligibility_duration_profile import _PROFILE_SCOPE
from app.research_label_eligibility_threshold_report_audit import _AUDIT_SCOPE
from app.research_label_gap_event_profile import _EVENT_SCOPE

_SCOPE = "GOLDCREST_LABEL_GAP_POLICY_EQUIVALENCE_AUDIT"
_EPS = 1e-9


def _check_status(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("status") not in _PASS:
        raise ValueError(f"{label}_status_not_pass")


def _check_flags(payload: Mapping[str, Any], label: str) -> None:
    for flag in ("inferential", "automatic_model_selection", "actionable", "execution_connected"):
        if payload.get(flag) is not False:
            raise ValueError(f"{label}_research_only_flag_invalid:{flag}")


def _float_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}_invalid")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label}_invalid")
    return result


def _date_key(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _mask_hash(rows: set[int]) -> str:
    body = "\n".join(str(value) for value in sorted(rows)) + "\n"
    return hashlib.sha256(body.encode("ascii")).hexdigest()


def _bucket(hours: float, caps: Sequence[float]) -> str:
    lower = None
    for cap in caps:
        if hours <= cap + _EPS:
            return f"<= {cap:g}" if lower is None else f"({lower:g}, {cap:g}]"
        lower = cap
    return f"> {caps[-1]:g}"


def _free_output_path(manifest_path: Path, run_id: str) -> Path:
    base = manifest_path.parent / f"{run_id}_gap_policy_equivalence_audit.json"
    if not base.exists():
        return base
    for n in range(2, 10002):
        candidate = manifest_path.parent / f"{run_id}-rerun-{n}_gap_policy_equivalence_audit.json"
        if not candidate.exists():
            return candidate
    raise ValueError("unable_to_find_free_gap_policy_audit_path")


def _threshold_profiles(audit: Mapping[str, Any]) -> dict[float, Mapping[str, Any]]:
    if audit.get("scope") != _AUDIT_SCOPE:
        raise ValueError("input_is_not_a_threshold_eligibility_profile_audit")
    _check_status(audit, "threshold_audit")
    _check_flags(audit, "threshold_audit")
    caps, profiles = audit.get("caps_audited_hours"), audit.get("cap_profiles")
    if not isinstance(caps, list) or not isinstance(profiles, list):
        raise ValueError("threshold_audit_profiles_missing")
    result: dict[float, Mapping[str, Any]] = {}
    for item in profiles:
        if not isinstance(item, dict):
            raise ValueError("threshold_audit_profile_invalid")
        cap = _float_positive(item.get("cap_hours"), "threshold_audit_cap")
        if cap in result:
            raise ValueError("threshold_audit_duplicate_cap")
        result[cap] = item
    normalized = {_float_positive(cap, "threshold_audit_cap") for cap in caps}
    if normalized != set(result):
        raise ValueError("threshold_audit_cap_list_mismatch")
    return result


def audit_gap_policy_equivalence(
    baseline_manifest_path: str | Path,
    duration_profile_report_path: str | Path,
    duration_gap_attribution_report_path: str | Path,
    gap_event_profile_report_path: str | Path,
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

    paths = {
        "duration_profile": Path(duration_profile_report_path).expanduser().resolve(),
        "duration_gap_attribution": Path(duration_gap_attribution_report_path).expanduser().resolve(),
        "gap_event_profile": Path(gap_event_profile_report_path).expanduser().resolve(),
        "threshold_audit": Path(threshold_audit_report_path).expanduser().resolve(),
    }
    inputs = {key: _read_json(path) for key, path in paths.items()}
    for key, expected_scope in (
        ("duration_profile", _PROFILE_SCOPE),
        ("duration_gap_attribution", _ATTRIBUTION_SCOPE),
        ("gap_event_profile", _EVENT_SCOPE),
    ):
        if inputs[key].get("scope") != expected_scope:
            raise ValueError(f"{key}_scope_mismatch")
        _check_status(inputs[key], key)
        _check_flags(inputs[key], key)

    raw_path = _resolve_artifact(manifest_path, baseline.get("raw_ohlc_path"), field="raw_ohlc_path")
    dataset_path = _resolve_artifact(manifest_path, baseline.get("dataset_path"), field="dataset_path")
    raw_hash, dataset_hash = baseline.get("raw_ohlc_sha256"), baseline.get("dataset_sha256")
    if not isinstance(raw_hash, str) or _sha256(raw_path) != raw_hash:
        raise ValueError("baseline_raw_ohlc_file_hash_mismatch")
    if not isinstance(dataset_hash, str) or _sha256(dataset_path) != dataset_hash:
        raise ValueError("baseline_dataset_file_hash_mismatch")

    for key in ("duration_profile", "duration_gap_attribution", "gap_event_profile"):
        payload = inputs[key]
        if payload.get("baseline_raw_ohlc_sha256") != raw_hash:
            raise ValueError(f"{key}_raw_ohlc_hash_mismatch")
        if payload.get("baseline_dataset_sha256") != dataset_hash:
            raise ValueError(f"{key}_dataset_hash_mismatch")
        if payload.get("baseline_manifest_path") and Path(str(payload["baseline_manifest_path"])).resolve() != manifest_path:
            raise ValueError(f"{key}_baseline_manifest_path_mismatch")
    if inputs["threshold_audit"].get("raw_ohlc_sha256") != raw_hash:
        raise ValueError("threshold_audit_raw_ohlc_hash_mismatch")
    if inputs["gap_event_profile"].get("duration_gap_attribution_report_sha256") != _sha256(paths["duration_gap_attribution"]):
        raise ValueError("gap_event_profile_attribution_report_hash_mismatch")

    timeframe = str(baseline.get("timeframe", "")).upper()
    if timeframe not in _TIMEFRAME_MINUTES:
        raise ValueError(f"baseline_timeframe_unsupported:{timeframe or 'missing'}")
    frame_seconds = _TIMEFRAME_MINUTES[timeframe] * 60
    horizon = baseline.get("label_horizon_bars")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("baseline_label_horizon_bars_invalid")
    raw_caps = inputs["duration_profile"].get("caps_hours")
    if not isinstance(raw_caps, list) or not raw_caps:
        raise ValueError("duration_profile_caps_missing")
    caps = sorted(_float_positive(value, "duration_profile_cap") for value in raw_caps)
    if len(caps) != len(set(caps)) or not {1.0, 1.5, 8.0}.issubset(set(caps)):
        raise ValueError("duration_profile_caps_invalid_or_missing_required_caps")

    raw_timestamps, errors = _load_raw_timestamps(raw_path)
    data_rows = _load_dataset(dataset_path, timeframe)
    warnings: list[str] = []
    for payload in inputs.values():
        warnings.extend(str(item) for item in payload.get("warnings", []) if isinstance(item, str))
    raw_index: dict[datetime, int] = {}
    for index, timestamp in enumerate(raw_timestamps):
        if timestamp in raw_index:
            errors.append("raw_ohlc_duplicate_timestamp")
        else:
            raw_index[timestamp] = index

    row_details: list[dict[str, Any]] = []
    missing_start = missing_target = horizon_mismatches = 0
    for row_number, (start, target) in enumerate(data_rows, start=2):
        start_index, target_index = raw_index.get(start), raw_index.get(target)
        if start_index is None:
            missing_start += 1
            errors.append(f"dataset_start_timestamp_missing_line_{row_number}")
            continue
        if target_index is None:
            missing_target += 1
            errors.append(f"dataset_label_maturity_timestamp_missing_line_{row_number}")
            continue
        if target_index - start_index != horizon:
            horizon_mismatches += 1
            errors.append(f"dataset_label_horizon_index_mismatch_line_{row_number}:{target_index-start_index}!={horizon}")
            continue
        elapsed = (target - start).total_seconds() / 3600.0
        if not math.isfinite(elapsed) or elapsed < 0:
            errors.append(f"dataset_label_elapsed_hours_invalid_line_{row_number}")
            continue
        row_details.append({
            "row": row_number, "start_index": start_index, "target_index": target_index,
            "elapsed_hours": elapsed, "bucket": _bucket(elapsed, caps),
        })
    all_rows = {row["row"] for row in row_details}
    if len(all_rows) != len(data_rows):
        errors.append(f"processed_row_count_mismatch:{len(all_rows)}!={len(data_rows)}")

    # Rebuild the gap-event-to-label map independently from raw timestamps.
    computed_events: dict[tuple[str, str], dict[str, Any]] = {}
    rows_by_start = {row["start_index"]: row for row in row_details}
    any_gap_rows: set[int] = set()
    raw_gaps_over_8h_rows: set[int] = set()
    gap_class_rows = {"excess_up_to_1h": set(), "excess_over_1_to_8h": set(), "excess_over_8h": set()}
    for right_idx in range(1, len(raw_timestamps)):
        left, right = raw_timestamps[right_idx - 1], raw_timestamps[right_idx]
        interval_seconds = (right - left).total_seconds()
        if interval_seconds <= frame_seconds + 1e-6:
            continue
        excess_hours = (interval_seconds - frame_seconds) / 3600.0
        klass = "excess_up_to_1h" if excess_hours <= 1 + _EPS else "excess_over_1_to_8h" if excess_hours <= 8 + _EPS else "excess_over_8h"
        affected = []
        for start_idx in range(max(0, right_idx - horizon), right_idx):
            row = rows_by_start.get(start_idx)
            if row is not None and row["target_index"] >= right_idx:
                affected.append(row["row"])
        affected_set = set(affected)
        any_gap_rows.update(affected_set)
        gap_class_rows[klass].update(affected_set)
        if interval_seconds / 3600.0 > 8 + _EPS:
            raw_gaps_over_8h_rows.update(affected_set)
        computed_events[(_date_key(left), _date_key(right))] = {
            "interval_hours": interval_seconds / 3600.0,
            "gap_excess_class": klass,
            "affected_label_rows": sorted(affected_set),
        }

    event_profile = inputs["gap_event_profile"]
    source_events = event_profile.get("gap_events")
    if not isinstance(source_events, list):
        raise ValueError("gap_event_profile_gap_events_missing")
    source_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for event in source_events:
        if not isinstance(event, dict):
            errors.append("gap_event_profile_event_invalid")
            continue
        key = (str(event.get("from_utc", "")), str(event.get("to_utc", "")))
        if key in source_by_key:
            errors.append(f"gap_event_duplicate:{key}")
        source_by_key[key] = event
    if set(source_by_key) != set(computed_events):
        errors.append(f"gap_event_timeline_mismatch:{len(computed_events)}!={len(source_by_key)}")

    event_failures = 0
    for key, current in computed_events.items():
        source = source_by_key.get(key)
        if source is None:
            continue
        source_rows = source.get("affected_label_rows")
        parsed_rows = sorted(v for v in source_rows if isinstance(v, int) and not isinstance(v, bool)) if isinstance(source_rows, list) else []
        checks = (
            parsed_rows == current["affected_label_rows"],
            source.get("gap_excess_class") == current["gap_excess_class"],
            isinstance(source.get("interval_hours"), (int, float))
            and not isinstance(source.get("interval_hours"), bool)
            and math.isclose(float(source["interval_hours"]), current["interval_hours"], rel_tol=0, abs_tol=1e-7),
        )
        if not all(checks):
            event_failures += 1
            errors.append(f"gap_event_exposure_mismatch:{key[0]}")

    bucket_counts: dict[str, int] = {}
    for row in row_details:
        bucket_counts[row["bucket"]] = bucket_counts.get(row["bucket"], 0) + 1
    profile_buckets = inputs["duration_profile"].get("duration_buckets")
    if not isinstance(profile_buckets, list):
        raise ValueError("duration_profile_buckets_missing")
    for item in profile_buckets:
        if isinstance(item, dict) and isinstance(item.get("interval_hours"), str):
            name = item["interval_hours"]
            if item.get("row_count") != bucket_counts.get(name, 0):
                errors.append(f"duration_profile_bucket_count_mismatch:{name}")
    for name in bucket_counts:
        if not any(isinstance(item, dict) and item.get("interval_hours") == name for item in profile_buckets):
            errors.append(f"duration_profile_bucket_missing:{name}")

    caps_eligible = {
        cap: {row["row"] for row in row_details if row["elapsed_hours"] <= cap + _EPS}
        for cap in caps
    }
    policies = {
        "baseline_all_labels": set(all_rows),
        "elapsed_cap_1h": caps_eligible[1.0],
        "exclude_any_non_nominal_gap": all_rows - any_gap_rows,
        "elapsed_cap_1_5h": caps_eligible[1.5],
        "elapsed_cap_8h": caps_eligible[8.0],
        "exclude_labels_crossing_raw_gap_over_8h": all_rows - raw_gaps_over_8h_rows,
    }

    summaries = []
    for name, eligible in policies.items():
        excluded = all_rows - eligible
        summaries.append({
            "policy": name,
            "dataset_rows": len(all_rows),
            "eligible_rows": len(eligible),
            "ineligible_rows": len(excluded),
            "eligible_fraction": round(len(eligible) / len(all_rows), 10) if all_rows else None,
            "eligible_row_numbers_sha256": _mask_hash(eligible),
            "ineligible_row_numbers_sha256": _mask_hash(excluded),
        })
    pairs = [
        ("elapsed_cap_1h", "exclude_any_non_nominal_gap"),
        ("elapsed_cap_1_5h", "elapsed_cap_8h"),
        ("elapsed_cap_1_5h", "exclude_labels_crossing_raw_gap_over_8h"),
        ("elapsed_cap_8h", "exclude_labels_crossing_raw_gap_over_8h"),
    ]
    equivalences = []
    for left_name, right_name in pairs:
        left, right = policies[left_name], policies[right_name]
        same = left == right
        equivalences.append({
            "left_policy": left_name, "right_policy": right_name,
            "same_eligible_mask": same,
            "eligible_row_symmetric_difference_count": len(left.symmetric_difference(right)),
            "eligible_rows_left_only": len(left - right),
            "eligible_rows_right_only": len(right - left),
        })
        if not same:
            warnings.append(f"eligibility_masks_differ:{left_name}:{right_name}")

    threshold = _threshold_profiles(inputs["threshold_audit"])
    if inputs["threshold_audit"].get("raw_ohlc_sha256") != raw_hash:
        errors.append("threshold_audit_raw_hash_mismatch")
    threshold_checks = []
    for cap in (1.0, 1.5, 8.0):
        src = threshold.get(cap)
        eligible_count = len(caps_eligible[cap])
        excluded_count = len(all_rows) - eligible_count
        match = isinstance(src, Mapping) and (
            src.get("dataset_rows") == len(all_rows)
            and src.get("eligible_rows") == eligible_count
            and src.get("ineligible_rows") == excluded_count
        )
        threshold_checks.append({
            "cap_hours": cap, "computed_eligible_rows": eligible_count,
            "computed_ineligible_rows": excluded_count,
            "reported_eligible_rows": src.get("eligible_rows") if src else None,
            "reported_ineligible_rows": src.get("ineligible_rows") if src else None,
            "counts_match": match,
        })
        if not match:
            errors.append(f"threshold_audit_count_mismatch:{cap:g}h")

    if event_profile.get("raw_non_nominal_gap_event_count") != len(computed_events):
        errors.append("gap_event_profile_event_count_mismatch")
    if event_profile.get("unique_label_rows_crossing_any_gap") != len(any_gap_rows):
        errors.append("gap_event_profile_exposed_label_count_mismatch")
    if event_profile.get("duration_bucket_counts_recomputed") != bucket_counts:
        errors.append("gap_event_profile_duration_bucket_counts_mismatch")
    if attribution := inputs["duration_gap_attribution"]:
        if attribution.get("rows_with_any_interbar_gap") != len(any_gap_rows):
            errors.append("duration_gap_attribution_exposed_label_count_mismatch")
        if attribution.get("rows_with_duration_gap_excess_reconciled") != len(all_rows):
            errors.append("duration_gap_attribution_reconciled_count_mismatch")

    warnings.extend(
        str(item) for item in inputs["threshold_audit"].get("warnings", [])
        if isinstance(item, str)
    )
    warnings.append(
        "Policy equivalence is dataset-specific. An observed timestamp gap does not prove missing data "
        "or establish an expected trading-session calendar."
    )
    warnings = list(dict.fromkeys(warnings))
    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")

    run_id = str(baseline.get("run_id") or manifest_path.stem.removesuffix("_pipeline_manifest"))
    destination = Path(output_path).expanduser().resolve() if output_path else _free_output_path(manifest_path, run_id)
    protected = {manifest_path, raw_path, dataset_path, *paths.values()}
    if destination in protected:
        raise ValueError("output_path_must_not_overwrite_input_artifact")
    if destination.exists():
        raise ValueError("output_path_already_exists_refusing_to_overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1, "scope": _SCOPE, "status": status,
        "baseline_manifest_path": str(manifest_path), "baseline_dataset_path": str(dataset_path),
        "baseline_dataset_sha256": dataset_hash, "baseline_raw_ohlc_path": str(raw_path),
        "baseline_raw_ohlc_sha256": raw_hash,
        "input_reports": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in paths.items()},
        "symbol": baseline.get("symbol"), "timeframe": timeframe, "label_horizon_bars": horizon,
        "nominal_horizon_hours": horizon * frame_seconds / 3600.0,
        "dataset_rows": len(data_rows), "processed_rows": len(row_details),
        "missing_start_timestamps": missing_start, "missing_label_maturity_timestamps": missing_target,
        "row_index_horizon_mismatches": horizon_mismatches,
        "gap_event_recomputation": {
            "computed_event_count": len(computed_events), "reported_event_count": len(source_events),
            "event_cross_check_failures": event_failures, "unique_gap_exposed_labels": len(any_gap_rows),
            "raw_gap_over_8h_exposed_labels": len(raw_gaps_over_8h_rows),
            "gap_class_event_counts": {
                name: sum(event["gap_excess_class"] == name for event in computed_events.values())
                for name in gap_class_rows
            },
            "labels_by_gap_excess_class": {name: len(rows) for name, rows in gap_class_rows.items()},
        },
        "duration_bucket_counts_recomputed": bucket_counts,
        "candidate_policies": summaries, "eligibility_mask_equivalence_checks": equivalences,
        "threshold_audit_count_cross_checks": threshold_checks,
        "errors": list(dict.fromkeys(errors)), "warnings": warnings,
        "inferential": False, "automatic_model_selection": False, "actionable": False,
        "execution_connected": False, "data_rewritten": False, "models_trained": False,
        "interpretation": (
            "Exact eligible-mask equality is compared by dataset row identity, not just counts. The audit "
            "does not train models, change targets, select a policy, repair OHLC data, or assert market-session rules."
        ),
        "report_path": str(destination),
    }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit row-level equivalence between elapsed-time caps and observed-gap policies.")
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--duration-profile-report", required=True)
    parser.add_argument("--duration-gap-attribution-report", required=True)
    parser.add_argument("--gap-event-profile-report", required=True)
    parser.add_argument("--threshold-audit-report", required=True)
    parser.add_argument("--output-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_gap_policy_equivalence(
            args.baseline_manifest, args.duration_profile_report,
            args.duration_gap_attribution_report, args.gap_event_profile_report,
            args.threshold_audit_report, output_path=args.output_path,
        )
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "scope": _SCOPE,
            "errors": [f"{type(exc).__name__}:{exc}"], "inferential": False,
            "automatic_model_selection": False, "actionable": False, "execution_connected": False},
            indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": report["status"], "report_path": report["report_path"],
        "dataset_rows": report["dataset_rows"], "processed_rows": report["processed_rows"],
        "gap_event_recomputation": report["gap_event_recomputation"],
        "duration_bucket_counts_recomputed": report["duration_bucket_counts_recomputed"],
        "candidate_policies": report["candidate_policies"],
        "eligibility_mask_equivalence_checks": report["eligibility_mask_equivalence_checks"],
        "threshold_audit_count_cross_checks": report["threshold_audit_count_cross_checks"],
        "errors": report["errors"], "warnings": report["warnings"],
        "inferential": False, "automatic_model_selection": False,
        "actionable": False, "execution_connected": False,
    }, indent=2, sort_keys=True))
    return 0 if report["status"] in _PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit elapsed-time label eligibility profiles from a completed threshold sweep.

This report-only tool reads existing threshold-sweep, seed-sweep, and pipeline manifests.
It does not retrain models, alter data, select a threshold, or connect to execution.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.research_strict_json import strict_json_loads

def _load_json_object(raw: bytes, *, source: str) -> dict[str, Any]:
    payload = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_must_be_object:{source}")
    return payload


_PASS = {"PASS", "PASS_WITH_WARNINGS"}
_SCOPE = "GOLDCREST_LABEL_ELIGIBILITY_THRESHOLD_SWEEP"
_AUDIT_SCOPE = "GOLDCREST_LABEL_ELIGIBILITY_THRESHOLD_PROFILE_AUDIT"


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = _load_json_object(Path(path).expanduser().resolve().read_bytes(), source=str(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"json_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"json_not_object:{path}")
    return value


def _numeric_cap(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label}_cap_invalid")
    return float(value)


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}_must_be_a_non_negative_integer")
    return value


def _validate_report_diagnostics(payload: Mapping[str, Any], label: str) -> list[str]:
    warnings: list[str] = []
    for field in ("errors", "warnings"):
        if field not in payload:
            continue
        value = payload[field]
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError(f"{label}_{field}_schema_invalid")
        if field == "errors" and value and isinstance(payload.get("status"), str) and payload.get("status") in _PASS:
            raise ValueError(f"{label}_reports_errors_despite_pass_status")
        if field == "warnings":
            warnings.extend(value)
    return warnings


def _same_cap(first: object, second: object) -> bool:
    return (
        isinstance(first, (int, float))
        and not isinstance(first, bool)
        and isinstance(second, (int, float))
        and not isinstance(second, bool)
        and math.isclose(float(first), float(second), rel_tol=0, abs_tol=1e-9)
    )


def _free_report_path(source_path: Path) -> Path:
    stem = source_path.name
    suffix = "_threshold_sweep_report.json"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    base = source_path.parent / f"{stem}_threshold_eligibility_audit_report.json"
    if not base.exists():
        return base
    for revision in range(2, 10002):
        candidate = source_path.parent / f"{stem}-rerun-{revision}_threshold_eligibility_audit_report.json"
        if not candidate.exists():
            return candidate
    raise ValueError("unable_to_find_free_threshold_eligibility_audit_path")


def _fold_eligibility_profile(cap_record: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    """Normalize both legacy signatures (with row counts) and window-only signatures.

    Seed-sweep fold signatures intentionally identify test windows only. Eligible-row counts
    are cap-dependent diagnostics, not fold identity, and newer threshold reports may omit
    them from the signature. Do not fabricate counts when they are unavailable.
    """
    separate_counts = cap_record.get("fold_eligibility_counts")
    signature = cap_record.get("fold_signature")
    if separate_counts is not None:
        if not isinstance(separate_counts, list) or not separate_counts:
            raise ValueError(f"{label}_fold_eligibility_counts_invalid")
        if any(not isinstance(item, dict) for item in separate_counts):
            raise ValueError(f"{label}_fold_eligibility_counts_entry_invalid")
        signature = [
            [item.get("fold_index"), item.get("test_start_utc"), item.get("test_end_utc"),
             item.get("eligible_test_rows")]
            for item in separate_counts
        ]
    if not isinstance(signature, list) or not signature:
        raise ValueError(f"{label}_fold_signature_missing")
    profile: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for row in signature:
        if not isinstance(row, list) or len(row) not in (3, 4):
            raise ValueError(f"{label}_fold_signature_entry_invalid")
        index = _count(row[0], f"{label}_fold_index")
        if index in seen_indices:
            raise ValueError(f"{label}_fold_index_duplicate")
        seen_indices.add(index)
        if not isinstance(row[1], str) or not row[1].strip() or not isinstance(row[2], str) or not row[2].strip():
            raise ValueError(f"{label}_fold_window_boundary_missing")
        eligible_rows = (
            _count(row[3], f"{label}_eligible_test_rows") if len(row) == 4 else None
        )
        profile.append({
            "fold_index": index,
            "test_start_utc": row[1],
            "test_end_utc": row[2],
            "eligible_test_rows": eligible_rows,
        })
    return sorted(profile, key=lambda item: item["fold_index"])


def _dataset_eligibility_for_cap(
    cap_record: Mapping[str, Any],
    *,
    cap_hours: float,
    expected_seeds: Sequence[int],
    expected_raw_hash: str,
) -> dict[str, Any]:
    label = f"cap_{format(cap_hours, 'g').replace('.', 'p')}h"
    child_path = cap_record.get("seed_sweep_report_path")
    if not isinstance(child_path, str) or not child_path:
        raise ValueError(f"{label}_seed_sweep_report_path_missing")
    child_path_resolved = Path(child_path).expanduser().resolve()
    child = _read_json(child_path_resolved)
    _validate_report_diagnostics(child, f"{label}_seed_sweep")
    if not isinstance(child.get("status"), str) or child.get("status") not in _PASS:
        raise ValueError(f"{label}_seed_sweep_status_not_pass")
    if not _same_cap(child.get("max_label_elapsed_hours"), cap_hours):
        raise ValueError(f"{label}_seed_sweep_cap_mismatch")
    if child.get("requested_seeds") != list(expected_seeds):
        raise ValueError(f"{label}_seed_list_mismatch")
    if child.get("raw_ohlc_sha256") != expected_raw_hash:
        raise ValueError(f"{label}_seed_sweep_raw_hash_mismatch")

    seed_results = child.get("seed_results")
    if not isinstance(seed_results, list) or len(seed_results) != len(expected_seeds):
        raise ValueError(f"{label}_seed_results_incomplete")
    by_seed: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for seed_result in seed_results:
        if not isinstance(seed_result, dict):
            raise ValueError(f"{label}_seed_result_invalid")
        seed = seed_result.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in seen_seeds:
            raise ValueError(f"{label}_seed_identity_invalid")
        seen_seeds.add(seed)
        _validate_report_diagnostics(seed_result, f"{label}_seed_{seed}")
        if not isinstance(seed_result.get("status"), str) or seed_result.get("status") not in _PASS:
            raise ValueError(f"{label}_seed_{seed}_status_not_pass")
        manifest_path = seed_result.get("capped_replay_manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path:
            raise ValueError(f"{label}_seed_{seed}_capped_manifest_path_missing")
        manifest = _read_json(manifest_path)
        if not _same_cap(manifest.get("max_label_elapsed_hours"), cap_hours):
            raise ValueError(f"{label}_seed_{seed}_pipeline_cap_mismatch")
        if manifest.get("raw_ohlc_sha256") != expected_raw_hash:
            raise ValueError(f"{label}_seed_{seed}_raw_hash_mismatch")
        metadata = manifest.get("label_eligibility_metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{label}_seed_{seed}_label_eligibility_metadata_missing")
        if metadata.get("mode") != "elapsed_time_cap":
            raise ValueError(f"{label}_seed_{seed}_eligibility_mode_mismatch")
        if not _same_cap(metadata.get("max_label_elapsed_hours"), cap_hours):
            raise ValueError(f"{label}_seed_{seed}_eligibility_cap_mismatch")
        eligible = _count(metadata.get("eligible_rows"), f"{label}_seed_{seed}_eligible_rows")
        ineligible = _count(metadata.get("ineligible_rows"), f"{label}_seed_{seed}_ineligible_rows")
        total = eligible + ineligible
        by_seed.append({
            "seed": seed,
            "dataset_rows": total,
            "eligible_rows": eligible,
            "ineligible_rows": ineligible,
            "pipeline_manifest_path": str(Path(manifest_path).expanduser().resolve()),
        })

    if seen_seeds != set(expected_seeds):
        raise ValueError(f"{label}_seed_results_do_not_match_requested_seed_set")
    profiles = {(item["dataset_rows"], item["eligible_rows"], item["ineligible_rows"]) for item in by_seed}
    if len(profiles) != 1:
        raise ValueError(f"{label}_dataset_eligibility_counts_differ_between_seeds")
    common = next(iter(profiles))
    if common[1] + common[2] != common[0]:
        raise ValueError(f"{label}_dataset_eligibility_count_arithmetic_invalid")

    fold_profile = _fold_eligibility_profile(cap_record, label)
    return {
        "cap_hours": cap_hours,
        "status": cap_record.get("status"),
        "seed_sweep_report_path": str(child_path_resolved),
        "seed_count": len(by_seed),
        "dataset_rows": common[0],
        "eligible_rows": common[1],
        "ineligible_rows": common[2],
        "eligible_fraction": round(common[1] / common[0], 10) if common[0] else None,
        "eligible_test_rows_across_fold_windows": (
            sum(row["eligible_test_rows"] for row in fold_profile)
            if all(row["eligible_test_rows"] is not None for row in fold_profile)
            else None
        ),
        "eligible_test_rows_by_fold": fold_profile,
        "seed_profiles": sorted(by_seed, key=lambda item: item["seed"]),
        "metric_summary": cap_record.get("metrics", []),
        "fold_signature": cap_record.get("fold_signature"),
        "fold_window_signature": cap_record.get("fold_window_signature"),
    }


def _compare_profiles(low: Mapping[str, Any], high: Mapping[str, Any]) -> dict[str, Any]:
    low_cap = float(low["cap_hours"])
    high_cap = float(high["cap_hours"])
    if high_cap <= low_cap:
        raise ValueError("profile_comparison_caps_must_be_increasing")
    same_dataset_rows = low["dataset_rows"] == high["dataset_rows"]
    eligible_delta = high["eligible_rows"] - low["eligible_rows"]
    ineligible_delta = high["ineligible_rows"] - low["ineligible_rows"]
    low_folds = low["eligible_test_rows_by_fold"]
    high_folds = high["eligible_test_rows_by_fold"]
    same_fold_shape = [
        (row["fold_index"], row["test_start_utc"], row["test_end_utc"])
        for row in low_folds
    ] == [
        (row["fold_index"], row["test_start_utc"], row["test_end_utc"])
        for row in high_folds
    ]
    fold_deltas: list[dict[str, Any]] = []
    fold_counts_available = same_fold_shape and all(
        row["eligible_test_rows"] is not None for row in low_folds + high_folds
    )
    if fold_counts_available:
        for low_fold, high_fold in zip(low_folds, high_folds):
            fold_deltas.append({
                "fold_index": low_fold["fold_index"],
                "eligible_test_rows_lower_cap": low_fold["eligible_test_rows"],
                "eligible_test_rows_higher_cap": high_fold["eligible_test_rows"],
                "eligible_test_rows_delta": (
                    high_fold["eligible_test_rows"] - low_fold["eligible_test_rows"]
                ),
            })
    same_fold_counts = (
        all(row["eligible_test_rows_delta"] == 0 for row in fold_deltas)
        if fold_counts_available else None
    )
    same_dataset_eligibility_profile = (
        same_dataset_rows
        and eligible_delta == 0
        and ineligible_delta == 0
        and same_fold_counts is not False
    )
    return {
        "lower_cap_hours": low_cap,
        "higher_cap_hours": high_cap,
        "dataset_rows_same": same_dataset_rows,
        "eligible_rows_lower_cap": low["eligible_rows"],
        "eligible_rows_higher_cap": high["eligible_rows"],
        "eligible_rows_delta": eligible_delta,
        "ineligible_rows_lower_cap": low["ineligible_rows"],
        "ineligible_rows_higher_cap": high["ineligible_rows"],
        "ineligible_rows_delta": ineligible_delta,
        "eligible_test_rows_fold_deltas": fold_deltas,
        "same_fold_window_shape": same_fold_shape,
        "same_fold_eligible_test_counts": same_fold_counts,
        "same_dataset_eligibility_profile": same_dataset_eligibility_profile,
        "metric_summary_identical": low.get("metric_summary") == high.get("metric_summary"),
        "automatic_model_selection": False,
        "actionable": False,
    }


def audit_threshold_report(
    threshold_report_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    source_path = Path(threshold_report_path).expanduser().resolve()
    source = _read_json(source_path)
    if source.get("scope") != _SCOPE:
        raise ValueError("input_is_not_a_label_eligibility_threshold_sweep_report")
    errors: list[str] = []
    warnings: list[str] = []
    try:
        warnings.extend(_validate_report_diagnostics(source, "source_threshold_sweep"))
    except ValueError as exc:
        errors.append(str(exc))
    if not isinstance(source.get("status"), str) or source.get("status") not in _PASS:
        errors.append("source_threshold_sweep_status_not_pass")
    for flag in ("actionable", "execution_connected", "automatic_model_selection", "inferential"):
        if source.get(flag) is not False:
            errors.append(f"source_research_only_flag_invalid:{flag}")
    raw_hash = source.get("raw_ohlc_sha256")
    if not isinstance(raw_hash, str) or not raw_hash:
        raise ValueError("source_raw_ohlc_hash_missing")
    seeds = source.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2 or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ) or len(set(seeds)) != len(seeds):
        raise ValueError("source_seed_list_invalid")
    cap_records = source.get("cap_results")
    if not isinstance(cap_records, list) or not cap_records:
        raise ValueError("source_cap_results_missing")

    profiles: list[dict[str, Any]] = []
    seen_caps: set[float] = set()
    failed_caps: list[float] = []
    for index, cap_record in enumerate(cap_records):
        if not isinstance(cap_record, dict):
            errors.append(f"cap_record_{index}_invalid")
            continue
        try:
            warnings.extend(_validate_report_diagnostics(cap_record, f"cap_record_{index}"))
            cap = _numeric_cap(cap_record.get("max_label_elapsed_hours"), f"cap_record_{index}")
            if cap in seen_caps:
                raise ValueError(f"duplicate_cap_record:{cap:g}")
            seen_caps.add(cap)
            if not isinstance(cap_record.get("status"), str) or cap_record.get("status") not in _PASS:
                failed_caps.append(cap)
                errors.append(f"cap_{format(cap, 'g').replace('.', 'p')}h_source_record_not_pass")
                continue
            profiles.append(_dataset_eligibility_for_cap(
                cap_record,
                cap_hours=cap,
                expected_seeds=seeds,
                expected_raw_hash=raw_hash,
            ))
        except Exception as exc:
            errors.append(f"cap_record_{index}:{type(exc).__name__}:{exc}")

    profiles.sort(key=lambda item: item["cap_hours"])
    comparisons: list[dict[str, Any]] = []
    for low, high in zip(profiles, profiles[1:]):
        pair = _compare_profiles(low, high)
        comparisons.append(pair)
        if not pair["dataset_rows_same"]:
            errors.append(
                f"dataset_row_count_changed_between_caps:{low['cap_hours']:g}h_and_{high['cap_hours']:g}h"
            )
        if pair["eligible_rows_delta"] < 0 or pair["ineligible_rows_delta"] > 0:
            errors.append(
                f"eligibility_counts_not_monotonic_between_caps:{low['cap_hours']:g}h_and_{high['cap_hours']:g}h"
            )
        fold_deltas = pair["eligible_test_rows_fold_deltas"]
        if pair["same_fold_window_shape"] and fold_deltas and any(
            item["eligible_test_rows_delta"] < 0 for item in fold_deltas
        ):
            errors.append(
                f"fold_eligibility_counts_not_monotonic_between_caps:{low['cap_hours']:g}h_and_{high['cap_hours']:g}h"
            )
        if pair["same_fold_window_shape"] and pair["same_fold_eligible_test_counts"] is None:
            warnings.append(
                f"fold_level_eligibility_counts_unavailable_for_{low['cap_hours']:g}h_and_{high['cap_hours']:g}h"
            )
        if pair["same_dataset_eligibility_profile"]:
            warnings.append(
                f"caps_{low['cap_hours']:g}h_and_{high['cap_hours']:g}h_have_identical_eligibility_profiles"
            )
        elif pair["metric_summary_identical"]:
            warnings.append(
                f"caps_{low['cap_hours']:g}h_and_{high['cap_hours']:g}h_have_identical_aggregate_metric_summaries_despite_different_eligibility_profiles"
            )

    expected_caps = source.get("requested_caps_hours")
    if isinstance(expected_caps, list):
        expected_numeric = []
        for value in expected_caps:
            try:
                expected_numeric.append(_numeric_cap(value, "requested"))
            except ValueError:
                errors.append("source_requested_caps_invalid")
                break
        if expected_numeric and set(expected_numeric) != seen_caps:
            errors.append("cap_records_do_not_match_requested_caps")

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    destination = Path(output_path).expanduser().resolve() if output_path else _free_report_path(source_path)
    if destination == source_path:
        raise ValueError("audit_output_path_must_not_overwrite_input_report")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": _AUDIT_SCOPE,
        "status": status,
        "threshold_sweep_report_path": str(source_path),
        "threshold_sweep_report_sha256": __import__("hashlib").sha256(source_path.read_bytes()).hexdigest(),
        "raw_ohlc_sha256": raw_hash,
        "caps_audited_hours": [profile["cap_hours"] for profile in profiles],
        "requested_seeds": seeds,
        "cap_profiles": profiles,
        "adjacent_cap_comparisons": comparisons,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(str(item) for item in warnings)),
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
        "interpretation": (
            "This audit reads existing artifact manifests and does not rerun training. Eligibility is "
            "defined by label_elapsed_hours <= cap on the same immutable dataset timestamps. For nested "
            "threshold masks on the same data, equal eligible/ineligible counts imply the higher cap "
            "added no rows. Repeated seeds are not independent market replications."
        ),
        "report_path": str(destination),
    }
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit eligibility profiles from an existing threshold-sweep report without retraining."
    )
    parser.add_argument("--threshold-report", required=True, help="Existing *_threshold_sweep_report.json")
    parser.add_argument("--output-path", help="Optional destination for the new immutable audit report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = audit_threshold_report(args.threshold_report, output_path=args.output_path)
    except Exception as exc:
        error = {
            "status": "FAIL",
            "scope": _AUDIT_SCOPE,
            "errors": [f"{type(exc).__name__}:{exc}"],
            "inferential": False,
            "automatic_model_selection": False,
            "actionable": False,
            "execution_connected": False,
        }
        print(json.dumps(error, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "threshold_sweep_report_path": result["threshold_sweep_report_path"],
        "report_path": result["report_path"],
        "caps_audited_hours": result["caps_audited_hours"],
        "cap_profiles": [
            {
                "cap_hours": profile["cap_hours"],
                "dataset_rows": profile["dataset_rows"],
                "eligible_rows": profile["eligible_rows"],
                "ineligible_rows": profile["ineligible_rows"],
                "eligible_fraction": profile["eligible_fraction"],
                "eligible_test_rows_across_fold_windows": profile["eligible_test_rows_across_fold_windows"],
            }
            for profile in result["cap_profiles"]
        ],
        "adjacent_cap_comparisons": result["adjacent_cap_comparisons"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
    }, indent=2, sort_keys=True))
    return 0 if result["status"] in _PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())

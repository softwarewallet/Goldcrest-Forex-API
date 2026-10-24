"""Review completed threshold-sweep metrics by unique label eligibility mask.

This report-only module consumes the existing gap-policy equivalence audit and
threshold sweep artifacts. It does not train models, select a policy, or connect
to execution. Repeated seeds remain same-history optimizer sensitivity checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_POLICY_SCOPE = "GOLDCREST_LABEL_GAP_POLICY_EQUIVALENCE_AUDIT"
_THRESHOLD_AUDIT_SCOPE = "GOLDCREST_LABEL_ELIGIBILITY_THRESHOLD_PROFILE_AUDIT"
_SWEEP_SCOPE = "GOLDCREST_LABEL_ELIGIBILITY_THRESHOLD_SWEEP"
_REVIEW_SCOPE = "GOLDCREST_LABEL_GAP_POLICY_PERFORMANCE_REVIEW"
_PASS = {"PASS", "PASS_WITH_WARNINGS"}
_REQUIRED_CAPS = (1.0, 1.5, 8.0)
_EPS = 1e-9


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"json_unreadable:{resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"json_not_object:{resolved}")
    return payload


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"artifact_unreadable:{path}") from exc


def _positive_cap(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}_invalid")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label}_invalid")
    return number


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label}_must_be_finite_number")
    return float(value)


def _check_pass(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("status") not in _PASS:
        raise ValueError(f"{label}_status_not_pass")


def _check_research_flags(payload: Mapping[str, Any], label: str) -> None:
    for flag in ("inferential", "automatic_model_selection", "actionable", "execution_connected"):
        if payload.get(flag) is not False:
            raise ValueError(f"{label}_research_only_flag_invalid:{flag}")


def _free_output_path(source_path: Path) -> Path:
    stem = source_path.stem
    if stem.endswith("_gap_policy_equivalence_audit"):
        stem = stem[: -len("_gap_policy_equivalence_audit")]
    base = source_path.parent / f"{stem}_gap_policy_performance_review.json"
    if not base.exists():
        return base
    for revision in range(2, 10002):
        candidate = source_path.parent / f"{stem}-rerun-{revision}_gap_policy_performance_review.json"
        if not candidate.exists():
            return candidate
    raise ValueError("unable_to_find_free_gap_policy_performance_review_path")


def _index_records(records: object, cap_field: str, label: str) -> dict[float, Mapping[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"{label}_records_missing")
    result: dict[float, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{label}_record_invalid")
        cap = _positive_cap(record.get(cap_field), f"{label}_cap")
        if cap in result:
            raise ValueError(f"{label}_duplicate_cap:{cap:g}")
        result[cap] = record
    return result


def _metric_map(metric_records: object, label: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    if not isinstance(metric_records, list) or not metric_records:
        raise ValueError(f"{label}_metrics_missing")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in metric_records:
        if not isinstance(item, dict):
            raise ValueError(f"{label}_metric_record_invalid")
        model = item.get("model_type")
        variant = item.get("feature_variant")
        metrics = item.get("metrics")
        if not isinstance(model, str) or not model or not isinstance(variant, str) or not variant:
            raise ValueError(f"{label}_experiment_identity_missing")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f"{label}_experiment_metrics_missing")
        key = (model, variant)
        if key in result:
            raise ValueError(f"{label}_duplicate_experiment:{model}/{variant}")
        result[key] = metrics
    return result


def _same_json(first: object, second: object) -> bool:
    return json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        second, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _load_seed_reports(
    sweep_record: Mapping[str, Any],
    *,
    cap_hours: float,
    requested_seeds: Sequence[int],
    expected_raw_hash: str,
) -> tuple[dict[int, Mapping[str, Any]], Path]:
    path_value = sweep_record.get("seed_sweep_report_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"cap_{cap_hours:g}h_seed_sweep_report_path_missing")
    path = Path(path_value).expanduser().resolve()
    report = _read_json(path)
    _check_pass(report, f"cap_{cap_hours:g}h_seed_sweep")
    _check_research_flags(report, f"cap_{cap_hours:g}h_seed_sweep")
    if report.get("scope") != "GOLDCREST_LABEL_ELIGIBILITY_MULTI_SEED_SENSITIVITY":
        raise ValueError(f"cap_{cap_hours:g}h_seed_sweep_scope_mismatch")
    recorded_cap = _positive_cap(report.get("max_label_elapsed_hours"), "seed_sweep_cap")
    if not math.isclose(recorded_cap, cap_hours, rel_tol=0, abs_tol=_EPS):
        raise ValueError(f"cap_{cap_hours:g}h_seed_sweep_cap_mismatch")
    if report.get("raw_ohlc_sha256") != expected_raw_hash:
        raise ValueError(f"cap_{cap_hours:g}h_seed_sweep_raw_hash_mismatch")
    if report.get("requested_seeds") != list(requested_seeds):
        raise ValueError(f"cap_{cap_hours:g}h_seed_sweep_seed_list_mismatch")
    if report.get("failed_seed_count") != 0 or report.get("successful_seed_count") != len(requested_seeds):
        raise ValueError(f"cap_{cap_hours:g}h_seed_sweep_incomplete")
    raw_results = report.get("seed_results")
    if not isinstance(raw_results, list):
        raise ValueError(f"cap_{cap_hours:g}h_seed_results_missing")
    results: dict[int, Mapping[str, Any]] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            raise ValueError(f"cap_{cap_hours:g}h_seed_result_invalid")
        seed = item.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in results:
            raise ValueError(f"cap_{cap_hours:g}h_seed_identity_invalid")
        if item.get("status") not in _PASS:
            raise ValueError(f"cap_{cap_hours:g}h_seed_{seed}_status_not_pass")
        experiments = item.get("experiments")
        if not isinstance(experiments, dict) or not experiments:
            raise ValueError(f"cap_{cap_hours:g}h_seed_{seed}_experiments_missing")
        results[seed] = item
    if set(results) != set(requested_seeds):
        raise ValueError(f"cap_{cap_hours:g}h_seed_result_set_mismatch")
    return results, path


def _extract_seed_metric(
    seed_result: Mapping[str, Any],
    experiment_key: tuple[str, str],
    metric_name: str,
    *,
    cap_hours: float,
    seed: int,
) -> dict[str, float]:
    experiments = seed_result.get("experiments")
    if not isinstance(experiments, dict):
        raise ValueError(f"cap_{cap_hours:g}h_seed_{seed}_experiments_invalid")
    identity = f"{experiment_key[0]}/{experiment_key[1]}"
    experiment = experiments.get(identity)
    if not isinstance(experiment, dict):
        raise ValueError(f"cap_{cap_hours:g}h_seed_{seed}_experiment_missing:{identity}")
    metrics = experiment.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get(metric_name), dict):
        raise ValueError(f"cap_{cap_hours:g}h_seed_{seed}_metric_missing:{identity}:{metric_name}")
    values = metrics[metric_name]
    return {
        name: _finite(values.get(name), f"cap_{cap_hours:g}h_seed_{seed}_{identity}_{metric_name}_{name}")
        for name in ("baseline", "capped", "delta")
    }


def _equivalence_group_map(policy_report: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Mapping[str, Any]]]:
    candidates = policy_report.get("candidate_policies")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("gap_policy_candidate_policies_missing")
    policies: dict[str, Mapping[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("gap_policy_candidate_record_invalid")
        name = item.get("policy")
        mask_hash = item.get("eligible_row_numbers_sha256")
        excluded_hash = item.get("ineligible_row_numbers_sha256")
        if not isinstance(name, str) or not name:
            raise ValueError("gap_policy_policy_name_missing")
        if name in policies:
            raise ValueError(f"gap_policy_duplicate_policy:{name}")
        if not isinstance(mask_hash, str) or len(mask_hash) != 64:
            raise ValueError(f"gap_policy_eligible_mask_hash_invalid:{name}")
        if not isinstance(excluded_hash, str) or len(excluded_hash) != 64:
            raise ValueError(f"gap_policy_ineligible_mask_hash_invalid:{name}")
        policies[name] = item
        group = groups.setdefault(mask_hash, {
            "eligible_row_numbers_sha256": mask_hash,
            "ineligible_row_numbers_sha256": excluded_hash,
            "policy_aliases": [],
            "dataset_rows": item.get("dataset_rows"),
            "eligible_rows": item.get("eligible_rows"),
            "ineligible_rows": item.get("ineligible_rows"),
            "eligible_fraction": item.get("eligible_fraction"),
        })
        if group["ineligible_row_numbers_sha256"] != excluded_hash:
            raise ValueError(f"same_eligible_mask_has_different_ineligible_hash:{name}")
        for field in ("dataset_rows", "eligible_rows", "ineligible_rows"):
            if group[field] != item.get(field):
                raise ValueError(f"same_eligible_mask_has_different_counts:{name}:{field}")
        group["policy_aliases"].append(name)
    return groups, policies


def review_gap_policy_performance(
    gap_policy_equivalence_report_path: str | Path,
    threshold_audit_report_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    policy_path = Path(gap_policy_equivalence_report_path).expanduser().resolve()
    threshold_audit_path = Path(threshold_audit_report_path).expanduser().resolve()
    policy_report = _read_json(policy_path)
    threshold_audit = _read_json(threshold_audit_path)

    if policy_report.get("scope") != _POLICY_SCOPE:
        raise ValueError("input_is_not_a_gap_policy_equivalence_audit")
    _check_pass(policy_report, "gap_policy_equivalence_audit")
    _check_research_flags(policy_report, "gap_policy_equivalence_audit")
    if threshold_audit.get("scope") != _THRESHOLD_AUDIT_SCOPE:
        raise ValueError("input_is_not_a_threshold_eligibility_profile_audit")
    _check_pass(threshold_audit, "threshold_eligibility_audit")
    _check_research_flags(threshold_audit, "threshold_eligibility_audit")

    input_reports = policy_report.get("input_reports")
    audit_ref = input_reports.get("threshold_audit") if isinstance(input_reports, dict) else None
    if not isinstance(audit_ref, dict):
        raise ValueError("gap_policy_report_threshold_audit_reference_missing")
    if Path(str(audit_ref.get("path", ""))).expanduser().resolve() != threshold_audit_path:
        raise ValueError("gap_policy_report_threshold_audit_path_mismatch")
    if audit_ref.get("sha256") != _sha256(threshold_audit_path):
        raise ValueError("gap_policy_report_threshold_audit_hash_mismatch")

    raw_hash = policy_report.get("baseline_raw_ohlc_sha256")
    dataset_hash = policy_report.get("baseline_dataset_sha256")
    if not isinstance(raw_hash, str) or not raw_hash or not isinstance(dataset_hash, str) or not dataset_hash:
        raise ValueError("gap_policy_report_source_hash_missing")
    if threshold_audit.get("raw_ohlc_sha256") != raw_hash:
        raise ValueError("threshold_audit_raw_ohlc_hash_mismatch")

    sweep_path_value = threshold_audit.get("threshold_sweep_report_path")
    if not isinstance(sweep_path_value, str) or not sweep_path_value:
        raise ValueError("threshold_sweep_report_path_missing")
    sweep_path = Path(sweep_path_value).expanduser().resolve()
    if not sweep_path.is_file():
        raise ValueError(f"threshold_sweep_report_missing:{sweep_path}")
    if _sha256(sweep_path) != threshold_audit.get("threshold_sweep_report_sha256"):
        raise ValueError("threshold_sweep_report_hash_mismatch")
    sweep = _read_json(sweep_path)
    if sweep.get("scope") != _SWEEP_SCOPE:
        raise ValueError("input_is_not_a_threshold_sweep_report")
    _check_pass(sweep, "threshold_sweep_report")
    _check_research_flags(sweep, "threshold_sweep_report")
    if sweep.get("raw_ohlc_sha256") != raw_hash:
        raise ValueError("threshold_sweep_raw_ohlc_hash_mismatch")
    if sweep.get("same_raw_ohlc_snapshot_across_caps") is not True:
        raise ValueError("threshold_sweep_raw_snapshot_not_verified")
    if sweep.get("same_fold_windows_across_caps") is not True:
        raise ValueError("threshold_sweep_fold_windows_not_verified")

    expected_seeds = threshold_audit.get("requested_seeds")
    if not isinstance(expected_seeds, list) or len(expected_seeds) < 2:
        raise ValueError("threshold_audit_seed_list_missing")
    if sweep.get("seeds") != expected_seeds:
        raise ValueError("threshold_sweep_seed_list_mismatch")

    cap_records = _index_records(sweep.get("cap_results"), "max_label_elapsed_hours", "threshold_sweep")
    cap_profiles = _index_records(threshold_audit.get("cap_profiles"), "cap_hours", "threshold_audit")
    for required_cap in _REQUIRED_CAPS:
        if required_cap not in cap_records or required_cap not in cap_profiles:
            raise ValueError(f"required_cap_missing:{required_cap:g}h")

    groups, policies = _equivalence_group_map(policy_report)
    cap_policy_names = {1.0: "elapsed_cap_1h", 1.5: "elapsed_cap_1_5h", 8.0: "elapsed_cap_8h"}
    for cap, policy_name in cap_policy_names.items():
        if policy_name not in policies:
            raise ValueError(f"gap_policy_mask_missing:{policy_name}")
        profile = cap_profiles[cap]
        policy_record = policies[policy_name]
        if profile.get("dataset_rows") != policy_record.get("dataset_rows"):
            raise ValueError(f"cap_{cap:g}h_dataset_row_count_mismatch")
        if profile.get("eligible_rows") != policy_record.get("eligible_rows"):
            raise ValueError(f"cap_{cap:g}h_eligible_row_count_mismatch")
        if profile.get("ineligible_rows") != policy_record.get("ineligible_rows"):
            raise ValueError(f"cap_{cap:g}h_ineligible_row_count_mismatch")

    # Confirm caps with equal row masks have equal eligible row-number hashes.
    if policies["elapsed_cap_1_5h"]["eligible_row_numbers_sha256"] != policies["elapsed_cap_8h"]["eligible_row_numbers_sha256"]:
        raise ValueError("1p5h_and_8h_row_mask_hashes_differ")
    equivalent_names = (
        ("elapsed_cap_1h", "exclude_any_non_nominal_gap"),
        ("elapsed_cap_1_5h", "elapsed_cap_8h"),
        ("elapsed_cap_1_5h", "exclude_labels_crossing_raw_gap_over_8h"),
        ("elapsed_cap_8h", "exclude_labels_crossing_raw_gap_over_8h"),
    )
    equivalence_records = policy_report.get("eligibility_mask_equivalence_checks")
    if not isinstance(equivalence_records, list):
        raise ValueError("gap_policy_equivalence_checks_missing")
    recorded_equivalences = {
        (item.get("left_policy"), item.get("right_policy")): item
        for item in equivalence_records if isinstance(item, dict)
    }
    for left_name, right_name in equivalent_names:
        item = recorded_equivalences.get((left_name, right_name))
        if item is None or item.get("same_eligible_mask") is not True or item.get("eligible_row_symmetric_difference_count") != 0:
            raise ValueError(f"required_mask_equivalence_not_verified:{left_name}:{right_name}")

    requested_metrics: dict[float, dict[tuple[str, str], Mapping[str, Any]]] = {}
    cap_seed_results: dict[float, dict[int, Mapping[str, Any]]] = {}
    seed_report_paths: dict[float, str] = {}
    expected_experiment_keys: set[tuple[str, str]] | None = None
    metric_names: set[str] | None = None
    warning_list = list(policy_report.get("warnings") or []) + list(threshold_audit.get("warnings") or [])

    for cap in _REQUIRED_CAPS:
        record = cap_records[cap]
        _check_pass(record, f"cap_{cap:g}h_threshold_record")
        if record.get("raw_ohlc_sha256") != raw_hash:
            raise ValueError(f"cap_{cap:g}h_raw_hash_mismatch")
        audit_profile = cap_profiles[cap]
        sweep_metrics = _metric_map(record.get("metrics"), f"cap_{cap:g}h_sweep")
        audit_metrics = _metric_map(audit_profile.get("metric_summary"), f"cap_{cap:g}h_audit")
        if not _same_json(
            [{"key": list(k), "metrics": v} for k, v in sorted(sweep_metrics.items())],
            [{"key": list(k), "metrics": v} for k, v in sorted(audit_metrics.items())],
        ):
            raise ValueError(f"cap_{cap:g}h_metric_summary_mismatch_between_audits")
        requested_metrics[cap] = sweep_metrics
        this_experiment_keys = set(sweep_metrics)
        if expected_experiment_keys is None:
            expected_experiment_keys = this_experiment_keys
        elif this_experiment_keys != expected_experiment_keys:
            raise ValueError("experiment_matrix_differs_between_caps")
        this_metric_names = {
            metric_name for payload in sweep_metrics.values() for metric_name in payload
        }
        if metric_names is None:
            metric_names = this_metric_names
        elif this_metric_names != metric_names:
            raise ValueError("metric_set_differs_between_caps")

        seed_results, child_path = _load_seed_reports(
            record, cap_hours=cap, requested_seeds=expected_seeds, expected_raw_hash=raw_hash
        )
        cap_seed_results[cap] = seed_results
        seed_report_paths[cap] = str(child_path)

    # Exact mask equality should imply the same scored data. Existing training runs
    # are not assumed deterministic: verify metric equality and warn if it does not hold.
    equivalent_cap_metric_checks: list[dict[str, Any]] = []
    for first, second in ((1.5, 8.0),):
        first_metrics, second_metrics = requested_metrics[first], requested_metrics[second]
        metrics_identical = _same_json(
            [{"key": list(k), "metrics": v} for k, v in sorted(first_metrics.items())],
            [{"key": list(k), "metrics": v} for k, v in sorted(second_metrics.items())],
        )
        per_seed_identical = True
        for seed in expected_seeds:
            first_experiments = cap_seed_results[first][seed]["experiments"]
            second_experiments = cap_seed_results[second][seed]["experiments"]
            if not _same_json(first_experiments, second_experiments):
                per_seed_identical = False
                break
        equivalent_cap_metric_checks.append({
            "cap_hours": [first, second],
            "same_eligible_row_mask": True,
            "aggregate_metric_summaries_identical": metrics_identical,
            "per_seed_results_identical": per_seed_identical,
        })
        if not metrics_identical or not per_seed_identical:
            warning_list.append(
                "equivalent_1p5h_and_8h_masks_have_different_metrics_or_seed_results; inspect run configuration and determinism"
            )

    baseline_refs_match = True
    baseline_reference_diffs: list[dict[str, Any]] = []
    assert expected_experiment_keys is not None and metric_names is not None
    for seed in expected_seeds:
        for experiment_key in sorted(expected_experiment_keys):
            for metric_name in sorted(metric_names):
                values_by_cap = {
                    cap: _extract_seed_metric(
                        cap_seed_results[cap][seed], experiment_key, metric_name, cap_hours=cap, seed=seed
                    )
                    for cap in _REQUIRED_CAPS
                }
                reference_values = [values_by_cap[cap]["baseline"] for cap in _REQUIRED_CAPS]
                same_ref = all(
                    math.isclose(reference_values[0], value, rel_tol=0, abs_tol=1e-8)
                    for value in reference_values[1:]
                )
                if not same_ref:
                    baseline_refs_match = False
                    baseline_reference_diffs.append({
                        "seed": seed,
                        "model_type": experiment_key[0],
                        "feature_variant": experiment_key[1],
                        "metric": metric_name,
                        "uncapped_baseline_metric_by_cap": {
                            f"{cap:g}h": round(values_by_cap[cap]["baseline"], 12)
                            for cap in _REQUIRED_CAPS
                        },
                    })

    paired_policy_differences: list[dict[str, Any]] = []
    for experiment_key in sorted(expected_experiment_keys):
        for metric_name in sorted(metric_names):
            per_seed: list[dict[str, Any]] = []
            for seed in expected_seeds:
                low = _extract_seed_metric(
                    cap_seed_results[1.0][seed], experiment_key, metric_name, cap_hours=1.0, seed=seed
                )
                high = _extract_seed_metric(
                    cap_seed_results[1.5][seed], experiment_key, metric_name, cap_hours=1.5, seed=seed
                )
                per_seed.append({
                    "seed": seed,
                    "cap_1h_capped_metric": round(low["capped"], 12),
                    "cap_1_5h_capped_metric": round(high["capped"], 12),
                    "uncapped_baseline_metric_cap_1h": round(low["baseline"], 12),
                    "uncapped_baseline_metric_cap_1_5h": round(high["baseline"], 12),
                    "uncapped_baselines_match": math.isclose(low["baseline"], high["baseline"], rel_tol=0, abs_tol=1e-8),
                    "cap_1h_delta_vs_its_uncapped_pair": round(low["delta"], 12),
                    "cap_1_5h_delta_vs_its_uncapped_pair": round(high["delta"], 12),
                    "difference_of_paired_deltas_1_5h_minus_1h": round(high["delta"] - low["delta"], 12),
                    "capped_metric_difference_1_5h_minus_1h": round(high["capped"] - low["capped"], 12),
                })
            delta_differences = [item["difference_of_paired_deltas_1_5h_minus_1h"] for item in per_seed]
            metric_differences = [item["capped_metric_difference_1_5h_minus_1h"] for item in per_seed]
            paired_policy_differences.append({
                "model_type": experiment_key[0],
                "feature_variant": experiment_key[1],
                "metric": metric_name,
                "seed_count": len(per_seed),
                "uncapped_baseline_reference_metrics_match_across_all_caps": baseline_refs_match,
                "mean_difference_of_paired_deltas": round(statistics.fmean(delta_differences), 12),
                "median_difference_of_paired_deltas": round(statistics.median(delta_differences), 12),
                "range_difference_of_paired_deltas": [
                    round(min(delta_differences), 12), round(max(delta_differences), 12)
                ],
                "mean_capped_metric_difference": round(statistics.fmean(metric_differences), 12),
                "median_capped_metric_difference": round(statistics.median(metric_differences), 12),
                "per_seed": per_seed,
                "interpretation": (
                    "Descriptive difference between existing same-seed cap runs on the same history and folds. "
                    "Use as a clean incremental policy comparison only when the uncapped reference metrics match "
                    "for the corresponding seed and experiment; no inference or model selection is performed."
                ),
            })

    if not baseline_refs_match:
        warning_list.append(
            "uncapped_baseline_metrics_differ_across_threshold_runs; cross-policy differences may include reference-run variability"
        )

    groups_by_hash: dict[str, list[float]] = {}
    for cap in _REQUIRED_CAPS:
        mask_hash = policies[cap_policy_names[cap]]["eligible_row_numbers_sha256"]
        groups_by_hash.setdefault(mask_hash, []).append(cap)
    unique_mask_groups: list[dict[str, Any]] = []
    for mask_hash, caps_in_group in sorted(groups_by_hash.items(), key=lambda item: min(item[1])):
        group_candidates = next(group for group in groups.values() if group["eligible_row_numbers_sha256"] == mask_hash)
        unique_mask_groups.append({
            "eligible_row_numbers_sha256": mask_hash,
            "ineligible_row_numbers_sha256": group_candidates["ineligible_row_numbers_sha256"],
            "policy_aliases": group_candidates["policy_aliases"],
            "caps_with_existing_performance_runs_hours": sorted(caps_in_group),
            "dataset_rows": group_candidates["dataset_rows"],
            "eligible_rows": group_candidates["eligible_rows"],
            "ineligible_rows": group_candidates["ineligible_rows"],
            "eligible_fraction": group_candidates["eligible_fraction"],
            "metrics_available": bool(caps_in_group),
        })

    warning_list.append(
        "Repeated optimizer seeds reuse the same market history and overlapping walk-forward windows; they are not independent market observations."
    )
    warning_list.append(
        "This review aggregates completed research runs only. Performance deltas do not establish live profitability or justify automatic policy selection."
    )
    warning_list = list(dict.fromkeys(str(item) for item in warning_list))
    status = "PASS_WITH_WARNINGS" if warning_list else "PASS"
    destination = Path(output_path).expanduser().resolve() if output_path else _free_output_path(policy_path)
    protected = {policy_path, threshold_audit_path, sweep_path}
    if destination in protected:
        raise ValueError("output_path_must_not_overwrite_input_artifact")
    if destination.exists():
        raise ValueError("output_path_already_exists_refusing_to_overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_version": 1,
        "scope": _REVIEW_SCOPE,
        "status": status,
        "gap_policy_equivalence_report_path": str(policy_path),
        "gap_policy_equivalence_report_sha256": _sha256(policy_path),
        "threshold_eligibility_audit_path": str(threshold_audit_path),
        "threshold_eligibility_audit_sha256": _sha256(threshold_audit_path),
        "threshold_sweep_report_path": str(sweep_path),
        "threshold_sweep_report_sha256": _sha256(sweep_path),
        "baseline_raw_ohlc_sha256": raw_hash,
        "baseline_dataset_sha256": dataset_hash,
        "requested_seeds": list(expected_seeds),
        "dataset_rows": policy_report.get("dataset_rows"),
        "unique_mask_count_in_policy_audit": len(groups),
        "unique_capped_mask_count_in_existing_sweep": len(groups_by_hash),
        "unique_capped_mask_groups": unique_mask_groups,
        "equivalent_cap_metric_checks": equivalent_cap_metric_checks,
        "uncapped_baseline_reference_metrics_match_across_caps": baseline_refs_match,
        "uncapped_baseline_reference_differences": baseline_reference_diffs,
        "paired_policy_metric_differences_1_5h_minus_1h": paired_policy_differences,
        "warnings": warning_list,
        "errors": [],
        "inferential": False,
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
        "models_trained": False,
        "interpretation": (
            "Caps are grouped by exact eligible row-mask identity. The 1.5-hour and 8-hour cap records are "
            "not counted as independent policies when they share a mask. Metric summaries are reported from "
            "existing matched seed sweeps; no reruns occur. Policy differences are descriptive and need matching "
            "uncapped baseline metrics to isolate cap-dependent effects from reference-run variability."
        ),
        "report_path": str(destination),
    }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review completed gap-policy performance by unique row mask without retraining."
    )
    parser.add_argument("--gap-policy-equivalence-report", required=True)
    parser.add_argument("--threshold-audit-report", required=True)
    parser.add_argument("--output-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = review_gap_policy_performance(
            args.gap_policy_equivalence_report,
            args.threshold_audit_report,
            output_path=args.output_path,
        )
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL", "scope": _REVIEW_SCOPE,
            "errors": [f"{type(exc).__name__}:{exc}"],
            "inferential": False, "automatic_model_selection": False,
            "actionable": False, "execution_connected": False,
        }, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": report["status"],
        "report_path": report["report_path"],
        "dataset_rows": report["dataset_rows"],
        "requested_seeds": report["requested_seeds"],
        "unique_mask_count_in_policy_audit": report["unique_mask_count_in_policy_audit"],
        "unique_capped_mask_count_in_existing_sweep": report["unique_capped_mask_count_in_existing_sweep"],
        "unique_capped_mask_groups": report["unique_capped_mask_groups"],
        "equivalent_cap_metric_checks": report["equivalent_cap_metric_checks"],
        "uncapped_baseline_reference_metrics_match_across_caps": report["uncapped_baseline_reference_metrics_match_across_caps"],
        "uncapped_baseline_reference_differences": report["uncapped_baseline_reference_differences"],
        "paired_policy_metric_differences_1_5h_minus_1h": report["paired_policy_metric_differences_1_5h_minus_1h"],
        "warnings": report["warnings"],
        "errors": report["errors"],
        "inferential": False, "automatic_model_selection": False,
        "actionable": False, "execution_connected": False,
    }, indent=2, sort_keys=True))
    return 0 if report["status"] in _PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())

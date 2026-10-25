from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.research_cross_cap_stability import audit_cross_cap_stability
from app.research_label_eligibility_common_cohort import _metrics, audit_common_heldout_cohort


CAPS = (1.0, 1.5, 8.0)
SEEDS = (20261021, 20261022)


def _source_metrics_for_cap(cap: float, *, equivalent_15_8: bool = False) -> tuple[float, float]:
    if equivalent_15_8 and cap == 8.0:
        return (0.2, 0.8)
    return {
        1.0: (0.4, 0.6),
        1.5: (0.2, 0.8),
        8.0: (0.45, 0.55),
    }[cap]


def _make_seed_reports(tmp_path: Path, seed: int, *, audit_status: str = "PASS",
                       missing_source: bool = False, bad_sample_count: bool = False,
                       equivalent_15_8: bool = False) -> Path:
    source_paths = []
    baseline_ids = []
    dataset_hash = format(seed, "064x")
    raw_hash = "a" * 64

    for cap in CAPS:
        report_path = tmp_path / f"source_seed_{seed}_cap_{str(cap).replace('.', 'p')}.json"
        variant_probabilities = _source_metrics_for_cap(cap, equivalent_15_8=equivalent_15_8)
        fold_results = []
        for fold_index in range(2):
            day = fold_index + 1
            timestamps = [
                f"2026-01-0{day}T00:00:00Z",
                f"2026-01-0{day}T01:00:00Z",
            ]
            fingerprint = hashlib.sha256(("\n".join(sorted(timestamps)) + "\n").encode()).hexdigest()
            paired_rows = [
                {
                    "timestamp_utc": timestamps[0],
                    "label": 0,
                    "baseline_probability_class_1": 0.3,
                    "variant_probability_class_1": variant_probabilities[0],
                },
                {
                    "timestamp_utc": timestamps[1],
                    "label": 1,
                    "baseline_probability_class_1": 0.7,
                    "variant_probability_class_1": variant_probabilities[1],
                },
            ]
            baseline_metrics = _metrics(
                [{"label": row["label"], "p": row["baseline_probability_class_1"]} for row in paired_rows], "p"
            )
            variant_metrics = _metrics(
                [{"label": row["label"], "p": row["variant_probability_class_1"]} for row in paired_rows], "p"
            )
            delta_metrics = {
                metric: variant_metrics[metric] - baseline_metrics[metric]
                for metric in (
                    "log_loss", "brier_score", "accuracy_at_fixed_half",
                    "prevalence", "mean_predicted_probability",
                )
            }
            fold_results.append({
                "fold_index": fold_index,
                "test_start_utc": f"2026-01-0{day}T00:00:00Z",
                "test_end_utc": f"2026-01-0{day}T02:00:00Z",
                "eligible_timestamp_fingerprint_sha256": fingerprint,
                "eligible_timestamp_predictions": paired_rows,
                "baseline_same_eligible_timestamps_metrics": baseline_metrics,
                "variant_eligible_test_metrics": variant_metrics,
                "paired_variant_minus_baseline_delta": delta_metrics,
            })
        report = {
            "status": "PASS_WITH_WARNINGS",
            "baseline_run_id": f"baseline-{seed}-{cap}",
            "baseline_dataset_sha256": dataset_hash,
            "variant_run_id": f"variant-{seed}-{cap}",
            "symbol": "EURUSD",
            "timeframe": "M15",
            "label_horizon_bars": 4,
            "raw_ohlc_sha256": raw_hash,
            "max_label_elapsed_hours": cap,
            "automatic_model_selection": False,
            "actionable": False,
            "execution_connected": False,
            "experiments": [{
                "model_type": "xgboost",
                "feature_variant": "all_features",
                "fold_results": fold_results,
            }],
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        source_paths.append(str(report_path))
        baseline_ids.append(report["baseline_run_id"])

    audit_path = tmp_path / f"common_cohort_audit_1h_1p5h_8h_seed_{seed}.json"
    audit = audit_common_heldout_cohort(
        source_paths,
        allow_baseline_id_variation=True,
    )
    assert audit["status"] == "PASS", audit
    if missing_source:
        audit["input_comparison_reports"][1] = str(tmp_path / "does_not_exist.json")
    if audit_status != "PASS":
        audit["status"] = audit_status
    if bad_sample_count:
        audit["experiments"][0]["fold_results"][0]["variant_metrics_on_common_cohort_by_cap"]["8.0"]["sample_count"] = 3
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return audit_path


def test_cross_cap_stability_summarizes_seed_level_deltas_and_allows_seed_hashes_to_differ(tmp_path):
    paths = [_make_seed_reports(tmp_path, seed) for seed in SEEDS]

    result = audit_cross_cap_stability(paths, expected_seeds=SEEDS)

    assert result["status"] == "PASS"
    assert result["observed_seed_ids"] == list(SEEDS)
    assert result["actionable"] is False
    assert result["execution_connected"] is False
    assert len(result["per_seed_results"]) == 2
    assert len(result["cross_seed_summary"]) == 6
    log_loss_15 = next(
        row for row in result["cross_seed_summary"]
        if row["cap_hours"] == 1.5 and row["metric"] == "log_loss"
    )
    assert log_loss_15["seed_favorable_count"] == 2
    assert log_loss_15["seed_adverse_count"] == 0
    assert log_loss_15["seed_level_mean_fold_delta_summary"]["count"] == 2
    log_loss_8 = next(
        row for row in result["cross_seed_summary"]
        if row["cap_hours"] == 8.0 and row["metric"] == "log_loss"
    )
    assert log_loss_8["seed_adverse_count"] == 2


def test_cross_cap_stability_fails_closed_when_a_common_cohort_audit_failed(tmp_path):
    paths = [
        _make_seed_reports(tmp_path, SEEDS[0], audit_status="FAIL"),
        _make_seed_reports(tmp_path, SEEDS[1]),
    ]

    result = audit_cross_cap_stability(paths, expected_seeds=SEEDS)

    assert result["status"] == "FAIL"
    assert any("status must be PASS" in error for error in result["errors"])
    assert result["actionable"] is False
    assert result["execution_connected"] is False


def test_cross_cap_stability_fails_when_a_source_comparison_report_is_missing(tmp_path):
    paths = [
        _make_seed_reports(tmp_path, SEEDS[0], missing_source=True),
        _make_seed_reports(tmp_path, SEEDS[1]),
    ]

    result = audit_cross_cap_stability(paths, expected_seeds=SEEDS)

    assert result["status"] == "FAIL"
    assert any("source comparison report not found" in error for error in result["errors"])


def test_cross_cap_stability_fails_when_cap_metrics_do_not_use_common_sample_count(tmp_path):
    paths = [
        _make_seed_reports(tmp_path, SEEDS[0], bad_sample_count=True),
        _make_seed_reports(tmp_path, SEEDS[1]),
    ]

    result = audit_cross_cap_stability(paths, expected_seeds=SEEDS)

    assert result["status"] == "FAIL"
    assert any("stored audit differs from recomputed source results for experiments" in error for error in result["errors"])


def test_cross_cap_stability_fails_when_expected_seed_set_is_incomplete(tmp_path):
    path = _make_seed_reports(tmp_path, SEEDS[0])

    result = audit_cross_cap_stability([path], expected_seeds=SEEDS)

    assert result["status"] == "FAIL"
    assert any("expected 2 common-cohort audit reports" in error for error in result["errors"])


def test_cap_equivalence_diagnostics_surface_same_cohort_prediction_mismatches(tmp_path):
    paths = [_make_seed_reports(tmp_path, seed) for seed in SEEDS]

    result = audit_cross_cap_stability(paths, expected_seeds=SEEDS)

    diagnostics = result["per_seed_results"][0]["cap_equivalence_diagnostics"]
    assert diagnostics["status"] == "REQUIRES_REVIEW"
    pair = diagnostics["counts_by_cap_pair"]["1.5_vs_8"]
    assert pair["equivalent_predictions_and_metrics"] == 0
    assert pair["same_eligibility_but_predictions_or_metrics_differ"] == 2
    assert all(issue["variant_predictions_match"] is False for issue in diagnostics["issues"] if issue["cap_pair"] == "1.5_vs_8")


def test_cap_equivalence_diagnostics_confirm_matching_timestamps_predictions_and_metrics(tmp_path):
    paths = [
        _make_seed_reports(tmp_path, seed, equivalent_15_8=True)
        for seed in SEEDS
    ]

    result = audit_cross_cap_stability(paths, expected_seeds=SEEDS)

    diagnostics = result["per_seed_results"][0]["cap_equivalence_diagnostics"]
    # Other cap pairs can legitimately differ; assert the specific 1.5h/8h pair.
    pair = diagnostics["counts_by_cap_pair"]["1.5_vs_8"]
    assert pair["equivalent_predictions_and_metrics"] == 2
    assert pair["same_eligibility_but_predictions_or_metrics_differ"] == 0

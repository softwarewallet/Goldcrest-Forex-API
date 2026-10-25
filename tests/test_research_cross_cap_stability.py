from __future__ import annotations

import json
from pathlib import Path

from app.research_cross_cap_stability import audit_cross_cap_stability


CAPS = (1.0, 1.5, 8.0)
SEEDS = (20261021, 20261022)


def _metrics(cap: float, sample_count: int = 2) -> dict:
    values = {
        1.0: (0.60, 0.20, 0.70),
        1.5: (0.50, 0.15, 0.75),
        8.0: (0.70, 0.25, 0.65),
    }[cap]
    return {
        "sample_count": sample_count,
        "positive_count": 1,
        "negative_count": sample_count - 1,
        "prevalence": 0.5,
        "log_loss": values[0],
        "brier_score": values[1],
        "accuracy_at_fixed_half": values[2],
        "mean_predicted_probability": 0.5,
    }


def _make_seed_reports(tmp_path: Path, seed: int, *, audit_status: str = "PASS",
                       missing_source: bool = False, bad_sample_count: bool = False) -> Path:
    source_paths = []
    baseline_ids = []
    dataset_hash = format(seed, "064x")
    raw_hash = "a" * 64
    for cap in CAPS:
        report_path = tmp_path / f"source_seed_{seed}_cap_{str(cap).replace('.', 'p')}.json"
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
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        source_paths.append(str(report_path))
        baseline_ids.append(report["baseline_run_id"])

    if missing_source:
        source_paths[1] = str(tmp_path / "does_not_exist.json")

    fold_results = []
    for fold_index in range(2):
        per_cap = {str(cap): _metrics(cap) for cap in CAPS}
        if bad_sample_count:
            per_cap["8.0"]["sample_count"] = 3
        fold_results.append({
            "fold_index": fold_index,
            "test_start_utc": f"2026-01-0{fold_index + 1}T00:00:00Z",
            "test_end_utc": f"2026-01-0{fold_index + 1}T02:00:00Z",
            "common_timestamp_count": 2,
            "common_cohort_timestamp_fingerprint_sha256": "c" * 64,
            "baseline_metrics_on_common_cohort": _metrics(1.0),
            "variant_metrics_on_common_cohort_by_cap": per_cap,
            "actionable": False,
        })
    audit_path = tmp_path / f"common_cohort_audit_1h_1p5h_8h_seed_{seed}.json"
    audit = {
        "status": audit_status,
        "scope": "GOLDCREST_COMMON_HELDOUT_COHORT_AUDIT",
        "baseline_identity_mode": "dataset_hash_equivalence",
        "baseline_dataset_sha256": dataset_hash,
        "baseline_run_id": None,
        "baseline_run_ids": baseline_ids,
        "symbol": "EURUSD",
        "timeframe": "M15",
        "label_horizon_bars": 4,
        "raw_ohlc_sha256": raw_hash,
        "caps_hours": list(CAPS),
        "input_comparison_reports": source_paths,
        "experiments": [{
            "model_type": "xgboost",
            "feature_variant": "all_features",
            "fold_count": 2,
            "fold_results": fold_results,
            "actionable": False,
        }],
        "automatic_model_selection": False,
        "actionable": False,
        "execution_connected": False,
    }
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
    assert any("not scored on the common timestamp count" in error for error in result["errors"])


def test_cross_cap_stability_fails_when_expected_seed_set_is_incomplete(tmp_path):
    path = _make_seed_reports(tmp_path, SEEDS[0])

    result = audit_cross_cap_stability([path], expected_seeds=SEEDS)

    assert result["status"] == "FAIL"
    assert any("expected 2 common-cohort audit reports" in error for error in result["errors"])

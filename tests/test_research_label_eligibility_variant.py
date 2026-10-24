from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.research_label_eligibility_variant import (
    _choose_variant_run_id,
    _comparison_summary_lines,
    _infer_fold_geometry,
    _variant_run_collisions,
)


def _write_dataset_timestamps(path: Path, count: int) -> list[str]:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [
        (start + timedelta(minutes=15 * index)).isoformat().replace("+00:00", "Z")
        for index in range(count)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"timestamp": timestamp} for timestamp in timestamps)
    return timestamps


def _matrix_for_geometry(timestamps: list[str]) -> dict:
    horizon = 4
    initial_train = 80
    validation = 30
    test = 40
    step = 20
    windows = []
    fold_results = []
    train_end = initial_train
    while train_end + validation + test <= len(timestamps):
        validation_end = train_end + validation
        test_start = validation_end + horizon
        test_end = validation_end + test - 1
        window = {
            "fold_index": len(windows),
            "test_start_utc": timestamps[test_start],
            "test_end_utc": timestamps[test_end],
            "test_rows": test - horizon,
        }
        windows.append(window)
        fold_results.append({
            "fold_index": len(fold_results),
            "validation_start_utc": timestamps[train_end + horizon],
            "test_start_utc": timestamps[test_start],
            "test_end_utc": timestamps[test_end],
            "trials_requested": 5,
        })
        train_end += step
    return {
        "common_fold_test_windows": windows,
        "experiments": [
            {"fold_results": fold_results},
            {"fold_results": [dict(fold) for fold in fold_results]},
        ],
    }


def test_infer_fold_geometry_recovers_exact_pre_purge_sizes_and_checks_all_windows(tmp_path):
    dataset = tmp_path / "prepared.csv"
    timestamps = _write_dataset_timestamps(dataset, 230)
    matrix = _matrix_for_geometry(timestamps)

    recovered = _infer_fold_geometry(dataset, matrix, horizon_bars=4)

    assert recovered == {
        "initial_train_bars": 80,
        "validation_bars": 30,
        "test_bars": 40,
        "step_bars": 20,
        "fold_count": 5,
        "dataset_rows": 230,
    }


def test_infer_fold_geometry_rejects_test_window_mismatch(tmp_path):
    dataset = tmp_path / "prepared.csv"
    timestamps = _write_dataset_timestamps(dataset, 230)
    matrix = _matrix_for_geometry(timestamps)
    matrix["common_fold_test_windows"][2]["test_end_utc"] = timestamps[200]

    with pytest.raises(ValueError, match="reconstructed_test_window_mismatch:fold_2"):
        _infer_fold_geometry(dataset, matrix, horizon_bars=4)


def test_infer_fold_geometry_rejects_timestamps_missing_from_dataset(tmp_path):
    dataset = tmp_path / "prepared.csv"
    timestamps = _write_dataset_timestamps(dataset, 230)
    matrix = _matrix_for_geometry(timestamps)
    matrix["experiments"][0]["fold_results"][0]["validation_start_utc"] = "2030-01-01T00:00:00Z"

    with pytest.raises(ValueError, match="fold_timestamp_not_in_dataset"):
        _infer_fold_geometry(dataset, matrix, horizon_bars=4)


def test_infer_fold_geometry_requires_multiple_windows_when_geometry_is_not_persisted(tmp_path):
    dataset = tmp_path / "prepared.csv"
    timestamps = _write_dataset_timestamps(dataset, 230)
    matrix = _matrix_for_geometry(timestamps)
    matrix["common_fold_test_windows"] = matrix["common_fold_test_windows"][:1]
    matrix["experiments"] = [
        {"fold_results": experiment["fold_results"][:1]}
        for experiment in matrix["experiments"]
    ]

    with pytest.raises(ValueError, match="without_two_common_test_windows"):
        _infer_fold_geometry(dataset, matrix, horizon_bars=4)



def test_default_variant_run_id_auto_versions_without_overwriting_existing_artifacts(tmp_path):
    requested = "baseline-elapsed-cap-1h"
    existing = tmp_path / f"{requested}_pipeline_manifest.json"
    existing.write_text('{"original": true}\\n', encoding="utf-8")

    chosen, was_auto_versioned = _choose_variant_run_id(
        tmp_path,
        requested,
        symbol="EURUSD",
        timeframe="M15",
        allow_auto_versioning=True,
    )

    assert chosen == "baseline-elapsed-cap-1h-rerun-2"
    assert was_auto_versioned is True
    assert existing.read_text(encoding="utf-8") == '{"original": true}\\n'
    assert _variant_run_collisions(
        tmp_path, chosen, symbol="EURUSD", timeframe="M15"
    ) == []


def test_default_variant_run_id_skips_multiple_occupied_namespaces(tmp_path):
    requested = "baseline-elapsed-cap-1h"
    (tmp_path / f"{requested}_pipeline_manifest.json").write_text("{}\\n", encoding="utf-8")
    (tmp_path / f"{requested}-rerun-2_label_eligibility_comparison.json").write_text(
        "{}\\n", encoding="utf-8"
    )

    chosen, was_auto_versioned = _choose_variant_run_id(
        tmp_path,
        requested,
        symbol="EURUSD",
        timeframe="M15",
        allow_auto_versioning=True,
    )

    assert chosen == "baseline-elapsed-cap-1h-rerun-3"
    assert was_auto_versioned is True


def test_explicit_variant_run_id_collision_fails_without_overwriting(tmp_path):
    requested = "operator-selected-cap-run"
    existing = tmp_path / f"{requested}_label_eligibility_variant_setup.json"
    existing.write_text('{"keep": "original"}\\n', encoding="utf-8")

    with pytest.raises(ValueError, match="variant_run_id_artifacts_already_exist"):
        _choose_variant_run_id(
            tmp_path,
            requested,
            symbol="EURUSD",
            timeframe="M15",
            allow_auto_versioning=False,
        )

    assert existing.read_text(encoding="utf-8") == '{"keep": "original"}\\n'



def test_comparison_summary_lines_prints_paired_metrics_and_eligibility(tmp_path):
    comparison = {
        "status": "PASS_WITH_WARNINGS",
        "raw_ohlc_audit_warning_diff": {
            "baseline_only": ["legacy_gap_rule"],
            "variant_only": ["new_gap_rule"],
        },
        "eligibility_summary": {
            "variant_eligible_rows": 900,
            "variant_dataset_rows": 1000,
            "variant_ineligible_rows": 100,
            "model_rows_scored_across_experiments_and_folds": 5000,
            "model_rows_excluded_across_experiments_and_folds": 600,
        },
        "experiments": [
            {
                "model_type": "xgboost",
                "feature_variant": "all_features",
                "mean_across_folds": {
                    "test_rows_counted_across_folds": 700,
                    "baseline_same_eligible_timestamps_metrics": {
                        "log_loss": 0.71,
                        "brier_score": 0.26,
                    },
                    "variant_eligible_test_metrics": {
                        "log_loss": 0.69,
                        "brier_score": 0.24,
                    },
                    "paired_variant_minus_baseline_delta_mean_across_folds": {
                        "log_loss": -0.02,
                        "brier_score": -0.02,
                        "accuracy_at_fixed_half": 0.03,
                    },
                },
                "fold_direction_stability": {
                    "fold_count": 3,
                    "metrics": {
                        "log_loss": {
                            "favorable_fold_count": 2,
                            "adverse_fold_count": 1,
                            "neutral_fold_count": 0,
                            "median_delta": -0.01,
                            "minimum_delta": -0.03,
                            "maximum_delta": 0.02,
                        },
                        "brier_score": {
                            "favorable_fold_count": 1,
                            "adverse_fold_count": 2,
                            "neutral_fold_count": 0,
                            "median_delta": 0.005,
                            "minimum_delta": -0.01,
                            "maximum_delta": 0.02,
                        },
                        "accuracy_at_fixed_half": {
                            "favorable_fold_count": 2,
                            "adverse_fold_count": 1,
                            "neutral_fold_count": 0,
                            "median_delta": 0.01,
                            "minimum_delta": -0.01,
                            "maximum_delta": 0.03,
                        },
                    },
                },
            }
        ],
    }

    lines = _comparison_summary_lines(comparison, tmp_path / "paired.json")
    joined = "\\n".join(lines)

    assert "Paired comparison status: PASS_WITH_WARNINGS" in joined
    assert "900/1000 dataset rows eligible; 100 ineligible" in joined
    assert "log_loss baseline/capped=+0.710000/+0.690000, delta=-0.020000" in joined
    assert "Brier baseline/capped=+0.260000/+0.240000, delta=-0.020000" in joined
    assert "accuracy@0.5 delta=+0.030000" in joined
    assert "fold direction log_loss: favorable=2/3, adverse=1, neutral=0" in joined
    assert "Raw OHLC audit warnings (baseline-only, 1):" in joined
    assert "legacy_gap_rule" in joined
    assert "new_gap_rule" in joined
    assert "Descriptive only; no model selection or execution." in joined


def test_comparison_summary_lines_handles_missing_optional_sections(tmp_path):
    lines = _comparison_summary_lines({"status": "FAIL", "errors": ["bad report"]}, tmp_path / "missing.json")

    assert lines[0] == "Paired comparison status: FAIL"
    assert any("Descriptive only" in line for line in lines)

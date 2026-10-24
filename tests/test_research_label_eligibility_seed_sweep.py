from __future__ import annotations

import json

import pytest

from app.research_label_eligibility_seed_sweep import (
    aggregate_seed_results,
    choose_seed_sweep_report_path,
    parse_random_seeds,
)


def _seed(seed: int, key: str, deltas: dict[str, float]) -> dict:
    return {
        "seed": seed,
        "experiments": {
            key: {
                "model_type": "xgboost",
                "feature_variant": "all_features",
                "fold_count": 7,
                "metrics": {
                    "log_loss": {"baseline": 0.7, "capped": 0.7 + deltas["log_loss"], "delta": deltas["log_loss"]},
                    "brier_score": {"baseline": 0.25, "capped": 0.25 + deltas["brier_score"], "delta": deltas["brier_score"]},
                    "accuracy_at_fixed_half": {"baseline": 0.5, "capped": 0.5 + deltas["accuracy_at_fixed_half"], "delta": deltas["accuracy_at_fixed_half"]},
                },
            }
        },
    }


def test_parse_random_seeds_accepts_distinct_values():
    assert parse_random_seeds("20261021, 20261022,20261023") == [
        20261021, 20261022, 20261023
    ]


@pytest.mark.parametrize(
    "value, message",
    [
        ("20261021", "at least two"),
        ("1,1", "unique"),
        ("1,,2", "comma-separated"),
        ("-1,2", "between 0"),
        ("1,4294967296", "between 0"),
        ("one,2", "comma-separated"),
        ([], "at least two"),
    ],
)
def test_parse_random_seeds_rejects_invalid_values(value, message):
    with pytest.raises(ValueError, match=message):
        parse_random_seeds(value)


def test_seed_sweep_report_path_is_collision_safe(tmp_path):
    first = choose_seed_sweep_report_path(tmp_path, "paired-seedcap-1h")
    assert first.name == "paired-seedcap-1h_seed_sweep_report.json"
    first.write_text('{"keep": true}\n', encoding="utf-8")
    second = choose_seed_sweep_report_path(tmp_path, "paired-seedcap-1h")
    assert second.name == "paired-seedcap-1h-rerun-2_seed_sweep_report.json"
    assert json.loads(first.read_text(encoding="utf-8")) == {"keep": True}


def test_aggregate_seed_results_reports_direction_counts_and_ranges():
    results = [
        _seed(1, "xgboost/all_features", {
            "log_loss": -0.01, "brier_score": -0.002, "accuracy_at_fixed_half": 0.01
        }),
        _seed(2, "xgboost/all_features", {
            "log_loss": 0.02, "brier_score": 0.003, "accuracy_at_fixed_half": -0.02
        }),
        _seed(3, "xgboost/all_features", {
            "log_loss": -0.005, "brier_score": 0.0, "accuracy_at_fixed_half": 0.0
        }),
    ]

    report = aggregate_seed_results(results)
    experiment = report["experiments"][0]

    assert experiment["metrics"]["log_loss"]["favorable_seed_count"] == 2
    assert experiment["metrics"]["log_loss"]["adverse_seed_count"] == 1
    assert experiment["metrics"]["log_loss"]["minimum_delta"] == -0.01
    assert experiment["metrics"]["log_loss"]["maximum_delta"] == 0.02
    assert experiment["metrics"]["brier_score"]["favorable_seed_count"] == 1
    assert experiment["metrics"]["brier_score"]["adverse_seed_count"] == 1
    assert experiment["metrics"]["brier_score"]["neutral_seed_count"] == 1
    assert experiment["metrics"]["accuracy_at_fixed_half"]["favorable_seed_count"] == 1
    assert report["inferential"] is False
    assert report["automatic_model_selection"] is False
    assert report["actionable"] is False


def test_aggregate_seed_results_rejects_inconsistent_experiment_matrices():
    first = _seed(1, "xgboost/all_features", {
        "log_loss": 0.0, "brier_score": 0.0, "accuracy_at_fixed_half": 0.0
    })
    second = _seed(2, "lightgbm/all_features", {
        "log_loss": 0.0, "brier_score": 0.0, "accuracy_at_fixed_half": 0.0
    })

    with pytest.raises(ValueError, match="experiment_matrix_differs"):
        aggregate_seed_results([first, second])

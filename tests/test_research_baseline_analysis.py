from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.research_baseline_analysis import _block_bootstrap, analyze_prediction_matrix


def _matrix():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    windows = []
    folds = []
    for fold_index in range(2):
        fold_start = start + timedelta(hours=3 * fold_index)
        rows = []
        for row_index in range(8):
            timestamp = fold_start + timedelta(minutes=15 * row_index)
            label = row_index % 2
            model_probability = 0.8 if label == 1 else 0.2
            rows.append({
                "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
                "label": label,
                "model_probability_class_1": model_probability,
                "constant_probability_class_1": 0.5,
                "matured_refit_prevalence_probability_class_1": 0.5,
                "matured_refit_majority_class": fold_index,
                "lagged_direction_probability_class_1": 0.65 if label == 1 else 0.35,
                "lagged_direction_prediction": label,
            })
        windows.append({
            "fold_index": fold_index,
            "test_start_utc": rows[0]["timestamp_utc"],
            "test_end_utc": rows[-1]["timestamp_utc"],
            "test_rows": len(rows),
        })
        folds.append({
            "fold_index": fold_index,
            "test_metrics": {"sample_count": len(rows)},
            "test_predictions": rows,
            "baseline_diagnostics": {
                "baseline_labels_used_test_data": False,
                "refit_rows_matured_before_test": 50,
            },
            "actionable": False,
        })
    return {
        "schema_version": 1,
        "run_id": "baseline-fixture",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "horizon_bars": 4,
        "class_1_means": "UP",
        "fold_count": 2,
        "common_fold_test_windows": windows,
        "experiments": [{
            "experiment_id": "baseline-fixture-xgboost-all_features",
            "model_type": "xgboost",
            "feature_variant": "all_features",
            "fold_count": 2,
            "fold_results": folds,
            "test_metrics_used_for_tuning": False,
            "actionable": False,
        }],
        "actionable": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    }


def test_analyzer_compares_models_to_causal_baselines_and_emits_uncertainty():
    result = analyze_prediction_matrix(_matrix(), bootstrap_resamples=100, random_seed=41)

    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert result["actionable"] is False
    assert result["execution_connected"] is False
    experiment = result["experiments"][0]
    assert experiment["model_metrics"]["sample_count"] == 16
    assert experiment["model_metrics"]["accuracy_at_fixed_half"] == 1.0
    assert len(experiment["probability_loss_comparisons"]) == 3
    constant = experiment["probability_loss_comparisons"][0]
    assert constant["baseline"] == "constant_probability_0_5"
    assert constant["metrics"]["log_loss"]["improvement_baseline_minus_model"] > 0
    assert len(constant["metrics"]["log_loss"]["confidence_interval_95"]) == 2
    assert constant["metrics"]["log_loss"]["interval_evidence"] == "positive_interval_excludes_zero"
    assert constant["metrics"]["log_loss"]["folds_with_positive_mean_improvement"] == 2
    assert constant["metrics"]["log_loss"]["folds_compared"] == 2
    assert len(experiment["fold_stability"]) == 2
    assert experiment["fold_stability"][0]["sample_count"] == 8
    assert experiment["fold_stability"][0]["probability_loss_improvements"]["constant_probability_0_5"]["log_loss_improvement"] > 0
    assert result["automatic_model_selection"] is False
    assert result["test_metrics_used_for_selection"] is False


def test_analyzer_fails_closed_when_prediction_records_are_missing():
    matrix = _matrix()
    matrix["experiments"][0]["fold_results"][0]["test_predictions"] = []

    result = analyze_prediction_matrix(matrix, bootstrap_resamples=100)

    assert result["status"] == "FAIL"
    assert any("test_predictions_missing" in error for error in result["errors"])
    assert result["actionable"] is False


def test_analyzer_rejects_baselines_that_claim_to_use_test_labels():
    matrix = _matrix()
    matrix["experiments"][0]["fold_results"][0]["baseline_diagnostics"]["baseline_labels_used_test_data"] = True

    result = analyze_prediction_matrix(matrix, bootstrap_resamples=100)

    assert result["status"] == "FAIL"
    assert any("baseline_provenance_invalid" in error for error in result["errors"])


def test_block_bootstrap_is_deterministic_and_detects_positive_paired_improvement():
    differences = [[0.02] * 20, [0.01] * 20]
    first = _block_bootstrap(differences, block_length=4, resamples=100, seed=7)
    second = _block_bootstrap(differences, block_length=4, resamples=100, seed=7)

    assert first == second
    assert first["observed_mean_improvement"] > 0
    assert first["confidence_interval_95"][0] > 0
    assert first["bootstrap_probability_improvement_positive"] == 1.0



def test_analyzer_marks_overlapping_zero_improvement_interval_as_inconclusive():
    matrix = _matrix()
    for fold in matrix["experiments"][0]["fold_results"]:
        for row in fold["test_predictions"]:
            row["model_probability_class_1"] = 0.5

    result = analyze_prediction_matrix(matrix, bootstrap_resamples=100, random_seed=7)

    constant = result["experiments"][0]["probability_loss_comparisons"][0]["metrics"]["log_loss"]
    assert constant["improvement_baseline_minus_model"] == 0
    assert constant["interval_evidence"] == "interval_includes_zero"
    assert constant["folds_with_positive_mean_improvement"] == 0
    assert constant["folds_with_zero_mean_improvement"] == 2

@pytest.mark.parametrize(
    "raw",
    [
        b'{"run_id":"first","run_id":"shadowed"}',
        b'{"metric":NaN}',
        b'{"metric":Infinity}',
        b'{"metric":-Infinity}',
    ],
)
def test_baseline_manifest_reader_rejects_ambiguous_json(tmp_path, raw):
    from app.research_baseline_analysis import _read_json

    path = tmp_path / "manifest.json"
    path.write_bytes(raw)

    with pytest.raises(ValueError):
        _read_json(path)


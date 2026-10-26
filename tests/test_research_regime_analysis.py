from __future__ import annotations

from copy import deepcopy

import pytest
from datetime import datetime, timedelta, timezone

from app.research_regime_analysis import analyze_prediction_regimes, _read_json


def _matrix():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    predictions = []
    for index in range(12):
        timestamp = start + timedelta(minutes=15 * index)
        label = index % 2
        predictions.append({
            "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "label": label,
            "model_probability_class_1": 0.7 if label else 0.3,
            "constant_probability_class_1": 0.5,
            "matured_refit_prevalence_probability_class_1": 0.51,
            "matured_refit_majority_class": 0,
            "lagged_direction_probability_class_1": 0.55 if label else 0.45,
            "lagged_direction_prediction": label,
        })
    fold = {
        "fold_index": 0,
        "test_start_utc": predictions[0]["timestamp_utc"],
        "test_end_utc": predictions[-1]["timestamp_utc"],
        "test_metrics": {"sample_count": len(predictions)},
        "test_predictions": predictions,
        "baseline_diagnostics": {
            "baseline_labels_used_test_data": False,
            "refit_rows_matured_before_test": 80,
        },
        "actionable": False,
    }
    return {
        "schema_version": 1,
        "run_id": "regime-fixture",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "horizon_bars": 4,
        "class_1_means": "UP",
        "fold_count": 1,
        "common_fold_test_windows": [{
            "fold_index": 0,
            "test_start_utc": predictions[0]["timestamp_utc"],
            "test_end_utc": predictions[-1]["timestamp_utc"],
            "test_rows": len(predictions),
        }],
        "experiments": [{
            "experiment_id": "regime-fixture-xgboost-all_features",
            "model_type": "xgboost",
            "feature_variant": "all_features",
            "fold_count": 1,
            "fold_results": [fold],
            "test_metrics_used_for_tuning": False,
            "actionable": False,
        }],
        "actionable": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
    }


def _ohlc_rows(matrix):
    test_start = datetime.fromisoformat(
        matrix["common_fold_test_windows"][0]["test_start_utc"].replace("Z", "+00:00")
    )
    history_times = [
        test_start - timedelta(minutes=15 * offset)
        for offset in range(45, 0, -1)
    ]
    test_times = [
        datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
        for row in matrix["experiments"][0]["fold_results"][0]["test_predictions"]
    ]
    timestamps = history_times + test_times
    close = 1.1
    rows = []
    for index, timestamp in enumerate(timestamps):
        amplitude = 0.00003 if index < 25 else 0.0003
        delta = amplitude if index % 2 else -amplitude * 0.8
        previous = close
        close = previous * (1 + delta)
        open_price = previous
        rows.append({
            "symbol": "EURUSD",
            "timeframe": "M15",
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "open": str(open_price),
            "high": str(max(open_price, close) * 1.0001),
            "low": str(min(open_price, close) * 0.9999),
            "close": str(close),
        })
    return rows


def test_regime_analysis_is_non_actionable_and_uses_prior_only_thresholds():
    matrix = _matrix()
    bars = _ohlc_rows(matrix)
    result = analyze_prediction_regimes(matrix, bars, volatility_window=10, bootstrap_resamples=100, random_seed=41)

    assert result["status"] in {"PASS", "PASS_WITH_WARNINGS"}
    assert result["errors"] == []
    assert result["actionable"] is False
    assert result["execution_connected"] is False
    assert result["automatic_model_selection"] is False
    assert result["test_metrics_used_for_selection"] is False
    assert result["bootstrap"]["method"] == "paired_circular_moving_block_bootstrap_stratified_by_walk_forward_fold_conditioned_on_regime"
    threshold = result["fold_thresholds"][0]
    assert threshold["pretest_volatility_observations"] >= 10
    assert threshold["low_threshold_q25"] <= threshold["high_threshold_q75"]
    assert sum(threshold["regime_counts"].values()) == 12
    metrics = result["experiments"][0]["regime_metrics"]
    assert sum(metrics[name]["sample_count"] for name in metrics) == 12
    assert all(metrics[name]["model"] is None or metrics[name]["model"]["sample_count"] > 0 for name in metrics)
    for regime_result in metrics.values():
        if regime_result["sample_count"]:
            uncertainty = regime_result["paired_bootstrap_uncertainty"]["constant_probability_0_5"]["log_loss"]
            assert uncertainty["bootstrap_resamples"] == 100
            assert uncertainty["folds_compared"] == 1
            assert len(uncertainty["confidence_interval_95"]) == 2
            assert uncertainty["interval_evidence"] in {
                "positive_interval_excludes_zero", "negative_interval_excludes_zero", "interval_includes_zero"
            }
            assert uncertainty["method"] == "paired_circular_moving_block_bootstrap_stratified_by_walk_forward_fold_conditioned_on_regime"


def test_regime_thresholds_do_not_use_test_period_volatility():
    matrix = _matrix()
    original_bars = _ohlc_rows(matrix)
    changed_bars = deepcopy(original_bars)
    test_start = datetime.fromisoformat(
        matrix["common_fold_test_windows"][0]["test_start_utc"].replace("Z", "+00:00")
    )
    for row in changed_bars:
        timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        if timestamp >= test_start:
            original_close = float(row["close"])
            modified_close = original_close * (1.0 + (0.02 if timestamp.minute in (0, 30) else -0.015))
            row["close"] = str(modified_close)
            open_price = float(row["open"])
            row["high"] = str(max(open_price, modified_close) * 1.0001)
            row["low"] = str(min(open_price, modified_close) * 0.9999)

    first = analyze_prediction_regimes(matrix, original_bars, volatility_window=10, bootstrap_resamples=100, random_seed=41)
    second = analyze_prediction_regimes(matrix, changed_bars, volatility_window=10, bootstrap_resamples=100, random_seed=41)

    assert first["status"] in {"PASS", "PASS_WITH_WARNINGS"}
    assert second["status"] in {"PASS", "PASS_WITH_WARNINGS"}
    first_threshold = first["fold_thresholds"][0]
    second_threshold = second["fold_thresholds"][0]
    assert first_threshold["low_threshold_q25"] == second_threshold["low_threshold_q25"]
    assert first_threshold["high_threshold_q75"] == second_threshold["high_threshold_q75"]


def test_regime_bootstrap_is_deterministic_and_does_not_enable_selection():
    matrix = _matrix()
    bars = _ohlc_rows(matrix)
    first = analyze_prediction_regimes(matrix, bars, volatility_window=10, bootstrap_resamples=100, random_seed=9)
    second = analyze_prediction_regimes(matrix, bars, volatility_window=10, bootstrap_resamples=100, random_seed=9)

    first_metrics = first["experiments"][0]["regime_metrics"]
    second_metrics = second["experiments"][0]["regime_metrics"]
    for regime in first_metrics:
        if first_metrics[regime]["sample_count"]:
            first_interval = first_metrics[regime]["paired_bootstrap_uncertainty"]["constant_probability_0_5"]["brier_score"]["confidence_interval_95"]
            second_interval = second_metrics[regime]["paired_bootstrap_uncertainty"]["constant_probability_0_5"]["brier_score"]["confidence_interval_95"]
            assert first_interval == second_interval
    assert first["automatic_model_selection"] is False
    assert first["test_metrics_used_for_selection"] is False
    assert first["actionable"] is False
    assert first["execution_connected"] is False


def test_regime_analysis_fails_closed_for_invalid_bootstrap_count():
    result = analyze_prediction_regimes(_matrix(), _ohlc_rows(_matrix()), volatility_window=10, bootstrap_resamples=99)

    assert result["status"] == "FAIL"
    assert any("bootstrap_resamples_must_be_between_100_and_20000" in error for error in result["errors"])
    assert result["actionable"] is False


def test_regime_analysis_fails_closed_for_malformed_ohlc():
    matrix = _matrix()
    bars = _ohlc_rows(matrix)
    bars[0]["close"] = "nan"

    result = analyze_prediction_regimes(matrix, bars, volatility_window=10)

    assert result["status"] == "FAIL"
    assert result["errors"]
    assert result["actionable"] is False
    assert result["execution_connected"] is False


@pytest.mark.parametrize("payload", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}'])
def test_read_json_rejects_ambiguous_or_nonstandard_json(tmp_path, payload):
    path = tmp_path / "input.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        _read_json(path)

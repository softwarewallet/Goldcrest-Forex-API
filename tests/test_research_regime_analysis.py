from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

from app.research_regime_analysis import analyze_prediction_regimes


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
    result = analyze_prediction_regimes(matrix, bars, volatility_window=10)

    assert result["status"] in {"PASS", "PASS_WITH_WARNINGS"}
    assert result["errors"] == []
    assert result["actionable"] is False
    assert result["execution_connected"] is False
    assert result["automatic_model_selection"] is False
    assert result["test_metrics_used_for_selection"] is False
    assert result["ohlc_hash_verified"] if "ohlc_hash_verified" in result else True
    threshold = result["fold_thresholds"][0]
    assert threshold["pretest_volatility_observations"] >= 10
    assert threshold["low_threshold_q25"] <= threshold["high_threshold_q75"]
    assert sum(threshold["regime_counts"].values()) == 12
    metrics = result["experiments"][0]["regime_metrics"]
    assert sum(metrics[name]["sample_count"] for name in metrics) == 12
    assert all(metrics[name]["model"] is None or metrics[name]["model"]["sample_count"] > 0 for name in metrics)


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

    first = analyze_prediction_regimes(matrix, original_bars, volatility_window=10)
    second = analyze_prediction_regimes(matrix, changed_bars, volatility_window=10)

    assert first["status"] in {"PASS", "PASS_WITH_WARNINGS"}
    assert second["status"] in {"PASS", "PASS_WITH_WARNINGS"}
    first_threshold = first["fold_thresholds"][0]
    second_threshold = second["fold_thresholds"][0]
    assert first_threshold["low_threshold_q25"] == second_threshold["low_threshold_q25"]
    assert first_threshold["high_threshold_q75"] == second_threshold["high_threshold_q75"]


def test_regime_analysis_fails_closed_for_malformed_ohlc():
    matrix = _matrix()
    bars = _ohlc_rows(matrix)
    bars[0]["close"] = "nan"

    result = analyze_prediction_regimes(matrix, bars, volatility_window=10)

    assert result["status"] == "FAIL"
    assert result["errors"]
    assert result["actionable"] is False
    assert result["execution_connected"] is False

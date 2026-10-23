from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.research_temporal_stability import analyze_prediction_temporal_stability


def _fixture():
    first = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    timestamps = []
    for index in range(12):
        if index < 5:
            timestamp = first + timedelta(minutes=15 * index)
        elif index == 5:
            timestamp = timestamps[4] + timedelta(hours=24)
        else:
            timestamp = timestamps[5] + timedelta(minutes=15 * (index - 5))
        timestamps.append(timestamp)

    bars = [{
        "symbol": "EURUSD",
        "timeframe": "M15",
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "open": "1.1000",
        "high": "1.1010",
        "low": "1.0990",
        "close": "1.1005",
    } for timestamp in timestamps]

    def prediction(index, label, probability):
        return {
            "timestamp_utc": timestamps[index].isoformat().replace("+00:00", "Z"),
            "label": label,
            "model_probability_class_1": probability,
            "constant_probability_class_1": 0.5,
            "matured_refit_prevalence_probability_class_1": 0.52,
            "lagged_direction_probability_class_1": 0.48,
        }

    matrix = {
        "schema_version": 1,
        "run_id": "temporal-stability-fixture",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "horizon_bars": 2,
        "experiments": [{
            "experiment_id": "fixture-xgboost",
            "model_type": "xgboost",
            "fold_results": [
                {"fold_index": 0, "test_predictions": [
                    prediction(2, 1, 0.65),
                    prediction(3, 0, 0.35),
                ]},
                {"fold_index": 1, "test_predictions": [
                    prediction(5, 1, 0.60),
                    prediction(6, 0, 0.40),
                ]},
            ],
        }],
    }
    return matrix, bars


def test_temporal_stability_identifies_forward_horizon_crossing_long_gap():
    matrix, bars = _fixture()

    result = analyze_prediction_temporal_stability(matrix, bars)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["errors"] == []
    assert result["long_gap_count"] == 1
    assert result["non_weekend_long_gap_count"] == 1
    assert result["prediction_count"] == 4
    assert result["prediction_horizon_crosses_long_gap_count"] == 1
    assert result["prediction_horizon_crosses_long_gap_fraction"] == 0.25
    experiment = result["experiments"][0]
    assert experiment["fold_count"] == 2
    assert experiment["gap_exposure"]["metrics_by_group"]["horizon_crosses_long_gap"]["sample_count"] == 1
    assert experiment["gap_exposure"]["metrics_by_group"]["horizon_does_not_cross_long_gap"]["sample_count"] == 3
    assert experiment["fold_consistency"]["constant_probability_0_5"]["log_loss_baseline_minus_model"]["fold_count"] == 2
    assert result["actionable"] is False
    assert result["automatic_model_selection"] is False
    assert result["execution_connected"] is False


def test_temporal_stability_warns_when_no_gap_exceeds_threshold():
    matrix, bars = _fixture()
    # Remove the large interval while preserving unique, strictly increasing timestamps.
    original_to_index = {
        datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")): index
        for index, row in enumerate(bars)
    }
    anchor = datetime.fromisoformat(bars[4]["timestamp"].replace("Z", "+00:00"))
    for index in range(5, len(bars)):
        timestamp = anchor + timedelta(minutes=15 * (index - 4))
        bars[index]["timestamp"] = timestamp.isoformat().replace("+00:00", "Z")
    new_timestamps = [
        datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        for row in bars
    ]
    for experiment in matrix["experiments"]:
        for fold in experiment["fold_results"]:
            for prediction in fold["test_predictions"]:
                old_timestamp = datetime.fromisoformat(prediction["timestamp_utc"].replace("Z", "+00:00"))
                index = original_to_index[old_timestamp]
                prediction["timestamp_utc"] = new_timestamps[index].isoformat().replace("+00:00", "Z")

    result = analyze_prediction_temporal_stability(matrix, bars)

    assert result["status"] == "PASS"
    assert result["long_gap_count"] == 0
    assert result["warnings"] == []
    assert result["prediction_horizon_crosses_long_gap_count"] == 0


def test_temporal_stability_fails_closed_when_prediction_timestamp_is_missing():
    matrix, bars = _fixture()
    matrix["experiments"][0]["fold_results"][0]["test_predictions"][0]["timestamp_utc"] = (
        "2026-02-01T00:00:00Z"
    )

    result = analyze_prediction_temporal_stability(matrix, bars)

    assert result["status"] == "FAIL"
    assert "prediction_timestamps_missing_from_ohlc:1" in result["errors"]
    assert result["actionable"] is False
    assert result["execution_connected"] is False

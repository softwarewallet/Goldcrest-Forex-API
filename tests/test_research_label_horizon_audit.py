from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.research_label_horizon_audit import analyze_label_horizon, _load_json_object


def _fixture():
    first = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    timestamps = []
    for index in range(12):
        if index < 5:
            timestamp = first + timedelta(minutes=15 * index)
        elif index == 5:
            timestamps.append(timestamps[4] + timedelta(hours=24))
            continue
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
        "run_id": "label-horizon-fixture",
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
    return matrix, bars, timestamps


def test_label_horizon_audit_measures_elapsed_time_without_changing_row_horizon():
    matrix, bars, timestamps = _fixture()

    result = analyze_label_horizon(matrix, bars, extended_time_threshold_hours=8.0)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert result["errors"] == []
    assert result["horizon_bars"] == 2
    assert result["nominal_elapsed_hours"] == 0.5
    assert result["prediction_model_row_count"] == 4
    assert result["unique_prediction_timestamp_count"] == 4
    assert result["elapsed_time_distribution_by_model_row"]["sample_count"] == 4
    assert result["elapsed_time_distribution_by_unique_timestamp"]["sample_count"] == 4
    assert result["counts_by_elapsed_bucket_model_rows"] == {
        "nominal_or_within_tolerance": 3,
        "extended_up_to_threshold": 0,
        "long_extension_over_threshold": 1,
    }
    assert result["prediction_rows_beyond_nominal_horizon_count"] == 1
    assert result["prediction_rows_over_extended_time_threshold_count"] == 1
    assert result["longest_elapsed_horizon_examples"][0]["elapsed_hours"] == 24.25
    assert result["longest_elapsed_horizon_examples"][0]["prediction_timestamp_utc"] == (
        timestamps[3].isoformat().replace("+00:00", "Z")
    )
    experiment = result["experiments"][0]
    assert experiment["counts_by_elapsed_bucket"] == {
        "nominal_or_within_tolerance": 3,
        "extended_up_to_threshold": 0,
        "long_extension_over_threshold": 1,
    }
    assert experiment["metrics_by_elapsed_bucket"]["long_extension_over_threshold"]["sample_count"] == 1
    assert experiment["metrics_by_elapsed_bucket"]["long_extension_over_threshold"]["model"]["sample_count"] == 1
    assert result["automatic_model_selection"] is False
    assert result["test_metrics_used_for_selection"] is False
    assert result["actionable"] is False
    assert result["execution_connected"] is False


def test_label_horizon_audit_counts_unique_timestamps_separately_from_model_rows():
    matrix, bars, _ = _fixture()
    other = matrix["experiments"][0]
    clone = {
        **other,
        "experiment_id": "fixture-lightgbm",
        "model_type": "lightgbm",
        "fold_results": [
            {
                "fold_index": fold["fold_index"],
                "test_predictions": [dict(row) for row in fold["test_predictions"]],
            }
            for fold in other["fold_results"]
        ],
    }
    matrix["experiments"].append(clone)

    result = analyze_label_horizon(matrix, bars)

    assert result["prediction_model_row_count"] == 8
    assert result["unique_prediction_timestamp_count"] == 4
    assert result["elapsed_time_distribution_by_model_row"]["sample_count"] == 8
    assert result["elapsed_time_distribution_by_unique_timestamp"]["sample_count"] == 4
    assert result["experiments"][0]["sample_count"] == 4
    assert result["experiments"][1]["sample_count"] == 4


def test_label_horizon_audit_fails_closed_when_target_row_is_unavailable():
    matrix, bars, _ = _fixture()
    last_timestamp = bars[-1]["timestamp"]
    matrix["experiments"][0]["fold_results"][1]["test_predictions"][1]["timestamp_utc"] = last_timestamp

    result = analyze_label_horizon(matrix, bars)

    assert result["status"] == "FAIL"
    assert any("label_horizon_target_missing_from_ohlc" in error for error in result["errors"])
    assert result["actionable"] is False
    assert result["execution_connected"] is False


def test_label_horizon_audit_fails_closed_for_out_of_order_bars():
    matrix, bars, _ = _fixture()
    bars[0], bars[1] = bars[1], bars[0]

    result = analyze_label_horizon(matrix, bars)

    assert result["status"] == "FAIL"
    assert any("strictly_increasing" in error for error in result["errors"])


def test_label_horizon_audit_rejects_unsupported_timeframe():
    matrix, bars, _ = _fixture()
    matrix["timeframe"] = "M2"

    result = analyze_label_horizon(matrix, bars)

    assert result["status"] == "FAIL"
    assert "matrix_timeframe_unsupported" in result["errors"]


def test_label_horizon_audit_reports_repeated_prediction_timestamps():
    matrix, bars, _ = _fixture()
    fold = matrix["experiments"][0]["fold_results"][0]
    fold["test_predictions"].append(dict(fold["test_predictions"][0]))

    result = analyze_label_horizon(matrix, bars)

    assert result["status"] == "PASS_WITH_WARNINGS"
    assert "repeated_prediction_timestamps_within_experiment_review_fold_overlap" in result["warnings"]
    assert result["experiments"][0]["repeated_timestamp_occurrence_count"] == 1
    assert result["repeated_prediction_timestamp_occurrence_count"] == 1


@pytest.mark.parametrize("raw", [
    b'{"status":"PASS","status":"FAIL"}', b'{"metric":NaN}',
    b'{"metric":Infinity}', b'{"metric":-Infinity}',
])
def test_strict_json_loader_rejects_ambiguous_inputs(raw):
    with pytest.raises(ValueError):
        _load_json_object(raw, source="label-horizon.json")

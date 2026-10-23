from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import research_dataset_builder, research_pipeline
from app.historical_tabular_research import load_historical_dataset
from app.research_validation import validate_prepared_dataset, summarize_experiment_matrix


def _write_raw_ohlc(path: Path) -> list[datetime]:
    first = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    timestamps = [first + timedelta(minutes=15 * i) for i in range(5)]
    timestamps.append(timestamps[-1] + timedelta(hours=24))
    timestamps.extend(timestamps[-1] + timedelta(minutes=15 * i) for i in (1, 2))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "open", "high", "low", "close"],
            lineterminator="\n",
        )
        writer.writeheader()
        for index, timestamp in enumerate(timestamps):
            close = 1.1 + index * 0.001
            writer.writerow({
                "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                "open": close,
                "high": close + 0.0005,
                "low": close - 0.0005,
                "close": close,
            })
    return timestamps


def _stub_features(monkeypatch, row_count: int):
    monkeypatch.setattr(
        research_dataset_builder,
        "build_causal_ohlc_features",
        lambda bars: SimpleNamespace(
            feature_names=("fixture_feature",),
            rows=[{"fixture_feature": float(index)} for index in range(row_count)],
            indicator_backend="fixture",
        ),
    )


def test_dataset_builder_preserves_timeline_and_marks_long_elapsed_labels(tmp_path, monkeypatch):
    raw = tmp_path / "raw.csv"
    timestamps = _write_raw_ohlc(raw)
    _stub_features(monkeypatch, len(timestamps))
    dataset = tmp_path / "capped.csv"
    manifest_path = tmp_path / "capped.manifest.json"

    manifest = research_dataset_builder.prepare_research_dataset(
        raw, dataset, manifest_path,
        symbol="EURUSD", timeframe="M15", horizon_bars=1, class_1_means="UP",
        data_source="fixture", source_version="fixture-v1",
        retrieved_at_utc="2026-01-06T00:00:00Z", terms_reference="fixture terms",
        max_label_elapsed_hours=1.0,
    )

    with dataset.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(rows) == len(timestamps) - 1
    assert [row["timestamp"] for row in rows] == [
        timestamp.isoformat().replace("+00:00", "Z") for timestamp in timestamps[:-1]
    ]
    assert sum(row["label_eligible"] == "1" for row in rows) == len(rows) - 1
    assert sum(row["label_eligible"] == "0" for row in rows) == 1
    long_row = next(row for row in rows if row["label_eligible"] == "0")
    assert float(long_row["label_elapsed_hours"]) == 24.0
    assert long_row["label_available_at"] == timestamps[5].isoformat().replace("+00:00", "Z")
    assert manifest["output_rows"] == len(rows)
    assert stored_manifest["label_eligibility"]["eligible_rows"] == len(rows) - 1
    assert stored_manifest["label_eligibility"]["ineligible_rows"] == 1
    assert stored_manifest["label_eligibility"]["all_rows_preserved_for_fixed_temporal_folds"] is True
    assert stored_manifest["label_eligibility"]["labels_rewritten"] is False
    assert manifest["feature_names"] == ["fixture_feature"]


def test_default_dataset_builder_keeps_legacy_columns_when_no_cap_is_given(tmp_path, monkeypatch):
    raw = tmp_path / "raw.csv"
    timestamps = _write_raw_ohlc(raw)
    _stub_features(monkeypatch, len(timestamps))

    dataset = tmp_path / "baseline.csv"
    research_dataset_builder.prepare_research_dataset(
        raw, dataset, tmp_path / "baseline.manifest.json",
        symbol="EURUSD", timeframe="M15", horizon_bars=1, class_1_means="UP",
        data_source="fixture", source_version="fixture-v1",
        retrieved_at_utc="2026-01-06T00:00:00Z", terms_reference="fixture terms",
    )

    with dataset.open("r", encoding="utf-8", newline="") as handle:
        columns = next(csv.reader(handle))
    assert "label_eligible" not in columns
    assert "label_elapsed_hours" not in columns


@pytest.mark.parametrize("cap", [0, -1, float("nan"), float("inf"), True])
def test_dataset_builder_rejects_invalid_elapsed_time_cap(tmp_path, cap):
    with pytest.raises(ValueError, match="max_label_elapsed_hours"):
        research_dataset_builder.prepare_research_dataset(
            tmp_path / "does-not-need-to-exist.csv",
            tmp_path / "unused.csv", tmp_path / "unused.json",
            symbol="EURUSD", timeframe="M15", horizon_bars=1, class_1_means="UP",
            data_source="fixture", source_version="fixture-v1",
            retrieved_at_utc="2026-01-06T00:00:00Z", terms_reference="fixture terms",
            max_label_elapsed_hours=cap,
        )


def test_pipeline_feature_discovery_and_loader_exclude_metadata_from_features(tmp_path):
    dataset = tmp_path / "capped.csv"
    dataset.write_text(
        "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,"
        "label_elapsed_hours,label_eligible,feature_x\n"
        "EURUSD,M15,2026-01-05T12:00:00Z,1.1,1.2,1.0,1.1,1,"
        "2026-01-05T13:00:00Z,1.0,1,0.25\n"
        "EURUSD,M15,2026-01-05T12:15:00Z,1.1,1.2,1.0,1.1,0,"
        "2026-01-06T12:15:00Z,24.0,0,0.50\n",
        encoding="utf-8",
    )

    features = research_pipeline._feature_names_from_dataset(dataset)
    bars, examples, _ = load_historical_dataset(dataset, features)

    assert features == ["feature_x"]
    assert len(bars) == len(examples) == 2
    assert [example.label_eligible for example in examples] == [True, False]
    assert examples[0].features == {"feature_x": 0.25}
    assert examples[1].features == {"feature_x": 0.50}


def test_completed_dataset_validation_checks_eligibility_metadata_and_cap(tmp_path):
    dataset = tmp_path / "capped.csv"
    dataset.write_text(
        "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,"
        "label_elapsed_hours,label_eligible,feature_x\n"
        "EURUSD,M15,2026-01-05T12:00:00Z,1.1,1.2,1.0,1.1,1,"
        "2026-01-05T13:00:00Z,1.0,1,0.25\n"
        "EURUSD,M15,2026-01-05T12:15:00Z,1.1,1.2,1.0,1.1,0,"
        "2026-01-06T12:15:00Z,24.0,0,0.50\n",
        encoding="utf-8",
    )

    result = validate_prepared_dataset(
        dataset,
        expected_symbol="EURUSD",
        expected_timeframe="M15",
        max_label_elapsed_hours=1.0,
    )

    assert result["status"] == "PASS"
    assert result["feature_names"] == ["feature_x"]
    assert result["label_eligibility"] == {
        "mode": "elapsed_time_cap",
        "max_label_elapsed_hours": 1.0,
        "eligible_row_count": 1,
        "ineligible_row_count": 1,
        "metadata_row_count": 2,
    }

    dataset.write_text(
        dataset.read_text(encoding="utf-8").replace("24.0,0,0.50", "24.0,1,0.50"),
        encoding="utf-8",
    )
    invalid = validate_prepared_dataset(dataset, max_label_elapsed_hours=1.0)
    assert invalid["status"] == "FAIL"
    assert any("label_eligibility_cap_mismatch" in error for error in invalid["errors"])


def test_baseline_dataset_validation_does_not_require_optional_eligibility_columns(tmp_path):
    dataset = tmp_path / "baseline.csv"
    dataset.write_text(
        "symbol,timeframe,timestamp,open,high,low,close,label,label_available_at,feature_x\n"
        "EURUSD,M15,2026-01-05T12:00:00Z,1.1,1.2,1.0,1.1,1,"
        "2026-01-05T13:00:00Z,0.25\n",
        encoding="utf-8",
    )

    result = validate_prepared_dataset(dataset, max_label_elapsed_hours=None)

    assert result["status"] == "PASS"
    assert result["feature_names"] == ["feature_x"]
    assert result["label_eligibility"]["mode"] == "all_labels_included"
    assert result["label_eligibility"]["eligible_row_count"] == 1
    assert result["label_eligibility"]["ineligible_row_count"] == 0


def test_experiment_matrix_validation_accepts_fewer_test_rows_when_cap_is_audited(tmp_path):
    report = {
        "schema_version": 1,
        "run_id": "elapsed-cap-validation-fixture",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "fold_count": 1,
        "label_eligibility_mode": "elapsed_time_cap",
        "max_label_elapsed_hours": 1.0,
        "actionable": False,
        "automatic_winner_selection": False,
        "test_metrics_used_for_selection": False,
        "common_fold_test_windows": [{
            "fold_index": 0,
            "test_start_utc": "2026-01-01T04:00:00Z",
            "test_end_utc": "2026-01-01T06:00:00Z",
            "test_rows": 3,
        }],
        "experiments": [{
            "experiment_id": "fixture-xgboost",
            "model_type": "xgboost",
            "feature_variant": "all_features",
            "actionable": False,
            "test_metrics_used_for_tuning": False,
            "fold_results": [{
                "fold_index": 0,
                "train_end_utc": "2026-01-01T01:00:00Z",
                "validation_start_utc": "2026-01-01T02:00:00Z",
                "validation_end_utc": "2026-01-01T03:00:00Z",
                "test_start_utc": "2026-01-01T04:00:00Z",
                "test_end_utc": "2026-01-01T06:00:00Z",
                "test_rows_available_before_eligibility": 3,
                "test_rows_excluded_ineligible": 1,
                "test_predictions": [{"timestamp_utc": "2026-01-01T04:00:00Z"}, {"timestamp_utc": "2026-01-01T06:00:00Z"}],
                "test_metrics": {
                    "sample_count": 2,
                    "positive_count": 1,
                    "negative_count": 1,
                    "brier_score": 0.25,
                    "log_loss": 0.69,
                    "accuracy_at_fixed_half": 0.5,
                    "expected_calibration_error": 0.1,
                    "prevalence": 0.5,
                    "mean_predicted_probability": 0.5,
                },
            }],
        }],
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = summarize_experiment_matrix(path)

    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert result["experiment_summaries"][0]["test_rows_counted_by_fold"] == 2


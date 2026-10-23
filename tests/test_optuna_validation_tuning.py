from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest

from app.model_walk_forward import TimedBinaryExample
from app.optuna_validation_tuning import tune_on_validation, write_tuning_report


def make_rows(start_index: int, count: int, *, delayed_last_label: bool = False):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for offset in range(count):
        index = start_index + offset
        timestamp = start + timedelta(minutes=15 * index)
        label = int(index % 2)
        features = {
            "momentum": float((index % 7) / 7),
            "volatility": float((index % 5) / 5),
        }
        available = timestamp + timedelta(minutes=15)
        if delayed_last_label and offset == count - 1:
            available = timestamp + timedelta(minutes=30)
        rows.append(TimedBinaryExample(
            timestamp=timestamp,
            features=features,
            label=label,
            label_available_at=available,
        ))
    return rows


@pytest.mark.parametrize("model_type", ["xgboost", "lightgbm"])
def test_tuner_uses_validation_only_and_returns_audit(model_type, tmp_path):
    train = make_rows(0, 20, delayed_last_label=True)
    validation = make_rows(25, 8)

    result = tune_on_validation(
        train, validation,
        feature_names=("momentum", "volatility"),
        model_type=model_type,
        n_trials=1,
        seed=17,
        study_name=f"test-{model_type}",
    )

    assert result.model_type == model_type
    assert result.trial_count_requested == 1
    assert result.trial_count_completed == 1
    assert result.train_rows_available == 19
    assert result.train_rows_excluded_unmatured == 1
    assert result.validation_rows == 8
    assert result.test_partition_used is False
    assert result.actionable is False
    assert result.best_validation_log_loss >= 0
    assert result.best_params
    path = tmp_path / "tuning.json"
    assert write_tuning_report(result, str(path)) == str(path)
    report = path.read_text(encoding="utf-8")
    assert '"actionable": false' in report
    assert '"test_partition_used": false' in report


def test_tuner_rejects_validation_overlap_and_bad_trial_count():
    train = make_rows(0, 12)
    overlapping_validation = make_rows(10, 6)
    with pytest.raises(ValueError, match="strictly after"):
        tune_on_validation(
            train, overlapping_validation,
            feature_names=("momentum", "volatility"),
            model_type="xgboost", n_trials=1,
        )
    with pytest.raises(ValueError, match="between 1 and 100"):
        tune_on_validation(
            train, make_rows(20, 6),
            feature_names=("momentum", "volatility"),
            model_type="xgboost", n_trials=101,
        )


def test_tuner_rejects_validation_without_both_classes():
    train = make_rows(0, 12)
    validation = make_rows(20, 6)
    validation = [row.__class__(
        timestamp=row.timestamp, features=row.features, label=1,
        label_available_at=row.label_available_at,
    ) for row in validation]
    with pytest.raises(ValueError, match="both classes"):
        tune_on_validation(
            train, validation,
            feature_names=("momentum", "volatility"),
            model_type="lightgbm", n_trials=1,
        )


def test_tuner_excludes_elapsed_time_ineligible_train_and_validation_rows():
    train = make_rows(0, 20, delayed_last_label=True)
    train = [
        replace(row, label_eligible=False) if index == 4 else row
        for index, row in enumerate(train)
    ]
    validation = make_rows(25, 8)
    validation = [
        replace(row, label_eligible=False) if index in {0, 1} else row
        for index, row in enumerate(validation)
    ]

    result = tune_on_validation(
        train, validation,
        feature_names=("momentum", "volatility"),
        model_type="xgboost",
        n_trials=1,
        seed=19,
        study_name="test-elapsed-time-eligibility",
    )

    assert result.train_rows_available == 18
    assert result.train_rows_excluded_unmatured == 1
    assert result.train_rows_excluded_ineligible == 1
    assert result.validation_rows == 6
    assert result.validation_rows_excluded_ineligible == 2
    assert result.actionable is False


def test_tuner_rejects_non_boolean_label_eligibility():
    train = make_rows(0, 12)
    train[3] = replace(train[3], label_eligible="no")

    with pytest.raises(ValueError, match="label_eligible must be a boolean"):
        tune_on_validation(
            train, make_rows(15, 6),
            feature_names=("momentum", "volatility"),
            model_type="xgboost", n_trials=1,
        )


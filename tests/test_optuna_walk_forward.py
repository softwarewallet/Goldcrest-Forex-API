from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.evaluation import ForexBar
from app.model_walk_forward import TimedBinaryExample
from app.optuna_walk_forward import run_optuna_walk_forward, write_optuna_walk_forward_report


def fixture_data(count=140, horizon=2):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = []
    examples = []
    for index in range(count):
        timestamp = start + timedelta(minutes=15 * index)
        close = 1.1 + index * 0.00001
        bars.append(ForexBar(
            symbol="EUR/USD",
            timeframe="M15",
            timestamp=timestamp,
            open=close,
            high=close + 0.0002,
            low=close - 0.0002,
            close=close,
        ))
        examples.append(TimedBinaryExample(
            timestamp=timestamp,
            features={
                "momentum": float(index % 7) / 7,
                "volatility": float(index % 5) / 5,
            },
            label=index % 2,
            label_available_at=timestamp + timedelta(minutes=15 * horizon),
        ))
    return bars, examples


@pytest.mark.parametrize("model_type", ["xgboost", "lightgbm"])
def test_optuna_walk_forward_tunes_then_scores_untouched_test(model_type, tmp_path):
    bars, examples = fixture_data()
    report = run_optuna_walk_forward(
        bars,
        examples,
        feature_names=("momentum", "volatility"),
        run_id=f"walk-forward-{model_type}",
        dataset_sha256="a" * 64,
        model_type=model_type,
        initial_train_bars=60,
        validation_bars=30,
        test_bars=25,
        step_bars=25,
        horizon_bars=2,
        n_trials_per_fold=1,
        random_seed=29,
        bootstrap_resamples=100,
    )

    assert report.actionable is False
    assert report.test_metrics_used_for_tuning is False
    assert report.fold_count >= 2
    assert len(report.folds) == report.fold_count
    for fold in report.folds:
        assert fold.actionable is False
        assert fold.trials_completed == 1
        assert fold.best_params
        assert fold.test_metrics["sample_count"] > 0
        assert fold.test_start_utc > fold.validation_end_utc
        assert fold.refit_rows_available >= fold.train_rows_available_for_tuning

    destination = tmp_path / "walk-forward.json"
    assert write_optuna_walk_forward_report(report, destination) == str(destination)
    import json
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["actionable"] is False
    assert saved["test_metrics_used_for_tuning"] is False
    assert len(saved["report_payload_sha256"]) == 64


def test_walk_forward_rejects_validation_labels_not_matured_before_test():
    bars, examples = fixture_data(count=120)
    # Make one validation outcome unavailable until after its test window begins.
    # With the known fold layout, row 70 is in the first fold's validation partition.
    row = examples[70]
    examples[70] = TimedBinaryExample(
        timestamp=row.timestamp,
        features=row.features,
        label=row.label,
        label_available_at=row.timestamp + timedelta(days=3),
    )
    with pytest.raises(ValueError, match="validation labels not matured by test start"):
        run_optuna_walk_forward(
            bars,
            examples,
            feature_names=("momentum", "volatility"),
            run_id="late-validation-label",
            dataset_sha256="b" * 64,
            model_type="xgboost",
            initial_train_bars=60,
            validation_bars=30,
            test_bars=25,
            step_bars=25,
            horizon_bars=2,
            n_trials_per_fold=1,
            bootstrap_resamples=100,
        )


def test_walk_forward_rejects_misaligned_bar_and_feature_timestamps():
    bars, examples = fixture_data(count=120)
    examples[0] = TimedBinaryExample(
        timestamp=examples[0].timestamp + timedelta(minutes=1),
        features=examples[0].features,
        label=examples[0].label,
        label_available_at=examples[0].label_available_at,
    )
    with pytest.raises(ValueError, match="exactly aligned"):
        run_optuna_walk_forward(
            bars,
            examples,
            feature_names=("momentum", "volatility"),
            run_id="misaligned-data",
            dataset_sha256="c" * 64,
            model_type="lightgbm",
            initial_train_bars=60,
            validation_bars=30,
            test_bars=25,
            step_bars=25,
            horizon_bars=2,
            n_trials_per_fold=1,
            bootstrap_resamples=100,
        )

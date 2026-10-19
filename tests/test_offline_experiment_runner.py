from dataclasses import dataclass

import pytest
from datetime import datetime, timedelta, timezone

from app.offline_experiment_runner import run_offline_experiment
from app.model_walk_forward import TimedBinaryExample
from app.prediction_ledger import PredictionLedger


@dataclass(frozen=True)
class Bar:
    timestamp: datetime


@dataclass(frozen=True)
class Fold:
    fold_index: int
    train: tuple
    validation: tuple
    test: tuple


@dataclass(frozen=True)
class Plan:
    symbol: str
    timeframe: str
    folds: tuple


@dataclass(frozen=True)
class Output:
    probability_class_1: float
    actionable: bool = False


class ConstantModel:
    model_type = "fixture_model"
    model_version = "fixture-v1"
    actionable = False

    def fit(self, feature_rows, labels, **kwargs):
        assert len(feature_rows) == len(labels)
        self.trained = True
        return self

    def predict(self, feature_row):
        assert self.trained
        return Output(0.7)


def test_runner_trains_records_compares_and_writes_non_actionable_report(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=5 * i) for i in range(36)]
    examples = [
        TimedBinaryExample(t, {"x": float(i % 3)}, i % 2, t)
        for i, t in enumerate(times)
    ]
    plan = Plan(
        "EUR/USD", "M5",
        (Fold(0, tuple(Bar(t) for t in times[:20]),
              tuple(Bar(t) for t in times[20:28]),
              tuple(Bar(t) for t in times[28:36])),),
    )
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    report_path = tmp_path / "reports" / "run.json"

    result = run_offline_experiment(
        plan, examples, feature_names=("x",),
        manifest={"run_id": "fixture-run", "dataset_id": "synthetic"},
        model_factories={"candidate": ConstantModel},
        ledger=ledger,
        evaluation_contract={
            "dataset_id": "synthetic-fixture",
            "label_definition": "alternating labels, test mechanics only",
            "timeframe": "M5",
            "horizon_bars": 1,
            "cost_assumptions": "not applicable to binary probability metrics",
        },
        exclusions={"unlabeled": 0, "invalid": 0},
        block_length=1, bootstrap_resamples=100, random_seed=7,
        report_path=report_path,
    )

    assert result.run_id == "fixture-run"
    assert result.actionable is False
    assert result.prediction_rows_inserted == 32
    assert result.report_path == str(report_path)
    assert report_path.is_file()
    assert result.report["interpretation"]["actionable"] is False
    assert len(result.report["paired_comparisons"]) == 2
    assert set(result.report["fold_results"]) == {"candidate", "training_prevalence"}
    # An identical retry is idempotent: it inserts no additional ledger rows.
    second = run_offline_experiment(
        plan, examples, feature_names=("x",),
        manifest={"run_id": "fixture-run", "dataset_id": "synthetic"},
        model_factories={"candidate": ConstantModel},
        ledger=ledger,
        evaluation_contract={
            "dataset_id": "synthetic-fixture",
            "label_definition": "alternating labels, test mechanics only",
            "timeframe": "M5",
            "horizon_bars": 1,
            "cost_assumptions": "not applicable to binary probability metrics",
        },
        exclusions={"unlabeled": 0, "invalid": 0},
        block_length=1, bootstrap_resamples=100, random_seed=7,
    )
    assert second.prediction_rows_inserted == 0
    assert ledger.get_run_status("fixture-run")["status"] == "completed"



def test_runner_rejects_fold_timestamps_missing_from_examples(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=5 * i) for i in range(12)]
    examples = [
        TimedBinaryExample(t, {"x": float(i)}, i % 2, t)
        for i, t in enumerate(times)
        if i != 3
    ]
    plan = Plan(
        "EUR/USD", "M5",
        (Fold(0, tuple(Bar(t) for t in times[:6]),
              tuple(Bar(t) for t in times[6:9]),
              tuple(Bar(t) for t in times[9:12])),),
    )
    with pytest.raises(ValueError, match="timestamps missing from examples"):
        run_offline_experiment(
            plan, examples, feature_names=("x",),
            manifest={"run_id": "missing-row-run"},
            model_factories={"candidate": ConstantModel},
            ledger=PredictionLedger(tmp_path / "ledger.sqlite"),
            evaluation_contract={"label_definition": "synthetic test only"},
            exclusions={"missing": 1}, block_length=1, bootstrap_resamples=100,
        )


def test_runner_rejects_overlapping_fold_partitions(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=5 * i) for i in range(12)]
    examples = [
        TimedBinaryExample(t, {"x": float(i)}, i % 2, t)
        for i, t in enumerate(times)
    ]
    plan = Plan(
        "EUR/USD", "M5",
        (Fold(0, tuple(Bar(t) for t in times[:6]),
              tuple(Bar(t) for t in times[5:9]),
              tuple(Bar(t) for t in times[9:12])),),
    )
    with pytest.raises(ValueError, match="must not share observation timestamps"):
        run_offline_experiment(
            plan, examples, feature_names=("x",),
            manifest={"run_id": "overlap-run"},
            model_factories={"candidate": ConstantModel},
            ledger=PredictionLedger(tmp_path / "ledger.sqlite"),
            evaluation_contract={"label_definition": "synthetic test only"},
            exclusions={"overlap": 1}, block_length=1, bootstrap_resamples=100,
        )

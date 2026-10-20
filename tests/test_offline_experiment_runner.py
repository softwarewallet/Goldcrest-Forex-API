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
    assert {(row["fold_index"], row["partition"]) for row in result.report["paired_comparisons"]} == {
        (0, "validation"), (0, "test"),
    }
    assert all(row["baseline_name"] == "training_prevalence" for row in result.report["paired_comparisons"])

    assert set(result.report["fold_results"]) == {"candidate", "training_prevalence"}
    # Completed run IDs are immutable. A rerun must use a new ID rather than
    # clearing the report digest or rewriting the completed attempt.
    completed_status = ledger.get_run_status("fixture-run")
    with pytest.raises(ValueError, match="already completed"):
        run_offline_experiment(
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
    assert ledger.get_run_status("fixture-run") == completed_status
    assert ledger.get_run_attempts("fixture-run")[0]["status"] == "completed"



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


def test_same_run_id_rejects_changed_feature_inputs_even_if_predictions_match(tmp_path):
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
    kwargs = dict(
        plan=plan, feature_names=("x",),
        manifest={"run_id": "provenance-run", "dataset_id": "fixture"},
        model_factories={"candidate": ConstantModel}, ledger=ledger,
        evaluation_contract={"label_definition": "synthetic fixture only"},
        exclusions={"invalid": 0}, block_length=1, bootstrap_resamples=100,
        random_seed=11,
    )
    run_offline_experiment(examples=examples, **kwargs)
    changed = list(examples)
    changed[0] = TimedBinaryExample(
        changed[0].timestamp, {"x": changed[0].features["x"] + 0.125},
        changed[0].label, changed[0].label_available_at,
    )
    with pytest.raises(ValueError, match="different manifest"):
        run_offline_experiment(examples=changed, **kwargs)
    assert ledger.get_run_status("provenance-run")["status"] == "completed"


class BrokenFitModel(ConstantModel):
    def fit(self, feature_rows, labels, **kwargs):
        raise RuntimeError("synthetic training failure")


def test_training_failure_is_durably_recorded(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=5 * i) for i in range(36)]
    examples = [TimedBinaryExample(t, {"x": float(i % 3)}, i % 2, t) for i, t in enumerate(times)]
    plan = Plan(
        "EUR/USD", "M5",
        (Fold(0, tuple(Bar(t) for t in times[:20]),
              tuple(Bar(t) for t in times[20:28]),
              tuple(Bar(t) for t in times[28:36])),),
    )
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run_offline_experiment(
            plan, examples, feature_names=("x",),
            manifest={"run_id": "training-failed-run", "dataset_id": "fixture"},
            model_factories={"candidate": BrokenFitModel}, ledger=ledger,
            evaluation_contract={"label_definition": "synthetic fixture only"},
            exclusions={"invalid": 0}, block_length=1, bootstrap_resamples=100,
        )
    status = ledger.get_run_status("training-failed-run")
    assert status["status"] == "failed"
    assert "synthetic training failure" in status["error"]


class ChangingVersionModel(ConstantModel):
    instance_count = 0

    def __init__(self):
        type(self).instance_count += 1
        self.model_version = f"fixture-v{type(self).instance_count}"


def test_model_version_change_across_folds_fails_closed_and_records_failure(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=5 * i) for i in range(60)]
    examples = [TimedBinaryExample(t, {"x": float(i % 3)}, i % 2, t) for i, t in enumerate(times)]
    plan = Plan(
        "EUR/USD", "M5",
        (
            Fold(0, tuple(Bar(t) for t in times[:20]), tuple(Bar(t) for t in times[20:25]), tuple(Bar(t) for t in times[25:30])),
            Fold(1, tuple(Bar(t) for t in times[:30]), tuple(Bar(t) for t in times[30:35]), tuple(Bar(t) for t in times[35:40])),
        ),
    )
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    ChangingVersionModel.instance_count = 0
    with pytest.raises(ValueError, match="model version changed across folds"):
        run_offline_experiment(
            plan, examples, feature_names=("x",),
            manifest={"run_id": "changing-version-run", "dataset_id": "fixture"},
            model_factories={"candidate": ChangingVersionModel}, ledger=ledger,
            evaluation_contract={"label_definition": "synthetic fixture only"},
            exclusions={"invalid": 0}, block_length=1, bootstrap_resamples=100,
        )
    assert ledger.get_run_status("changing-version-run")["status"] == "failed"


def test_runner_rejects_temporally_reordered_validation_and_test(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=5 * i) for i in range(36)]
    examples = [TimedBinaryExample(t, {"x": float(i % 3)}, i % 2, t) for i, t in enumerate(times)]
    plan = Plan(
        "EUR/USD", "M5",
        (Fold(0, tuple(Bar(t) for t in times[:20]),
              tuple(Bar(t) for t in times[28:36]),
              tuple(Bar(t) for t in times[20:28])),),
    )
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    with pytest.raises(ValueError, match="strictly chronological"):
        run_offline_experiment(
            plan, examples, feature_names=("x",),
            manifest={"run_id": "reordered-fold-run", "dataset_id": "fixture"},
            model_factories={"candidate": ConstantModel}, ledger=ledger,
            evaluation_contract={"label_definition": "synthetic fixture only"},
            exclusions={"invalid": 0}, block_length=1, bootstrap_resamples=100,
        )
    assert ledger.get_run_status("reordered-fold-run")["status"] == "failed"


class ChangingConfigModel(ConstantModel):
    next_config = 0

    def __init__(self):
        type(self).next_config += 1
        self.parameter = type(self).next_config
        self.model_version = "stable-version"


def test_runner_rejects_configuration_drift_across_folds_and_records_failure(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=5 * i) for i in range(60)]
    examples = [TimedBinaryExample(t, {"x": float(i % 3)}, i % 2, t) for i, t in enumerate(times)]
    plan = Plan(
        "EUR/USD", "M5",
        (
            Fold(0, tuple(Bar(t) for t in times[:20]), tuple(Bar(t) for t in times[20:25]), tuple(Bar(t) for t in times[25:30])),
            Fold(1, tuple(Bar(t) for t in times[:30]), tuple(Bar(t) for t in times[30:35]), tuple(Bar(t) for t in times[35:40])),
        ),
    )
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    ChangingConfigModel.next_config = 0
    with pytest.raises(ValueError, match="model configuration changed across folds"):
        run_offline_experiment(
            plan, examples, feature_names=("x",),
            manifest={"run_id": "config-drift-run", "dataset_id": "fixture"},
            model_factories={"candidate": ChangingConfigModel}, ledger=ledger,
            evaluation_contract={"label_definition": "synthetic fixture only"},
            exclusions={"invalid": 0}, block_length=1, bootstrap_resamples=100,
        )
    assert ledger.get_run_status("config-drift-run")["status"] == "failed"


def test_runner_report_contains_environment_and_model_fingerprints(tmp_path):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=5 * i) for i in range(36)]
    examples = [TimedBinaryExample(t, {"x": float(i % 3)}, i % 2, t) for i, t in enumerate(times)]
    plan = Plan(
        "EUR/USD", "M5",
        (Fold(0, tuple(Bar(t) for t in times[:20]),
              tuple(Bar(t) for t in times[20:28]),
              tuple(Bar(t) for t in times[28:36])),),
    )
    result = run_offline_experiment(
        plan, examples, feature_names=("x",),
        manifest={"run_id": "reproducibility-run", "dataset_id": "fixture"},
        model_factories={"candidate": ConstantModel},
        ledger=PredictionLedger(tmp_path / "ledger.sqlite"),
        evaluation_contract={"label_definition": "synthetic fixture only"},
        exclusions={"invalid": 0}, block_length=1, bootstrap_resamples=100,
    )
    reproducibility = result.report["manifest"]["reproducibility"]
    assert result.actionable is False
    assert reproducibility["actionable"] is False
    assert len(reproducibility["manifest_sha256"]) == 64
    assert reproducibility["environment"]["python_version"]
    assert "xgboost" in reproducibility["environment"]["packages"]
    assert len(reproducibility["initial_model_configuration"]["candidate"]["configuration_sha256"]) == 64

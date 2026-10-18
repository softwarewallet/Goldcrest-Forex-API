from datetime import datetime, timedelta, timezone

import pytest

from app.prediction_ledger import PredictionLedger, PredictionRecord


START = datetime(2025, 1, 1, tzinfo=timezone.utc)


def record(model, index, *, probability=0.8, label=None, fold=0, partition="test"):
    return PredictionRecord(
        run_id="run-abc", symbol="EUR/USD", timeframe="M5",
        observation_timestamp=START + timedelta(minutes=5 * index),
        fold_index=fold, partition=partition, model_name=model,
        model_version="v1", probability=probability, label=label,
        label_available_at=(START + timedelta(minutes=5 * index + 5)) if label is not None else None,
        actionable=False,
    )


def test_insert_is_idempotent_and_conflicting_retry_fails(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    item = record("candidate", 0, probability=0.7)
    assert ledger.record(item) is True
    assert ledger.record(item) is False
    with pytest.raises(ValueError, match="conflicting"):
        ledger.record(record("candidate", 0, probability=0.9))


def test_compare_pairs_by_observation_identity_and_matured_labels(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    rows = []
    for i, label in enumerate([0, 1] * 15):
        rows.append(record("candidate", i, probability=0.1 if label == 0 else 0.9, label=label))
        rows.append(record("baseline", i, probability=0.4 if label == 0 else 0.6, label=label))
    assert ledger.record_many(rows) == len(rows)
    result = ledger.compare(
        run_id="run-abc", symbol="EUR/USD", timeframe="M5", fold_index=0,
        partition="test", candidate_name="candidate", candidate_version="v1",
        baseline_name="baseline", baseline_version="v1", block_length=3,
        bootstrap_resamples=200, random_seed=7,
    )
    assert result.sample_count == 30
    assert result.actionable is False
    assert result.comparison.candidate_brier_score < result.comparison.baseline_brier_score


def test_unmatured_labels_are_not_silently_scored(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    for i in range(4):
        ledger.record(record("candidate", i, label=i % 2))
        ledger.record(record("baseline", i, label=i % 2))
    # No hidden data inference: explicit mature labels are required, and sample size
    # requirements for block bootstrap must be satisfied by the paired observations.
    result = ledger.compare(
        run_id="run-abc", symbol="EUR/USD", timeframe="M5", fold_index=0,
        partition="test", candidate_name="candidate", candidate_version="v1",
        baseline_name="baseline", baseline_version="v1", block_length=2,
        bootstrap_resamples=100,
    )
    assert result.sample_count == 4


@pytest.mark.parametrize("kwargs", [
    {"probability": float("nan")},
    {"probability": 1.2},
    {"label": True},
])
def test_record_rejects_invalid_probability_or_label(tmp_path, kwargs):
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    with pytest.raises(ValueError):
        ledger.record(record("candidate", 0, **kwargs))


def test_comparison_rejects_mismatched_observation_identity(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    ledger.record(record("candidate", 0, label=1))
    ledger.record(record("baseline", 1, label=1))
    with pytest.raises(ValueError, match="identities"):
        ledger.compare(
            run_id="run-abc", symbol="EUR/USD", timeframe="M5", fold_index=0,
            partition="test", candidate_name="candidate", candidate_version="v1",
            baseline_name="baseline", baseline_version="v1", block_length=1,
            bootstrap_resamples=100,
        )


def test_research_rows_cannot_be_actionable(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    with pytest.raises(ValueError, match="actionable=False"):
        ledger.record(PredictionRecord(
            run_id="r", symbol="EUR/USD", timeframe="M5", observation_timestamp=START,
            fold_index=0, partition="test", model_name="m", model_version="v1",
            probability=0.5, actionable=True,
        ))

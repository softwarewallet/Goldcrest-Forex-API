from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from app.prediction_ledger import PredictionLedger, PredictionRecord
from app.evaluation import ForexBar
from app.run_manifest import ExperimentRegistry, build_run_manifest


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
    assert isinstance(result.observation_identity_sha256, str)
    assert len(result.observation_identity_sha256) == 64
    expected_identities = [
        (START + timedelta(minutes=5 * i)).isoformat().replace("+00:00", "Z")
        for i in range(30)
    ]
    import hashlib
    import json
    expected_digest = hashlib.sha256(json.dumps(
        expected_identities, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    assert result.observation_identity_sha256 == expected_digest


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



def test_run_lifecycle_rejects_manifest_reuse_and_records_completion(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    ledger.start_run("run-lifecycle", {"dataset": "fixture", "version": 1})
    assert ledger.get_run_status("run-lifecycle")["status"] == "running"
    with pytest.raises(ValueError, match="different manifest"):
        ledger.start_run("run-lifecycle", {"dataset": "different"})
    ledger.finish_run("run-lifecycle", "a" * 64)
    status = ledger.get_run_status("run-lifecycle")
    assert status["status"] == "completed"
    assert status["report_sha256"] == "a" * 64
    assert status["error"] is None


def test_run_lifecycle_records_each_failed_and_retried_attempt(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    manifest = {"dataset": "fixture"}
    ledger.start_run("run-failed", manifest)
    ledger.fail_run("run-failed", "OSError: report write failed")
    status = ledger.get_run_status("run-failed")
    assert status["status"] == "failed"
    assert status["error"] == "OSError: report write failed"
    ledger.start_run("run-failed", manifest)
    assert ledger.get_run_status("run-failed")["status"] == "running"
    ledger.fail_run("run-failed", "RuntimeError: retry interrupted")
    attempts = ledger.get_run_attempts("run-failed")
    assert [attempt["status"] for attempt in attempts] == ["failed", "failed"]
    assert [attempt["attempt_number"] for attempt in attempts] == [1, 2]
    assert "report write failed" in attempts[0]["error"]
    assert "retry interrupted" in attempts[1]["error"]


def test_completed_run_cannot_be_reopened_or_have_report_fingerprint_erased(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    manifest = {"dataset": "fixture"}
    ledger.start_run("immutable-run", manifest)
    ledger.finish_run("immutable-run", "a" * 64)
    before = ledger.get_run_status("immutable-run")
    with pytest.raises(ValueError, match="already completed"):
        ledger.start_run("immutable-run", manifest)
    assert ledger.get_run_status("immutable-run") == before
    attempts = ledger.get_run_attempts("immutable-run")
    assert len(attempts) == 1
    assert attempts[0]["status"] == "completed"
    assert attempts[0]["report_sha256"] == "a" * 64


def test_concurrent_running_attempt_is_rejected(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    ledger.start_run("active-run", {"dataset": "fixture"})
    with pytest.raises(ValueError, match="running attempt"):
        ledger.start_run("active-run", {"dataset": "fixture"})
    assert len(ledger.get_run_attempts("active-run")) == 1


def test_completed_report_verification_checks_payload_and_ledger_fingerprint(tmp_path):
    import hashlib
    import json

    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    report = {"run_id": "report-run", "actionable": False, "metrics": {"brier": 0.2}}
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    ledger.start_run("report-run", {"dataset": "fixture"})
    ledger.finish_run("report-run", report["report_sha256"])

    assert ledger.verify_completed_report("report-run", report) is True
    tampered = {**report, "metrics": {"brier": 0.9}}
    assert ledger.verify_completed_report("report-run", tampered) is False
    assert ledger.verify_completed_report("different-run", report) is False


def test_finish_run_rejects_non_hex_report_digest(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    ledger.start_run("bad-digest", {"dataset": "fixture"})
    with pytest.raises(ValueError, match="hex digest"):
        ledger.finish_run("bad-digest", "z" * 64)


def _shared_db_manifest():
    bar = ForexBar(
        symbol="EUR/USD", timeframe="M5", timestamp=START,
        open=1.1, high=1.101, low=1.099, close=1.1005,
        bid=1.1004, ask=1.1006,
    )
    return build_run_manifest(
        [bar], code_commit="fixture-commit", model_id="fixture-model",
        model_revision="v1", dataset_id="fixture-dataset",
        evaluation_config={"target": "fixture-only"},
    )


@pytest.mark.parametrize("registry_first", [True, False])
def test_manifest_registry_and_prediction_ledger_can_share_sqlite_file(tmp_path, registry_first):
    path = tmp_path / "shared.sqlite"
    if registry_first:
        registry = ExperimentRegistry(path)
        ledger = PredictionLedger(path)
    else:
        ledger = PredictionLedger(path)
        registry = ExperimentRegistry(path)

    manifest = _shared_db_manifest()
    assert registry.register(manifest) is True
    ledger.start_run("ledger-run", {"dataset": "fixture", "run": 1})
    assert registry.get(manifest.run_id)["status"] == "REGISTERED"
    assert ledger.get_run_status("ledger-run")["status"] == "running"
    ledger.finish_run("ledger-run", "a" * 64)
    registry.record_result(manifest.run_id, status="COMPLETED", outcome_summary={"ok": True})
    assert ledger.get_run_status("ledger-run")["status"] == "completed"
    assert registry.get(manifest.run_id)["outcome_summary"] == {"ok": True}


def test_legacy_prediction_ledger_lifecycle_rows_are_migrated(tmp_path):
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE experiment_runs (
            run_id TEXT PRIMARY KEY, manifest_sha256 TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
            report_sha256 TEXT, error TEXT, updated_at TEXT NOT NULL
        )""")
        connection.execute(
            "INSERT INTO experiment_runs VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy-run", "b" * 64, "completed", "c" * 64, None, "2026-01-01T00:00:00Z"),
        )
    ledger = PredictionLedger(path)
    status = ledger.get_run_status("legacy-run")
    assert status["status"] == "completed"
    assert status["manifest_sha256"] == "b" * 64
    assert status["report_sha256"] == "c" * 64
    # The old table is renamed so the independent manifest registry can now initialize.
    registry = ExperimentRegistry(path)
    assert registry.register(_shared_db_manifest()) is True
    assert ledger.get_run_status("legacy-run")["status"] == "completed"



def test_recovery_marks_only_stale_running_attempts_interrupted(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    ledger.start_run("stale-run", {"dataset": "fixture"})
    ledger.start_run("fresh-run", {"dataset": "fixture"})
    ledger.start_run("completed-run", {"dataset": "fixture"})
    ledger.finish_run("completed-run", "c" * 64)
    cutoff = datetime(2020, 1, 1, tzinfo=timezone.utc)
    # Backdate only one active run to simulate a process crash.
    with ledger._connect() as connection:
        connection.execute(
            "UPDATE prediction_ledger_runs SET updated_at=? WHERE run_id=?",
            ("2000-01-01T00:00:00Z", "stale-run"),
        )
    recovered = ledger.mark_interrupted_runs(older_than=cutoff)
    assert recovered == ["stale-run"]
    assert ledger.get_run_status("stale-run")["status"] == "failed"
    assert ledger.get_run_attempts("stale-run")[0]["status"] == "failed"
    assert ledger.get_run_status("fresh-run")["status"] == "running"
    assert ledger.get_run_status("completed-run")["status"] == "completed"
    # A failed stale attempt can be safely retried with the identical contract.
    ledger.start_run("stale-run", {"dataset": "fixture"})
    assert [x["status"] for x in ledger.get_run_attempts("stale-run")] == ["failed", "running"]


def test_recovery_requires_aware_cutoff(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    with pytest.raises(ValueError, match="timezone-aware"):
        ledger.mark_interrupted_runs(older_than=datetime(2026, 1, 1))


def test_finish_run_rejects_missing_expected_prediction_rows(tmp_path):
    ledger = PredictionLedger(tmp_path / "expected-count.sqlite")
    ledger.start_run("expected-count", {"dataset": "fixture"}, expected_prediction_rows=1)
    with pytest.raises(ValueError, match="prediction row count mismatch: expected 1, found 0"):
        ledger.finish_run("expected-count", "a" * 64)
    assert ledger.get_run_status("expected-count")["status"] == "running"


def test_finish_run_completes_when_expected_prediction_count_matches(tmp_path):
    ledger = PredictionLedger(tmp_path / "expected-count-ok.sqlite")
    ledger.start_run("expected-count-ok", {"dataset": "fixture"}, expected_prediction_rows=1)
    ledger.record(PredictionRecord(
        run_id="expected-count-ok", symbol="EUR/USD", timeframe="M5",
        observation_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        fold_index=0, partition="test", model_name="candidate", model_version="v1",
        probability=0.6,
    ))
    ledger.finish_run("expected-count-ok", "a" * 64)
    assert ledger.get_run_status("expected-count-ok")["status"] == "completed"


def test_run_contract_rejects_changed_expected_prediction_count(tmp_path):
    ledger = PredictionLedger(tmp_path / "expected-count-retry.sqlite")
    manifest = {"dataset": "fixture"}
    ledger.start_run("expected-count-retry", manifest, expected_prediction_rows=2)
    ledger.fail_run("expected-count-retry", "simulated failure")
    with pytest.raises(ValueError, match="different manifest|different expected prediction count"):
        ledger.start_run("expected-count-retry", manifest, expected_prediction_rows=3)

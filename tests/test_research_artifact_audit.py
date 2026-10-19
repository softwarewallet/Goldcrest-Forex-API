from datetime import datetime, timezone

import pytest

from app.experiment_report import build_experiment_report, write_experiment_report
from app.prediction_ledger import PredictionLedger, PredictionRecord
from app.research_artifact_audit import verify_research_artifacts


def _fixture(tmp_path):
    ledger = PredictionLedger(tmp_path / "audit.sqlite")
    run_id = "audit-fixture"
    manifest = {"run_id": run_id, "dataset": "synthetic-only"}
    ledger.start_run(run_id, {"manifest": manifest})
    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for model, probability in (("candidate", 0.7), ("training_prevalence", 0.5)):
        for partition in ("validation", "test"):
            observed_at = timestamp if partition == "validation" else datetime(2025, 1, 1, 0, 5, tzinfo=timezone.utc)
            ledger.record(PredictionRecord(
                run_id=run_id, symbol="EUR/USD", timeframe="M5",
                observation_timestamp=observed_at,
                fold_index=0, partition=partition, model_name=model, model_version="fixture-v1",
                probability=probability, label=1, label_available_at=observed_at,
            ))
    metrics = {"sample_count": 1}
    report = build_experiment_report(
        manifest=manifest,
        fold_results={
            model: [{
                "fold_index": 0, "model_type": "fixture", "model_version": "fixture-v1",
                "validation": metrics, "test": metrics, "actionable": False,
            }]
            for model in ("candidate", "training_prevalence")
        },
        evaluation_contract={"fixture": "synthetic mechanics only"},
    )
    ledger.finish_run(run_id, report["report_sha256"])
    report_path = tmp_path / "report.json"
    write_experiment_report(report, report_path)
    return ledger, report_path


def test_audit_verifies_report_and_ledger_consistency(tmp_path):
    ledger, report_path = _fixture(tmp_path)
    result = verify_research_artifacts(report_path, ledger=ledger)
    assert result.verified is True
    assert result.actionable is False
    assert result.expected_prediction_rows == result.actual_prediction_rows == 4
    assert result.actionable_rows == 0
    assert result.ledger_status == "completed"


def test_audit_rejects_tampered_report(tmp_path):
    ledger, report_path = _fixture(tmp_path)
    report_path.write_text(report_path.read_text().replace('"scope": "OFFLINE_RESEARCH_ONLY"', '"scope": "LIVE"'))
    with pytest.raises(ValueError, match="hash or completed-run"):
        verify_research_artifacts(report_path, ledger=ledger)


def test_audit_rejects_ledger_row_count_mismatch(tmp_path):
    ledger, report_path = _fixture(tmp_path)
    with ledger._connect() as connection:
        connection.execute(
            "DELETE FROM oos_predictions WHERE run_id=? AND model_name=? AND partition=?",
            ("audit-fixture", "candidate", "test"),
        )
    with pytest.raises(ValueError, match="group mismatch"):
        verify_research_artifacts(report_path, ledger=ledger)


def test_audit_rejects_incomplete_run(tmp_path):
    ledger, report_path = _fixture(tmp_path)
    with ledger._connect() as connection:
        connection.execute(
            "UPDATE prediction_ledger_runs SET status='failed' WHERE run_id=?",
            ("audit-fixture",),
        )
    with pytest.raises(ValueError, match="hash or completed-run"):
        verify_research_artifacts(report_path, ledger=ledger)


def test_audit_rejects_misplaced_rows_even_when_total_count_matches(tmp_path):
    ledger, report_path = _fixture(tmp_path)
    # Keep the total row count unchanged but move a prediction to an unexpected model group.
    with ledger._connect() as connection:
        connection.execute(
            "UPDATE oos_predictions SET model_name=? WHERE run_id=? AND model_name=? AND partition=?",
            ("unexpected-model", "audit-fixture", "candidate", "test"),
        )
    with pytest.raises(ValueError, match="group mismatch"):
        verify_research_artifacts(report_path, ledger=ledger)



def test_audit_rejects_attempt_history_report_digest_mismatch(tmp_path):
    ledger, report_path = _fixture(tmp_path)
    with ledger._connect() as connection:
        connection.execute(
            "UPDATE prediction_ledger_attempts SET report_sha256=? WHERE run_id=? AND status='completed'",
            ("f" * 64, "audit-fixture"),
        )
    with pytest.raises(ValueError, match="attempt history does not reconcile"):
        verify_research_artifacts(report_path, ledger=ledger)


def test_audit_rejects_running_attempt_under_completed_run(tmp_path):
    ledger, report_path = _fixture(tmp_path)
    with ledger._connect() as connection:
        connection.execute(
            """INSERT INTO prediction_ledger_attempts
               (run_id, attempt_number, status, started_at, finished_at, report_sha256, error)
               VALUES (?, ?, 'running', ?, NULL, NULL, NULL)""",
            ("audit-fixture", 2, "2026-01-01T00:00:00Z"),
        )
    with pytest.raises(ValueError, match="attempt history does not reconcile"):
        verify_research_artifacts(report_path, ledger=ledger)

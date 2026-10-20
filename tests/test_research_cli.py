import json

from app import research_cli
from app.research_artifact_audit import ResearchArtifactAudit


def test_cli_audit_emits_machine_readable_success_and_zero_exit(tmp_path, monkeypatch, capsys):
    report = tmp_path / "report.json"
    ledger = tmp_path / "research.sqlite"
    report.write_text("{}", encoding="utf-8")
    ledger.write_bytes(b"")
    result = ResearchArtifactAudit(
        run_id="fixture-run",
        report_sha256="a" * 64,
        expected_prediction_rows=8,
        actual_prediction_rows=8,
        actionable_rows=0,
        ledger_status="completed",
    )
    monkeypatch.setattr(research_cli, "PredictionLedger", lambda _: object())
    monkeypatch.setattr(research_cli, "verify_research_artifacts", lambda *_args, **_kwargs: result)

    code = research_cli.main(["audit", "--report", str(report), "--ledger", str(ledger)])
    captured = capsys.readouterr()

    assert code == 0
    output = json.loads(captured.out)
    assert output["verified"] is True
    assert output["actionable"] is False
    assert output["expected_prediction_rows"] == output["actual_prediction_rows"] == 8
    assert captured.err == ""


def test_cli_audit_fails_with_nonzero_exit_for_missing_report(tmp_path, capsys):
    ledger = tmp_path / "research.sqlite"
    ledger.write_bytes(b"")

    code = research_cli.main([
        "audit", "--report", str(tmp_path / "missing.json"), "--ledger", str(ledger)
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "report file does not exist" in captured.err
    assert captured.out == ""


def test_cli_audit_fails_closed_when_artifact_verification_raises(tmp_path, monkeypatch, capsys):
    report = tmp_path / "report.json"
    ledger = tmp_path / "research.sqlite"
    report.write_text("{}", encoding="utf-8")
    ledger.write_bytes(b"")
    monkeypatch.setattr(research_cli, "PredictionLedger", lambda _: object())

    def reject(*_args, **_kwargs):
        raise ValueError("report hash or completed-run ledger binding is invalid")

    monkeypatch.setattr(research_cli, "verify_research_artifacts", reject)
    code = research_cli.main(["audit", "--report", str(report), "--ledger", str(ledger)])
    captured = capsys.readouterr()

    assert code == 1
    error = json.loads(captured.out)
    assert error["verified"] is False
    assert error["actionable"] is False
    assert "hash" in error["error"]
    assert captured.err == ""


def test_cli_status_emits_run_and_attempt_history_without_mutation(tmp_path, monkeypatch, capsys):
    ledger_path = tmp_path / "research.sqlite"
    ledger_path.write_bytes(b"")
    run = {"run_id": "fixture-run", "status": "failed", "expected_prediction_rows": 8}
    attempts = [{"attempt_number": 1, "status": "failed"}, {"attempt_number": 2, "status": "failed"}]

    class FakeLedger:
        def __init__(self, _path):
            pass

        def get_run_status(self, run_id):
            assert run_id == "fixture-run"
            return run

        def get_run_attempts(self, run_id):
            assert run_id == "fixture-run"
            return attempts

    monkeypatch.setattr(research_cli, "PredictionLedger", FakeLedger)
    code = research_cli.main(["status", "--run-id", "fixture-run", "--ledger", str(ledger_path)])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["found"] is True
    assert payload["run"]["status"] == "failed"
    assert payload["attempts"] == attempts
    assert payload["actionable"] is False
    assert captured.err == ""


def test_cli_status_returns_two_for_unknown_run(tmp_path, monkeypatch, capsys):
    ledger_path = tmp_path / "research.sqlite"
    ledger_path.write_bytes(b"")

    class FakeLedger:
        def __init__(self, _path):
            pass

        def get_run_status(self, _run_id):
            return None

    monkeypatch.setattr(research_cli, "PredictionLedger", FakeLedger)
    code = research_cli.main(["status", "--run-id", "unknown", "--ledger", str(ledger_path)])
    captured = capsys.readouterr()

    assert code == 2
    assert json.loads(captured.out) == {"attempts": [], "found": False, "run_id": "unknown"}
    assert captured.err == ""


def test_cli_status_returns_two_for_missing_ledger(tmp_path, capsys):
    code = research_cli.main([
        "status", "--run-id", "fixture-run", "--ledger", str(tmp_path / "missing.sqlite")
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "ledger file does not exist" in captured.err
    assert captured.out == ""

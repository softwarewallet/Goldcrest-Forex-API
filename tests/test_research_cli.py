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
    error = json.loads(captured.err)
    assert error["verified"] is False
    assert error["actionable"] is False
    assert "hash" in error["error"]
    assert captured.out == ""

import json

from app import research_cli
from app.experiment_report import build_experiment_report, write_experiment_report


def _write_report(path):
    report = build_experiment_report(
        manifest={"run_id": "cli-inspection-run", "dataset_id": "fixture"},
        fold_results={
            "candidate": [{
                "fold_index": 0,
                "validation": {"sample_count": 4, "brier_score": 0.2},
                "test": {"sample_count": 4, "brier_score": 0.22},
                "actionable": False,
            }]
        },
        exclusions={"unlabeled": 2},
        warnings=["Fixture only."],
    )
    return write_experiment_report(report, path)


def test_cli_inspect_emits_summary_and_zero_exit(tmp_path, capsys):
    report = _write_report(tmp_path / "report.json")
    code = research_cli.main(["inspect", "--report", str(report)])
    captured = capsys.readouterr()

    assert code == 0
    summary = json.loads(captured.out)
    assert summary["verified"] is True
    assert summary["actionable"] is False
    assert summary["run_id"] == "cli-inspection-run"
    assert summary["models"][0]["folds"][0]["test"]["brier_score"] == 0.22
    assert summary["interpretation"]["pooled_test_metrics"] is False
    assert captured.err == ""


def test_cli_inspect_returns_one_for_tampered_report(tmp_path, capsys):
    report = _write_report(tmp_path / "report.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    data["warnings"].append("changed after fingerprint")
    report.write_text(json.dumps(data), encoding="utf-8")

    code = research_cli.main(["inspect", "--report", str(report)])
    captured = capsys.readouterr()

    assert code == 1
    payload = json.loads(captured.out)
    assert payload["verified"] is False
    assert payload["actionable"] is False
    assert "fingerprint mismatch" in payload["error"]
    assert captured.err == ""


def test_cli_inspect_returns_two_for_missing_report(tmp_path, capsys):
    code = research_cli.main(["inspect", "--report", str(tmp_path / "missing.json")])
    captured = capsys.readouterr()

    assert code == 2
    assert "report file does not exist" in captured.err
    assert captured.out == ""

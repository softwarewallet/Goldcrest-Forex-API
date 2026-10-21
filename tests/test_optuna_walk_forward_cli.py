from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from app import optuna_walk_forward_cli


def test_cli_writes_provenance_bound_non_actionable_report(tmp_path, monkeypatch, capsys):
    import app.optuna_walk_forward_cli as cli

    monkeypatch.setattr(cli, "load_historical_dataset", lambda *_args: (
        [SimpleNamespace(symbol="EUR/USD", timeframe="M15")],
        [SimpleNamespace()],
        "a" * 64,
    ))
    monkeypatch.setattr(cli, "run_optuna_walk_forward", lambda *_args, **_kwargs: SimpleNamespace(
        to_dict=lambda: {
            "schema_version": 1,
            "run_id": "fixture-run",
            "symbol": "EUR/USD",
            "timeframe": "M15",
            "actionable": False,
            "test_metrics_used_for_tuning": False,
        },
        fold_count=2,
    ))

    report_path = tmp_path / "report.json"
    code = optuna_walk_forward_cli.main([
        "--csv", str(tmp_path / "data.csv"),
        "--features", "momentum,volatility",
        "--run-id", "fixture-run",
        "--report", str(report_path),
        "--data-source", "fixture-provider",
        "--dataset-version", "fixture-v1",
        "--retrieved-at-utc", "2026-10-21T10:00:00Z",
        "--timestamp-timezone", "UTC",
        "--duplicate-policy", "reject",
        "--adjustment-policy", "none",
        "--license-or-terms-reference", "fixture-license",
        "--initial-train-bars", "60",
        "--validation-bars", "30",
        "--test-bars", "25",
        "--step-bars", "25",
        "--horizon-bars", "2",
        "--trials-per-fold", "1",
        "--bootstrap-resamples", "100",
    ])
    captured = capsys.readouterr()

    assert code == 0
    output = json.loads(captured.out)
    assert output["actionable"] is False
    assert output["fold_count"] == 2
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["dataset_provenance"]["source_name"] == "fixture-provider"
    assert payload["dataset_provenance_verification"]["verified"] is True
    assert payload["test_metrics_used_for_tuning"] is False
    expected_hash = payload.pop("report_payload_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert expected_hash == hashlib.sha256(canonical.encode("utf-8")).hexdigest()

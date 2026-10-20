from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone

import pytest

from app import historical_research_pipeline as pipeline


def write_ohlc(path, count=80):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "symbol", "timeframe", "timestamp", "open", "high", "low", "close",
        ])
        writer.writeheader()
        for i in range(count):
            close = 1.1 + i * 0.0002 + (i % 3) * 0.00001
            writer.writerow({
                "symbol": "EUR/USD", "timeframe": "M15",
                "timestamp": (start + timedelta(minutes=15 * i)).isoformat().replace("+00:00", "Z"),
                "open": close - 0.00005, "high": close + 0.0001,
                "low": close - 0.0001, "close": close,
            })


def test_pipeline_runs_preparation_and_passes_generated_features_to_evaluation(tmp_path, monkeypatch):
    raw = tmp_path / "raw.csv"
    write_ohlc(raw)
    captured = {}

    def fake_research(args):
        captured["args"] = args
        report_path = args[args.index("--report") + 1]
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump({"run_id": "pipeline-test", "actionable": False}, handle)
        return 0

    monkeypatch.setattr(pipeline, "run_research", fake_research)
    summary = pipeline.run_historical_research_pipeline(
        raw,
        features_csv=tmp_path / "features.csv",
        labeled_csv=tmp_path / "labeled.csv",
        feature_report=tmp_path / "features.json",
        label_report=tmp_path / "labels.json",
        report=tmp_path / "combined.json",
        run_id="pipeline-test", data_source="verified test fixture",
        dataset_version="fixture-v1", horizon_bars=3, neutral_band_bps=0,
        neutral_policy="up", initial_train_bars=30, validation_bars=10,
        test_bars=10, step_bars=10, block_length=2,
        spread_cost_notes="test only", slippage_and_commission_notes="test only",
        ledger=tmp_path / "ledger.sqlite",
    )

    args = captured["args"]
    assert args[args.index("--features") + 1] == "return_1,return_3,return_5,range_pct,body_pct,close_location,volatility_14"
    assert summary["actionable"] is False
    assert summary["evaluation"]["run_id"] == "pipeline-test"
    assert (tmp_path / "features.csv").exists()
    assert (tmp_path / "labeled.csv").exists()
    assert json.loads((tmp_path / "combined.json").read_text())["raw_input_sha256"] == summary["raw_input_sha256"]


def test_pipeline_rejects_path_collisions_before_writing(tmp_path):
    raw = tmp_path / "raw.csv"
    write_ohlc(raw)
    with pytest.raises(ValueError, match="distinct"):
        pipeline.run_historical_research_pipeline(
            raw, features_csv=tmp_path / "features.csv", labeled_csv=tmp_path / "features.csv",
            feature_report=tmp_path / "features.json", label_report=tmp_path / "labels.json",
            report=tmp_path / "combined.json", run_id="x", data_source="source",
            dataset_version="v1", horizon_bars=3, neutral_band_bps=0,
            neutral_policy="up", initial_train_bars=30, validation_bars=10,
            test_bars=10, step_bars=10, block_length=2,
            spread_cost_notes="notes", slippage_and_commission_notes="notes",
            ledger=tmp_path / "ledger.sqlite",
        )

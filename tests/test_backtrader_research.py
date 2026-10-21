from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from app.backtrader_research import (
    _load_signal_frame,
    run_backtrader_replay,
    write_backtrader_report,
)


def _write_signals(path, signals=None):
    signals = signals or [0, 1, 1, 0, -1, 0]
    rows = []
    for index, signal in enumerate(signals):
        close = 1.10 + index * 0.002
        rows.append({
            "timestamp": pd.Timestamp("2026-01-01T00:00:00Z")
                + pd.Timedelta(minutes=15 * index),
            "open": close - 0.0005,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "signal": signal,
            "volume": 100,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_signal_loader_rejects_invalid_signal(tmp_path):
    path = tmp_path / "signals.csv"
    _write_signals(path, [2])
    with pytest.raises(ValueError, match="signal must be"):
        _load_signal_frame(path)


def test_signal_loader_rejects_duplicate_timestamps(tmp_path):
    path = tmp_path / "signals.csv"
    pd.DataFrame([
        {"timestamp": "2026-01-01T00:00:00Z", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "signal": 0},
        {"timestamp": "2026-01-01T00:00:00Z", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "signal": 0},
    ]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="strictly chronological"):
        _load_signal_frame(path)


def test_synthetic_replay_executes_local_simulated_orders_and_returns_json_safe_report(tmp_path):
    pytest.importorskip("backtrader")
    path = tmp_path / "signals.csv"
    _write_signals(path)

    report = run_backtrader_replay(
        path,
        initial_cash=100_000,
        stake=10,
        commission_rate=0.0002,
        slippage_rate=0.0001,
    )

    assert report["scope"] == "OFFLINE_BACKTRADER_REPLAY"
    assert report["actionable"] is False
    assert report["observation_count"] == 6
    assert report["source_csv_sha256"]
    assert report["execution_assumption"].find("cheat-on-close disabled") >= 0
    assert math.isfinite(report["initial_cash"])
    assert math.isfinite(report["final_value"])
    json.dumps(report, allow_nan=False)

    report_path = tmp_path / "replay-report.json"
    write_backtrader_report(report, report_path)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["actionable"] is False
    assert len(persisted["report_payload_sha256"]) == 64


def test_replay_rejects_invalid_friction():
    with pytest.raises(ValueError, match="commission_rate and slippage_rate"):
        run_backtrader_replay("does-not-need-to-exist.csv", commission_rate=1.0)

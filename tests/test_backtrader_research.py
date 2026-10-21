from __future__ import annotations

import pandas as pd
import pytest

from app.backtrader_research import _load_signal_frame


def test_signal_loader_rejects_invalid_signal(tmp_path):
    path = tmp_path / "signals.csv"
    pd.DataFrame([{
        "timestamp": "2026-01-01T00:00:00Z",
        "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "signal": 2,
    }]).to_csv(path, index=False)
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

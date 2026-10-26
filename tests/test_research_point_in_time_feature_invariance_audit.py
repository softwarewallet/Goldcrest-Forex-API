from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import research_point_in_time_feature_invariance_audit as audit_module
from app.causal_ta_features import CausalFeatureBatch


def _ohlc_file(path: Path, rows: int = 48) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["symbol", "timeframe", "timestamp", "open", "high", "low", "close"],
            lineterminator="\n",
        )
        writer.writeheader()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index in range(rows):
            close = 1.1 + index * 0.0005
            open_ = close - 0.0001
            writer.writerow({
                "symbol": "EURUSD",
                "timeframe": "M15",
                "timestamp": (start + timedelta(minutes=15 * index)).isoformat().replace("+00:00", "Z"),
                "open": open_,
                "high": close + 0.0002,
                "low": open_ - 0.0002,
                "close": close,
            })
    return path


def test_prefix_invariance_passes_for_causal_ohlc_features(tmp_path):
    source = _ohlc_file(tmp_path / "ohlc.csv")

    report = audit_module.audit_point_in_time_invariance(
        source, symbol="EURUSD", timeframe="M15", max_cutoffs=6
    )

    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert report["rows"] == 48
    assert report["cutoff_count"] >= 1
    assert report["feature_values_compared"] > 0
    assert report["source_sha256"]
    assert report["actionable"] is False
    assert report["execution_connected"] is False


def test_prefix_invariance_detects_future_dependent_feature(tmp_path, monkeypatch):
    source = _ohlc_file(tmp_path / "ohlc.csv", rows=12)

    def leaky_builder(bars):
        last_close = float(bars[-1].close)
        return CausalFeatureBatch(
            timestamps_utc=tuple(
                bar.timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                for bar in bars
            ),
            feature_names=("future_leak",),
            rows=tuple({"future_leak": last_close} for _ in bars),
            indicator_backend="test_future_leak",
        )

    monkeypatch.setattr(audit_module, "build_causal_ohlc_features", leaky_builder)

    report = audit_module.audit_point_in_time_invariance(
        source, symbol="EURUSD", timeframe="M15", max_cutoffs=4
    )

    assert report["status"] == "FAIL"
    assert any(item["code"] == "future_data_changed_past_feature" for item in report["errors"])
    assert report["actionable"] is False


def test_audit_rejects_symbol_mismatch_and_invalid_cutoff_count(tmp_path):
    source = _ohlc_file(tmp_path / "ohlc.csv")
    with pytest.raises(ValueError, match="symbol_or_timeframe_mismatch"):
        audit_module.audit_point_in_time_invariance(source, symbol="GBPUSD", timeframe="M15")
    with pytest.raises(ValueError, match="max_cutoffs_must_be_integer"):
        audit_module.audit_point_in_time_invariance(
            source, symbol="EURUSD", timeframe="M15", max_cutoffs=True
        )


def test_cli_publishes_audit_and_refuses_to_overwrite(tmp_path, capsys):
    source = _ohlc_file(tmp_path / "ohlc.csv")
    output = tmp_path / "invariance.json"
    args = [
        "--ohlc-csv", str(source), "--symbol", "EURUSD", "--timeframe", "M15",
        "--output", str(output), "--max-cutoffs", "4",
    ]

    assert audit_module.main(args) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    assert audit_module.main(args) == 2
    assert "output_already_exists" in capsys.readouterr().out


def test_cli_refuses_output_aliasing_input(tmp_path, capsys):
    source = _ohlc_file(tmp_path / "ohlc.csv")
    original = source.read_bytes()
    args = [
        "--ohlc-csv", str(source), "--symbol", "EURUSD", "--timeframe", "M15",
        "--output", str(source),
    ]

    assert audit_module.main(args) == 2
    assert source.read_bytes() == original
    assert "output_must_not_alias_ohlc_input" in capsys.readouterr().out

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import app.research_feature_temporal_integrity_audit as audit
from app.evaluation import ForexBar


def make_bars(count: int = 120) -> list[ForexBar]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(count):
        close = 1.1 + index * 0.00007 + (index % 7) * 0.000013
        open_ = close - 0.00002
        rows.append(ForexBar(
            symbol="EUR/USD",
            timeframe="M15",
            timestamp=start + timedelta(minutes=15 * index),
            open=open_,
            high=close + 0.00015,
            low=open_ - 0.00012,
            close=close,
        ))
    return rows


def test_temporal_integrity_audit_passes_causal_prefix_and_perturbation_checks():
    report = audit.audit_feature_temporal_integrity(
        make_bars(),
        source_data_sha256=hashlib.sha256(b"fixture").hexdigest(),
        horizon_bars=3,
        sample_count=8,
    )

    assert report["status"] == "PASS_WITH_WARNINGS"
    assert report["errors"] == []
    assert report["prefix_invariance"]["checks"] >= 8
    assert report["prefix_invariance"]["all_passed"] is True
    assert report["future_perturbation_invariance"]["all_passed"] is True
    partitions = report["chronological_partition_isolation"]
    assert partitions["strictly_ordered_non_overlapping"] is True
    assert partitions["purge_gaps_sufficient_for_horizon"] is True
    assert partitions["rows_purged_between_train_and_validation"] >= 6
    assert partitions["rows_purged_between_validation_and_test"] >= 6
    assert report["actionable"] is False
    assert report["automatic_model_selection"] is False
    assert report["inferential"] is False
    assert report["execution_connected"] is False
    assert report["model_training_performed"] is False


def test_temporal_integrity_audit_detects_future_dependent_feature(monkeypatch):
    causal_builder = audit.build_causal_ohlc_features

    def leaking_builder(bars):
        batch = causal_builder(bars)
        rows = [dict(row) for row in batch.rows]
        rows[0]["return_1"] = float(bars[-1].close)
        return SimpleNamespace(
            timestamps_utc=batch.timestamps_utc,
            feature_names=batch.feature_names,
            rows=tuple(rows),
            indicator_backend=batch.indicator_backend,
        )

    monkeypatch.setattr(audit, "build_causal_ohlc_features", leaking_builder)
    report = audit.audit_feature_temporal_integrity(
        make_bars(),
        source_data_sha256=hashlib.sha256(b"fixture").hexdigest(),
        sample_count=4,
    )

    assert report["status"] == "FAIL"
    assert any(error.startswith("prefix_invariance_failed:") for error in report["errors"])
    assert any(error.startswith("future_perturbation_invariance_failed:") for error in report["errors"])


def test_temporal_integrity_audit_rejects_invalid_source_hash():
    with pytest.raises(ValueError, match="source_data_sha256_invalid"):
        audit.audit_feature_temporal_integrity(
            make_bars(),
            source_data_sha256="not-a-sha",
        )


def test_input_reader_accepts_bars_envelope_and_hashes_exact_source_bytes(tmp_path):
    import json

    payload = {
        "bars": [{
            "symbol": "EUR/USD",
            "timeframe": "M15",
            "timestamp": "2026-01-01T00:00:00Z",
            "open": 1.1,
            "high": 1.2,
            "low": 1.0,
            "close": 1.15,
        }]
    }
    raw = json.dumps(payload).encode("utf-8")
    path = tmp_path / "bars.json"
    path.write_bytes(raw)

    bars, digest = audit._read_bars(path)

    assert len(bars) == 1
    assert bars[0].timestamp.utcoffset() == timedelta(0)
    assert digest == hashlib.sha256(raw).hexdigest()


def test_input_reader_rejects_missing_bars(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="input_bars_missing"):
        audit._read_bars(path)



def test_input_reader_accepts_csv_with_default_instrument_metadata(tmp_path):
    raw = (
        "Timestamp,Open,High,Low,Close\n"
        "2026-01-01T00:00:00Z,1.1,1.2,1.0,1.15\n"
        "2026-01-01T00:15:00Z,1.15,1.25,1.05,1.2\n"
    ).encode("utf-8")
    path = tmp_path / "bars.csv"
    path.write_bytes(raw)

    bars, digest = audit._read_bars(path, default_symbol="EUR/USD", default_timeframe="M15")

    assert len(bars) == 2
    assert bars[0].symbol == "EUR/USD"
    assert bars[0].timeframe == "M15"
    assert digest == hashlib.sha256(raw).hexdigest()

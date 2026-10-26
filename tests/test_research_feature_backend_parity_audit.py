from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

import app.research_feature_backend_parity_audit as audit
from app.evaluation import ForexBar


def make_bars(count: int = 100) -> list[ForexBar]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = []
    for index in range(count):
        close = 1.08 + index * 0.00003 + (index % 9) * 0.000017
        open_ = close - 0.00004
        result.append(ForexBar(
            symbol="EUR/USD",
            timeframe="M15",
            timestamp=start + timedelta(minutes=15 * index),
            open=open_,
            high=close + 0.00011,
            low=open_ - 0.00013,
            close=close,
        ))
    return result


def test_backend_parity_audit_returns_safety_metadata_and_backend_results():
    report = audit.audit_feature_backend_parity(
        make_bars(),
        source_data_sha256=hashlib.sha256(b"fixture").hexdigest(),
    )

    assert report["status"] in {"PASS_WITH_WARNINGS", "FAIL", "NOT_COMPARABLE"}
    assert report["dataset_rows"] == 100
    assert report["native_backend"] in {"talib", "numpy_fallback"}
    assert report["actionable"] is False
    assert report["automatic_model_selection"] is False
    assert report["inferential"] is False
    assert report["execution_connected"] is False
    assert report["model_training_performed"] is False
    if report["comparison_performed"]:
        assert set(report["feature_results"]) == set(audit.build_causal_ohlc_features(make_bars()).feature_names)
        assert all("finite_mask_mismatch_rows" in result for result in report["feature_results"].values())
        assert report["status"] == ("PASS_WITH_WARNINGS" if not report["errors"] else "FAIL")
    else:
        assert report["status"] == "NOT_COMPARABLE"


def test_forced_numpy_backend_is_deterministically_selected():
    result = audit._build_numpy_fallback(make_bars())

    assert result.indicator_backend == "numpy_fallback"
    assert len(result.rows) == 100
    assert len(result.feature_names) > 0


def test_backend_parity_audit_rejects_invalid_hash_and_tolerances():
    bars = make_bars()
    with pytest.raises(ValueError, match="source_data_sha256_invalid"):
        audit.audit_feature_backend_parity(bars, source_data_sha256="bad")
    with pytest.raises(ValueError, match="absolute_tolerance_must_be_finite_and_nonnegative"):
        audit.audit_feature_backend_parity(
            bars,
            source_data_sha256=hashlib.sha256(b"fixture").hexdigest(),
            absolute_tolerance=-1,
        )
    with pytest.raises(ValueError, match="relative_tolerance_must_be_finite_and_nonnegative"):
        audit.audit_feature_backend_parity(
            bars,
            source_data_sha256=hashlib.sha256(b"fixture").hexdigest(),
            relative_tolerance=float("nan"),
        )


def test_backend_parity_audit_rejects_empty_dataset():
    with pytest.raises(ValueError, match="bars_must_not_be_empty"):
        audit.audit_feature_backend_parity(
            [],
            source_data_sha256=hashlib.sha256(b"fixture").hexdigest(),
        )


@pytest.mark.parametrize("raw", [
    b'{"bars":[],"bars":[{}]}', b'{"bars":[{"close":NaN}]}',
    b'{"bars":[{"close":Infinity}]}', b'{"bars":[{"close":-Infinity}]}',
])
def test_strict_json_parser_rejects_ambiguous_values(raw):
    with pytest.raises(ValueError):
        audit._loads_strict_json(raw)

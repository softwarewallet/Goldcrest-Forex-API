from __future__ import annotations

import hashlib

import pytest

from app.research_backtest_result_integrity_audit import audit_backtest_result_integrity, _load_json_object


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def manifest():
    return {
        "initial_equity": 10000.0,
        "trades": [
            {
                "trade_id": "t1", "entry_time": "2025-01-02T10:00:00+00:00",
                "close_time": "2025-01-02T11:00:00+00:00", "side": "LONG",
                "quantity": 1, "entry_price": 1.1, "exit_price": 1.11,
                "gross_pnl": 100, "commission": 2, "swap": 0, "net_pnl": 98,
            },
            {
                "trade_id": "t2", "entry_time": "2025-01-03T10:00:00+00:00",
                "close_time": "2025-01-03T11:00:00+00:00", "side": "SHORT",
                "quantity": 1, "entry_price": 1.12, "exit_price": 1.13,
                "gross_pnl": -50, "commission": 2, "swap": 0, "net_pnl": -52,
            },
        ],
        "reported_summary": {
            "trade_count": 2, "net_pnl": 46, "win_rate": 0.5, "profit_factor": 100 / 50,
        },
        "equity_curve": [
            {"timestamp": "2025-01-01T00:00:00+00:00", "equity": 10000},
            {"timestamp": "2025-01-02T11:00:00+00:00", "equity": 10098},
            {"timestamp": "2025-01-03T11:00:00+00:00", "equity": 10046},
        ],
    }


def audit(data, **kwargs):
    return audit_backtest_result_integrity(data, manifest_sha256=sha(b"manifest"), **kwargs)


def codes(report):
    return {item["code"] for item in report["errors"]}


def test_consistent_ledger_summary_and_equity_curve_pass():
    report = audit(manifest())
    assert report["status"] == "PASS"
    assert report["trades_validated"] == 2
    assert report["net_pnl_recomputed_from_valid_trades"] == 46
    assert report["equity_curve"]["final_equity"] == 10046
    assert report["execution_connected"] is False


def test_net_pnl_and_summary_mismatch_fail():
    data = manifest()
    data["trades"][0]["net_pnl"] = 101
    data["reported_summary"]["net_pnl"] = 999
    report = audit(data)
    assert "net_pnl_reconciliation_mismatch" in codes(report)
    assert "reported_net_pnl_mismatch" in codes(report)


def test_duplicate_trade_id_and_invalid_timestamps_fail():
    data = manifest()
    data["trades"][1]["trade_id"] = "t1"
    data["trades"][1]["entry_time"] = "not-a-timestamp"
    report = audit(data)
    assert "duplicate_trade_id" in codes(report)
    assert "trade_timestamps_invalid" in codes(report)


def test_invalid_equity_curve_and_final_equity_mismatch_fail():
    data = manifest()
    data["equity_curve"][-1]["equity"] = 11000
    data["equity_curve"][1]["timestamp"] = data["equity_curve"][0]["timestamp"]
    report = audit(data)
    assert "equity_curve_final_equity_mismatch" in codes(report)
    assert "equity_curve_timestamps_not_strictly_increasing" in codes(report)


def test_profit_factor_and_win_rate_mismatch_fail():
    data = manifest()
    data["reported_summary"]["profit_factor"] = 1
    data["reported_summary"]["win_rate"] = 0.9
    report = audit(data)
    assert "reported_profit_factor_mismatch" in codes(report)
    assert "reported_win_rate_mismatch" in codes(report)


def test_missing_equity_curve_is_explicit_warning():
    data = manifest()
    data.pop("equity_curve")
    report = audit(data)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert "equity_curve_not_supplied; drawdown path cannot be independently checked" in report["warnings"]


def test_invalid_costs_and_side_fail_closed():
    data = manifest()
    data["trades"][0]["commission"] = -1
    data["trades"][1]["side"] = "SIDEWAYS"
    report = audit(data)
    assert "commission_must_be_nonnegative" in codes(report)
    assert "trade_side_invalid" in codes(report)


def test_invalid_hash_and_tolerance_rejected():
    with pytest.raises(ValueError, match="manifest_sha256_invalid"):
        audit_backtest_result_integrity(manifest(), manifest_sha256="bad")
    with pytest.raises(ValueError, match="equity_tolerance"):
        audit(manifest(), equity_tolerance=-1)


@pytest.mark.parametrize("raw", [
    b'{"status":"PASS","status":"FAIL"}',
    b'{"metric":NaN}',
    b'{"metric":Infinity}',
    b'{"metric":-Infinity}',
])
def test_json_loader_rejects_duplicate_keys_and_nonstandard_constants(raw):
    with pytest.raises(ValueError):
        _load_json_object(raw, source="test-manifest.json")

"""Offline integrity audit for backtest trade ledgers and reported performance metrics.

Validates accounting identities and chronology in supplied artifacts. It does not fetch
prices, reconstruct trades, train/select models, or connect to broker execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

_FALSE_FLAGS = {
    "actionable": False,
    "inferential": False,
    "statistical_significance_claimed": False,
    "automatic_model_selection": False,
    "model_training_performed": False,
    "execution_connected": False,
    "automatic_repair_performed": False,
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdefABCDEF" for char in value
    )


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed
    except ValueError:
        return None


def audit_backtest_result_integrity(
    manifest: dict[str, Any],
    *,
    manifest_sha256: str,
    pnl_tolerance: float = 0.01,
    equity_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Validate trade-level arithmetic, timestamp ordering, and summary reconciliation."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest_must_be_object")
    if not _valid_hash(manifest_sha256):
        raise ValueError("manifest_sha256_invalid")
    for name, value in (("pnl_tolerance", pnl_tolerance), ("equity_tolerance", equity_tolerance)):
        if not _finite(value) or float(value) < 0:
            raise ValueError(f"{name}_must_be_finite_nonnegative_number")

    trades = manifest.get("trades")
    if not isinstance(trades, list) or not trades:
        raise ValueError("trades_must_be_nonempty_list")
    initial_equity = manifest.get("initial_equity")
    if not _finite(initial_equity) or float(initial_equity) <= 0:
        raise ValueError("initial_equity_must_be_positive_finite_number")

    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    last_close: datetime | None = None
    total_net = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    wins = losses = 0

    for index, trade in enumerate(trades):
        if not isinstance(trade, dict):
            errors.append({"code": "trade_record_invalid", "index": index})
            continue
        tid = trade.get("trade_id")
        entry = _timestamp(trade.get("entry_time"))
        close = _timestamp(trade.get("close_time"))
        side = trade.get("side")
        quantity = trade.get("quantity")
        entry_price = trade.get("entry_price")
        exit_price = trade.get("exit_price")
        gross_pnl = trade.get("gross_pnl")
        commission = trade.get("commission", 0.0)
        swap = trade.get("swap", 0.0)
        reported_net = trade.get("net_pnl")
        item_errors: list[str] = []

        if not isinstance(tid, str) or not tid.strip():
            item_errors.append("trade_id_invalid")
        elif tid in seen_ids:
            item_errors.append("duplicate_trade_id")
        else:
            seen_ids.add(tid)
        if entry is None or close is None or (entry is not None and close is not None and entry > close):
            item_errors.append("trade_timestamps_invalid")
        elif last_close is not None and entry < last_close:
            item_errors.append("trade_chronology_overlap_or_out_of_order")
        if close is not None:
            last_close = close
        if side not in {"LONG", "SHORT", "BUY", "SELL"}:
            item_errors.append("trade_side_invalid")
        for key, value in (
            ("quantity", quantity), ("entry_price", entry_price),
            ("exit_price", exit_price), ("gross_pnl", gross_pnl),
            ("commission", commission), ("swap", swap), ("net_pnl", reported_net),
        ):
            if not _finite(value):
                item_errors.append(f"{key}_not_finite_number")
        if _finite(quantity) and float(quantity) <= 0:
            item_errors.append("quantity_must_be_positive")
        if _finite(entry_price) and float(entry_price) <= 0:
            item_errors.append("entry_price_must_be_positive")
        if _finite(exit_price) and float(exit_price) <= 0:
            item_errors.append("exit_price_must_be_positive")
        if _finite(commission) and float(commission) < 0:
            item_errors.append("commission_must_be_nonnegative")

        calculated_net = None
        if all(_finite(v) for v in (gross_pnl, commission, swap, reported_net)):
            calculated_net = float(gross_pnl) - float(commission) + float(swap)
            if abs(calculated_net - float(reported_net)) > float(pnl_tolerance):
                item_errors.append("net_pnl_reconciliation_mismatch")
        for code in item_errors:
            errors.append({"code": code, "index": index, "trade_id": tid if isinstance(tid, str) else None})
        normalized.append({
            "trade_id": tid if isinstance(tid, str) else f"invalid_{index}",
            "net_pnl": float(reported_net) if _finite(reported_net) else None,
            "gross_pnl": float(gross_pnl) if _finite(gross_pnl) else None,
            "valid": not item_errors,
        })
        if not item_errors:
            net = float(reported_net)
            total_net += net
            if net > 0:
                wins += 1
                gross_profit += net
            elif net < 0:
                losses += 1
                gross_loss += abs(net)

    valid_trades = [trade for trade in normalized if trade["valid"]]
    if not valid_trades:
        errors.append({"code": "no_valid_trades_for_reconciliation"})

    summary = manifest.get("reported_summary")
    if summary is not None:
        if not isinstance(summary, dict):
            errors.append({"code": "reported_summary_must_be_object"})
        else:
            reported_count = summary.get("trade_count")
            if not isinstance(reported_count, int) or isinstance(reported_count, bool) or reported_count < 0:
                errors.append({"code": "reported_trade_count_invalid"})
            elif reported_count != len(trades):
                errors.append({"code": "reported_trade_count_mismatch", "reported": reported_count, "observed": len(trades)})
            if _finite(summary.get("net_pnl")) and abs(float(summary["net_pnl"]) - total_net) > float(pnl_tolerance):
                errors.append({"code": "reported_net_pnl_mismatch", "reported": float(summary["net_pnl"]), "recomputed": total_net})
            elif "net_pnl" in summary and not _finite(summary.get("net_pnl")):
                errors.append({"code": "reported_net_pnl_invalid"})
            if _finite(summary.get("win_rate")):
                observed_win_rate = wins / len(valid_trades) if valid_trades else 0.0
                if not 0 <= float(summary["win_rate"]) <= 1:
                    errors.append({"code": "reported_win_rate_out_of_range"})
                elif abs(float(summary["win_rate"]) - observed_win_rate) > 1e-6:
                    errors.append({"code": "reported_win_rate_mismatch", "reported": float(summary["win_rate"]), "recomputed": observed_win_rate})
            elif "win_rate" in summary:
                errors.append({"code": "reported_win_rate_invalid"})
            if _finite(summary.get("profit_factor")):
                observed_pf = gross_profit / gross_loss if gross_loss > 0 else (None if gross_profit == 0 else math.inf)
                if observed_pf is None or not math.isfinite(observed_pf):
                    warnings.append("profit_factor_not_comparable_due_to_zero_gross_loss_or_zero_pnl")
                elif abs(float(summary["profit_factor"]) - observed_pf) > 1e-6:
                    errors.append({"code": "reported_profit_factor_mismatch", "reported": float(summary["profit_factor"]), "recomputed": observed_pf})
            elif "profit_factor" in summary:
                errors.append({"code": "reported_profit_factor_invalid"})

    equity_curve = manifest.get("equity_curve")
    equity_result: dict[str, Any] = {"supplied": equity_curve is not None}
    if equity_curve is not None:
        if not isinstance(equity_curve, list) or not equity_curve:
            errors.append({"code": "equity_curve_must_be_nonempty_list"})
        else:
            parsed_curve: list[tuple[datetime, float]] = []
            for index, point in enumerate(equity_curve):
                if not isinstance(point, dict):
                    errors.append({"code": "equity_point_invalid", "index": index})
                    continue
                stamp = _timestamp(point.get("timestamp"))
                value = point.get("equity")
                if stamp is None or not _finite(value) or float(value) <= 0:
                    errors.append({"code": "equity_point_invalid", "index": index})
                    continue
                parsed_curve.append((stamp, float(value)))
            if any(a[0] >= b[0] for a, b in zip(parsed_curve, parsed_curve[1:])):
                errors.append({"code": "equity_curve_timestamps_not_strictly_increasing"})
            if parsed_curve:
                expected_final = float(initial_equity) + total_net
                actual_final = parsed_curve[-1][1]
                equity_result.update({
                    "points_validated": len(parsed_curve),
                    "initial_equity": parsed_curve[0][1],
                    "final_equity": actual_final,
                    "expected_final_equity_from_trade_ledger": expected_final,
                })
                if abs(parsed_curve[0][1] - float(initial_equity)) > float(equity_tolerance):
                    errors.append({"code": "equity_curve_initial_equity_mismatch"})
                if abs(actual_final - expected_final) > float(equity_tolerance):
                    errors.append({"code": "equity_curve_final_equity_mismatch", "reported": actual_final, "recomputed": expected_final})
                peak = parsed_curve[0][1]
                max_drawdown = 0.0
                for _, value in parsed_curve:
                    peak = max(peak, value)
                    if peak > 0:
                        max_drawdown = max(max_drawdown, (peak - value) / peak)
                equity_result["maximum_drawdown_fraction"] = max_drawdown
            else:
                equity_result["points_validated"] = 0
    else:
        warnings.append("equity_curve_not_supplied; drawdown path cannot be independently checked")

    if len(valid_trades) != len(trades):
        warnings.append("summary_reconciliation_uses_valid_trades_only; inspect trade-level errors")
    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "schema_version": 1,
        "audit_name": "backtest_result_integrity",
        "manifest_sha256": manifest_sha256.lower(),
        "status": status,
        "trades_supplied": len(trades),
        "trades_validated": len(valid_trades),
        "net_pnl_recomputed_from_valid_trades": total_net,
        "gross_profit_from_valid_trades": gross_profit,
        "gross_loss_from_valid_trades": gross_loss,
        "win_count": wins,
        "loss_count": losses,
        "win_rate_from_valid_trades": wins / len(valid_trades) if valid_trades else None,
        "equity_curve": equity_result,
        "errors": errors,
        "warnings": warnings,
        **_FALSE_FLAGS,
        "scope_limitations": [
            "Reconciles supplied ledger and summary fields; it does not verify fills against broker records or market prices.",
            "Assumes net_pnl = gross_pnl - commission + swap with commission represented as a nonnegative cost.",
            "Equity-curve reconciliation assumes no external deposits, withdrawals, funding, or unlisted fees.",
            "A passing integrity audit does not prove strategy profitability, generalization, or deployment readiness.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pnl-tolerance", type=float, default=0.01)
    parser.add_argument("--equity-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
        report = audit_backtest_result_integrity(
            manifest,
            manifest_sha256=_sha256(raw),
            pnl_tolerance=args.pnl_tolerance,
            equity_tolerance=args.equity_tolerance,
        )
        output = args.output or args.input.with_name(args.input.stem + "_backtest_integrity_audit.json")
        if output.exists():
            raise ValueError(f"output_already_exists:{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "report_path": str(output.resolve()),
            "trades_validated": report["trades_validated"],
            "error_count": len(report["errors"]),
            "warning_count": len(report["warnings"]),
            "execution_connected": False,
        }, indent=2))
        return 0 if report["status"] != "FAIL" else 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

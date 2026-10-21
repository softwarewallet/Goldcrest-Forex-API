"""Isolated Backtrader replay for externally supplied historical signals.

Orders are simulated locally and submitted after the current bar's close, so the default
Backtrader fill model uses a later bar (normally next open). No broker/network adapters are
configured. Signal generation and label construction are deliberately outside this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "signal")


def _load_signal_frame(csv_path: str | Path) -> tuple[pd.DataFrame, str]:
    path = Path(csv_path)
    raw = path.read_bytes()
    frame = pd.read_csv(path)
    frame.columns = [str(name).strip().lower() for name in frame.columns]
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"signal CSV missing columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if frame.empty:
        raise ValueError("signal CSV must not be empty")
    if frame["timestamp"].duplicated().any() or not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps must be unique and strictly chronological")
    for name in ("open", "high", "low", "close", "signal"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if frame[list(REQUIRED_COLUMNS[1:])].isna().any().any():
        raise ValueError("OHLC/signal values must not be null")
    if not frame["signal"].isin([-1, 0, 1]).all():
        raise ValueError("signal must be -1 (short), 0 (flat), or 1 (long)")
    if (frame["close"] <= 0).any() or (frame["high"] < frame[["open", "close"]].max(axis=1)).any() or (frame["low"] > frame[["open", "close"]].min(axis=1)).any() or (frame["high"] < frame["low"]).any():
        raise ValueError("invalid OHLC relationships")
    frame = frame.set_index("timestamp")
    frame["volume"] = pd.to_numeric(frame.get("volume", 0.0), errors="coerce").fillna(0.0) if "volume" in frame else 0.0
    return frame, hashlib.sha256(raw).hexdigest()


def run_backtrader_replay(
    csv_path: str | Path,
    *,
    initial_cash: float = 100_000.0,
    stake: int = 1,
    commission_rate: float = 0.0002,
    slippage_rate: float = 0.0001,
) -> dict[str, Any]:
    """Replay externally generated -1/0/1 signals with explicit friction assumptions."""
    for name, value in (("initial_cash", initial_cash), ("stake", stake)):
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= commission_rate < 1 or not 0 <= slippage_rate < 1:
        raise ValueError("commission_rate and slippage_rate must be in [0, 1)")
    frame, source_sha256 = _load_signal_frame(csv_path)
    try:
        import backtrader as bt
    except ImportError as exc:
        raise RuntimeError("backtrader is required; install the project requirements") from exc

    class SignalPandasData(bt.feeds.PandasData):
        lines = ("signal",)
        params = (("signal", "signal"),)

    class ReplayStrategy(bt.Strategy):
        params = (("stake", stake),)

        def next(self):
            signal = int(self.data.signal[0])
            target = signal * self.p.stake
            if self.position.size != target:
                self.order_target_size(target=target)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(ReplayStrategy, stake=stake)
    cerebro.adddata(SignalPandasData(dataname=frame))
    cerebro.broker.setcash(float(initial_cash))
    cerebro.broker.setcommission(commission=float(commission_rate))
    cerebro.broker.set_slippage_perc(float(slippage_rate))
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.broker.set_coc(False)
    start_value = float(cerebro.broker.getvalue())
    strategies = cerebro.run()
    strategy = strategies[0]
    end_value = float(cerebro.broker.getvalue())

    def safe_analysis(name: str) -> Any:
        try:
            return getattr(strategy.analyzers, name).get_analysis()
        except (AttributeError, TypeError, ValueError):
            return {}

    payload = {
        "schema_version": 1,
        "scope": "OFFLINE_BACKTRADER_REPLAY",
        "actionable": False,
        "source_csv_sha256": source_sha256,
        "observation_count": int(len(frame)),
        "first_timestamp_utc": frame.index[0].isoformat().replace("+00:00", "Z"),
        "last_timestamp_utc": frame.index[-1].isoformat().replace("+00:00", "Z"),
        "initial_cash": start_value,
        "final_value": end_value,
        "net_return_fraction": (end_value / start_value) - 1.0,
        "stake_units": stake,
        "commission_rate": commission_rate,
        "slippage_rate": slippage_rate,
        "execution_assumption": "Signal read at bar close; cheat-on-close disabled; orders are simulated on subsequent bars.",
        "returns_analysis": safe_analysis("returns"),
        "drawdown_analysis": safe_analysis("drawdown"),
        "trade_analysis": safe_analysis("trades"),
        "sharpe_analysis": safe_analysis("sharpe"),
        "limitations": [
            "Signals are externally supplied and must be point-in-time valid.",
            "This is a local bar replay, not broker execution or a guarantee of realistic fills.",
            "Instrument contract size, financing, spread dynamics, margin, swaps, and market impact require instrument-specific modeling.",
            "Predictive accuracy and simulated returns do not guarantee future profitability.",
        ],
    }
    # Ensure analysis objects can be written as JSON without silently coercing unknown objects.
    json.dumps(payload, default=str, allow_nan=False)
    return payload


def write_backtrader_report(report: dict[str, Any], path: str | Path) -> str:
    if report.get("actionable") is not False:
        raise ValueError("research reports must remain non-actionable")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    report["report_payload_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")
    return str(destination)

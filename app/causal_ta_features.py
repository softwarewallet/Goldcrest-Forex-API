"""Causal, offline OHLC feature engineering for historical research.

Every feature at row t uses only bars at or before t. Warm-up values are None rather
than silently imputed. TA-Lib is used when installed; deterministic NumPy fallbacks keep
CI and lightweight environments usable. This module has no broker or live dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from typing import Sequence

import numpy as np

from app.evaluation import ForexBar


FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "returns": ("return_1", "return_3", "return_5"),
    "candle_structure": ("range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct"),
    "volatility": ("volatility_5", "volatility_10", "atr_14_pct", "bollinger_width_20"),
    "momentum": ("rsi_14", "macd_hist_pct", "ema_gap_12_26_pct", "close_zscore_20"),
}


@dataclass(frozen=True)
class CausalFeatureBatch:
    timestamps_utc: tuple[str, ...]
    feature_names: tuple[str, ...]
    rows: tuple[dict[str, float | None], ...]
    indicator_backend: str
    causal_rule: str = "Features at timestamp t use only OHLC bars with timestamp <= t."
    actionable: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamps_utc": list(self.timestamps_utc),
            "feature_names": list(self.feature_names),
            "rows": list(self.rows),
            "indicator_backend": self.indicator_backend,
            "causal_rule": self.causal_rule,
            "actionable": False,
        }


def _rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        if np.isfinite(window).all():
            result[i] = float(np.std(window, ddof=0))
    return result


def _rolling_mean(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        if np.isfinite(window).all():
            result[i] = float(np.mean(window))
    return result


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    if len(values) < period:
        return result
    alpha = 2.0 / (period + 1.0)
    result[period - 1] = float(np.mean(values[:period]))
    for i in range(period, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result


def _fallback_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    result = np.full(len(close), np.nan, dtype=float)
    if len(close) <= period:
        return result
    delta = np.diff(close)
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    def value(gain: float, loss: float) -> float:
        if loss == 0:
            return 100.0 if gain > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + gain / loss)
    result[period] = value(avg_gain, avg_loss)
    for i in range(period + 1, len(close)):
        avg_gain = ((period - 1) * avg_gain + gains[i - 1]) / period
        avg_loss = ((period - 1) * avg_loss + losses[i - 1]) / period
        result[i] = value(avg_gain, avg_loss)
    return result


def _as_optional_rows(matrix: dict[str, np.ndarray], count: int) -> tuple[dict[str, float | None], ...]:
    rows: list[dict[str, float | None]] = []
    for i in range(count):
        row: dict[str, float | None] = {}
        for name, values in matrix.items():
            value = float(values[i])
            row[name] = value if np.isfinite(value) else None
        rows.append(row)
    return tuple(rows)


def build_causal_ohlc_features(bars: Sequence[ForexBar]) -> CausalFeatureBatch:
    """Build causal features from strictly chronological bars for one instrument/timeframe."""
    if not bars:
        raise ValueError("bars must not be empty")
    if len({(bar.symbol, bar.timeframe) for bar in bars}) != 1:
        raise ValueError("features require exactly one symbol/timeframe")
    timestamps = [bar.timestamp for bar in bars]
    if any(ts.tzinfo is None or ts.utcoffset() is None for ts in timestamps):
        raise ValueError("bar timestamps must be timezone-aware")
    utc_times = [ts.astimezone(timezone.utc) for ts in timestamps]
    if utc_times != sorted(utc_times) or len(set(utc_times)) != len(utc_times):
        raise ValueError("bars must be strictly chronological with unique timestamps")
    for bar in bars:
        values = (bar.open, bar.high, bar.low, bar.close)
        if not all(np.isfinite(v) and v > 0 for v in values) or bar.low > min(bar.open, bar.close) or bar.high < max(bar.open, bar.close) or bar.high < bar.low:
            raise ValueError("bars contain invalid OHLC values")

    open_ = np.asarray([b.open for b in bars], dtype=float)
    high = np.asarray([b.high for b in bars], dtype=float)
    low = np.asarray([b.low for b in bars], dtype=float)
    close = np.asarray([b.close for b in bars], dtype=float)
    n = len(bars)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan

    def lag_return(period: int) -> np.ndarray:
        result = np.full(n, np.nan, dtype=float)
        if n > period:
            result[period:] = close[period:] / close[:-period] - 1.0
        return result

    true_range = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    true_range[0] = high[0] - low[0]
    atr14 = _rolling_mean(true_range, 14)
    log_returns = np.full(n, np.nan, dtype=float)
    if n > 1:
        log_returns[1:] = np.diff(np.log(close))
    vol5 = _rolling_std(log_returns, 5) * np.sqrt(5.0)
    vol10 = _rolling_std(log_returns, 10) * np.sqrt(10.0)
    mean20 = _rolling_mean(close, 20)
    std20 = _rolling_std(close, 20)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    ema12_26 = ema12 - ema26
    macd = ema12 - ema26
    signal = _ema(np.where(np.isfinite(macd), macd, 0.0), 9)
    # Suppress signal warm-up until MACD itself is available.
    signal[:33] = np.nan
    macd_hist = macd - signal

    backend = "numpy_fallback"
    try:
        import talib  # type: ignore[import-not-found]
        rsi = talib.RSI(close, timeperiod=14)
        atr14 = talib.ATR(high, low, close, timeperiod=14)
        macd_line, signal_line, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_hist = hist
        ema12_26 = talib.EMA(close, timeperiod=12) - talib.EMA(close, timeperiod=26)
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        mean20 = middle
        std20 = (upper - lower) / 4.0
        backend = "talib"
    except ImportError:
        rsi = _fallback_rsi(close, 14)

    matrix = {
        "return_1": lag_return(1),
        "return_3": lag_return(3),
        "return_5": lag_return(5),
        "range_pct": (high - low) / close,
        "body_pct": (close - open_) / open_,
        "upper_wick_pct": (high - np.maximum(open_, close)) / close,
        "lower_wick_pct": (np.minimum(open_, close) - low) / close,
        "volatility_5": vol5,
        "volatility_10": vol10,
        "atr_14_pct": atr14 / close,
        "bollinger_width_20": (4.0 * std20) / mean20,
        "rsi_14": np.asarray(rsi, dtype=float) / 100.0,
        "macd_hist_pct": macd_hist / close,
        "ema_gap_12_26_pct": ema12_26 / close,
        "close_zscore_20": (close - mean20) / std20,
    }
    # Explicitly avoid infinities from zero-variance windows.
    for name, values in matrix.items():
        matrix[name] = np.where(np.isfinite(values), values, np.nan)
    return CausalFeatureBatch(
        timestamps_utc=tuple(ts.isoformat().replace("+00:00", "Z") for ts in utc_times),
        feature_names=tuple(matrix),
        rows=_as_optional_rows(matrix, n),
        indicator_backend=backend,
    )


def feature_ablation_sets(feature_names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Return deterministic all-feature and leave-one-family-out feature sets."""
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")
    unknown = set(names) - {name for family in FEATURE_FAMILIES.values() for name in family}
    if unknown:
        raise ValueError(f"unknown causal feature names: {sorted(unknown)}")
    sets: dict[str, tuple[str, ...]] = {"all_features": names}
    for family_name, family in FEATURE_FAMILIES.items():
        excluded = set(family)
        retained = tuple(name for name in names if name not in excluded)
        if retained:
            sets[f"without_{family_name}"] = retained
    return sets

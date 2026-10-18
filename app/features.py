from math import isfinite

from app.schemas import Candle


def _ema(values: list[float], period: int) -> float:
    alpha = 2.0 / (period + 1)
    value = sum(values[:period]) / period
    for current in values[period:]:
        value = alpha * current + (1 - alpha) * value
    return value


def calculate_features(candles: list[Candle]) -> dict[str, float | str]:
    """Calculate a transparent starter feature set, not a validated trading strategy."""
    closes = [float(c.close) for c in candles]
    highs = [float(c.high) for c in candles]
    lows = [float(c.low) for c in candles]
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent = changes[-14:]
    gains = [max(x, 0.0) for x in recent]
    losses = [max(-x, 0.0) for x in recent]
    gain = sum(gains) / max(1, len(gains))
    loss = sum(losses) / max(1, len(losses))
    rsi = 50.0 if gain == 0 and loss == 0 else (100.0 if loss == 0 else 100.0 - 100.0 / (1.0 + gain / loss))
    true_ranges = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                   for i in range(1, len(candles))]
    atr = sum(true_ranges[-14:]) / max(1, len(true_ranges[-14:]))
    ema9, ema21 = _ema(closes, 9), _ema(closes, 21)
    values = {"last_close": closes[-1], "ema_9": ema9, "ema_21": ema21,
              "rsi_14": rsi, "atr_14": atr,
              "return_over_window": (closes[-1] - closes[0]) / closes[0]}
    if not all(isfinite(v) for v in values.values()):
        raise ValueError("Non-finite market features")
    regime = "TRENDING_UP" if ema9 > ema21 else "TRENDING_DOWN" if ema9 < ema21 else "UNCERTAIN"
    return {**values, "quantitative_regime": regime}

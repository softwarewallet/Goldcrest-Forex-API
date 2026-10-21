# Causal TA-Lib feature research

## Scope and safety boundary

`app.causal_ta_features` builds point-in-time features from a single chronological OHLC
series for offline research. It has no data-fetching, broker, inference-serving, or order
execution code. Every result and feature batch is explicitly `actionable=false`.

## Feature families

- **Returns:** 1-, 3-, and 5-bar close returns.
- **Candle structure:** normalized high-low range, body, upper wick, and lower wick.
- **Volatility:** 5-/10-bar log-return volatility, normalized 14-bar ATR, and 20-bar Bollinger width.
- **Momentum:** normalized RSI(14), normalized MACD histogram, normalized EMA(12)-EMA(26) gap,
  and 20-bar close z-score.

TA-Lib is used when importable. A NumPy fallback is provided for lightweight environments.
The report identifies the backend actually used. Warm-up windows are represented as
JSON-safe `null` values, not silently imputed values. A downstream experiment must exclude
or explicitly handle warm-up rows using a rule fixed before examining test results.

## Causality and validation

- Rows must be timezone-aware, unique, strictly chronological, and from one symbol/timeframe.
- No future bars are used to calculate a row's features.
- The test suite changes future bars and asserts all earlier feature rows remain identical.
- This verifies temporal computation, but cannot prove that source OHLC data or any
  separately prepared labels are historically correct.
- This module does not create labels or decide class semantics.

## Feature ablation

`feature_ablation_sets` produces a deterministic full feature set and leave-one-family-out
sets for returns, candle structure, volatility, and momentum. Compare these variants using
identical chronological folds, the same label definition, costs/selection protocol, and
independent test windows. Do not choose a feature set by repeatedly optimizing on a final
test partition.

## Example

```python
from app.causal_ta_features import build_causal_ohlc_features, feature_ablation_sets

batch = build_causal_ohlc_features(bars)
feature_variants = feature_ablation_sets(batch.feature_names)
print(batch.indicator_backend, batch.feature_names)
```

The feature generator is intentionally separate from the existing inference API and
Auto Live/trade execution paths. Predictive metrics from downstream experiments are not
profitability evidence.

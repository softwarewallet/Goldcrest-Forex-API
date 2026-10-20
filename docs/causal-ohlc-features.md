# Causal OHLC features (offline research)

`app/causal_features.py` provides deterministic feature rows computed at the
close of each finalized OHLC bar. It does not load data, create labels, train
models, serve predictions, or place orders.

## Features

- Close-to-close fractional returns over 1, 3, and 5 bars.
- High-low range and close-open body, each scaled by the previous close.
- Population standard deviation of close-to-close returns over trailing 5 and
  10 return windows.

The first output is at bar index 10 because ten complete close-to-close returns
are required. No future bar is read to construct a row, and adding future bars
must not change already emitted feature rows.

## Data contract and limitations

- Input bars must represent one instrument and one timeframe, be strictly
  chronological, use timezone-aware timestamps, and have finite positive OHLC
  values with consistent high/low bounds.
- The timestamp denotes the finalized bar close / point at which the row is
  available. Do not use an unfinished bar for evaluation.
- The function deliberately does not resample, fill gaps, normalize across
  instruments, infer missing bars, or impute warm-up values.
- Callers must audit source timestamps, revisions, session boundaries, and
  whether each feature would truly have been available at decision time.
- These are baseline candidate features only. No predictive benefit, calibrated
  probability, profitability, or trading readiness is implied.
- Synthetic tests establish deterministic mechanics, not market performance.
- No API, broker, order execution, or live prediction path is modified.
  Research output remains non-actionable and serving remains `NO_TRADE`.


## Extended causal indicator set

The current preparer excludes the first 34 rows for indicator initialization and emits
12 features: returns over 1/3/5/10 bars, normalized range and candle body, close location,
14-return volatility, normalized EMA(12)-EMA(26) gap, Wilder RSI(14), normalized
Wilder ATR(14), and normalized MACD histogram (12/26 with a 9-period signal EMA).
The JSON audit records each feature definition and warm-up policy. All indicators use
only the current completed candle and earlier candles; the causality regression test
perturbs the final source candle and verifies that earlier feature rows remain unchanged.

These are candidate features for research, not a claim that any indicator improves
out-of-sample prediction. Freeze feature selection before final-test evaluation.

# Unified research data preparation

This phase connects the existing historical-data adapters and causal feature engine to the
existing research CSV contract. It is deliberately offline after the source snapshot exists.
It does not authenticate to cTrader, fetch data, create predictions, or connect to execution.

## Pipeline

1. Retrieve historical bars explicitly through the existing cTrader Open API client or the
   opt-in yfinance adapter.
2. For cTrader Open API trendbars, pass the already-returned response objects to
   `app.ctrader_history.normalize_ctrader_trendbars`. It emits canonical UTC OHLC CSV and
   a SHA-256 manifest. It does not open a connection or authenticate.
3. Pass the canonical OHLC CSV to `app.research_dataset_builder`.
4. The builder validates the data without silently sorting or imputing; creates causal
   TA-Lib/NumPy features; creates forward-close direction labels; and records the timestamp
   at which each label becomes available.
5. Feed the output CSV into the existing historical tabular research / Optuna walk-forward
   CLI. It expects `symbol,timeframe,timestamp,open,high,low,close,label,label_available_at`
   plus numeric feature columns.
6. Run walk-forward tests and the fixed-fold experiment matrix separately. Research outputs
   remain non-actionable.

## cTrader normalization example

Call this function with trendbars already returned by the installed cTrader Open API SDK:

```python
from app.ctrader_history import normalize_ctrader_trendbars

manifest = normalize_ctrader_trendbars(
    trendbars,  # response trendbars from the existing API client
    symbol="EURUSD",
    timeframe="M15",
    output_csv="data/research/EURUSD_M15_ctrader.csv",
    price_scale=100000,
)
```

The protobuf trendbar encoding uses a base `low` and price deltas. The adapter defaults to
100000 but requires operators to verify the scale against the instrument's precision and
the SDK response representation. It rejects invalid OHLC, duplicate timestamps, and
out-of-order bars rather than silently repairing them.

## Build a model-ready dataset

```bash
python -m app.research_dataset_builder \
  --input-csv data/research/EURUSD_M15_ctrader.csv \
  --output-csv data/research/EURUSD_M15_dataset.csv \
  --manifest data/research/EURUSD_M15_dataset.manifest.json \
  --symbol EURUSD \
  --timeframe M15 \
  --horizon-bars 4 \
  --class-1-means UP \
  --data-source "cTrader Open API" \
  --source-version "0.9.2" \
  --retrieved-at-utc "2026-10-21T12:00:00Z" \
  --terms-reference "<actual applicable provider terms reference>"
```

Replace the retrieval time and terms reference with the actual values for the source snapshot.
Do not invent licensing/terms metadata. For yfinance, provide its exported canonical CSV and
identify the source as Yahoo Finance via yfinance.

The binary label is direction of the close at horizon `t + horizon_bars` relative to close at
`t`. `label_available_at` is the future bar timestamp, so the existing walk-forward engine
can exclude labels that had not matured by each training/validation/test cutoff. Equality is
class 0. `class_1_means` explicitly determines which direction maps to class 1.

## Guarantees and limitations

- Every technical feature at timestamp t uses bars timestamped at or before t.
- Feature warm-up rows and trailing rows without a complete label horizon are excluded.
- Source and output CSVs receive SHA-256 hashes; the manifest is hash-bound and marked
  `actionable=false`.
- No missing data imputation or automatic sorting is performed.
- A dataset build does not prove the provider source is authentic; source metadata is
  operator-supplied.
- This phase does not yet run model fitting, PyTorch, llama.cpp, Vega, or Nova-hunting. Those
  require separate, explicitly tested orchestration adapters and should not be described as
  integrated until they have executed in a recorded run.
- No Auto Live, broker authentication, risk control, or order-execution code is changed.

## Tests

```bash
python -m pytest -q tests/test_research_orchestrator.py
```

Tests use synthetic data and local objects; they do not contact cTrader or Yahoo Finance.

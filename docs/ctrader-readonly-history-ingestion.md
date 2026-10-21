# Read-only cTrader history ingestion

This command downloads historical OHLC from cTrader Open API for offline research. It uses
the installed `ctrader-open-api==0.9.2` SDK and Twisted callbacks. It has no order API
imports and cannot place, modify, or cancel trades.

## Credentials

Export credentials in the terminal session. Do not commit a .env file, token, client secret,
or account-specific credential file:

```bash
export CTRADER_CLIENT_ID="your-application-client-id"
export CTRADER_CLIENT_SECRET="your-application-client-secret"
export CTRADER_ACCESS_TOKEN="your-cTrader-access-token"
export CTRADER_ACCOUNT_ID="your-demo-account-id"
export CTRADER_HOST="demo"
```

The command requires all four credential values. The host is deliberately restricted to
DEMO; a LIVE value fails closed. No credential value is printed, added to output manifests,
or persisted to disk. Use an access token valid for the selected demo account.

## Download a historical snapshot

```bash
python -m app.ctrader_history_ingest \
  --symbol EURUSD \
  --timeframe M15 \
  --lookback-days 365 \
  --output-csv data/research/EURUSD_M15_ctrader.csv
```

Supported periods: M1, M5, M15, M30, H1, H4, D1. The exact symbol name must exist on the
selected cTrader demo account. The program authenticates the API application, authorizes the
demo account, resolves the symbol through the account symbol list, and requests trendbars.
The snapshot CSV and a JSON manifest are written locally. The manifest includes the request
window, data-source scope, row count, and SHA-256 snapshot hash.

cTrader trendbar values are decoded using the documented representation: `low` is the base
price and open/high/close are base plus their deltas, divided by 100000. If the broker/SDK
returns a different representation, the normalizer rejects invalid OHLC instead of silently
repairing it. Validate a sample snapshot against cTrader's own chart before using it in
experiments.

## Continue into feature/label preparation

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
  --source-version "ctrader-open-api 0.9.2" \
  --retrieved-at-utc "<actual UTC retrieval time>" \
  --terms-reference "<actual applicable provider terms reference>"
```

Use the actual retrieval time and applicable provider terms. Never put credentials in metadata.

## Safety and diagnostics

- Default and only allowed endpoint: DEMO.
- Read-only requests only: application auth, account auth, symbol list, historical trendbars.
- No Auto Live or broker execution modules are imported or changed.
- Errors print only the diagnostic exception type to avoid accidentally exposing request or
  authentication payloads.
- Empty, duplicate, out-of-order, or invalid OHLC responses fail closed.
- Output is historical research data, never an actionable signal.

## Tests

```bash
python -m pytest -q tests/test_ctrader_history_ingest.py tests/test_research_orchestrator.py
```

The tests use synthetic responses and do not contact cTrader.

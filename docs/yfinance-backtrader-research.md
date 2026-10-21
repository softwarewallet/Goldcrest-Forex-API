# yfinance and Backtrader research adapters

## yfinance historical snapshots

`app.yfinance_history.download_yfinance_ohlc` is an explicit, opt-in downloader for historical
Yahoo Finance OHLCV. Common FX pair symbols such as `EUR/USD` map to `EURUSD=X`. The adapter
disables auto-adjustment, validates timestamps/OHLC, exports a stable CSV snapshot, and can
write a JSON manifest containing the snapshot SHA-256 and coverage window.

Yahoo Finance coverage, timestamp semantics, adjustment behavior, rate limits, terms, and
intraday history availability vary by symbol and interval. Treat the downloaded CSV and its
manifest as an immutable research input. Do not infer missing labels or point-in-time features
from provider metadata alone. A raw OHLC snapshot is not directly compatible with the labelled
research CLI until explicit labels and causal features have been separately prepared and audited.

## Backtrader local replay

`app.backtrader_research.run_backtrader_replay` accepts a CSV with timestamp, OHLC and an
externally prepared `signal` column restricted to -1/0/1. It uses a local Cerebro engine only,
with configurable initial cash, fixed units, commission and slippage. Cheat-on-close is disabled;
a signal read at bar close is submitted for a subsequent simulated bar.

This is not a live broker integration and does not create signals. The default unit sizing is
a research convention, not instrument-accurate position sizing. FX contract size, financing,
swaps, spread changes, margin, order rejection, liquidity, and gap/slippage behavior must be
modeled for the intended instrument before interpreting results. No simulated result is evidence
of guaranteed future profitability.

## Safety and methodology

- Both adapters are offline research only; every report says `actionable=false`.
- No order execution, Auto Live, live inference, or risk-gate paths are imported or changed.
- Keep data hashes, source symbol, interval, snapshot times, and assumptions with each experiment.
- Keep labels and signals externally defined; never use a test result to tune the same test window.

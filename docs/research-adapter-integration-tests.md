# Research adapter integration tests

The yfinance and Backtrader adapters are opt-in and isolated from live execution.

## Automated coverage

- yfinance network calls are mocked in tests; CI never contacts Yahoo Finance.
- The download test checks symbol mapping, disabled auto-adjustment, OHLC normalization,
  persisted CSV/manifest, and the manifest's snapshot hash.
- Invalid OHLC is rejected before output is written.
- Backtrader is exercised with a small deterministic synthetic CSV and local Cerebro replay.
  The test validates the non-actionable JSON-safe report and report hash; it does not claim
  that synthetic fills reproduce real FX execution.
- Invalid signals, duplicate timestamps, and invalid friction assumptions are rejected.

Run locally with `python -m pytest -q tests/test_yfinance_history.py tests/test_backtrader_research.py`.
No test performs network requests or invokes a broker, Auto Live, or production risk controls.

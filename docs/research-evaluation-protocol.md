# Research Evaluation Protocol — Foundation

Status: research-only foundation; no validated directional strategy is claimed.

## Purpose

Define the minimum evidence required before Goldcrest Forex API publishes prediction-quality metrics. The current API is deliberately conservative: LLM BUY/SELL suggestions remain `NO_TRADE` until a quantitative decision rule is validated. The research summary endpoint reports record counts and coverage only.

## Current telemetry

`GET /api/v1/research/summary` provides:
- total stored prediction records;
- number of records with an outcome and pending-outcome count;
- outcome coverage (outcomes recorded / predictions total);
- grouped counts by stored decision, outcome label, and model identifier.

These are operational/research bookkeeping measures, not directional accuracy, probability calibration, expected return, or profitability. Empty datasets return a null coverage ratio rather than divide by zero. `GET /api/v1/performance` uses database-side aggregate counts to avoid loading every record into application memory.

## Requirements before scoring

1. **Freeze the target definition.** Specify instrument, timeframe, prediction timestamp, evaluation horizon and exact target event. Do not mix incompatible horizons.
2. **Use timestamped market data.** Outcomes must be derived from market data strictly after the prediction timestamp. Store the source and time range used to evaluate the outcome.
3. **Specify price-side and costs.** Define bid/ask side, spread, commission, swap/financing where applicable, slippage assumptions, and treatment of missing or invalid quotes.
4. **Define event precedence.** If target and stop are both touched inside one candle and tick data is unavailable, use a documented conservative rule or mark the sample ambiguous; do not assume the favourable ordering.
5. **Avoid duplicate scoring.** One immutable evaluation result per prediction and protocol version. Corrections should be explicit, audited revisions rather than silent overwrites.
6. **Prevent leakage.** Features, labels, normalization and model selection must use information available at the prediction timestamp. Keep tuning data separate from validation and final test windows.
7. **Use chronological walk-forward evaluation.** Preserve fold boundaries and report per-fold metrics. Never pool overlapping windows in a way that double-counts the same market event.
8. **Report uncertainty and coverage.** Include sample count, exclusions, missing outcomes, regime and instrument breakdowns. Small samples must not be presented as reliable evidence.
9. **Calibrate probabilities before using them.** Do not interpret LLM self-reported confidence as a probability. Use Brier score, log loss and calibration analysis only when forecasts contain valid, pre-outcome probabilities and labels match the defined event.
10. **Evaluate economic results separately.** Any backtest must include realistic costs and out-of-sample results. Past research performance is not a guarantee of future results.

## Allowed metrics today

- Record counts and outcome coverage.
- Counts by stored decision, model identifier and outcome label.
- Inference/API operational reliability, when measured from actual logs.

## Not allowed until protocol implementation and validation

- Calling outcome-label counts "accuracy".
- Reporting calibrated probability scores without a defined probabilistic target.
- Reporting strategy profitability or trade expectancy from `TARGET_FIRST`, `STOP_FIRST`, or `HORIZON_EXPIRED` labels alone.
- Enabling BUY/SELL as actionable solely because an LLM proposed it.

## Next exit criteria

- A versioned, machine-readable evaluation protocol.
- A reproducible evaluator that derives labels from timestamped post-prediction prices.
- Tests for horizon boundaries, bid/ask handling, ambiguous target/stop ordering, missing data, duplicate evaluation, and future-data leakage.
- Chronological walk-forward reports with separate tuning, validation and final test results.
- Reviewable evidence before any metric is promoted to the dashboard.

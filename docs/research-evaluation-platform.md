# Research Evaluation Platform Phase

## Included capabilities

- Offline deterministic OHLC barrier evaluator with explicit TARGET_FIRST, STOP_FIRST, AMBIGUOUS, HORIZON_EXPIRED, and INSUFFICIENT_DATA labels.
- Expanding chronological walk-forward fold generator. Test windows must be non-overlapping; the default step equals test size.
- Authenticated read-only `GET /api/v1/research/model-comparison` endpoint, grouped by stored model identifier and outcome labels.
- Jeffreys beta-binomial credible intervals, minimum sample floor, Fisher exact pairwise tests, and Holm-Bonferroni adjustment.
- Automated tests and this evaluation protocol.

## Validity requirements

Only recorded TARGET_FIRST and STOP_FIRST labels are treated as resolved barrier outcomes. The endpoint cannot infer labels from OHLC itself; the offline evaluator must be supplied with correctly timestamped bars and frozen target/stop definitions. Compare models only when instrument, timeframe, horizon, target/stop convention, cost assumptions, sample period, and inclusion criteria are comparable. A minimum of 30 observations is a guardrail, not a statistical power guarantee. Corrected p-values do not remove selection bias, data leakage, or market dependence.

The output is descriptive research telemetry, not directional prediction accuracy, calibrated probabilities, expected returns, or profitability. No actionable trading signal is generated. Auto Live, execution, broker integration, and risk controls are outside this change.

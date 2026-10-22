# Research Dashboard Integration Phase

## Delivered

- Authenticated `GET /api/v1/research/model-comparison` groups recorded TARGET_FIRST and STOP_FIRST outcomes by model, symbol, and timeframe.
- Each model group includes resolved sample size, target-first share, Jeffreys credible interval, and a minimum-sample eligibility flag.
- Pairwise model comparisons use two-sided Fisher exact tests and Holm-Bonferroni correction within each symbol/timeframe stratum.
- The dashboard queries the API with the configured base URL and API key, exposes a sample-floor control, renders uncertainty and pairwise results, and handles unavailable/empty data without fabricating examples.
- API contract tests cover authentication, response structure, and invalid query parameters.

## Interpretation limits

This endpoint compares barrier outcome frequencies, not directional prediction accuracy or profitability. Stratifying by symbol and timeframe avoids directly pooling unlike instruments/timeframes, but does not control market regime, overlapping observations, selection bias, costs, or all temporal dependence. The results must not be used as an automatic model selector or to enable actionable signals.

The endpoint is read-only. No broker, Auto Live, execution, or risk-control code is changed.

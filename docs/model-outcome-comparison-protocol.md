# Model Outcome Comparison Protocol

## Purpose and scope

The model comparison helper is research-only. It compares recorded TARGET_FIRST and STOP_FIRST outcome counts using two-sided Fisher exact tests and Holm-Bonferroni correction across pairwise comparisons. Each model's target-first share is accompanied by a Jeffreys beta-binomial credible interval.

## Minimum evidence

- Default minimum: 30 resolved barrier outcomes per model before inferential pairwise tests are emitted. This is a guardrail, not a guarantee of statistical power.
- Outcomes labelled AMBIGUOUS, HORIZON_EXPIRED, INSUFFICIENT_DATA, INVALIDATED, or UNKNOWN must be excluded from resolved barrier counts and reported separately by the caller.
- Comparisons require comparable instruments, timeframes, horizons, target/stop definitions, costs, market periods, and inclusion criteria; otherwise results are confounded.
- Holm-Bonferroni adjusted p-values address multiple pairwise tests, not selection bias, repeated experimentation, leakage, or non-independent observations.
- Report sample counts and uncertainty alongside p-values. Statistical significance is not economic significance.

## Interpretation restrictions

The output is not directional prediction accuracy, probability calibration, expected return, or profitability. A target-first share is descriptive under the supplied barrier definitions. No actionable trading decision may be enabled solely from these results.

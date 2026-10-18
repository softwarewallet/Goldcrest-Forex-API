# Probability diagnostics and paired model comparison

`app/probability_diagnostics.py` scores explicitly supplied, aligned out-of-sample
binary labels and class-1 probabilities. It is an offline diagnostic only; it does
not fit, calibrate, threshold, promote, or serve models.

## Diagnostics

- Brier score and log loss (lower is better).
- Accuracy at the fixed 0.5 threshold, reported for context only.
- Prevalence and mean predicted probability.
- Equal-width reliability bins with counts, mean prediction, and observed positive
  rate. Empty bins remain visible with null summaries.
- Expected calibration error (ECE), weighted by bin frequency.

ECE depends on binning and sample size; it is descriptive, not proof of calibration.
Probability outputs must come from predictions made without fitting on the evaluated
observations.

## Paired comparison

`compare_probability_models` compares candidate and baseline Brier/log-loss values
on the same labels and observation order. It reports candidate-minus-baseline paired
loss differences and moving-block-bootstrap percentile intervals. Negative differences
favor the candidate. The block length and seed are explicit, and repeated calls are
reproducible for fixed inputs.

The observations must be aligned by identical instrument/timeframe/timestamp/fold
identity before calling the function; this module cannot infer or verify those
identities from bare arrays. Use a block length justified by the data and report
sensitivity to alternatives. Block bootstrap does not correct nonstationarity,
multiple testing, data snooping, or a poor sampling design.

## Required evaluation discipline

- Compare LightGBM, XGBoost, and any baseline on the exact same chronological folds,
  labels, feature schema, and held-out observations.
- Keep validation data for decisions and retain the final test set until model and
  feature choices are frozen.
- A constant-prevalence baseline must use prevalence estimated only from the
  corresponding training fold; never use test-set prevalence to construct it.
- Use only point-in-time causal features and labels whose outcome availability is
  explicitly recorded.
- Synthetic tests validate implementation mechanics only. No predictive advantage,
  calibration, profitability, or live readiness is implied.
- Research outputs remain non-actionable. No API, broker, or execution path changes;
  serving remains `NO_TRADE`.

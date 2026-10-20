# Calibration uncertainty diagnostics

Walk-forward validation and test metrics now include an optional
`calibration_uncertainty` record. It estimates a percentile interval for the
10-bin equal-width expected calibration error (ECE) by resampling contiguous
blocks from the chronological out-of-sample sequence. The default is 2,000
resamples and a 95% interval; evaluation callers may declare the block length,
confidence level, resample count, and deterministic seed.

Intervals are withheld when a scored partition has fewer than 30 observations.
The block length is capped at the partition sample count. Invalid labels,
probabilities, block lengths, and bootstrap settings fail closed. A fixed seed
makes repeated runs deterministic for the same ordered inputs and configuration.

Interpretation limits:

- ECE is a binned, sample-size-sensitive descriptive statistic; its interval does
  not establish that probabilities are calibrated.
- The interval depends on bin count, block length, stationarity, sample size and
  bootstrap assumptions. It is not robust to every dependence structure.
- The interval is computed on out-of-sample observations only. It does not
  calibrate or change model predictions, tune a threshold, or select a model.
- Fold windows can overlap; intervals from different folds are not independent.
- Synthetic fixtures validate software mechanics only. Genuine historical data
  with documented provenance and a predeclared evaluation protocol are needed
  before interpreting empirical market performance.
- The API remains offline, research-only and non-actionable. No broker,
  execution, model promotion, or GoldcrestForex Auto Live code is involved.

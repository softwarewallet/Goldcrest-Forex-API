# Fold-consistent model benchmarking

`app/model_comparison.py` runs candidate binary research models through the same
pre-generated chronological `WalkForwardPlan` and the same timed examples. It also
fits a constant training-prevalence baseline independently inside every fold.

## Comparison contract

- Every model is freshly constructed for every fold.
- The shared walk-forward evaluator restricts training to feature timestamps in the
  fold's training partition and excludes labels that had not matured by the training
  cutoff.
- Validation and test examples are scored but are never passed to `fit`.
- Candidate and baseline must report identical validation/test sample counts and fold
  identities. The benchmark fails closed when these differ.
- Per-fold candidate-minus-baseline differences are reported for Brier score, log loss,
  fixed-0.5 accuracy, and mean predicted probability. Negative Brier/log-loss differences
  favor the candidate; positive accuracy differences favor the candidate.
- The baseline probability is the positive-label prevalence in that fold's matured
  training labels only. Validation/test prevalence is never used to construct it.

## Recommended use

Provide `LightGBMResearchModel` and `XGBoostResearchModel` factories configured with
the same feature names and explicit, fixed hyperparameters. Include a training-prevalence
baseline. Use identical data, folds, labels, label-maturity rules, and feature schema.
Do not tune hyperparameters against the final test results.

The returned per-fold metrics are descriptive and do not retain per-row prediction
vectors. Therefore, this API does not claim a paired statistical significance test.
For uncertainty estimates, add an identity-keyed out-of-sample prediction ledger and
compare aligned per-observation losses using the probability-diagnostics module.
A model winning on a small number of folds is not evidence of generalization.

## Scope and limitations

- The caller remains responsible for the source data, point-in-time feature correctness,
  target definition, horizon, label availability, data gaps, and instrument/session rules.
- This utility does not fetch data, train automatically, calibrate probabilities, select
  thresholds, promote models, or connect to serving or trading.
- Synthetic tests validate mechanics only; no market performance or profitability is
  claimed.
- Research outputs remain non-actionable; serving remains `NO_TRADE`.

# Paired model-comparison evidence in the report viewer

The offline experiment runner compares each candidate against the fold-local training-prevalence baseline using identical out-of-sample observation identities. Every paired-comparison record now carries the fold index and partition (`validation` or `test`) so the viewer can show exactly which window the comparison describes.

The local read-only dashboard displays candidate-minus-baseline Brier and log-loss differences with moving-block bootstrap intervals, sample count, fold, partition, and the recorded interpretation. Negative loss differences favor the candidate. An interval crossing zero is not clear evidence of a difference; metric disagreement is also possible.

## Interpretation rules

- The baseline probability is computed from matured labels in the training partition only.
- Validation comparisons are diagnostic and must not be used to repeatedly tune against final test windows.
- Test comparisons are per-fold; if test windows overlap, repeated timestamps are not independent observations.
- The moving-block bootstrap depends on the chosen block length and does not remove multiple-testing bias, nonstationarity, source-data defects, or external feature leakage.
- These are predictive probability comparisons, not cost-adjusted P&L or proof of profitability.
- The API remains research-only and non-actionable; no execution or GoldcrestForex Auto Live code is involved.

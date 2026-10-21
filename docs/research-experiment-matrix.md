# Unified model and feature-ablation experiment matrix

`app.research_experiment_matrix.run_experiment_matrix` runs XGBoost and LightGBM research
variants against one shared purged chronological walk-forward plan. When enabled, feature
variants include the complete causal TA feature set and leave-one-family-out variants.

## Fair comparison contract

- The same source OHLC bars, label definitions, dataset hash, fold boundaries, and test windows
  are used for every experiment.
- Optuna is run independently within each fold and feature/model candidate.
- Test rows are passed only to the scoring stage, never to Optuna.
- All test windows are recorded explicitly so comparisons can be audited.
- The runner does not rank candidates or declare a winner. Repeated comparison on the same
  test windows creates selection bias; final selection requires a separate, untouched period.
- Reports include feature-schema fingerprints, per-fold validation/test metrics, and an
  integrity hash. Every report is non-actionable.

## Resource controls

The matrix has a configurable `max_experiments` guard. With four feature families, the default
matrix is up to five feature variants multiplied by two model types (ten experiments). Each
experiment can perform multiple Optuna trials per fold, so use small trial budgets for initial
smoke tests and increase only after reviewing runtime and available memory.

## Inputs and data provenance

The function expects bars and labelled examples that already align exactly by timestamp and
contain the full feature schema. Labels and class semantics must be externally defined, and
label availability must be correctly timestamped. Use the yfinance snapshot manifest to track
source history, then prepare causal features and labels in a separate audited preprocessing step.
A raw yfinance download does not itself create labels or guarantee point-in-time feature validity.

The matrix is offline research only. It does not connect to inference serving, Auto Live, broker,
risk, or execution paths. Predictive metrics and simulated results are not profitability guarantees.

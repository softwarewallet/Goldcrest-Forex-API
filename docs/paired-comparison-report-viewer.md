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


## Reproducibility manifest

Each experiment run now records a nested reproducibility manifest containing Python/platform metadata, versions of NumPy, scikit-learn, LightGBM and XGBoost when installed, model factory defaults/serializable closure configuration, model public configuration, and SHA-256 fingerprints of model implementation source when source is inspectable.

The runner fingerprints these details into the ledger run contract and report. It uses a preflight model instance for the first fold, then requires a fresh instance with the same model version and configuration in later folds. Reused instances or configuration drift fail closed. Non-serializable factory closures must be replaced with explicit JSON-compatible configuration rather than being silently ignored.

The report inspector verifies the aggregate reproducibility fingerprint and the nested factory/model configuration fingerprints. This detects accidental or inconsistent changes; an unsigned self-fingerprint is not proof against a deliberate editor who can recompute hashes. Environment snapshots also do not guarantee bitwise repeatability across hardware, native libraries, or parallel execution. The data-source provenance and causal-feature limitations remain unchanged.

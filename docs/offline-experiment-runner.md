# End-to-end offline experiment runner

The runner in `app/offline_experiment_runner.py` connects existing research components
without adding a serving route or any trading integration.

## Contract

The caller supplies a validated walk-forward plan, chronological feature/label examples,
feature names, a content-addressed manifest containing `run_id`, model factories,
an evaluation-contract mapping, explicit exclusions, and a block-bootstrap length.
Each example must include an explicitly created binary label and its
`label_available_at` timestamp. The runner does not infer targets, horizons, costs,
market-data sources, symbols, or timeframes.

For each fold it fits a fresh candidate and a training-prevalence baseline on labels
matured by the end of the training window; scores validation and test partitions; records
identity-keyed out-of-sample probabilities to SQLite; compares candidates to the baseline
within each fold/partition using the configured moving-block bootstrap; and writes a
fingerprinted JSON report if a path is provided. Existing ledger idempotency means
identical retries do not duplicate rows, while conflicting content fails closed.

## Important limits

- Research outputs always have `actionable=false`; there is no API/strategy integration.
- This runner consumes prebuilt examples and a fold plan. It does not fetch data or build
  labels/features for you. Use the causal feature builder and explicit target process
  separately, then verify timestamps and label maturity.
- Model factories must be deterministic/reproducible for identical run inputs. Record
  relevant random seeds and model parameters in the manifest/evaluation contract.
- The supplied manifest must describe the exact data/config/code/model run; the runner
  checks for a run ID but does not independently certify manifest provenance.
- Block length and resample count are explicit research choices. Sensitivity analysis
  and overlapping-window/dependence review remain necessary.
- Synthetic fixture tests verify software behavior only, not market predictiveness.
- Keep final-test metrics untouched during model/feature selection. No automatic model
  promotion, production deployment, or trading authorization occurs.


## Durable run lifecycle and retry integrity

The prediction ledger also stores an experiment-run record with a manifest/configuration
fingerprint, status (`running`, `completed`, or `failed`), report SHA-256, bounded error
diagnostic, and update timestamp. The runner fingerprints the manifest together with the
evaluation contract, exclusions, feature/model names, symbol/timeframe, fold identities,
and bootstrap settings. Reusing a run ID with a different configuration is rejected.

A run is marked completed only after comparisons and report generation/writing succeed.
Failures in the persistence/comparison/report phase are recorded as failed; any already
persisted prediction rows remain auditable, not silently rolled back as if the run never
occurred. A retry with the same run ID and same configuration may safely replay identical
ledger writes. Conflicting prediction content still fails closed. The caller must ensure
the manifest itself accurately fingerprints source data, labels, feature-generation code,
model code/parameters, and other run inputs; the runner cannot independently prove those
claims.

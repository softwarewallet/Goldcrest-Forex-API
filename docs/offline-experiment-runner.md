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


## Input-level provenance and retry protection

In addition to the caller-provided manifest and evaluation contract, the runner computes a
canonical SHA-256 fingerprint over the actual supplied feature/label examples (including
UTC observation and label-maturity timestamps), exact train/validation/test timestamp
membership for every fold, feature names, symbol/timeframe, and model-factory qualified
identities. These values are included in the durable run contract. Reusing a completed run
ID with altered inputs is rejected before ledger rows are changed, even if a simple test
model happens to produce identical probabilities for the altered data.

This is an integrity check on the inputs received by the runner, not a guarantee that the
caller sourced those inputs correctly or that model code/hyperparameters are fully captured.
Record source-data hashes, code commit, library versions, model revisions and parameters
in the caller's manifest/evaluation contract. Factory identity is not a substitute for a
code-content fingerprint. Training failures that occur before the durable run contract is
registered are still surfaced as exceptions; the runner does not claim that every possible
preflight/training failure is durably registered as a failed run.


## Full run lifecycle and fitted-model identity

The durable run record is registered after deterministic input-contract construction and
before fold validation/model fitting. Failures during fold checks, fitting, scoring, ledger
persistence, comparison, or report generation are recorded as `failed` when the ledger
remains available; the original experiment exception is preserved if failure logging also
fails. Static preflight errors rejected before registration are still ordinary exceptions.

The runner records each version from the actual fitted model instance used to produce the
out-of-sample probabilities. It no longer constructs an extra, unfitted model just to
read its version. If a factory reports a different model version across folds under the
same run contract, the run fails closed rather than comparing against a version that does
not match the stored prediction rows.


## Temporal partition integrity

Both the end-to-end runner and the standalone binary walk-forward evaluator now reject
folds whose training, validation, and test windows are not strictly forward in time,
whose partitions overlap, or whose referenced timestamps are missing from the supplied
examples. The standalone evaluator also rejects duplicate/invalid fold indices and labels
that are numerically equal to 0/1 but are not actual integers. These checks prevent a
malformed caller-supplied plan from silently creating a leakage-prone evaluation.


## Retry and recovery semantics

A failed attempt may be retried under the same run ID only when the complete run contract
matches. Every attempt is retained in `prediction_ledger_attempts`; previous failure
diagnostics are not overwritten. An already-running attempt blocks a second concurrent
start, and a completed run ID is immutable, including its report digest. Use a new run ID
for a deliberate rerun after completion. Partial prediction rows from a failed attempt
remain in the ledger for audit and must reconcile with the run status/report before any
result is interpreted. The runner does not silently resume at an arbitrary fold boundary;
retry replays deterministically and relies on idempotent prediction identities, rejecting
any conflicting outputs.

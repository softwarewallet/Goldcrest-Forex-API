# Identity-keyed out-of-sample prediction ledger

`app/prediction_ledger.py` provides an offline SQLite ledger for storing model
probabilities with explicit observation identity and later comparing models on exactly
paired observations. It is not connected to FastAPI serving or any trading system.

## Identity and immutability

A row is uniquely identified by run manifest ID, symbol, timeframe, UTC-normalized
observation timestamp, fold index, partition, model name, and model version. Identical
retries are idempotent; conflicting content under the same identity fails closed.
Batch inserts are transactional. Each record is explicitly non-actionable.

Labels and label-availability timestamps are optional as a pair, allowing a prediction
to be recorded before the outcome matures. A comparison excludes unlabeled rows and
requires candidate and baseline to have identical matured observation identities and
matching labels/maturity timestamps. It then invokes the existing paired Brier/log-loss
diagnostics and moving-block bootstrap.

## Example

```python
from app.prediction_ledger import PredictionLedger, PredictionRecord

ledger = PredictionLedger("data/research_predictions.sqlite")
ledger.record(PredictionRecord(
    run_id="<registered-manifest-run-id>",
    symbol="EUR/USD",
    timeframe="M5",
    observation_timestamp=observation_time,  # timezone-aware
    fold_index=0,
    partition="validation",
    model_name="candidate",
    model_version="model-revision",
    probability=0.61,
    actionable=False,
))
```

Record the baseline for the exact same observation identity (except model name/version).
Once labels have matured, store the label and its actual availability time in the initial
record; this first version treats rows as immutable, so it does not support later label
updates. A prediction initially written without a label must be re-created in a separate
run/identity once outcome data is finalized, or the schema must be explicitly extended
with a reviewed label-finalization operation. Do not mutate historical labels silently.

## Evaluation safeguards

- Compare one run, symbol, timeframe, fold, and partition at a time.
- Candidate and baseline must have the same labeled observation identities. Different
  timestamps are not paired merely because row counts match.
- Fold index is part of identity. Do not concatenate overlapping test folds and count
  repeated market observations as independent samples. Define a deduplication/aggregation
  policy before any cross-fold summary.
- Use validation for development. Preserve final test results until model/feature choices
  are frozen.
- Block-bootstrap interval quality depends on sample size, serial dependence, and block
  length; use sensitivity analysis and do not interpret intervals as proof of profitability.
- Labels, target/horizon, costs, market data provenance, and point-in-time feature rules
  remain explicit caller responsibilities.

All outputs remain research-only with `actionable=False`. No market data is fetched,
no models are promoted, no broker credentials are used, and no orders are placed.
Synthetic tests validate software behavior only, not market performance.


## Shared SQLite database and compatibility

The prediction ledger stores its lifecycle records in `prediction_ledger_runs`, separate
from `ExperimentRegistry`'s `experiment_runs` table. Both components may therefore use the same
SQLite file without conflicting table schemas, regardless of initialization order. On
startup, the ledger detects the legacy ledger-specific `experiment_runs` schema by its
columns and copies its lifecycle records into `prediction_ledger_runs`; it does not reinterpret
or overwrite the manifest registry's differently shaped table.


## Durable attempt history and completed-run immutability

Each call that starts a run creates a durable attempt row in
`prediction_ledger_attempts`. Failed retries append a new attempt; prior failures remain
available with their bounded diagnostic and timestamps. Only one attempt may be running
for a run ID at a time. A completed run ID cannot be restarted or have its report digest
cleared: create a new run ID for an intentional rerun. The current run status remains a
convenient summary, while attempt rows preserve the lifecycle history.

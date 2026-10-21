# Optuna walk-forward research orchestration

## Purpose

`app.optuna_walk_forward_cli` runs an end-to-end, offline research experiment on an
already-labeled historical CSV. It connects the Optuna tuner to the existing purged
walk-forward fold generator and model metrics. This is a research capability, not a
live inference or trading path.

## CLI example

```bash
python -m app.optuna_walk_forward_cli \
  --csv data/research/EURUSD_M15_labeled.csv \
  --features return_1,return_3,return_5,range_pct,body_pct,volatility_5,volatility_10 \
  --run-id eurusd-m15-optuna-20261021-a \
  --report reports/eurusd-m15-optuna-20261021-a.json \
  --data-source provider-export \
  --dataset-version export-2026-10-21-v1 \
  --retrieved-at-utc 2026-10-21T10:00:00Z \
  --timestamp-timezone UTC \
  --duplicate-policy reject \
  --adjustment-policy none-document-with-provider \
  --license-or-terms-reference provider-contract-or-license-id \
  --initial-train-bars 2000 \
  --validation-bars 400 \
  --test-bars 400 \
  --step-bars 400 \
  --horizon-bars 12 \
  --model-type xgboost \
  --trials-per-fold 20 \
  --random-seed 20261021 \
  --bootstrap-resamples 500
```

The values above are an example of command structure, not assertions about the user's
actual data source or license. Replace all dataset provenance fields with truthful
information from the real export. The input must already contain `label` and
`label_available_at`; this command does not invent targets. Use a new run ID and report
path for each immutable experiment.

## Fold procedure

1. Validate chronological OHLC, labels, finite features, and timestamp alignment.
2. Generate expanding train/validation/test folds with the configured horizon purge.
3. Tune each fold using only its training and validation partitions.
4. Reject a fold if any validation outcome has not matured by test start.
5. Refit the selected model on training plus validation examples whose labels matured
   before test start.
6. Score the test partition once, recording predictive metrics and calibration
   uncertainty.
7. Write a JSON report containing dataset SHA-256, dataset provenance, feature-schema
   fingerprint, per-fold hyperparameters, row-exclusion counts, test metrics, resource
   profile, and an integrity hash.

## Interpretation and limits

- Each fold's test partition is not supplied to Optuna.
- Test outcomes from earlier folds must not be used to change parameters or feature
  choices for later folds if the report is being interpreted as a single unbiased
  evaluation. Fold-wise tuning only uses the current fold's training and validation
  partitions, but the research operator still controls the overall experiment design.
- The reported metrics are predictive-probability diagnostics, not a cost-aware trading
  backtest or proof of profitability.
- Reports are marked `actionable=false`; no model is promoted and no broker/execution
  API is called.
- Feature timestamp correctness and label semantics must be guaranteed by the dataset
  producer. The command can verify structure, not reconstruct unavailable source history.
- The input must be one symbol/timeframe and already chronologically ordered. Bad rows
  are rejected rather than silently sorted or imputed.
- Use conservative trial counts on constrained machines. Avoid running multiple heavy
  Optuna studies concurrently with Qwen/PyTorch workloads until peak memory and runtime
  have been measured.

The CLI does not change the existing inference API, LLM gateway, live configuration,
risk controls, or Auto Live/trade execution methods.

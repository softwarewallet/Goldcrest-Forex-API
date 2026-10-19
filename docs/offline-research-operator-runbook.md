# Offline research operator runbook

This runbook connects GoldcrestAI's existing offline components into a controlled
research workflow. It is an operator checklist, not an automatic data-ingestion,
model-promotion, or live-trading procedure.

## Safety boundary

- All model outputs and reports from this workflow are offline research artifacts.
- The runner and artifact audit do not call broker APIs, submit orders, or authorize a
  strategy. Keep `actionable=false`.
- A passing software test or artifact audit is not evidence of market edge, calibration,
  profitability, or future performance.
- Do not connect this workflow to GoldcrestForex execution. No production deployment or
  service restart is required to run offline research.

## Workflow map

1. **Approve the experiment contract.** Specify the actual source and version of market
   data, instruments, timeframe, timezone, target/label definition, prediction horizon,
   label-maturity rule, feature-generation revision, cost model, exclusions, baseline,
   metrics, bootstrap settings, and model hyperparameters. Do not leave placeholder
   values in a run intended to support a decision.
2. **Prepare point-in-time inputs.** Validate OHLC rows and strict chronology; create
   causal features; label outcomes under the approved target policy; and record when
   each label became available. Do not use future bars, future labels, or revised data
   unavailable at the observation time.
3. **Create purged chronological folds.** Keep training before validation and validation
   before test. Purge observations whose outcome horizon crosses a boundary. Freeze the
   final test partition before model/feature selection.
4. **Run the offline experiment.** Use `app/offline_experiment_runner.py` with a
   `WalkForwardPlan`, `TimedBinaryExample` inputs, feature names, a manifest containing a
   unique `run_id`, model factories, an explicit evaluation contract, exclusions, and a
   justified block length. The runner trains a fresh model per fold, uses matured training
   labels only, evaluates a training-prevalence baseline, and writes out-of-sample
   predictions and a fingerprinted report.
5. **Audit the persisted artifacts.** Run `verify_research_artifacts` from
   `app/research_artifact_audit.py` against the report and the same SQLite ledger. Stop if
   verification raises an exception or does not pass.
6. **Review research results.** Inspect per-fold validation and test metrics, sample counts,
   exclusions, model versions, baseline differences, uncertainty intervals, and warnings.
   Do not pool overlapping test windows. Use validation data for iteration; reserve final
   test results for a frozen evaluation.
7. **Archive the experiment.** Preserve the report, SQLite ledger, source-data hash,
   feature/label code revision, Git commit, Python/package versions, model parameters,
   random seeds, and the exact evaluation contract. Record any deviations before rerunning.

## Pre-run checklist

Before training, verify all of the following:

- [ ] Data source/version and data snapshot hash are recorded.
- [ ] Timestamps are timezone-aware, normalized to UTC, strictly ordered, and duplicate
      observations are resolved by an explicit policy.
- [ ] OHLC values are finite, positive, and internally consistent.
- [ ] Features are causal and their warm-up/invalid rows are accounted for.
- [ ] Labels use the approved target policy; label availability timestamps are explicit.
- [ ] Train, validation, and test partitions are disjoint and strictly chronological.
- [ ] Training labels are mature by the training cutoff; fold boundaries are purged for
      the target horizon.
- [ ] Instrument, timeframe, horizon, spread, slippage, commission, and any financing
      assumptions are explicitly specified where applicable.
- [ ] Candidate models and the baseline use identical fold definitions and observation
      identities.
- [ ] Block length, resample count, confidence level, random seed, and sensitivity checks
      are recorded and justified.
- [ ] The final test set has not been repeatedly inspected to select models or features.
- [ ] Manifest fields identify the source data, code revision, model revision/parameters,
      library versions, and experiment configuration.

## Run identity, retries, and artifacts

Use a new unique `run_id` for each deliberate experiment. The runner fingerprints the
supplied examples, labels and label-maturity timestamps, fold membership, feature names,
symbol/timeframe, model-factory identities, and run contract. Reusing a run ID with
changed inputs/configuration fails closed. A completed run is immutable; do not overwrite
its report or reinterpret a conflicting retry as the same experiment.

A failed run can leave out-of-sample rows that were committed before failure. Treat those
rows as incomplete and not reportable until run lifecycle and artifact reconciliation
succeed. Retry only with the same contract; use a new run ID for a deliberate changed
experiment. Inspect recorded attempt diagnostics rather than manually editing the ledger.

The artifact audit checks the report fingerprint and completed-run binding, report/fold
research-only flags, and counts grouped by model name, fitted model version, fold, and
validation/test partition. It is read-only. A successful audit proves internal consistency
of these artifacts only; it does not authenticate the market data or prove the experiment
is statistically valid.

## Interpretation and decision gates

- Compare candidates with the training-prevalence baseline on matched out-of-sample
  observations; reject comparisons with mismatched identities or labels.
- Examine each fold and partition separately, plus the number of observations and
  exclusions. Avoid survivor-only reporting.
- Evaluate uncertainty with dependence-aware methods and justify block-length choices;
  run sensitivity analyses. Block bootstrap does not solve non-stationarity, multiple
  testing, or every dependence structure.
- Report costs in explicit units and subtract them consistently. Do not silently assume
  zero spread/slippage/commission.
- Do not infer probability calibration from accuracy or from a high mean predicted
  probability. Evaluate proper scoring rules and calibration diagnostics separately.
- Do not claim profitability from synthetic tests, a small sample, a single market regime,
  a positive confidence interval, or a successful artifact audit.
- Model selection and any future deployment require a separately approved protocol and
  independent review. This runbook never promotes a model or enables trading.

## Relevant implementation references

- `app/causal_features.py` — deterministic causal OHLC features.
- `app/outcome_labeling.py` — explicit deterministic outcome labels and ambiguous-bar policy.
- `app/walk_forward.py` — chronological purged fold planning.
- `app/model_walk_forward.py` — leakage-aware binary model evaluation.
- `app/offline_experiment_runner.py` — end-to-end offline evaluation and ledger writes.
- `app/experiment_report.py` — deterministic fingerprinted JSON reports.
- `app/prediction_ledger.py` — identity-keyed out-of-sample predictions and run lifecycle.
- `app/research_artifact_audit.py` — read-only report/ledger reconciliation.

The existing component-level documentation remains the source for API details. This
runbook does not prescribe an unapproved market-data provider, target, horizon, baseline,
or cost assumption.

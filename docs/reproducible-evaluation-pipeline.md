# Reproducible Offline Evaluation Pipeline

**Status:** research-only plumbing. No production prediction behavior changes, no broker access, no order execution, and no automatic promotion. GoldcrestAI remains independent from GoldcrestForex; serving decisions remain `NO_TRADE` until a separately reviewed and explicitly authorized change.

## Components

- `app/evaluation_pipeline.py:prepare_research_run` validates the complete bar set against an explicit `EvaluationConfig`, rejects malformed/duplicate/non-monotonic records, checks exact symbols and timeframe, creates purged expanding walk-forward plans per symbol, fingerprints dataset/config/code/model metadata, and idempotently registers the run in SQLite.
- `compare_scored_outcomes` applies the existing explicit round-trip cost model to candidate and baseline outcomes, excludes ambiguous/missing outcomes, requires exact timestamp/symbol/timeframe pairing, and computes a moving-block bootstrap comparison of aligned net price-unit outcomes.
- Existing primitives remain separate and testable: `outcome_labeling`, `cost_aware_evaluation`, `walk_forward`, `block_bootstrap`, and `run_manifest`.

## Explicit inputs — never guessed

Before a real historical evaluation, an operator must version and document the exact instruments/venue, data provider and dataset version, timeframe, prediction timing, horizon, entry/exit convention, target/stop rules, ambiguous-bar policy, spread/slippage/commission (and financing if applicable), baseline definitions, primary/secondary metrics, and untouched test period. Costs in current scoring primitives are raw price units, not pips or account-currency P&L. Confirm the provider's quote semantics and avoid double-counting spread.

The current orchestration expects bars to be supplied in chronological order for each symbol. It does not download data, silently sort/fill gaps, fit features, train LightGBM/XGBoost/Qwen, generate candidate signals, or infer strategy outputs. Fold plans define data partitions; model fitting and prediction replay must use only information available at each prediction timestamp and remain separate work.

## Reading results

All scored returns are raw price-unit changes after the configured costs. Candidate and baseline must be aligned observation-by-observation. Ambiguous outcomes are excluded, not assigned a favorable result. Block length and cost assumptions require documented sensitivity analysis. A confidence interval above zero is not proof of profitability and does not correct for all nonstationarity, multiple testing, model selection, or execution risk.

## Safety and promotion gate

Synthetic tests verify mechanics only. No real market dataset has been approved by this change. No endpoint is added and the live API response is unchanged. A future promotion must be a separate reviewed change with an untouched final test, calibration/uncertainty analysis as appropriate, complete per-instrument/per-fold reporting, and explicit user authorization. Until then, do not use research model outputs as actionable signals; `NO_TRADE` stays enforced.

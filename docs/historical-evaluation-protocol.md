# Historical Evaluation Protocol (Draft)

**Status:** Protocol draft; evaluation is not yet authorized to produce trading signals.  
**Scope:** Offline research and measurement for GoldcrestAI only. This document does not connect GoldcrestAI to GoldcrestForex or any broker, and does not authorize order placement.

## 1. Objectives and non-claims

The evaluation framework must measure whether a frozen prediction method adds reproducible information beyond explicit baselines. It must report sample sizes, uncertainty, costs, data quality, and failure cases. It must not imply profitability from directional accuracy alone.

Until the protocol's TBD items are approved and implemented, the service remains fail-closed: decisions returned to callers remain `NO_TRADE`, with no direction or calibrated probability.

## 2. Decisions requiring approval before a scored run

Do not silently choose these values. Record approved values in a versioned evaluation configuration.

| Decision | Required definition | Current status |
| --- | --- | --- |
| Instruments | Exact symbols, venue, contract/spot definition, and symbol mapping | TBD |
| Historical data source | Provider, licensing/retention limits, timezone, and data version | TBD |
| Price data | Bid/ask availability; candle timeframe; missing/duplicate bar policy | TBD |
| Prediction cadence | When each prediction is generated and which information is available at that timestamp | TBD |
| Evaluation horizon | Fixed elapsed time or number of bars; exact boundary convention | TBD |
| Entry reference | Bid/ask or other executable-side price; timing/latency assumption | TBD |
| Exit/outcome rules | Target/stop/horizon/invalidation definitions, including gaps | TBD |
| Costs | Spread, commission, financing/swap if applicable, slippage, and latency assumptions | TBD |
| Ambiguous candles | If target and stop both fall inside one OHLC bar, use finer data or mark ambiguous; never select the favorable result | TBD |
| Baselines | Naive/no-change, simple non-ML baseline, and relevant regime baseline | TBD |
| Metrics | Primary metric, secondary metrics, uncertainty intervals, and minimum sample requirements | TBD |
| Data window | Training/development, validation, and untouched final test periods | TBD |

A scored evaluation must not begin while any decision needed to interpret its outcomes remains unresolved.

## 3. Data integrity and point-in-time correctness

1. Preserve original timestamps with explicit timezone and normalize internally to UTC.
2. Enforce chronological order, uniqueness, and instrument/timeframe consistency.
3. Record source, retrieval time, dataset/version fingerprint, and validation findings.
4. Never forward-fill a price across an unobserved gap without an explicit, documented rule.
5. A prediction at time *t* may use only data that was available by *t*. Features, normalization, regime labels, and model prompts must not use future bars or future-derived values.
6. Keep a strict separation between prediction-time features and outcome labels. Outcome labels are attached only after the evaluation horizon is complete.
7. Quarantine malformed, stale, incomplete, or contradictory records; report exclusions and reasons instead of silently dropping them.
8. Do not use the same request/prediction identifiers to represent distinct market snapshots.

## 4. Temporal splits and walk-forward testing

- Split data chronologically, never by random row-level shuffling.
- Keep an untouched final test period that is not used to select features, prompts, thresholds, models, or costs.
- Use rolling or expanding walk-forward folds for development/validation.
- Apply an embargo/purge interval where label horizons overlap adjacent folds; set its duration from the approved horizon and data cadence.
- Fit any learned preprocessing only on the training portion of each fold.
- Freeze model identifier/revision, prompt version, feature version, parameters, and evaluation configuration for each run.
- If any choice changes after examining validation/test results, treat the affected period as development data and obtain a new untouched test period.
- Report results per fold and per instrument as well as aggregate results; do not hide regime-specific failures.

## 5. Outcome and execution assumptions

Outcome labels such as `TARGET_FIRST`, `STOP_FIRST`, `HORIZON_EXPIRED`, `INVALIDATED`, and `UNKNOWN` require deterministic, versioned definitions before scoring. Store the label rule version and the exact observation window.

Use executable-side prices where available: buying is assessed against ask-side entry and bid-side exit; selling against bid-side entry and ask-side exit. If only mid/close data is available, explicitly label the result as a proxy and do not describe it as executable performance.

Include spread and all applicable costs in net results. Slippage must be modeled conservatively and with sensitivity analysis. If candle data cannot resolve the order of target/stop touches, mark the observation ambiguous or resolve with finer-grained data; do not assume the favorable ordering. Missing exit data must not be converted into a win.

## 6. Required reporting

Every evaluation report should include:

- Evaluation ID and UTC creation time.
- Dataset/source/version and a reproducible data fingerprint.
- Model ID/revision, prompt hash, feature version, configuration hash, and code commit.
- Instruments, date ranges, timeframe, horizon, and sample counts.
- Counts of valid, excluded, ambiguous, missing, and unknown outcomes with reasons.
- Baseline comparison and per-fold/per-instrument/per-regime breakdowns.
- Metrics with uncertainty intervals where appropriate, and explicit limitations.
- Gross and net results after costs, plus cost/slippage sensitivity.
- A record of all protocol deviations and failed runs.

Do not report a single accuracy percentage without its denominator, class distribution, baseline, time window, and uncertainty. Do not infer profitability from accuracy alone. Avoid selecting the best result from many experiments without accounting for multiple testing and selection bias.

## 7. Minimum evaluation safeguards

- Use a reproducible run manifest and immutable run ID.
- Fail the run on data leakage, non-monotonic timestamps, unresolved required fields, or invalid label definitions.
- Keep test-set access restricted until the method and configuration are frozen.
- Log exceptions and exclusions without storing API keys or other secrets.
- Never change the production prediction response based on offline results automatically.
- Any future strategy promotion requires a separate reviewed change, evidence from untouched data, and explicit user authorization.

## 8. Implementation sequence

1. Approve the TBD protocol choices.
2. Define a versioned evaluation configuration schema and run manifest.
3. Add offline dataset validation and leakage checks.
4. Add deterministic outcome labeling with tests for gaps and ambiguous bars.
5. Add chronological/walk-forward split generation with overlap purging.
6. Add baseline and metric computation with uncertainty reporting.
7. Add reproducibility tests and synthetic fixtures before evaluating real data.
8. Review reports and limitations before any decision about future use.

## 9. Current safety boundary

This protocol is documentation for offline research. It does not change API behavior, enable BUY/SELL, add broker access, modify GoldcrestForex, or claim that the current model has predictive edge.

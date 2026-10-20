# Multi-model research stack

## Current components

- **XGBoost** and **LightGBM** are independent tabular binary-classifier candidates. Use
  `build_tabular_model_factories(feature_names)` with the existing offline walk-forward
  runner to compare them on the same chronological folds and feature schema.
- **Llama/Qwen** are represented by one configured LLM gateway result. The gateway's
  returned model ID is recorded for provenance. Qwen should be orchestrated behind this
  same gateway if the chosen serving runtime supports that workflow; this application
  does not create a second Qwen endpoint or assume that a plain llama.cpp model can call
  another model on its own.
- **Research comparison** reports each tabular candidate's fixed-threshold direction
  beside the LLM direction. It does not average probabilities, infer calibrated
  confidence, automatically select a winner, or authorize trading.

## Historical CSV evaluation CLI

A reproducible, offline evaluation entry point is available as
`python -m app.historical_tabular_research`. It uses the existing expanding-window,
purged walk-forward pipeline and compares XGBoost and LightGBM against a
training-prevalence baseline. The CLI writes a JSON report and a SQLite prediction
ledger; every output remains `actionable=false`.

Example invocation (replace the paths, dataset details, feature names and split sizes
with values appropriate to the supplied historical dataset):

```bash
python -m app.historical_tabular_research \
  --csv data/historical/EURUSD_M15.csv \
  --features ema_gap,rsi_14,atr_14,return_5 \
  --run-id eurusd-m15-xgb-lgbm-2026-10-20-01 \
  --data-source "documented provider/export" \
  --dataset-version "export-revision-1" \
  --label-definition "Externally prepared label definition; state exact horizon and tie/neutral rule" \
  --class-1-means UP \
  --initial-train-bars 2000 \
  --validation-bars 400 \
  --test-bars 400 \
  --step-bars 400 \
  --horizon-bars 3 \
  --block-length 10 \
  --report reports/eurusd-m15-research.json \
  --spread-cost-notes "Document quote source and spread assumptions" \
  --slippage-and-commission-notes "Document instrument-specific assumptions"
```

The CSV must include these columns:

- `symbol,timeframe,timestamp,open,high,low,close`
- `label,label_available_at`
- Every numeric feature named in `--features`

Requirements:

1. Each row must represent one timestamp for a single symbol/timeframe; timestamps must
   be timezone-aware ISO-8601 and already chronological. The loader does not sort,
   silently drop rows, impute values, or create labels.
2. `label` must be explicitly precomputed as integer `0` or `1`; `label_available_at`
   records when its outcome became known. The CLI requires the exact label definition
   and whether class 1 means UP or DOWN. The dataset producer must guarantee features
   were available at the row timestamp; the CLI cannot prove feature provenance from
   numeric values alone.
3. Choose `--horizon-bars` to match the label's actual outcome horizon. The walk-forward
   planner uses it to purge partition boundaries, and the evaluator excludes training
   labels that had not matured by the training cutoff.
4. Dataset SHA-256, feature schema fingerprint, split parameters, label contract, and
   assumptions are recorded for reproducibility. Use a new `--run-id` for each run;
   completed ledger run IDs are immutable.
5. The current metrics are predictive probability metrics (Brier score, log loss, and
   fixed-0.5 accuracy), not strategy P&L. Spread/slippage/commission notes are captured
   as provenance only; this CLI does not simulate trades or apply costs to the metrics.
   Do not infer profitability from these results.
6. Treat validation as the model/feature development partition. Keep final-test results
   untouched until decisions are frozen. There is no automatic model selection or
   promotion.

## Important configuration contract

1. Pin the feature names and their order in the dataset/run manifest.
2. Explicitly declare whether label class 1 means UP or DOWN.
3. Use mature labels, point-in-time features, dataset provenance, and purged chronological
   walk-forward partitions already required by the offline evaluation pipeline.
4. Compare XGBoost and LightGBM against the training-prevalence baseline and retain
   validation and test metrics separately. Do not tune thresholds on the final test set.
5. Keep LLM outputs as an independent research observation. If a Llama gateway internally
   orchestrates Qwen, identify that orchestration in gateway-side configuration and logs.
   If the gateway cannot orchestrate Qwen, do not claim Qwen participated in the result.
6. Treat all outputs as `actionable=false`. A model's inclusion is not evidence of
   improved accuracy. Promote no model automatically; require reproducible out-of-sample
   evidence, cost-aware evaluation, calibration diagnostics, and explicit review.

## Boundary

This is an offline/research utility. It does not change `/api/v1/predictions`, the
current fail-closed `NO_TRADE` policy, broker integration, GoldcrestForex, Auto Live,
position sizing, or execution/risk gates. No training occurs in API request handlers.


## Preparing raw historical OHLC data

Before labels are defined, a label-free preparation utility can validate a provider-exported
OHLC CSV and compute a transparent causal feature set:

```bash
python -m app.historical_feature_preparation \
  --input-csv data/historical/EURUSD_M15_raw.csv \
  --output-csv data/historical/EURUSD_M15_features.csv \
  --report reports/EURUSD_M15_feature_audit.json
```

Input columns are `symbol,timeframe,timestamp,open,high,low,close`. The input must be
timezone-aware, chronological, valid OHLC for one symbol/timeframe, with at least 16
rows. The utility excludes the first 15 warm-up rows and emits `return_1`, `return_3`,
`return_5`, `range_pct`, `body_pct`, `close_location`, and `volatility_14`.
These use only current and historical OHLC; no future candle is used for feature values.
The JSON report records the input SHA-256 and row counts.

This command intentionally does **not** create labels. It is not a replacement for the
labeled CSV contract above. The prediction horizon, neutral/tie handling, and label
availability timestamps must be defined explicitly before the evaluation CLI can run.
The resulting features and report are offline research artifacts only.


## Explicit target-label preparation

After generating causal features, prepare labels with a separate command. This command
requires the horizon, neutral return band, and treatment of neutral outcomes to be
explicit. It does not drop neutral rows: choose `up` to assign neutral outcomes class 1,
`down` to assign them class 0, or `error` to stop if any neutral outcome is found.

```bash
python -m app.historical_label_preparation \
  --input-csv data/historical/EURUSD_M15_features.csv \
  --output-csv data/historical/EURUSD_M15_labeled.csv \
  --report reports/EURUSD_M15_label_audit.json \
  --horizon-bars 3 \
  --neutral-band-bps 0.5 \
  --neutral-policy error
```

The target compares `close[t]` with `close[t+horizon_bars]`. Class 1 means UP and class
0 means DOWN, outside the configured neutral band. `label_available_at` is the timestamp
of the future close that resolves the outcome. The final `horizon_bars` rows are excluded
because their outcome is not yet mature. Existing label columns are rejected rather than
overwritten. The JSON report records the input hash, exact target rule, horizon, neutral
policy, class counts, and excluded tail count.

This is a configurable research target, not a recommendation that a particular horizon
or neutral band is predictive. Select parameters based on the intended research question,
document them, and freeze them before final-test evaluation. This target is close-to-close
price direction, not a cost-adjusted trade return; costs and execution assumptions need
separate analysis. The generated data remains `actionable=false`.


## One-command historical research pipeline

The individual tools can be orchestrated with
`python -m app.historical_research_pipeline`. This performs the same explicit stages
in order: validate raw OHLC and compute causal features, prepare matured forward-return
labels under the specified horizon/neutral policy, then run chronological purged
walk-forward evaluation of XGBoost and LightGBM against the training-prevalence baseline.

Example (use real, documented provider data and sizes appropriate to its history):

```bash
python -m app.historical_research_pipeline \
  --input-csv data/historical/EURUSD_M15_raw.csv \
  --features-csv data/historical/EURUSD_M15_features.csv \
  --labeled-csv data/historical/EURUSD_M15_labeled.csv \
  --feature-report reports/EURUSD_M15_features.json \
  --label-report reports/EURUSD_M15_labels.json \
  --report reports/EURUSD_M15_pipeline.json \
  --run-id eurusd-m15-pipeline-2026-10-20-01 \
  --data-source "documented provider/export" \
  --dataset-version "export-revision-1" \
  --horizon-bars 3 \
  --neutral-band-bps 0.5 \
  --neutral-policy error \
  --initial-train-bars 2000 \
  --validation-bars 400 \
  --test-bars 400 \
  --step-bars 400 \
  --block-length 10 \
  --spread-cost-notes "Record spread source and assumptions" \
  --slippage-and-commission-notes "Record instrument-specific assumptions"
```

The tool also writes a separate `*.evaluation.json` report and the SQLite ledger. All
artifact paths, including the ledger, must be distinct. The pipeline uses exactly the
features emitted by the current feature preparer:
`return_1,return_3,return_5,range_pct,body_pct,close_location,volatility_14`.
It does not silently fetch data, synthesize data, sort or impute rows, skip neutral
labels, or promote a model. With `--neutral-policy error`, any neutral outcome stops
the run; choose another explicit policy only if that target definition is justified.

A successful run means the pipeline completed, not that a model is profitable or more
accurate in live markets. Use genuine historical data, review the combined provenance
report and separate validation/test metrics, and retain final-test results for evaluation
rather than tuning. The entire workflow remains offline and `actionable=false`.


## Historical OHLC quality and cadence audit

The one-command research pipeline now writes a source quality report before feature
preparation. The standalone utility is also available:

```bash
python -m app.historical_data_quality \
  --input-csv data/historical/EURUSD_M15_raw.csv \
  --report reports/EURUSD_M15_quality.json
```

The audit fingerprints the exact source bytes and reports invalid rows, duplicate and
non-monotonic timestamps, observed cadence versus the nominal timeframe, long intervals,
and an estimate of bars between timestamps only when the interval is an exact multiple
of the nominal timeframe. The pipeline stops before feature generation if structural
checks fail and leaves the quality report for inspection.

**Important:** a long interval is not proof of missing data. Weekends, market sessions,
holidays, daylight-saving changes and provider conventions can explain gaps. The audit
never sorts, fills, imputes, drops or synthesizes records, and it does not treat its
estimated gap count as an instruction to repair data. All audit artifacts remain
offline-only and `actionable=false`.


## Dashboard report review

The dashboard's **Historical research report review** panel accepts a combined pipeline
JSON report or an evaluation JSON report. It reads the selected file locally in the
browser; it does not upload the report to the API. When the combined report is loaded,
the panel summarizes raw-row quality, timestamp duplicates, cadence gaps, feature and
label row counts, and model fold metrics where present. Per-fold metrics are displayed
as descriptive fold averages only, not as a pooled out-of-sample score.

The panel rejects execution-scoped/actionable reports and clearly states that predictive
metrics do not establish profitability. It is a review surface, not a model promotion,
prediction-serving, or order-execution control.

# End-to-end offline research validation

The research pipeline writes a separate `<run-id>_validation_report.json` after building
the pipeline manifest. The report combines artifact integrity checks, prepared-dataset
quality, walk-forward result structure, fold-level test-metric summaries, and a local
runtime preflight. It does not modify the trading application or publish actionable signals.

## cTrader DEMO credentials: where to put them

The configuration loader reads the ignored `.env` file at the root of the
`Goldcrest-Forex-API` repository (the same directory as `requirements.txt` and
`.env.example`). It now loads that file explicitly, independent of the shell's current
working directory. Existing non-empty environment variables take precedence over values in
`.env`.

From the repository root, create or edit the file:

```bash
cp .env.example .env   # only if .env does not already exist
nano .env
```

Add or fill in these entries using the actual cTrader Open API application credentials,
a valid access token, and the numeric ID of the DEMO trading account:

```dotenv
CTRADER_HOST=demo
CTRADER_CLIENT_ID=your_application_client_id
CTRADER_CLIENT_SECRET=your_application_client_secret
CTRADER_ACCESS_TOKEN=your_valid_demo_access_token
CTRADER_ACCOUNT_ID=123456789
```

Replace every example value; do not paste credentials into GitHub, source code, screenshots,
or this chat. Keep the real `.env` file local; it is already excluded by `.gitignore`.
`CTRADER_ACCOUNT_ID` must be the numeric cTrader account ID, not the login/email or the
application client ID. `CTRADER_HOST` must be exactly `demo`; this research ingestion
rejects LIVE.

Install the added dotenv dependency and check that configuration is being read:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.research_validation --check-ctrader-config --json
```

A `PASS` from this command confirms that the file/environment has the expected values and
that the host is DEMO. It does **not** make a network request or prove that the token is
accepted.

If the historical download fails with `CH_CTID_TRADER_ACCOUNT_NOT_FOUND`, do not guess the
account ID or substitute the broker's displayed login number. Discover the account IDs granted
to the access token on the DEMO endpoint:

```bash
python -m app.ctrader_account_discovery --json
```

Choose an account from the report with `"demo_eligible": true`, and set that exact
`account_id` as `CTRADER_ACCOUNT_ID` in the local `.env`. LIVE accounts are identified but
are never authenticated or used by this research workflow. If no account is marked eligible
for DEMO, verify the token's granted account access and reauthorize the intended DEMO account.
This discovery command authenticates the application and requests the account list only; it
does not authorize any account or request market data.

After setting the account ID returned by the discovery command, test the historical download
with a modest lookback:

```bash
python -m app.ctrader_history_ingest \
  --symbol EURUSD --timeframe M15 --lookback-days 7 \
  --output-csv data/research/EURUSD_M15_demo.csv
```

The output should report a row count and a snapshot hash. The manifest now records requested
time bounds, window count, page count, any duplicate bars removed at overlapping window
boundaries, and whether the SDK exposed the response's `hasMore` flag. The downloader divides
long lookbacks into timeframe-bounded windows and requests historical bars in count-limited
pages, proceeding backward from each page's earliest timestamp. The pinned
`ctrader-open-api==0.9.2` SDK does not expose `ProtoOAGetTrendbarsRes.hasMore`, so a full
page triggers a conservative backwards request as a fallback. Returned rows are timestamp
sorted and deduplicated before CSV normalization; the final manifest records the result as
complete only after every requested time window has been traversed.

The downloader observes the official cTrader historical-data request limit by issuing pages
sequentially on the same connection (the SDK throttles message sending). Conservative
timeframe-specific windows stay below documented maximum timestamp spans. If this network
check fails after configuration passes, check token expiry/permissions, the account ID, the
exact instrument name in the DEMO account, and network access to the cTrader DEMO endpoint.
The error output is deliberately sanitized; never print or share token/secret values when debugging.

## cTrader TLS and connection failures

The CLI includes the sanitized cTrader error class and, for recognized API error responses,
the error code. API-provided descriptions are deliberately omitted from diagnostics so that
messages cannot accidentally expose sensitive fields. If the Twisted TLS warning mentions a
missing `service_identity` module, pull the latest requirements and reinstall dependencies:

```bash
git pull origin main
python -m pip install -r requirements.txt
python -m pip show service-identity
```

The package improves hostname verification; it does not by itself prove that authentication
or connectivity succeeds. Use the reported error code to distinguish application-credential
errors, account/token authorization errors, request timeouts, and disconnects. Never disable
TLS certificate verification to work around a connection problem.

## 0. Audit a downloaded raw OHLC snapshot before preparing features

After a cTrader historical download completes, run the local integrity audit against its CSV
and companion manifest. This audit does not connect to cTrader, modify the input, or train a model.

```bash
python -m app.research_data_audit \
  --csv data/research/EURUSD_M365_demo.csv \
  --symbol EURUSD --timeframe M15 \
  --output data/research/EURUSD_M365_demo.audit.json --json
```

The audit verifies that the raw file matches the manifest SHA-256, checks the manifest's DEMO
and non-actionable flags, confirms the recorded row count and timestamp bounds, and inspects
chronology, duplicate timestamps, timeframe alignment, and OHLC constraints. It reports the
observed time span, interval statistics, pagination metadata, and large time gaps. Gap warnings
are intentionally not treated as corruption automatically because FX weekends, holidays,
instrument inactivity, and provider behavior can create legitimate gaps. Inspect any reported
large gaps before model evaluation.

A `PASS` or `PASS_WITH_WARNINGS` means the structural audit completed without a hard error;
it does not prove that every expected market candle exists or that a dataset is suitable for
training. The input CSV and manifest are kept unchanged, and the output report is non-actionable.

## 1. Preflight the Ubuntu runtime

From the repository root with the project virtual environment active:

```bash
python -m app.research_validation --json
```

This discovers installed Python modules and records the resource-aware CPU/RAM profile.
It does not import every ML framework, test drivers, or connect to llama.cpp.

To verify that the cTrader DEMO environment variables are present and valid:

```bash
python -m app.research_validation --check-ctrader-config --json
```

This reads `CTRADER_CLIENT_ID`, `CTRADER_CLIENT_SECRET`, `CTRADER_ACCESS_TOKEN`,
`CTRADER_ACCOUNT_ID`, and the optional `CTRADER_HOST`; it reports presence/status only,
never credential values, and makes no network request. The config check is not a connectivity
certification. An actual historical download requires the explicit `--fetch-ctrader`
pipeline flag and remains restricted to cTrader DEMO.

Optional preflight receipt:

```bash
python -m app.research_validation --check-ctrader-config \
  --output reports/research-preflight.json --json
```

## 2. Run the research pipeline

Use an existing canonical OHLC CSV for a fully offline run, or explicitly request cTrader
DEMO ingestion with `--fetch-ctrader`. Example using a local CSV:

```bash
python -m app.research_pipeline \
  --ohlc-csv data/research/EURUSD_M15_ctrader.csv \
  --symbol EURUSD --timeframe M15 \
  --horizon-bars 4 --class-1-means UP \
  --retrieved-at-utc "2026-10-21T12:00:00Z" \
  --terms-reference "<actual applicable data-provider terms reference>" \
  --source-version "operator-export-v1" \
  --run-id eurusd-m15-validation-001 \
  --initial-train-bars 2000 --validation-bars 300 \
  --test-bars 300 --step-bars 300 \
  --trials-per-fold 5 --strict-validation
```

Replace the sample date, path, provider terms, data version, instrument, and fold sizes with
actual values. Do not invent data provenance. Fold sizes describe pre-purge bars and must fit
the sample after horizon-based purging. For a resource-constrained laptop, begin with fewer
trials and smaller experiments; do not start several model jobs in parallel.

When `--ohlc-csv` points to a cTrader downloader output with the adjacent
`<csv-path>.manifest.json`, the pipeline now derives the retrieval timestamp and cTrader DEMO
source identity from that receipt. You can omit `--retrieved-at-utc` in that case; the audit
gate validates the manifest's host, symbol, timeframe, pagination completeness, row count,
timestamp bounds, and snapshot hash before feature generation. A local CSV without a cTrader
receipt still requires an explicit `--retrieved-at-utc` and receives a provenance warning.

Before any feature or label generation, the pipeline now runs the raw OHLC integrity audit and
writes `<run-id>_raw_ohlc_audit.json`. A `FAIL` stops the pipeline before dataset preparation or
model experiments. Large time gaps remain warnings (not automatic failures), because FX weekends
and market closures are expected. cTrader downloads must have a matching DEMO pagination/hash
manifest. An operator-supplied local CSV can be audited without a companion manifest, but it is
explicitly marked with a provenance warning because its input hash cannot be independently
verified against a retrieval receipt.

The experiment-matrix provenance audit compares the shared test windows against each model's fold
records. All of these UTC timestamps are normalized to the same ISO-8601 `Z` representation before
comparison, so equivalent `+00:00` and `Z` offsets do not create false audit failures. The pipeline
also prints `provenance_audit_errors` and `provenance_audit_warnings` in its JSON output to make
future failures diagnosable without copying the entire response.

The pipeline also runs a chronological temporal calibration audit from the persisted experiment
matrix and writes `<run-id>_temporal_calibration_stability.json`. This receipt verifies the
matrix's embedded payload hash, validates timezone-aware common test windows, checks that each
experiment covers every declared fold, and reports held-out ECE by chronological fold with its
moving-block uncertainty interval where available. It does not refit a calibrator or change model
metrics. Missing calibration intervals or fewer than three eligible periods are explicit warnings;
invalid matrix provenance or incomplete fold coverage is a hard audit failure. The path, exact
file SHA-256, status, errors, and warnings are stored in the pipeline manifest and checked by
`app.research_run_audit` during consolidated validation.

Every successful pipeline run now emits:
- raw OHLC audit receipt with status, warnings, and a SHA-256 stored in the pipeline manifest;
- canonical raw OHLC snapshot (when fetching from cTrader);
- prepared feature/label dataset and its manifest;
- experiment matrix with XGBoost and LightGBM fold results;
- structural provenance audit receipt;
- temporal calibration stability receipt tied to the exact experiment-matrix file hash;
- pipeline manifest with artifact hashes for the available run artifacts;
- consolidated `<run-id>_validation_report.json`.

File names based on an instrument are sanitized separately from the symbol value, and
`run-id` is restricted to a safe filename token. Pipeline artifact paths are stored as
absolute paths to make validation robust when invoked from another working directory.

## 3. Revalidate an existing completed run

```bash
python -m app.research_validation \
  data/research/runs/<run-id>_pipeline_manifest.json \
  --output data/research/runs/<run-id>_validation_report.json --json
```

The validator checks recorded hashes, cross-file run IDs and data hashes, OHLC invariants,
strict timestamp order/uniqueness, label availability, finite feature values, binary label
counts, and the structure and numerical ranges of fold test metrics. The report summarizes
Brier score, log loss, fixed-threshold accuracy, prevalence, mean predicted probability, and
expected calibration error using descriptive fold-level mean/minimum/maximum values.

Test-window overlap is detected and reported: overlapping fold-level scores are not independent
observations and must not be pooled as if they were. No candidate is ranked or selected by this
report. The validator also records package discovery and resource profile details. The optional
cTrader config check only inspects environment configuration; it is not a broker/network test.

## 5. Temporal stability and label-gap sensitivity

Use the temporal stability diagnostic after the baseline and volatility-regime reports:

```bash
python -m app.research_temporal_stability \
  --manifest data/research/runs/<run-id>_pipeline_manifest.json \
  --ohlc-csv data/research/<exact-raw-ohlc-file>.csv \
  --output data/research/runs/<run-id>_temporal_stability.json \
  --gap-threshold-hours 8
```

The tool verifies the run manifest, exact raw OHLC file hash, and experiment-matrix hash before
scoring held-out predictions. It reports per-fold log loss and Brier score, fold counts where
model loss improves or worsens versus each persisted baseline, and sensitivity groups based on
whether the forward label horizon spans an OHLC interval longer than the configured threshold.
It also separates exposures crossing weekend-closure-like gaps from non-weekend long gaps and
reports whether a crossed interval carries Christmas/New Year's fixed-date context. The
`prediction_count` / `prediction_model_row_count` totals one row per model forecast; because models
share held-out windows, `unique_prediction_timestamp_count` separately reports the deduplicated
number of evaluation timestamps. Category exposure counts therefore may be larger than the number
of unique timestamps when multiple models score the same forecast windows.

The label horizon is interpreted as a number of rows in the original chronologically ordered OHLC
snapshot, matching the dataset builder's forward-close label definition. A nominal four-bar horizon
can therefore span far more than four timeframe durations if bars are separated by a market closure
or a data gap. Check `metrics_by_exposure_category` and `crosses_holiday_context_gap` groups before
interpreting aggregate scores.

Long-gap classifications use the same conservative weekend-window heuristic and fixed-date context
annotations as the raw OHLC audit. Those calendar tags are not an authoritative session calendar.
Gap-exposed and unexposed samples may differ in time-of-week, label prevalence, volatility, and
sample size, so score differences are descriptive rather than causal. Fold-level consistency is
not an inferential significance test. No model is ranked or selected by this report; outputs remain
non-actionable and disconnected from execution.

## 4. Produce a compact model-results summary

To avoid copying the entire pipeline JSON, create a short summary from the completed pipeline
manifest and its experiment-matrix report:

```bash
python -m app.research_results_summary \
  --manifest data/research/runs/<run-id>_pipeline_manifest.json \
  --output data/research/runs/<run-id>_results_summary.json
```

The command verifies the pipeline-manifest fingerprint and the experiment-matrix file hash,
then prints per-experiment fold-level means and ranges for log loss, Brier score, fixed-threshold
accuracy, expected calibration error, label prevalence, and mean predicted probability. It also
compares average log loss and Brier score to a constant 0.5-probability reference (log loss
`ln(2)`, Brier score `0.25`). Positive improvements mean the average metric is lower than that
reference. This is a descriptive reference, not a trading strategy or a profitability benchmark.
The command never ranks or automatically selects a model.

## Status semantics

- `PASS`: all structural and data checks passed with no warnings.
- `PASS_WITH_WARNINGS`: checks did not find a hard integrity/data error, but a warning remains
  (for example, class imbalance, overlapping test windows, or unavailable local packages).
- `FAIL`: an integrity check, data-quality check, or experiment-report validation failed.
- Pipeline CLI exit code `4`: `--strict-validation` was supplied and consolidated validation failed.
- Pipeline CLI exit code `5`: `--strict-temporal-calibration-audit` was supplied and the temporal calibration audit failed.
- Validation CLI exit code `3`: the requested validation failed.

The pipeline still preserves diagnostic artifacts on validation failure. Review errors and
warnings before trusting reported metrics. A passing report does not establish economic
profitability, predictive edge, full absence of feature leakage, provider data rights, or
production readiness. All outputs remain non-actionable and disconnected from broker execution.



## Market-data gaps and expected FX closures

The raw OHLC integrity audit now retains the total count and examples of intervals greater than
eight hours, while also classifying likely FX weekend-closure intervals separately from other long
gaps. The weekend classification is a conservative UTC timestamp heuristic: it requires a late
Friday start, a Sunday-evening or early-Monday resumption, and a duration of 24-72 hours. It is not
an authoritative trading-session calendar and does not prove that a missing interval was expected.

The audit reports `weekend_closure_like_gap_count_over_8_hours`,
`non_weekend_long_gap_count_over_8_hours`, and examples for both categories. It additionally tags
gaps that include the UTC calendar date of Christmas Day or New Year's Day with
`calendar_context`, summarized by `holiday_context_gap_count_over_8_hours` and
`holiday_context_gap_examples`. These are informational fixed-date annotations only: they do not
change `gap_category`, suppress the non-weekend warning, prove that the market was closed, or
confirm that the provider returned every expected candle. The two holiday-context dates are not a
substitute for a broker/instrument-specific session calendar. Non-weekend long gaps still produce a
warning for review. Do not treat a warning-free weekend classification as proof of complete data
coverage; verify the provider's history and the applicable market calendar.

## 6. Audit actual elapsed time of row-based forward labels

The temporal stability report revealed that the existing four-row M15 horizon has a nominal
duration of one hour, but some targets can be almost 50 hours after the prediction timestamp
when the forward row sequence spans a weekend closure. This is a property of row-index labels;
it does not by itself mean that the label builder is wrong. The following audit measures that
duration without rewriting the dataset, labels, historical manifest, predictions, or scores.

After pulling the latest code, run:

```bash
python -m app.research_label_horizon_audit \
  --manifest data/research/runs/<run-id>_pipeline_manifest.json \
  --ohlc-csv data/research/<exact-raw-ohlc-file>.csv \
  --output data/research/runs/<run-id>_label_horizon_elapsed_audit.json \
  --extended-time-threshold-hours 8 \
  --longest-examples-limit 25
```

The command verifies the research manifest, raw OHLC SHA-256, experiment-matrix SHA-256, and
run identity. It maps each prediction timestamp to its source OHLC row, then audits the target at
`prediction_row_index + horizon_bars`, exactly matching the existing row-based horizon semantics.
The nominal elapsed duration is derived from `horizon_bars` and the declared timeframe. It reports
elapsed-time quantiles, the fraction of labels exceeding nominal duration, counts and descriptive
probability metrics for elapsed-time buckets, per-fold summaries, and examples of the longest
horizons.

Elapsed-time buckets are `nominal_or_within_tolerance`, `extended_up_to_threshold`, and
`long_extension_over_threshold`. The threshold is diagnostic only; the default is eight hours.
The report separately counts model prediction rows and unique timestamps because multiple models
score the same held-out windows. Repeated timestamps within an experiment are surfaced as a warning
for possible fold overlap. A prediction whose target row cannot be resolved is an error, not silently
discarded.

This report is offline, descriptive, non-actionable, and disconnected from execution. It does not
change the label-generation rule or automatically select a model. The bucket metrics are exploratory:
long-duration buckets may be small and may differ in market regime and label prevalence. If timing
semantics are to change, create a distinct versioned dataset and a new pipeline run so the old run
remains auditable; do not silently overwrite historical artifacts.

## 7. Run a separately versioned elapsed-time label-eligibility experiment

The elapsed-time audit can reveal that a row-based forward target is more than the nominal
horizon away because of timestamp gaps. For example, an M15 target four rows ahead has a
nominal one-hour horizon, but a row sequence may cross a weekend closure. Preserve the original
run and use a new run ID for a capped eligibility experiment.

For an existing baseline, the repository now has a manifest-driven runner that verifies the run's artifact hashes, reconstructs walk-forward partition sizes from the prepared dataset timestamps and persisted test windows, derives the trial count and feature variants from the experiment matrix, and creates a unique capped run ID. From the repository root, run:

```bash
python -m app.research_label_eligibility_variant \
  --baseline-manifest data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_pipeline_manifest.json \
  --max-label-elapsed-hours 1.0
```

The runner does not download data, modify the baseline, connect to a broker, or select a model. It reuses the original raw OHLC CSV and requires the variant's raw hash, model/feature pairs, and common test windows to match the baseline. If the default run ID already has any pipeline or variant receipt files, the runner automatically chooses the next free suffix (`-rerun-2`, `-rerun-3`, and so on) rather than overwriting files; the chosen ID and whether it was auto-versioned are recorded in the receipts. When you explicitly provide `--run-id`, a collision remains a hard error so the requested immutable namespace is never silently changed. After the capped run finishes, it automatically runs the paired label-eligibility comparator on identical eligible held-out timestamps and writes both `<variant-run-id>_label_eligibility_comparison.json` and `<variant-run-id>_label_eligibility_variant_setup.json`. If the baseline is an older manifest that did not persist all configuration fields, the runner records explicit assumptions; in particular, a missing original random seed or bootstrap count is not silently claimed as recovered. Supply `--random-seed` and/or `--bootstrap-resamples` to override those values when the original settings are known. Use `--plan-only` to inspect the reconstructed configuration without starting model experiments.

For new runs, the pipeline manifest stores a `research_config` object containing fold sizes, trials per fold, random seed, bootstrap count, ablation configuration, and strict-audit flags so future sensitivity runs need fewer assumptions.

The equivalent manual procedure remains available. Use the same raw OHLC snapshot, model/feature settings, walk-forward sizes, random seed, and
trial count as the baseline. The optional `--max-label-elapsed-hours 1.0` cap marks labels
ineligible when the existing row-index target is more than one elapsed UTC hour from the
prediction timestamp:

```bash
python -m app.research_pipeline \
  --ohlc-csv data/research/EURUSD_M365_demo.csv \
  --symbol EURUSD \
  --timeframe M15 \
  --horizon-bars 4 \
  --class-1-means UP \
  --terms-reference "<the same actual provider terms reference as the baseline>" \
  --source-version "<the same source version as the baseline>" \
  --output-dir data/research/runs \
  --run-id eurusd-m15-demo-m365-baseline-validation-20261023-elapsed-cap-1h \
  --initial-train-bars <same baseline value> \
  --validation-bars <same baseline value> \
  --test-bars <same baseline value> \
  --step-bars <same baseline value> \
  --trials-per-fold <same baseline value> \
  --random-seed <same baseline value> \
  --bootstrap-resamples <same baseline value> \
  --max-experiments 2 \
  --no-ablation \
  --max-label-elapsed-hours 1.0
```

Replace every angle-bracket value with the exact setting used for the completed baseline run.
Use a unique run ID, and do not overwrite its baseline artifacts. Keep `--no-ablation` and
`--max-experiments` consistent with the baseline matrix configuration.

The cap is applied to actual UTC elapsed time between the prediction timestamp and existing
forward target timestamp. The builder retains every output row and the original forward label,
writing `label_elapsed_hours` and `label_eligible` metadata for the capped run. This preserves
the underlying timestamp timeline, so the chronological fold boundaries remain comparable.
Ineligible rows are excluded from model fitting, validation-based tuning, final refitting, and
held-out scoring. Reports separately count excluded rows for each partition and the dataset
manifest records the cap and eligibility counts. With no cap specified, the builder retains its
legacy CSV schema and all labels remain eligible.

After the variant completes, compare its experiment matrix and validation report with the
baseline. Review held-out sample counts, label prevalence, fold-by-fold log loss/Brier score,
eligibility exclusions, and whether the test windows still match. A one-hour cap is an
eligibility sensitivity test, not a claim that the target semantics are now correct. The
remaining predictions may still have other data-quality or leakage limitations; no model is
selected automatically and no execution path is connected.

## 8. Compare the capped run against the baseline on paired timestamps

After the baseline and cap run have both completed, create a paired comparison report:

```bash
python -m app.research_label_eligibility_comparison \
  --baseline-manifest data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_pipeline_manifest.json \
  --variant-manifest data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023-elapsed-cap-1h_pipeline_manifest.json \
  --output data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_label_eligibility_comparison.json
```

Use the exact variant manifest path created by your run ID. Do not type the JSON report output into
the shell; the command prints a compact status summary and saves the detailed report to the output
file.

The comparator validates both pipeline-manifest fingerprints and referenced artifact hashes,
checks that the original raw OHLC snapshots match, and confirms that the shared dataset timestamps,
OHLC, labels, target-availability timestamps, and feature values remain identical. It also requires
the same symbol/timeframe, source metadata, label horizon, class semantics, feature schema, model/feature
experiment pairs, and chronological fold/test windows.

The immutable raw-snapshot hash and audit status must match. The audit warning text itself can differ
between historical runs when audit rules or warning categories have changed since the baseline was
created. In that case, the paired comparison continues with PASS_WITH_WARNINGS, records both warning
lists, and sets raw_ohlc_audit_warning_sets_identical to false; review these warnings before
interpreting the metrics. Raw snapshot identity, successful per-run audit integrity, and no hard audit
errors remain mandatory.

For each matching model/feature variant and fold, the report contains three views:

- The baseline's original held-out metrics across its complete test window.
- The baseline's metrics recomputed only on timestamps that remain eligible in the capped dataset.
- The capped model's metrics on those exact same eligible timestamps.

The paired deltas use `variant_minus_baseline`. Negative log-loss and Brier-score deltas, or a positive
fixed-threshold accuracy delta, favor the capped variant descriptively. Because both models are then
evaluated on the same eligible timestamps, this paired comparison separates model differences from the
mechanical change in which test labels are included. The baseline-full metrics are kept as context and
must not be compared as if they came from the same sample.

Fold means are unweighted descriptive summaries. If test windows overlap, rows across folds are not
independent; do not pool them as independent observations or interpret the means as inferential
significance. The comparator is not a model-selection tool and does not establish profitability.
All outputs remain `actionable=false` and `execution_connected=false`.


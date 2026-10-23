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

The output should report a row count and a snapshot hash. If this network check fails after
configuration passes, check token expiry/permissions, the account ID, the exact instrument
name in the DEMO account, and network access to the cTrader DEMO endpoint. The error output
is deliberately sanitized; never print or share token/secret values when debugging.

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

Every successful pipeline run now emits:
- canonical raw OHLC snapshot (when fetching from cTrader);
- prepared feature/label dataset and its manifest;
- experiment matrix with XGBoost and LightGBM fold results;
- structural provenance audit receipt;
- pipeline manifest with artifact hashes;
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

## Status semantics

- `PASS`: all structural and data checks passed with no warnings.
- `PASS_WITH_WARNINGS`: checks did not find a hard integrity/data error, but a warning remains
  (for example, class imbalance, overlapping test windows, or unavailable local packages).
- `FAIL`: an integrity check, data-quality check, or experiment-report validation failed.
- Pipeline CLI exit code `4`: `--strict-validation` was supplied and consolidated validation failed.
- Validation CLI exit code `3`: the requested validation failed.

The pipeline still preserves diagnostic artifacts on validation failure. Review errors and
warnings before trusting reported metrics. A passing report does not establish economic
profitability, predictive edge, full absence of feature leakage, provider data rights, or
production readiness. All outputs remain non-actionable and disconnected from broker execution.

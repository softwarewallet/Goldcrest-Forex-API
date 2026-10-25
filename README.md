# Goldcrest Forex API

Standalone AI market-analysis API for GoldcrestForex. Prediction and analysis only: no broker credentials and no order-placement endpoints.

## Stack
- Python + FastAPI + Pydantic
- llama.cpp OpenAI-compatible API, default `http://192.168.29.80:8080`
- SQLite through SQLAlchemy
- pytest

GoldcrestForex remains responsible for execution, sizing, and all risk gates. GoldcrestAI is developed and validated independently before any integration is authorized.

## Development setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Replace API_KEY with a newly generated secret before using the API.
python -m pytest -q
uvicorn app.main:app --host 127.0.0.1 --port 8090
```

For cTrader historical research, put `CTRADER_HOST`, `CTRADER_CLIENT_ID`, `CTRADER_CLIENT_SECRET`, `CTRADER_ACCESS_TOKEN`, and the numeric `CTRADER_ACCOUNT_ID` in the repository-root `.env` file (same folder as `requirements.txt`). The research loader reads that file explicitly; see [cTrader DEMO setup and verification](docs/end-to-end-research-validation.md#ctrader-demo-credentials-where-to-put-them). The real `.env` remains ignored by Git. Only `CTRADER_HOST=demo` is accepted by this research workflow.

The API bind address and port are set by the Uvicorn command, not by the `GOLDCREST_BIND_HOST` / `GOLDCREST_PORT` settings in `.env`. For trusted-LAN access, run Uvicorn with `--host 0.0.0.0 --port 8090` and configure the host firewall to permit port 8090 only from the required private subnet. Never expose this API or llama.cpp directly to the public Internet. The production systemd service should use explicit, reviewed bind and port arguments.

## Routes
- `GET /api/v1/health`
- `GET /api/v1/models`
- `POST /api/v1/predictions`
- `GET /api/v1/predictions/{prediction_id}`
- `POST /api/v1/outcomes`
- `GET /api/v1/performance`
- `GET /api/v1/audit`

Protected routes use `X-API-Key`. Set `LLAMA_MODEL_ID` to the exact ID returned by llama.cpp `/v1/models`, or leave it empty to auto-select only when exactly one model is served.

## Prediction status
The initial prediction service is scaffolding, not a validated trading strategy. It deliberately returns `NO_TRADE` even when the LLM suggests BUY or SELL, until a quantitative rule is validated. Model self-reported confidence is not used as a calibrated probability. No broker credentials are accepted or stored.

See `docs/phase-1-audit.md` for current audit findings and remaining work.


## Resource-aware Ubuntu runtime

For constrained laptops, run the lightweight environment preflight before starting research:

```bash
python -m app.runtime_diagnostics
python -m app.runtime_diagnostics --json
```

XGBoost and LightGBM use a conservative RAM-aware CPU thread limit. Override it for
controlled benchmarks with `GOLDCREST_ML_THREADS=1` (or another positive integer).
See `docs/resource-aware-runtime.md` for limits, diagnostics, and safeguards.


## Offline research provenance audit

Each completed run writes a separate `<run-id>_provenance_audit.json` receipt beside the experiment-matrix report. The pipeline manifest records the receipt path, SHA-256 fingerprint, and audit status; the original experiment-matrix report is not modified. The auditor checks structural consistency and recorded safeguards only. It does not recompute metrics, prove absence of data leakage, validate provider data rights, or establish profitability.

By default, a failed structural audit is recorded in the receipt and manifest while preserving the research artifacts for inspection. Add `--strict-provenance-audit` to make the CLI return exit code 3 when the audit status is `FAIL`. This mode still writes the audit receipt and pipeline manifest. The research pipeline remains offline unless `--fetch-ctrader` is explicitly selected, and no research output is connected to broker execution.


## Temporal calibration stability audit

Every research-pipeline run now creates a separate
`<run-id>_temporal_calibration_stability.json` receipt from the exact persisted experiment
matrix. The pipeline manifest records its path, SHA-256, status, errors, and warnings. The
receipt verifies the matrix's embedded payload hash, UTC-normalizes test-window boundaries,
requires complete fold coverage, and summarizes chronological out-of-sample expected
calibration error (ECE) with the stored moving-block uncertainty intervals. Missing ECE or
uncertainty fields are surfaced as warnings; invalid source structure or incomplete fold
coverage fails the audit. Existing run IDs are refused before pipeline artifacts are changed.

To run the audit directly against a stored matrix:

```bash
python -m app.temporal_calibration_stability \
  --experiment-matrix data/research/runs/<run-id>_experiment_matrix.json \
  --output data/research/runs/<run-id>_temporal_calibration_stability-manual.json
```

The standalone command refuses to overwrite an existing output file. Use
`--strict-temporal-calibration-audit` with `app.research_pipeline` to return exit code 5
when its generated audit has status `FAIL`. The audit is descriptive only: endpoint interval
separation is not a significance test, folds may be dependent, and it does not fit a
calibrator, select a model, estimate profitability, or authorize execution.

## Research run integrity audit

After a completed offline research run, verify all artifact hashes and cross-file consistency:

```bash
python -m app.research_run_audit data/research/runs/<run-id>_pipeline_manifest.json
python -m app.research_run_audit data/research/runs/<run-id>_pipeline_manifest.json --json
```

Exit code 0 indicates integrity checks passed (possibly with warnings); exit code 3 indicates failure. This audit checks recorded hashes and safeguards but does not certify model quality, absence of leakage, data rights, or profitability. See `docs/research-run-integrity-audit.md`.


## End-to-end research validation

Run a no-network environment preflight with `python -m app.research_validation --json`; add `--check-ctrader-config` to verify cTrader DEMO environment-variable presence without printing secret values or connecting to cTrader. Each completed research pipeline run now writes a consolidated `<run-id>_validation_report.json` containing artifact integrity checks, dataset quality checks, walk-forward metric summaries, and runtime diagnostics. Use `--strict-validation` to return a nonzero exit code when consolidated validation fails. See `docs/end-to-end-research-validation.md` for the end-to-end workflow and status semantics.




## Consolidated research evidence

After generating the diagnostics available for a run, combine integrity, validation, temporal calibration, baseline analysis, volatility-regime analysis, and label-gap sensitivity into one non-actionable evidence report:

```bash
python -m app.research_evidence_report \
  --manifest data/research/runs/<run-id>_pipeline_manifest.json
```

The report is written as `<run-id>_consolidated_evidence.json` beside the manifest and refuses to overwrite an existing output. Optional reports are discovered by their standard run-scoped names or can be supplied with CLI flags. Missing optional diagnostics are explicitly reported as coverage gaps; cross-run/hash mismatches and invalid safety flags fail closed. See [consolidated research evidence](docs/consolidated-research-evidence.md) for schema checks and legacy-run usage.

## Volatility-regime diagnostics

After reviewing baseline and fold stability, inspect conditional performance by past-only
historical-volatility regime without retraining:

```bash
python -m app.research_regime_analysis \
  --manifest data/research/runs/<run-id>_pipeline_manifest.json \
  --ohlc-csv data/research/<exact-raw-ohlc-file>.csv \
  --output data/research/runs/<run-id>_regime_analysis.json \
  --volatility-window 20
```

The command requires the exact OHLC CSV whose SHA-256 is recorded by that run's manifest.
Each fold's low/mid/high thresholds use volatility observations strictly before the test start.
Results are descriptive and non-actionable; the tool neither ranks models nor connects to
execution. See [volatility-regime diagnostics](docs/research-regime-analysis.md).

## Temporal stability and label-gap sensitivity

After baseline and volatility-regime analysis, inspect per-fold metric consistency and whether a
forward label horizon crosses an unusually long interval in the raw OHLC timestamps:

```bash
python -m app.research_temporal_stability \
  --manifest data/research/runs/<run-id>_pipeline_manifest.json \
  --ohlc-csv data/research/<exact-raw-ohlc-file>.csv \
  --output data/research/runs/<run-id>_temporal_stability.json \
  --gap-threshold-hours 8
```

The report shows each model's fold-level log loss/Brier score and consistency of improvement over
fixed baselines, plus separate descriptive metrics for labels whose forward horizon does or does
not cross a long timestamp gap. A positive loss improvement favors the model over that baseline.
This is not a significance test, a causal estimate of gap effects, or a model-selection mechanism.
Outputs remain non-actionable and disconnected from execution.

## Time-series baseline and uncertainty analysis

New offline research runs store observation-level held-out predictions so model forecasts can be compared on identical timestamps with constant-probability, matured-refit prevalence, matured-refit majority-class, and lagged-bar-direction baselines. Analyze a completed run with:

```bash
python -m app.research_baseline_analysis \
  --manifest data/research/runs/<run-id>_pipeline_manifest.json \
  --output data/research/runs/<run-id>_baseline_analysis.json \
  --bootstrap-resamples 1000
```

The analyzer uses paired circular moving-block bootstrap intervals for descriptive uncertainty. It does not rank or promote models or connect outputs to broker execution. Older matrix files without per-observation predictions must be regenerated by rerunning the offline pipeline. See `docs/time-series-baseline-analysis.md`.

## Cross-cap label-duration stability

After the three-cap common-cohort audits pass for seeds 20261021–20261025, aggregate
same-cohort log-loss, Brier-score, and fixed-0.5 accuracy deltas across 1h, 1.5h, and 8h:

```bash
python -m app.research_cross_cap_stability \
  --audit-report data/research/runs/common_cohort_audit_1h_1p5h_8h_seed_20261021.json \
  --audit-report data/research/runs/common_cohort_audit_1h_1p5h_8h_seed_20261022.json \
  --audit-report data/research/runs/common_cohort_audit_1h_1p5h_8h_seed_20261023.json \
  --audit-report data/research/runs/common_cohort_audit_1h_1p5h_8h_seed_20261024.json \
  --audit-report data/research/runs/common_cohort_audit_1h_1p5h_8h_seed_20261025.json \
  --output data/research/runs/cross_cap_stability_1h_1p5h_8h_five_seed.json
```

The audit is descriptive and non-actionable; it summarizes fold deltas within each seed before
cross-seed aggregation and does not authorize model selection or trading. See
`docs/research-cross-cap-stability.md` for interpretation and safeguards.


## Offline model evaluation integrity gate

Validate persisted walk-forward experiment matrices for metric arithmetic, fold coverage, timezone-aware temporal boundaries, and explicit non-actionable/no-test-selection safeguards without changing the source report. See `docs/research-model-evaluation-gate.md` and run `python -m app.research_model_evaluation_gate --experiment-matrix data/research/runs/<run-id>_experiment_matrix.json --json`.


The offline pipeline also writes a hash-addressed `<run-id>_model_evaluation_gate.json` receipt from the exact persisted experiment matrix. Use `--strict-model-evaluation-gate` to return exit code 6 if that independent structural/metric-integrity gate fails. The receipt is diagnostic only and does not select a model or authorize execution. See `docs/research-model-evaluation-gate.md`.


The consolidated evidence report now verifies and summarizes the model evaluation gate receipt alongside provenance and temporal calibration evidence. For new runs it checks the receipt's recorded hash, run identity, and source-matrix hash; legacy manifests without the gate are marked as incomplete evidence rather than treated as having passed the gate.

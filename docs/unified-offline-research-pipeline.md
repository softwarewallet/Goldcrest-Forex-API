# Unified offline research pipeline

`app.research_pipeline` connects the existing canonical OHLC input, causal feature/label
builder, purged walk-forward plan, Optuna tuning, and XGBoost/LightGBM feature-ablation
comparison into one explicit CLI. Every output is research-only; it does not promote a model,
publish a live signal, change Auto Live configuration, or call order APIs.

## Mode A: use an existing OHLC snapshot (no network)

The input must be canonical CSV with `timestamp,open,high,low,close`; optional `symbol`
and `timeframe` columns must match the command. Supply the actual snapshot retrieval time
and applicable provider terms, rather than inventing provenance.

```bash
python -m app.research_pipeline \
  --ohlc-csv data/research/EURUSD_M15_ctrader.csv \
  --symbol EURUSD \
  --timeframe M15 \
  --horizon-bars 4 \
  --class-1-means UP \
  --retrieved-at-utc "2026-10-21T12:00:00Z" \
  --terms-reference "<actual applicable provider terms reference>" \
  --source-version "ctrader-open-api==0.9.2" \
  --run-id eurusd-m15-research-001 \
  --initial-train-bars 2000 \
  --validation-bars 300 \
  --test-bars 300 \
  --step-bars 300 \
  --trials-per-fold 5
```

The timestamp above is an example only: replace it with the actual retrieval timestamp.
Use fold sizes appropriate to the amount and cadence of data available. The command will
fail rather than silently sort or impute invalid data.

## Mode B: explicitly fetch cTrader DEMO data

Network access happens only when `--fetch-ctrader` is supplied. It uses the existing
environment variables `CTRADER_CLIENT_ID`, `CTRADER_CLIENT_SECRET`,
`CTRADER_ACCESS_TOKEN`, and `CTRADER_ACCOUNT_ID`; `CTRADER_HOST` must be `demo`.

```bash
python -m app.research_pipeline \
  --fetch-ctrader \
  --symbol EURUSD \
  --timeframe M15 \
  --lookback-days 365 \
  --horizon-bars 4 \
  --class-1-means UP \
  --terms-reference "<actual applicable provider terms reference>" \
  --source-version "ctrader-open-api==0.9.2" \
  --run-id eurusd-m15-research-001 \
  --initial-train-bars 2000 \
  --validation-bars 300 \
  --test-bars 300 \
  --step-bars 300 \
  --trials-per-fold 5
```

This DEMO fetch is historical/read-only. A successful CI run does not prove that the user's
credentials or network connection work; the first real run still needs to be verified locally.

## Outputs

The output directory defaults to `data/research/runs`. It contains:
- canonical raw OHLC snapshot when fetched from cTrader;
- causal-feature and mature-label dataset CSV plus its provenance manifest;
- fixed-fold experiment matrix JSON report;
- a pipeline manifest binding the input, dataset, and report hashes.

The experiment matrix compares XGBoost and LightGBM on shared purged chronological folds.
Feature-family ablations are enabled by default; use `--no-ablation` to run only the full
feature set. `--max-experiments` limits the experiment count. Trial count, fold geometry,
and bootstrap resamples are explicit resource controls.

## Safety and limitations

- Exactly one input mode is required: local `--ohlc-csv` or explicit `--fetch-ctrader`.
- cTrader retrieval remains DEMO-only and read-only.
- No Auto Live, broker execution, or risk-control code is imported for writes or modified.
- Test metrics are descriptive; no automatic winner selection occurs.
- Reports are non-actionable and are not evidence of profitability.
- The research command does not use news, PyTorch, Llama/Qwen, Vega, Nova, or Titan yet. The completed-run validator records package discovery but does not automatically add those systems to model training. Use `python -m app.research_validation --json` for a no-network runtime preflight and add `--strict-validation` to make the research CLI fail when consolidated validation fails.
- Run tests with `python -m pytest -q tests/test_research_pipeline.py tests/test_research_orchestrator.py`.

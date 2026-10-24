# Multi-seed label-eligibility sensitivity run

## Purpose

Use this runner after a single matched uncapped/capped replay to check whether the observed metric changes are sensitive to the optimizer random seed. It creates several fresh paired runs. Within every pair, the uncapped and capped variants use the same seed, reconstructed chronological fold geometry, model/feature matrix, tuning-trial budget, bootstrap count, and raw OHLC snapshot.

The historical baseline manifest is used only to reconstruct settings and verify source artifacts. If the historical manifest did not preserve its original random seed, the seed sweep does not recover that unknown seed; it explicitly runs new, matched replay pairs instead.

## Run

From the repository root with its virtual environment active:

```bash
python -m app.research_label_eligibility_seed_sweep \
  --baseline-manifest data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_pipeline_manifest.json \
  --max-label-elapsed-hours 1.0
```

The default configuration uses five distinct seeds. To run a smaller diagnostic sweep or a different prespecified list, use `--seeds` with at least two distinct non-negative 32-bit integers, for example `--seeds 20261021,20261022,20261023`. Each seed triggers two fresh research runs, so runtime increases linearly with the number of seeds.

The runner writes new immutable run artifacts, per-seed paired comparison reports, run receipts, and one aggregate `*_seed_sweep_report.json`. It selects collision-free run IDs and output-report names; existing files are not overwritten.

## Interpretation

The aggregate report includes per-metric mean/median deltas, ranges, and counts of seeds in which the cap direction is favorable, adverse, or neutral. Negative log-loss/Brier deltas and positive fixed-threshold accuracy deltas favor the cap descriptively.

All seeds reuse the same historical OHLC snapshot and walk-forward test windows. Repeated seeds measure optimizer/retraining sensitivity; they do not add independent market observations and must not be interpreted as independent replication, statistical significance, or profitability evidence. The runner does not rank, select, promote, or connect a model to execution.

Each pair's raw data, manifest hashes, test-window identity, validation, provenance, and comparison status remain independently auditable. Any failed seed makes the overall sweep status `FAIL`; inspect the partial report and per-seed receipts rather than relying on an incomplete aggregate.

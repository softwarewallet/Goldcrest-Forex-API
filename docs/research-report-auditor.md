# Offline research report auditor

`app.research_report_audit` verifies the integrity and consistency of artifacts emitted by
`app.research_pipeline`. It is useful when historical data is not available at development
time: the auditor can be tested against synthetic fixtures and later run against a real
research run without fitting models or making network requests.

## Usage

After a pipeline run:

```bash
python -m app.research_report_audit \
  --pipeline-manifest data/research/runs/eurusd-m15-research-001_pipeline_manifest.json \
  --output data/research/runs/eurusd-m15-research-001_audit.json
```

The command exits with:
- `0` when all checks pass;
- `1` when the audit runs but finds failed checks;
- `2` when it cannot run because of an input or runtime error.

Use `--base-dir` when artifact paths in a manifest should be resolved against a specific
directory. The tool prints a JSON report to stdout even when `--output` is supplied.

## Checks performed

- Recomputes the pipeline manifest's canonical SHA-256 integrity hash.
- Confirms referenced raw OHLC, prepared dataset, dataset manifest, and experiment matrix
  artifacts exist and match the hashes recorded in the pipeline manifest.
- Recomputes the dataset-manifest and experiment-matrix internal hashes.
- Checks that pipeline and matrix manifests remain explicitly non-actionable, disconnected
  from execution, and do not automatically select a winner from test metrics.
- Confirms matrix dataset provenance, fold counts, and common test windows are consistent.
- Summarizes available fold-level Brier score, log loss, fixed-0.5 accuracy, and expected
  calibration error using mean, population standard deviation, minimum, and maximum.

The summary is descriptive only. It does not rank candidates, tune thresholds, declare a
profitable strategy, or promote a model. Metric variability is reported, not automatically
interpreted as a pass/fail profitability threshold.

## Offline and safety boundaries

- No network access or credentials are used.
- No model is trained and no market data is fetched.
- No broker, Auto Live, risk-control, or execution code is imported.
- An audit pass means artifact integrity and selected research invariants passed; it is not
  proof that the data source is correct, features are truly causal, model results generalize,
  or a strategy is profitable.
- Keep source CSVs and reports unchanged after an experiment; rerun the audit if any file is
  regenerated or edited.

## Tests

```bash
python -m pytest -q tests/test_research_report_audit.py
```

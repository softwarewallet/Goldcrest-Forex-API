# Consolidated research evidence report

The evidence report joins diagnostics that were created independently for the same immutable offline research run. It does not retrain models, recompute metrics, rank candidates, fit calibration transforms, estimate profitability, or connect to broker execution.

## Build a report

Run from the repository root with the virtual environment active:

```bash
python -m app.research_evidence_report \
  --manifest data/research/runs/<run-id>_pipeline_manifest.json
```

By default, the report is written beside the manifest as
`<run-id>_consolidated_evidence.json`. The output writer refuses to overwrite an existing report. Use a new output filename for a subsequent review.

The report checks the pipeline manifest and its hash-bound artifacts, then summarizes available provenance, consolidated validation, temporal calibration, baseline comparisons, volatility-regime diagnostics, and long-gap sensitivity. It automatically looks for the optional reports under their standard run-scoped names:

- `<run-id>_baseline_analysis.json`
- `<run-id>_regime_analysis.json`
- `<run-id>_temporal_stability.json`

Optional reports may also be supplied with `--baseline-analysis`, `--regime-analysis`, or `--temporal-stability`. Each provided report is checked for expected scope, run ID, symbol/timeframe when present, manifest path when present, safety flags, and matrix SHA-256 when reported. A provided report tied to another run or matrix fails the consolidated report.

For legacy runs that were created before the temporal calibration receipt was integrated into the pipeline manifest, provide an independently generated receipt explicitly:

```bash
python -m app.research_evidence_report \
  --manifest data/research/runs/<run-id>_pipeline_manifest.json \
  --temporal-calibration-audit data/research/runs/<run-id>_temporal_calibration_stability-<suffix>.json \
  --output data/research/runs/<run-id>_consolidated_evidence-review.json
```

The supplied calibration receipt must have the expected scope and run ID and must reference the exact experiment-matrix file hash. It is not rewritten or added to the original manifest.

## Status semantics

- `PASS`: all supplied/core evidence checks passed and no warnings were reported.
- `PASS_WITH_WARNINGS`: integrity checks passed, but optional diagnostics are absent or a source report carries warnings.
- `FAIL`: a core integrity check failed, a required artifact is missing, a report does not match the requested run/matrix, or a safety flag is invalid.

Missing optional analyses are reported as coverage gaps rather than negative model findings. A report can therefore remain useful while later diagnostic stages are still being run. Explicitly supplied-but-missing paths fail closed.

The report emits section status, SHA-256, compact summaries, coverage information, errors, and warnings. The summaries preserve per-experiment and per-regime data without an aggregate score or model ranking. No choice is made on the operator's behalf.

## Interpretation boundaries

ECE drift, paired bootstrap intervals, fold consistency, volatility-conditioned metrics, and label-gap exposure are descriptive diagnostics. Overlapping windows can be dependent, optional reports may represent different analyses, and none of these measures alone establishes predictive edge, economic profitability, or deployment readiness. Preserve the source reports and the original run artifacts unchanged.

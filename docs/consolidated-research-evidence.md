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


## Cross-diagnostic interpretation

The consolidated output includes an `evidence_interpretation` object to make the shape of the evidence easier to review without introducing a composite score or model ranking.

- `baseline_comparison` recomputes interval classification from each available 95% interval: above zero, below zero, includes zero, or unavailable. The underlying improvement is baseline loss minus model loss. The report does not call an interval that includes zero evidence of a clear separation.
- `temporal_calibration` summarizes the direction of each first-to-last held-out ECE change. An observed upward ECE change is a descriptive drift finding, not an automatic calibration-failure verdict.
- `volatility_regime_consistency` records whether baseline-minus-model loss improvements are positive, negative, or mixed across observed regimes for each experiment, baseline, and metric. It now also summarizes the source report's conditional paired moving-block bootstrap intervals, including the number that include zero or exclude it on either side, plus regime-level interval summaries. It does not recompute bootstrap samples or convert interval evidence into a profitability or deployment claim.
- `label_gap_exposure` repeats the gap counts and prediction-horizon exposure already recorded by the temporal-stability report; it does not suppress weekend closures or non-weekend gaps.
- `chronological_oos_robustness` derives early/middle/late period summaries from the immutable matrix's stored walk-forward test-fold metrics, including Brier score, log loss, and ECE. It requires a matching matrix SHA-256, uses only common complete test folds, preserves candidate-specific summaries, and never ranks or selects models.

These assessments use the already-generated reports and do not recompute model predictions or bootstrap intervals. The chronological OOS period diagnostic derives medians and ranges from stored per-fold metrics and validates the source matrix hash before doing so. A missing or malformed regime interval is reported as unavailable rather than treated as evidence of no effect. They preserve each experiment separately, do not calculate a universal "winner" score, and do not change status to authorize model promotion. `promotion_readiness` is deliberately reported as `NOT_ESTABLISHED_BY_THIS_REPORT`; predictive evidence and profitability including transaction costs and execution effects require separate evaluation.

## Interpretation boundaries

ECE drift, paired bootstrap intervals, fold consistency, volatility-conditioned metrics, and label-gap exposure are descriptive diagnostics. Overlapping windows can be dependent, optional reports may represent different analyses, and none of these measures alone establishes predictive edge, economic profitability, or deployment readiness. Preserve the source reports and the original run artifacts unchanged.

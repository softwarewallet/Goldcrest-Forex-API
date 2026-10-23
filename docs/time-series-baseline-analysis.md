# Time-series baseline and uncertainty analysis

This stage evaluates the stored held-out predictions from the offline research pipeline.
It does not promote a model, pick a winner, recommend a trade, or connect to broker execution.

## Required artifacts

Run the latest code before running the baseline analyzer. The experiment matrix must contain
per-observation test predictions for every fold and experiment. Earlier matrix files created
before prediction capture was added cannot be analyzed by this command; rerun the offline
pipeline to create a new, versioned research run rather than editing a historical report.

## Run

From the repository root with the virtual environment active:

```bash
python -m app.research_baseline_analysis \
  --manifest data/research/runs/<run-id>_pipeline_manifest.json \
  --output data/research/runs/<run-id>_baseline_analysis.json \
  --bootstrap-resamples 1000
```

The analyzer first verifies the pipeline manifest and experiment matrix against the run's
integrity checks. It then validates each persisted held-out prediction against its fold's
test-window boundaries and the recorded sample count. Missing prediction records, malformed
probabilities, invalid fold metadata, or incorrect safety flags fail closed.

## Baselines

- **Constant probability:** class-1 probability of 0.5 for every observation.
- **Matured refit prevalence:** class-1 prevalence calculated from only the training and
  validation labels that had matured before the fold's test start.
- **Matured refit majority class:** the majority class based on those same matured labels,
  used for the accuracy comparison only.
- **Lagged bar direction:** direction of the latest available close-to-close move, with its
  hit rate estimated using a Laplace-smoothed calculation from only matured refit labels.
  Flat moves and the first available observation fall back to the matured-refit prevalence.

The class meaning (UP or DOWN) is passed from the pipeline into the directional baseline.
Test labels are used only for scoring the baseline, not for fitting its parameters.

## Uncertainty

Log-loss and Brier-score improvements are paired on the same held-out observations and defined
as baseline loss minus model loss. Positive values favor the model. The analyzer reports a
95% interval and the share of bootstrap draws with positive improvement using a circular
moving-block bootstrap, stratified by walk-forward fold. The block length defaults to the
label horizon; override it with `--block-length` only when you have a defensible dependence
assumption. Set `--seed` to reproduce the draws.

This bootstrap is a descriptive uncertainty diagnostic, not proof of statistical significance.
Overlapping forecast windows, regime shifts, repeated timestamps, model-search bias, and
other forms of temporal dependence can still limit interpretation. Reported metrics are not
profitability metrics and do not include transaction costs, spread, slippage, or execution risk.

## Safety boundaries

All output flags remain non-actionable and execution-disconnected. The analyzer reports each
experiment's results independently and deliberately does not rank models or automatically
select a candidate. Do not use these same held-out test periods to choose a production model;
reserve a later, untouched evaluation period for any future confirmation.


## Fold-level stability and evidence interpretation

The analyzer reports each fold's model and baseline metrics, mean paired log-loss/Brier-score
improvements, and whether each fold favors the model or baseline. Aggregated comparisons also
include the count of positive, negative, and zero-improvement folds, plus the minimum, maximum,
and mean fold improvement.

Each 95% interval is labelled `positive_interval_excludes_zero`,
`negative_interval_excludes_zero`, or `interval_includes_zero`. If the interval includes zero,
the result is inconclusive under this descriptive bootstrap. These labels are diagnostics only;
they are not automatic model-selection gates or proof of statistical significance. Review
performance consistency across time windows alongside baseline comparisons and the known caveats
around temporal dependence and model-search bias.

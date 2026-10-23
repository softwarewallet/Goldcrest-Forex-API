# Volatility-regime diagnostics for held-out predictions

This stage is an offline, descriptive follow-up to the baseline and fold-stability reports. It
helps show whether conditional forecast quality differs in quieter versus more volatile market
periods. It does not choose a model, estimate trading profitability, or connect to execution.

## Run

Use the exact OHLC CSV supplied to the existing research pipeline. The command verifies its
SHA-256 against the pipeline manifest, verifies the experiment-matrix hash, and checks the stored
held-out prediction structure before scoring anything:

```bash
python -m app.research_regime_analysis \
  --manifest data/research/runs/<run-id>_pipeline_manifest.json \
  --ohlc-csv data/research/<exact-raw-ohlc-file>.csv \
  --output data/research/runs/<run-id>_regime_analysis.json \
  --volatility-window 20
```

This stage does not retrain any model. It works from the stored out-of-sample prediction records
and the original OHLC input.

## Regime definition

For each bar, volatility is the population standard deviation of the trailing window of
close-to-close log returns. The calculation is causal: it uses only returns available at that
bar's close. For each walk-forward test fold, the 25th and 75th percentile thresholds are
estimated only from historical volatility observations strictly before that fold's test start.
Values below the lower threshold are marked `low_volatility`; values above the upper threshold
are `high_volatility`; boundary ties and values between thresholds remain `mid_volatility`.

The output gives model and baseline accuracy, log loss, and Brier score inside each regime, with
paired differences shown descriptively. Empty regimes are emitted with a zero sample count rather
than being silently omitted. Fold threshold reports include the sample counts and calibration
history size so results can be audited.

## Interpretation and limits

Regime-specific sample sizes can be much smaller than the total held-out sample, particularly
when a regime appears only briefly. Treat all conditional metrics as exploratory diagnostics.
Do not choose the better-looking regime, feature set, model, probability threshold, or trading rule
using the same test windows. Preserve a later untouched evaluation period for any future
confirmation. These metrics are classification/probability metrics; they do not include spreads,
commissions, slippage, market impact, or execution risk.

The tool fails closed if the raw OHLC hash does not match the pipeline manifest, the experiment
matrix hash does not match its manifest, prediction validation fails, timestamps cannot be joined,
or the pre-test volatility history is insufficient. A propagated raw-data gap warning is not proof
of data corruption; it means market closures and coverage should be reviewed separately.

All output remains `actionable: false`, `automatic_model_selection: false`,
`test_metrics_used_for_selection: false`, and `execution_connected: false`.

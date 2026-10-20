# Temporal calibration stability audit

The report now describes fold-level expected calibration error (ECE) across
chronologically ordered validation and test periods, using the declared
`fold_temporal_boundaries` in the report manifest. It records each period's ECE,
available moving-block interval, first-to-last change, observed range, and
adjacent interval overlap counts.

At least three explicitly bounded periods are required for temporal coverage.
If first and last ECE intervals are disjoint, the audit labels the direction as
an endpoint separation observation; this is descriptive, not a significance
test. Missing intervals prevent endpoint-separation interpretation. Periods
without declared boundaries are not ordered by guesswork.

Cautions: ECE is binned and sample-size sensitive; interval estimates depend on
the bootstrap/block-length assumptions. Walk-forward folds may overlap, so
folds are not necessarily independent. This audit does not infer market regimes,
change model probabilities, calibrate or select models, tune thresholds, or
make profitability claims. Use genuine historical data with documented
provenance for empirical conclusions. All outputs remain offline,
research-only, and `actionable=false`.

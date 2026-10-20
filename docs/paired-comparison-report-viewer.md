# Paired model-comparison evidence in the report viewer

The offline experiment runner compares each candidate against the fold-local training-prevalence baseline using identical out-of-sample observation identities. Every paired-comparison record now carries the fold index and partition (`validation` or `test`) so the viewer can show exactly which window the comparison describes.

The local read-only dashboard displays candidate-minus-baseline Brier and log-loss differences with moving-block bootstrap intervals, sample count, fold, partition, and the recorded interpretation. Negative loss differences favor the candidate. An interval crossing zero is not clear evidence of a difference; metric disagreement is also possible.

## Interpretation rules

- The baseline probability is computed from matured labels in the training partition only.
- Validation comparisons are diagnostic and must not be used to repeatedly tune against final test windows.
- Test comparisons are per-fold; if test windows overlap, repeated timestamps are not independent observations.
- The moving-block bootstrap depends on the chosen block length and does not remove multiple-testing bias, nonstationarity, source-data defects, or external feature leakage.
- These are predictive probability comparisons, not cost-adjusted P&L or proof of profitability.
- The API remains research-only and non-actionable; no execution or GoldcrestForex Auto Live code is involved.


## Reproducibility manifest

Each experiment run now records a nested reproducibility manifest containing Python/platform metadata, versions of NumPy, scikit-learn, LightGBM and XGBoost when installed, model factory defaults/serializable closure configuration, model public configuration, and SHA-256 fingerprints of model implementation source when source is inspectable.

The runner fingerprints these details into the ledger run contract and report. It uses a preflight model instance for the first fold, then requires a fresh instance with the same model version and configuration in later folds. Reused instances or configuration drift fail closed. Non-serializable factory closures must be replaced with explicit JSON-compatible configuration rather than being silently ignored.

The report inspector verifies the aggregate reproducibility fingerprint and the nested factory/model configuration fingerprints. This detects accidental or inconsistent changes; an unsigned self-fingerprint is not proof against a deliberate editor who can recompute hashes. Environment snapshots also do not guarantee bitwise repeatability across hardware, native libraries, or parallel execution. The data-source provenance and causal-feature limitations remain unchanged.


## Semantic integrity checks

The read-only report inspector also validates the meaning and identity of each paired
comparison, not just the report's self-fingerprint. Candidate and baseline names must
resolve to models in the report; the fold and partition must exist for both; the
comparison's sample count must equal each side's recorded partition sample count; and
duplicate candidate/baseline/fold/partition comparisons are rejected. These checks catch
dangling references and internally inconsistent reports even if a report was edited and
its unsigned SHA-256 fingerprint recomputed.

These are structural consistency checks only. They do not independently recompute
metrics from the SQLite ledger, authenticate the input data, prove feature causality,
or establish profitability. The API remains offline and non-actionable.


## Exact paired-observation identity fingerprints

Each scored validation/test metric now records a SHA-256 fingerprint of the sorted UTC
observation timestamps used for that partition. The SQLite ledger independently records
the same fingerprint on each paired comparison after confirming that candidate and
baseline rows contain identical matured observation identities, labels, and label
maturity timestamps.

The report inspector checks that candidate and baseline metric fingerprints agree and,
when the paired comparison carries a fingerprint, that it matches the corresponding
fold/partition metrics. This detects reports that retain equal sample counts while
silently associating a comparison with a different observation set. Existing reports
without these optional fingerprints remain readable for backward compatibility; new
runner-generated reports include them.

A timestamp-set fingerprint does not prove that source data or features are correct,
that timestamps reflect real market availability, or that metrics are profitable. It
is an integrity check for pairing, not a market-data authenticity certificate.


## Fold-to-fold stability diagnostics

The report inspector now derives a descriptive stability summary for each candidate/baseline/partition pair represented in the paired-comparison records. It compares Brier score and log-loss separately within each common fold and reports the number of folds where the candidate is lower, tied, or higher, plus the median candidate-minus-baseline difference and the exact fold indices included.

Negative Brier/log-loss differences favor the candidate. The median is a robust descriptive center, not a pooled metric or a significance estimate. Only common folds with equal sample counts are compared; inconsistent counts fail closed. Missing metric values are omitted for that metric rather than imputed.

Fold counts are not independent sample counts. Repeated/overlapping test windows, small numbers of folds, regime changes, repeated model experimentation, and source-data limitations still constrain interpretation. These summaries intentionally calculate no p-values, do not correct multiple testing, do not select a model, and do not establish trading performance or profitability. The research API remains non-actionable.


## Validation-only model-selection audit

New experiment reports include a deterministic `model_selection_audit`. Candidate roles are derived from paired-comparison baseline metadata. Ranking uses the median of fold-level validation log-loss, then median validation Brier score, then candidate name as a deterministic tie-break. Only fold indices with finite validation log-loss and Brier metrics for every candidate are included. At least two complete common validation folds are required; otherwise selection is withheld.

The audit explicitly records that test metrics were not used for selection. The report inspector recomputes the audit from the report's validation metrics and rejects a mismatch, even when the outer report fingerprint has been recomputed. Legacy reports without this optional field remain readable. The dashboard displays the protocol, common-fold count, ranking, and selection-withheld status.

This is a transparent research protocol, not a guarantee against every form of adaptive overfitting. Repeated experiments, human decisions, feature leakage, overlapping windows, regime changes, and small validation samples remain risks. Final test partitions must not be used to tune or choose models; report inspection does not prove that a human followed that rule outside the recorded artifact. The selected name is not a model promotion or execution authorization, and all outputs remain non-actionable.


## Chronological out-of-sample period robustness audit

New reports include `oos_period_robustness_audit`, a descriptive post-selection diagnostic
based only on per-fold test metrics. It requires at least three common fold indices with
finite Brier and log-loss for all candidate models, then divides the chronologically
ordered fold indices into balanced EARLY, MIDDLE, and LATE groups. It reports per-period
median Brier/log-loss, worst-fold Brier/log-loss, and each candidate's full test-fold
metric range. It does not pool predictions or observations, rank models, choose a model,
or run a significance test. The report inspector recomputes the diagnostic and rejects
a mismatch even if the outer report fingerprint was recomputed.

The groups are based on fold index order supplied by the evaluation plan; they are not
market-regime labels and do not prove independent samples. If test windows overlap,
underlying observations may repeat across folds. This audit cannot establish robustness
to actual economic regimes, profitability, or generalization. Genuine historical OHLC
data and a predeclared evaluation protocol remain necessary; synthetic fixtures validate
software mechanics only. All outputs remain research-only and non-actionable.


## Fold uncertainty envelopes and declared regime stratification

New reports include regime_uncertainty_audit. It reports per-fold test metric minimum, first quartile, median, third quartile, maximum and range for Brier score, log-loss, fixed-threshold accuracy and ECE. These are descriptive fold-level dispersion summaries, not confidence intervals or estimates of sampling uncertainty.

Optional regime-stratified summaries require an explicit manifest.fold_regime_annotations object with source, method, a 64-character hexadecimal source_snapshot_sha256, and a folds mapping whose keys are non-negative fold-index strings. Each fold value must contain label, available_at_utc, valid_from_utc, and valid_through_utc. The runner records test_start_utc/test_end_utc for each fold from the actual walk-forward plan. The inspector requires the annotation to have been available by test start and its validity interval to cover the entire test window. Timestamps must explicitly use UTC (Z or +00:00). The system does not infer or fabricate market regimes.

Legacy string-only labels remain readable for audit continuity but are marked LEGACY_LABELS_WITHOUT_TEMPORAL_PROOF and never used to produce stratified metrics. Missing fold boundaries withhold stratification. These checks enforce internal temporal consistency and content-addressed source metadata; they do not authenticate that the external source is truthful or prove that the classification method itself is causal. The report inspector recomputes the audit from the manifest and fold metrics and rejects inconsistencies.

Only labels on common complete test folds are used; each candidate/regime group needs at least two folds or is withheld. Two folds is a minimum for descriptive output, not evidence of statistical power. Overlapping test windows may reuse observations; no pooling, significance testing, confidence interval, model selection, profitability inference or execution authorization is produced.

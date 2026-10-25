# Multi-threshold label-eligibility sensitivity

## Purpose

The completed one-hour cap sweep showed consistent degradation in LightGBM probability metrics and mostly adverse log-loss/Brier changes in XGBoost across five matched optimizer seeds. This runner compares predefined alternative thresholds to test whether the result depends on excluding every label that takes longer than the nominal hour or only on excluding the longer elapsed-time tail.

Default candidate caps are 1 hour, 1.5 hours, and 8 hours. One hour is the nominal four-bar duration for the M15 four-bar horizon; 1.5 hours is close to the previously observed p99 elapsed duration; 8 hours is the existing long-gap audit boundary. These are diagnostic thresholds, not assumptions about a broker session calendar.

## Reuse the existing one-hour sweep

Run from the repository root with the virtual environment active:

    python -m app.research_label_eligibility_threshold_sweep \
      --baseline-manifest data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_pipeline_manifest.json \
      --reference-seed-sweep-report data/research/runs/eurusd-m15-demo-m365-baseline-validation-2026102-seedcap-1h_seed_sweep_report.json \
      --reference-cap-hours 1.0 \
      --caps 1.0,1.5,8.0 \
      --seeds 20261021,20261022,20261023,20261024,20261025

The reference report is reused only after checking the cap, seed list, research-only flags, source hash, fold signature, and model/feature matrix. The 1.5-hour and 8-hour caps each run fresh five-seed matched pairs. Existing artifacts are preserved. If your local reference report has a different path, pass the exact path to it.

A new threshold report is written with collision-safe naming, and progress is saved after each threshold. The report compares per-model, per-metric seed deltas and checks that source data, chronological fold boundaries, and experiment identities match across thresholds. Eligible-row counts are retained in each cap's fold signature and may differ by design; they are excluded only from the cross-cap boundary comparison.

## Audit a completed sweep without retraining

After the sweep completes, the saved threshold report can be audited against its existing seed-sweep and pipeline manifests. This checks dataset-level eligible/ineligible row counts for each cap, verifies that each seed used the recorded cap and same raw OHLC hash, compares fold-level eligible test counts, and flags threshold pairs with identical eligibility profiles. It does not retrain models or overwrite the original report.

    python -m app.research_label_eligibility_threshold_report_audit \
      --threshold-report data/research/runs/eurusd-m15-demo-m365-bas-cap-sweep_threshold_sweep_report.json

The audit writes a new collision-safe report next to the threshold report. A pair of nested caps with identical eligible/ineligible counts on the same dataset has an unchanged eligibility profile; identical metrics across those caps are then expected rather than evidence that one cap is better.

## Profile exact elapsed-time durations around the caps

The profile audit independently recomputes each row's elapsed duration from `timestamp` to `label_available_at` in the immutable uncapped baseline dataset. It verifies the source dataset and raw OHLC hashes, reports duration quantiles and non-overlapping buckets between the audited caps, and cross-checks the computed eligible counts with the existing eligibility audit. It does not retrain models or rewrite the dataset.

    python -m app.research_label_eligibility_duration_profile \
      --baseline-manifest data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_pipeline_manifest.json \
      --threshold-audit-report data/research/runs/eurusd-m15-demo-m365-bas-cap-sweep_threshold_eligibility_audit_report.json

The output is a new collision-safe report beside the baseline manifest. If the eligibility counts at 1.5 and 8 hours match, inspect the `(1.5, 8]` duration bucket: a zero count confirms no labels mature in that interval for this dataset; labels excluded by the 8-hour cap are all beyond 8 hours. The profile is dataset-specific and does not establish a market-session rule.

## Attribute duration extensions to observed OHLC gaps

After generating the duration profile, run this follow-up audit. It checks every labelled row against the immutable raw timeline, confirms that `label_available_at` is exactly the configured row-index horizon after `timestamp`, and reconciles elapsed time above the nominal horizon with the sum of observed inter-bar intervals above one nominal timeframe. The report groups those findings by the same elapsed-duration buckets and provides a small set of the largest gap-exposure examples.

    python -m app.research_label_duration_gap_attribution \
      --baseline-manifest data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_pipeline_manifest.json \
      --duration-profile-report data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_label_duration_cap_boundary_profile.json

Use the exact duration-profile path printed by the previous command if collision-safe naming produced a rerun filename. The follow-up checks both source hashes and refuses to overwrite existing input artifacts. A `FAIL` indicates a hash, timestamp alignment, row-index horizon, or elapsed-time decomposition inconsistency that must be investigated before drawing conclusions. A successful reconciliation is arithmetic evidence that elapsed extensions coincide with observed timestamp spacing; it does not prove that an interval represents missing candles, a broken provider feed, or an expected open-market session. No labels are changed and no models are retrained.

## Interpretation

All caps reuse the same historical OHLC snapshot and chronological test windows. Repeated seeds measure optimizer/retraining sensitivity; they are not independent market observations. The sweep makes no inferential or profitability claims, does not select the best threshold or model, does not promote a model, and does not connect to broker/execution services. A threshold should not be preferred merely because it performs best after trying multiple values; follow-up decisions require a separately prespecified protocol.


## Profile individual gap events and affected label windows

After the duration-gap attribution audit reconciles elapsed-time excess, run this event-level profile. It enumerates each non-nominal interval in the raw OHLC timeline and maps it to the dataset rows whose configured forward-label window crosses that interval. It reports unique event counts, affected labels per event, overlaps where a label window crosses multiple gaps, duration-cap bucket counts, and descriptive weekend-like/fixed-date context flags for review.

    python -m app.research_label_gap_event_profile \
      --baseline-manifest data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_pipeline_manifest.json \
      --duration-gap-attribution-report data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_label_duration_gap_attribution.json

Use the exact report path printed by the previous duration-gap attribution command if collision-safe naming produced a rerun filename. The event profile checks both input hashes, the forward-bar horizon, and exposure counts against the prior audit. Its classifications are descriptive only and do not prove missing candles, provider failure, or an expected session closure. No dataset is rewritten and no model is trained.


## Verify gap-policy equivalence without retraining

The event-profile audit found 259 non-nominal raw timestamp intervals: 205 shorter-gap events and 54 long-gap events. Each event was associated with four label windows, for 1,036 unique labels crossing at least one gap. No label window crossed multiple events. The follow-up audit compares row-level masks—not only aggregate counts—between elapsed-time caps and gap-driven exclusion policies.

    python -m app.research_label_gap_policy_equivalence_audit \
      --baseline-manifest data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_pipeline_manifest.json \
      --duration-profile-report data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_label_duration_cap_boundary_profile.json \
      --duration-gap-attribution-report data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_label_duration_gap_attribution.json \
      --gap-event-profile-report data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_label_gap_event_profile.json \
      --threshold-audit-report data/research/runs/eurusd-m15-demo-m365-bas-cap-sweep_threshold_eligibility_audit_report.json

Use the exact report filenames printed by previous runs if collision-safe rerun names were created. The audit independently reconstructs gap-to-label associations and verifies source hashes, duration buckets, and existing threshold-audit counts. It compares the complete eligible row masks for: retaining all labels; applying 1-hour, 1.5-hour, or 8-hour caps; excluding labels crossing any non-nominal raw interval; and excluding labels crossing raw intervals longer than eight hours.

Exact mask equivalence means those policies select the same rows for this particular snapshot and label definition. It does not show that they have equal predictive performance or that a gap is missing data. No training, label rewrite, threshold selection, or execution connection occurs.


## Review existing model metrics by unique eligibility mask

After the gap-policy equivalence audit passes, this report-only review reuses the completed threshold sweep rather than retraining. It groups caps by the exact eligible-row mask, checks the threshold audit and sweep hashes, verifies repeated-seed results, confirms whether caps with identical masks also produced identical metrics, and compares the two distinct capped masks.

    python -m app.research_label_gap_policy_performance_review \
      --gap-policy-equivalence-report data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_gap_policy_equivalence_audit.json \
      --threshold-audit-report data/research/runs/eurusd-m15-demo-m365-bas-cap-sweep_threshold_eligibility_audit_report.json

Use the exact equivalence-audit report path printed by the previous command if collision-safe naming produced a rerun filename. The review reports existing per-model/per-feature metric summaries and per-seed paired deltas, and checks whether the uncapped reference metrics match between cap runs. If they do not match, differences between cap-level deltas may include reference-run variability and should not be interpreted as a clean incremental policy effect.

Identical row masks are grouped as one policy for this dataset; therefore the 1.5-hour and 8-hour caps are not treated as independent candidates when their masks are identical. Repeated seeds use the same historical period and overlapping walk-forward windows, so they are optimizer-sensitivity checks rather than independent market replications. The report makes no model selection, profitability, or live-execution decision.


## Performance-review input integrity

The downstream gap-policy performance review validates the diagnostic schema of every passing
input and seed result, rejects contradictory passing reports that still contain errors, checks that
eligibility row counts reconcile arithmetically, verifies eligible fractions against those counts,
and requires row-mask fingerprints to be hexadecimal SHA-256 strings. These are report-integrity
checks only: they do not retrain models, choose a cap, or establish predictive superiority.


## Threshold-audit diagnostic schema checks

The threshold eligibility report auditor now validates optional `errors` and `warnings` fields
on the source sweep, cap records, seed-sweep reports, and per-seed records. Diagnostic fields must
be lists of non-empty strings; reports marked passing cannot carry unresolved errors. Malformed
inputs become explicit audit failures instead of being silently converted (for example, treating a
warning string as a list of individual characters). No model is retrained or selected by these checks.


## Common-cohort report output safety

The common held-out cohort audit refuses an output path that resolves to any supplied comparison-report input, and it never replaces an existing report. It stages JSON in the destination directory and publishes with a no-clobber filesystem link; if another process creates the destination during publication, the audit fails and preserves that file. Choose a fresh output path for each audit run. This protects the original comparison reports and keeps rerun outputs distinct.


## Common-cohort temporal integrity checks

Before intersecting held-out predictions, the common-cohort audit parses fold boundaries and prediction timestamps as timezone-aware UTC values, rejects reversed windows and timestamps outside the declared fold window, and detects duplicate instants even when two strings encode the same UTC time. It canonicalizes accepted timestamps for cross-report matching while validating each source report's timestamp fingerprint against its original serialized timestamps. Missing model or feature-variant identities are rejected rather than coerced to placeholder strings. These checks protect the evaluation cohort's temporal identity; they do not prove that model training itself was leakage-free.

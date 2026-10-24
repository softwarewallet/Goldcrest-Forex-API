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


## Audit equivalence of elapsed-time caps and raw-gap policies

After the duration profile, duration-gap attribution, and event-level gap profile complete, use the combined audit to compare exact row-level masks rather than comparing cap counts alone. It recomputes gap events from the hashed raw OHLC timeline, checks each event's affected label rows, validates the prior report counts, and compares the eligibility masks for the following policies:

- Keep every label.
- Exclude labels with elapsed duration above 1 hour.
- Exclude labels whose forward window crosses any observed non-nominal timestamp interval.
- Exclude labels above 1.5 hours or 8 hours.
- Exclude labels whose forward window crosses an observed timestamp interval longer than 8 hours.

    python -m app.research_label_gap_policy_equivalence_audit \
      --baseline-manifest data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_pipeline_manifest.json \
      --duration-profile-report data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_label_duration_cap_boundary_profile.json \
      --duration-gap-attribution-report data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_label_duration_gap_attribution.json \
      --gap-event-profile-report data/research/runs/eurusd-m15-demo-m365-baseline-validation-20261023_label_gap_event_profile.json \
      --threshold-audit-report data/research/runs/eurusd-m15-demo-m365-bas-cap-sweep_threshold_eligibility_audit_report.json

Use the exact report paths emitted by the preceding commands if collision-safe rerun naming generated different names. The audit is immutable and report-only. Equality is established using row-identity hashes and symmetric differences, not just aggregate eligible counts. It does not evaluate model quality or decide which policy to adopt. An exact mask equality means the corresponding policies are redundant on this raw snapshot and label definition; it is not a general statement about other dates, instruments, timeframes, or data providers.

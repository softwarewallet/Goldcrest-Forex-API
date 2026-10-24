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

## Interpretation

All caps reuse the same historical OHLC snapshot and chronological test windows. Repeated seeds measure optimizer/retraining sensitivity; they are not independent market observations. The sweep makes no inferential or profitability claims, does not select the best threshold or model, does not promote a model, and does not connect to broker/execution services. A threshold should not be preferred merely because it performs best after trying multiple values; follow-up decisions require a separately prespecified protocol.

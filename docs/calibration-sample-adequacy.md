# Calibration sample adequacy audit

The offline experiment report includes a per-model, per-fold, per-partition screen for calibration sample adequacy. It records sample count, explicit positive/negative counts and prevalence where internally consistent, plus populated and sparse reliability-bin counts where available.

The fixed review flags are fewer than 30 observations, fewer than 10 observations in the minority class, and fewer than 5 observations in a populated reliability bin. Missing class counts are never inferred. Empty reliability bins are not counted as populated.

These flags are descriptive review aids, not validated universal statistical thresholds. Passing them does not establish calibration or model quality. ECE depends on binning choices; this milestone does not claim bin-choice robustness because raw per-observation probabilities are not retained in the report contract. No data are filtered, no calibration thresholds are tuned, and no model is selected or promoted. All results remain actionable=false and OFFLINE_RESEARCH_ONLY.

The report inspector independently recomputes the audit from fold metrics and rejects an inconsistent embedded result. Synthetic fixtures test software behavior only, not market performance.

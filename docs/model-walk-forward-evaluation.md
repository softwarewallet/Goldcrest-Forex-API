# Model Walk-Forward Evaluation

This module adds a model-agnostic offline evaluator for binary research classifiers. It accepts precomputed, explicitly specified feature rows and labels, creates a fresh model per existing walk-forward fold, and reports validation/test Brier score, log loss, fixed-0.5 accuracy, prevalence, mean predicted probability, expected calibration error (ECE), and equal-width reliability bins for each partition.

## Point-in-time safeguards

- Feature timestamps must be timezone-aware, unique, and strictly chronological.
- Feature schemas and finite numeric values are validated.
- Every label includes `label_available_at`; a training row is used only if its label had matured by the final timestamp in that fold's training partition. Unmatured training labels are counted and excluded.
- A model is freshly instantiated and fit separately for every fold. Validation/test records are only scored, never passed to `fit`.
- Training requires both binary classes. Probabilities are validated and are explicitly research-only and non-actionable.

## Important limitations

The evaluator cannot prove that externally supplied features were genuinely available at their timestamps. Feature generation must be independently audited for look-ahead leakage, and target/label generation must follow the approved protocol. Validation/test scores are not automatically combined into profitability claims. Brier score/log loss measure probabilistic prediction quality, not executable returns. No probability-calibration transformation, threshold tuning, feature engineering, model promotion, API serving, broker integration, or trading is performed here. ECE and reliability bins are diagnostics only: they depend on binning and sample size and do not prove calibration or generalization.

Synthetic fixtures test mechanics only. Use only after the real data source, features, labels, horizon, costs, baselines, and untouched test protocol have been approved. GoldcrestAI remains independent of GoldcrestForex and live serving remains fail-closed with `NO_TRADE`.

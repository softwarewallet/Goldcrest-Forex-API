# Multi-model research stack

## Current components

- **XGBoost** and **LightGBM** are independent tabular binary-classifier candidates. Use
  `build_tabular_model_factories(feature_names)` with the existing offline walk-forward
  runner to compare them on the same chronological folds and feature schema.
- **Llama/Qwen** are represented by one configured LLM gateway result. The gateway's
  returned model ID is recorded for provenance. Qwen should be orchestrated behind this
  same gateway if the chosen serving runtime supports that workflow; this application
  does not create a second Qwen endpoint or assume that a plain llama.cpp model can call
  another model on its own.
- **Research comparison** reports each tabular candidate's fixed-threshold direction
  beside the LLM direction. It does not average probabilities, infer calibrated
  confidence, automatically select a winner, or authorize trading.

## Important configuration contract

1. Pin the feature names and their order in the dataset/run manifest.
2. Explicitly declare whether label class 1 means UP or DOWN.
3. Use mature labels, point-in-time features, dataset provenance, and purged chronological
   walk-forward partitions already required by the offline evaluation pipeline.
4. Compare XGBoost and LightGBM against the training-prevalence baseline and retain
   validation and test metrics separately. Do not tune thresholds on the final test set.
5. Keep LLM outputs as an independent research observation. If a Llama gateway internally
   orchestrates Qwen, identify that orchestration in gateway-side configuration and logs.
   If the gateway cannot orchestrate Qwen, do not claim Qwen participated in the result.
6. Treat all outputs as `actionable=false`. A model's inclusion is not evidence of
   improved accuracy. Promote no model automatically; require reproducible out-of-sample
   evidence, cost-aware evaluation, calibration diagnostics, and explicit review.

## Boundary

This is an offline/research utility. It does not change `/api/v1/predictions`, the
current fail-closed `NO_TRADE` policy, broker integration, GoldcrestForex, Auto Live,
position sizing, or execution/risk gates. No training occurs in API request handlers.

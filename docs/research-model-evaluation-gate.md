# Offline model evaluation integrity gate

The evaluation gate checks a persisted experiment-matrix JSON without retraining models or modifying the source report.

## Run

```bash
python -m app.research_model_evaluation_gate \
  --experiment-matrix data/research/runs/<run-id>_experiment_matrix.json \
  --output data/research/runs/<run-id>_model_evaluation_gate.json \
  --json
```

The output path must not already exist. Exit code `0` means the checked report passed; exit code `3` means it failed. The source matrix is read-only.

## Checks

- Explicit non-actionable and no-test-selection/no-test-tuning flags.
- Unique experiment IDs and unique fold indices.
- Declared fold count agrees with each experiment's recorded folds.
- Complete timezone-aware train, validation, and test boundaries, with each window's start preceding its end.
- Finite metric values and expected ranges for sample counts, prevalence, Brier score, log loss, fixed-threshold accuracy, mean probability, and expected calibration error.
- Positive and negative counts sum to sample count; prevalence agrees with positive-count/sample-count within a small floating-point tolerance.

## Interpretation and limits

A `PASS` means the listed structure and arithmetic checks are internally consistent. It does **not** prove that the features are point-in-time correct, that all data leakage is absent, that model results are statistically significant, or that any strategy is profitable. The gate does not compare models, select a winner, fit a calibrator, alter persisted artifacts, connect to a broker, or authorize execution. Treat it as one independent integrity layer alongside provenance, temporal calibration, and research-run audits.

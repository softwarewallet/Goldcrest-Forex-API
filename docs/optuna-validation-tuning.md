# Optuna validation-only hyperparameter tuning

Goldcrest-Forex-API now provides an optional offline tuner for XGBoost and
LightGBM. It deliberately accepts only a training partition and a later validation
partition; it has no test-data argument. The final test partition must remain untouched
until features, label semantics, model family, and hyperparameters have been frozen.

## Example

Call the function from a research script using the same point-in-time feature and
matured-label records used by the walk-forward pipeline:

```python
from app.historical_tabular_research import load_historical_dataset
from app.optuna_validation_tuning import tune_on_validation, write_tuning_report

# Load the same audited CSV and choose the exact chronological slices using your
# frozen research protocol. Do not pass the final test partition to this function.
_, examples, _ = load_historical_dataset(
    "data/historical/EURUSD_M15_labeled_features.csv",
    ("return_1", "return_3", "rsi_14", "atr_14_pct"),
)
train_examples = examples[:2000]
validation_examples = examples[2200:2600]  # respect the protocol's purge/gap

result = tune_on_validation(
    train_examples,
    validation_examples,
    feature_names=("return_1", "return_3", "rsi_14", "atr_14_pct"),
    model_type="xgboost",  # or "lightgbm"
    n_trials=25,
    seed=20261021,
    study_name="eurusd-m15-xgb-validation-v1",
)
write_tuning_report(result, "reports/eurusd-m15-xgb-tuning.json")
print(result.to_dict())
```

The example indices are illustrative only; use the actual frozen fold plan and its
purge boundaries rather than copying arbitrary slice positions.

## Safeguards

- Search space is deliberately bounded; trial count is restricted to 1–100.
- The final training rows are filtered by `label_available_at`; outcomes that had not
  matured by the training cutoff are excluded.
- Training and validation timestamps must be strictly ordered and disjoint.
- Both classes must exist in matured training data and validation data.
- Trials are sequential and use the resource-aware CPU thread limit.
- The fixed selection objective is validation log-loss. The result records the seed,
  feature schema fingerprint, trial counts, best parameters, validation window, and
  number of unmatured training labels excluded.
- The report is explicitly `actionable=false` and `test_partition_used=false`.
- Tuning is not a trading strategy, a profitability estimate, or permission to promote
  a model. Re-evaluate frozen choices on untouched, later test folds with the existing
  purged walk-forward evaluation and retain the existing `NO_TRADE` fail-closed policy.

Optuna is used for the parameter search only. PyTorch and TA-Lib are not automatically
loaded by this utility: introduce additional frameworks or indicators only when a
separate experiment has a clear hypothesis, provenance, and tests. Qwen stays behind
the existing configured LLM gateway; this tuner does not create a second model
connection or alter any prediction API or broker/execution code.

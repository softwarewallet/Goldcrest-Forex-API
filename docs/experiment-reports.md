# Reproducible experiment reports

`app/experiment_report.py` assembles existing offline research artifacts into a
deterministic JSON report. It does not fetch data, train models, choose a winner, or
change prediction serving.

## Contents

- Registered run manifest and run ID.
- Per-model, per-fold validation and test metrics, with fold order normalized.
- Identity-paired model comparisons already computed from the prediction ledger.
- Explicit exclusion counts, evaluation-contract metadata, and warnings.
- A SHA-256 fingerprint over the canonical report payload.
- Explicit research-only interpretation and `actionable: false`.

## Example

```python
from app.experiment_report import build_experiment_report, write_experiment_report

report = build_experiment_report(
    manifest=registered_manifest,
    fold_results=walk_forward_results_by_model,
    paired_comparisons=ledger_comparisons,
    exclusions={"unlabeled": 12, "invalid": 2},
    evaluation_contract={
        "dataset_id": "provider/version",
        "timeframe": "M5",
        "target_definition": "supply the approved target definition",
        "horizon_bars": 3,
        "cost_assumptions": "explicitly document spread/slippage/commission",
    },
    warnings=["Review block-length sensitivity before interpreting intervals."],
)
write_experiment_report(report, "reports/<run-id>.json")
```

The example values are placeholders, not a prescribed target, horizon, data source, or
cost model. The caller must supply actual choices and evidence; this module does not
invent missing evaluation assumptions.

## Interpretation safeguards

- Fold metrics remain separate. Do not pool overlapping test folds into a larger sample
  or treat repeated observations as independent.
- Use validation metrics for development. Final test metrics are for a frozen evaluation,
  not repeated model/feature selection.
- A report hash establishes payload integrity/reproducibility, not correctness of source
  data, feature timing, target labels, or market assumptions.
- Confidence intervals depend on sample size, dependence assumptions, and block length.
- No report constitutes evidence of profitability or authorization for live trading.

All report outputs are offline research artifacts and remain non-actionable.

# Walk-Forward Fold Orchestration

**Status:** Offline fold generation only. No strategy scoring, model training, serving changes, or promotion.

The `app.walk_forward.generate_walk_forward_folds` helper creates expanding-window folds for one instrument/timeframe. Each fold has train, validation, and forward test partitions. Input is not sorted or silently cleaned; duplicate, non-monotonic, invalid, or mixed-instrument data fails closed.

## Required explicit inputs

- Initial training bars
- Validation bars per fold
- Test bars per fold
- Step size between fold origins
- Outcome horizon in bars (used as the purge width)

Partition sizes refer to the pre-purge index ranges. The helper purges `horizon_bars` from the end of train and validation and from the start of validation and test, so returned partition sizes are smaller than requested. Train windows expand as the fold index advances. Test windows are later than the corresponding train/validation windows.

## Interpretation and limitations

- Folds are infrastructure, not evidence of model skill.
- A candidate must be frozen before its test partition is scored; test results must not be used to tune that fold's candidate.
- The helper does not select target/stop values, transaction costs, data sources, instruments, or horizons.
- Overlapping test periods can occur when `step_bars < test_bars`; downstream aggregation must account for repeated observations and time dependence.
- The historical evaluation protocol's unresolved assumptions must be approved before any real-data scored run.
- GoldcrestAI remains `NO_TRADE`; no automatic promotion or GoldcrestForex integration is introduced.

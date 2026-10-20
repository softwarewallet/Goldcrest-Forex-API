# Walk-forward temporal integrity audit

The historical tabular research CLI embeds `temporal_integrity_audit` in its run manifest. It records every fold's UTC train/validation/test boundaries, partition row counts, purged rows, and how many training labels were excluded because their outcomes had not matured by the training cutoff.

## Repeated test timestamps

When the expanding-window step is smaller than the test window, a timestamp can appear in more than one fold's test partition. The audit reports total test observations, unique test timestamps, repeated observations, and a bounded list of overlapping UTC timestamps. It does not silently deduplicate fold results. Per-fold metrics are descriptive; do not pool repeated observations as independent samples.

## Integrity boundaries

The audit rejects duplicate fold indices, missing example timestamps, partition timestamp overlap, and non-chronological partitions. It independently recomputes training-label maturity counts using the same rule as model evaluation. The audit is diagnostic: it does not establish that supplied features were genuinely available at their timestamps, authenticate the provider export, validate legal data rights, remove model-selection bias, or prove profitability. Those require external evidence and additional research.

All output remains `OFFLINE_RESEARCH_ONLY` and `actionable=false`. No prediction-serving, broker, order execution, or GoldcrestForex code is changed.

# Research operations CLI

The CLI provides operator-friendly, read-only inspection of offline research artifacts.
It does not run experiments, fetch market data, fit models, change reports, or authorize
trading.

## Audit a report

From the repository root, with the project environment active:

```bash
python -m app.research_cli audit \
  --report reports/<run-id>.json \
  --ledger artifacts/research.sqlite
```

The command emits a JSON summary. A verified artifact returns exit code `0`; a report
or ledger consistency failure returns `1`; missing input files return `2`. The audit
fails closed on report hash, completed-run binding, attempt history, model/fold/partition
counts, or actionability mismatches.

## Inspect a run and its attempts

```bash
python -m app.research_cli status \
  --run-id <run-id> \
  --ledger artifacts/research.sqlite
```

This command reports the durable run status, manifest fingerprint, expected prediction
row count, stored report digest/error, and append-only attempt history. It does not recover,
retry, mark interrupted runs, or mutate the database. Exit code `0` means the run was
found; `2` means the ledger or run was not found; `1` indicates an inspection error.
All JSON output explicitly reports `actionable: false` for status inspection.

## Interpretation and limitations

The audit proves internal consistency between report and ledger, not the correctness of
market data, labels, features, assumptions, or model behavior. Status is a lifecycle
snapshot, not a model-quality assessment. Neither command establishes calibration,
profitability, or generalization, and neither permits live trading.

Preserve source-data hashes, evaluation assumptions, model/code revisions, and package
versions alongside reports. All outputs remain offline research artifacts with
`actionable=false`.

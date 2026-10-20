# Research artifact audit CLI

The CLI provides an operator-friendly wrapper around the existing offline report/ledger
consistency check. It does not run an experiment, fetch data, fit a model, change a report,
or authorize trading.

## Usage

From the repository root, with the project environment active:

```bash
python -m app.research_cli audit \
  --report reports/<run-id>.json \
  --ledger artifacts/research.sqlite
```

The command emits a JSON summary. A verified artifact returns exit code `0`; a report
or ledger consistency failure returns exit code `1`; missing input files return exit
code `2`. This makes it suitable for a manual checklist or an external CI step.

The result includes run ID, report digest, expected and actual prediction row counts,
actionable-row count, and completed ledger status. The CLI fails closed on a report hash,
run binding, attempt history, model/fold/partition count, or actionability mismatch.

## Important limitation

The command verifies internal report/ledger consistency only. It does not authenticate
the underlying market data, prove that labels or features were correct, establish model
calibration or profitability, or permit live trading. Preserve source-data hashes,
evaluation assumptions, model/code revisions, and package versions alongside the report.
All outputs remain offline research artifacts with `actionable=false`.

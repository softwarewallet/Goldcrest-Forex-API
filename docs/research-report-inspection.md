# Offline research report inspection

Inspect a generated report without fitting a model or changing its ledger:

```bash
python -m app.research_cli inspect --report reports/<run-id>.json
```

The command validates the report's canonical SHA-256 fingerprint, schema version,
research-only scope, non-actionable status, fold identities, and separate validation/test
metric objects. It emits a JSON summary sorted by model and fold. Exclusions, paired
comparisons, warnings, and the evaluation contract are preserved for operator review.

Exit codes: `0` means the report passed integrity and structural checks; `1` means the
report is invalid or could not be inspected; `2` means the report file is missing.

## Interpretation safeguards

- Validation and test results remain separate for each fold. No pooled test score is
  calculated because test windows may overlap.
- The fingerprint detects accidental or deliberate changes after report generation; it
  does not authenticate market data or prove that the original inputs were correct.
- A structurally valid report does not establish model calibration, profitability, or
  generalization.
- All summaries remain `actionable=false`; this command cannot authorize live trading.

For ledger-to-report reconciliation, also run the `audit` command documented in
`research-artifact-audit-cli.md`. Use `status` to inspect run lifecycle and attempt history.

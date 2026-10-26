# Label-eligibility cross-report integrity audit

Use this audit after generating at least two paired label-eligibility comparison reports for different elapsed-time caps. It records the exact SHA-256 of each input report, along with its baseline/variant run identifiers, cap, market identity, horizon, and raw/dataset provenance hashes. It then runs the existing common-held-out-cohort consistency checks and fails closed if any report is invalid or incompatible.

Example:

```bash
python -m app.research_label_eligibility_cross_report_integrity \
  --comparison-report data/research/runs/<cap-1-comparison>.json \
  --comparison-report data/research/runs/<cap-8-comparison>.json \
  --output data/research/runs/<new-cross-report-integrity-audit>.json
```

For deliberately distinct baseline run IDs, the optional `--allow-baseline-id-variation-with-matching-dataset-hash` flag allows comparison only when the common-cohort auditor confirms matching baseline dataset hashes. This does not establish identical training configurations or training samples.

The audit snapshots each input's bytes and verifies that the same input identity/hash remains present after the underlying cohort audit. If an input changes while the audit is running, the result fails. The output refuses to overwrite an existing file or an input report.

This is an integrity/provenance diagnostic, not a model-selection gate or profitability claim. Outputs preserve `automatic_model_selection=false`, `actionable=false`, and `execution_connected=false`. Identical predictions or metrics remain diagnostic observations and are not automatically classified as defects.

Input comparison reports are parsed with strict JSON object handling. Duplicate keys are rejected so provenance fields cannot be shadowed by later duplicate entries.

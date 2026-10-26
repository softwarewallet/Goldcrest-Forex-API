# Research run integrity audit

After a research pipeline run, verify that the manifest and its five core artifacts
still match their recorded SHA-256 hashes and that their run IDs, dataset hashes, and audit
receipts agree. Newer runs also include a raw OHLC audit receipt and temporal calibration
stability receipt; each optional receipt is verified whenever its path/hash is present in the
manifest. Newer runs also produce a separate pipeline-completion receipt, which is checked
against the manifest's run ID and self-hash when present. Older schema-v1 manifests and runs
without a completion receipt remain readable.

From the repository root, with the project virtual environment active:

```bash
python -m app.research_run_audit data/research/runs/<run-id>_pipeline_manifest.json
python -m app.research_run_audit data/research/runs/<run-id>_pipeline_manifest.json --json
```

The command checks the raw OHLC snapshot, prepared dataset, dataset manifest, experiment
matrix report, and provenance audit receipt. For newer runs, it also verifies the raw OHLC
audit and temporal calibration stability receipts against their recorded hashes. The temporal
receipt must bind to the exact experiment-matrix SHA-256, have a verified embedded matrix
payload hash, retain non-actionable/no-execution safeguards, and agree with the manifest status.
It also verifies the pipeline-manifest fingerprint. Exit code 0 means the cross-artifact checks
passed (possibly with warnings); exit code 3 means the audit failed. Invalid or missing files
are reported as findings rather than exposing tracebacks.

This is a consistency audit, not a model-quality certification. It does not recompute
metrics, prove that feature engineering is leakage-free, verify provider data rights, or
establish profitability. It never contacts cTrader, llama.cpp, or a broker and never changes
research outputs. Run it against a completed run directory before sharing its reports.

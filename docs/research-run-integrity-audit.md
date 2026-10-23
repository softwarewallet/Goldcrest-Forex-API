# Research run integrity audit

After a research pipeline run, verify that the manifest and all five referenced artifacts
still match their recorded SHA-256 hashes and that their run IDs, dataset hashes, and audit
receipt agree.

From the repository root, with the project virtual environment active:

```bash
python -m app.research_run_audit data/research/runs/<run-id>_pipeline_manifest.json
python -m app.research_run_audit data/research/runs/<run-id>_pipeline_manifest.json --json
```

The command checks the raw OHLC snapshot, prepared dataset, dataset manifest, experiment
matrix report, and provenance audit receipt. It also verifies the pipeline-manifest
fingerprint and the recorded non-actionable / no-execution safeguards. Exit code 0 means
the cross-artifact checks passed (possibly with warnings); exit code 3 means the audit
failed. Invalid or missing files are reported as findings rather than exposing tracebacks.

This is a consistency audit, not a model-quality certification. It does not recompute
metrics, prove that feature engineering is leakage-free, verify provider data rights, or
establish profitability. It never contacts cTrader, llama.cpp, or a broker and never changes
research outputs. Run it against a completed run directory before sharing its reports.

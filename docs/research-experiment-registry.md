# Cross-run research experiment registry

The cross-run registry validates a set of completed offline research runs before they are
combined into a comparative evidence pack. It requires at least two runs, clean individual
artifact-integrity audits, clean required audit statuses, a common research protocol, matching
raw-data and prepared-dataset fingerprints, and distinct random seeds.

## Run the registry

Supply each run manifest explicitly and choose a new output path:

```bash
python -m app.research_experiment_registry \
  --manifest data/research/runs/<run-a>_pipeline_manifest.json \
  --manifest data/research/runs/<run-b>_pipeline_manifest.json \
  --output data/research/evidence/<registry-id>_experiment_registry.json
```

The report is suitable for the `experiment_registry` role in
`app.research_evidence_pack_audit`. The registry report includes `runs_supplied`,
`runs_validated`, per-run status/errors, reconciled dataset fingerprints, and explicit
non-actionable safety flags. The output is created atomically without replacing an existing
file. Exit code 0 means every run passed; exit code 1 means a completed registry found
qualification failures; exit code 2 means invalid input or unsafe output handling.

## Protocol invariants

Runs must agree on instrument, timeframe, data source/version, provider-terms reference hash,
raw OHLC and prepared-dataset hashes, feature schema/implementation fingerprints, label
definition, and the walk-forward/tuning/ablation configuration. Random seeds must be valid and
distinct. Any mismatch fails the registry rather than silently pooling non-comparable runs.

Each run must have clean PASS statuses for the required manifest-level audits, a clean
reproducibility receipt, a clean completion receipt, explicit false execution/selection flags,
and a clean cross-artifact integrity audit. A run with unresolved warnings is not accepted as a
validated registry member.

## Interpretation limits

This registry verifies recorded protocol and artifact consistency; it does not recompute
training metrics, establish model superiority, estimate profitability, establish statistical
significance, prove absence of all leakage, or authorize deployment. The registry does not
modify the source runs or choose a model. If runs use different datasets or protocols, create
separate registries rather than relaxing the consistency checks.

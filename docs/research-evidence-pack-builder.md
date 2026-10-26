# Research evidence-pack assembly

Use the evidence-pack builder to calculate report hashes, create a confined manifest, and run the evidence-pack audit in one command. It reads supplied reports only; it does not rerun upstream audits or modify the reports.

## Example

Run from the repository root:

    python -m app.research_evidence_pack_builder \
      --pack-id eurusd-m15-reviewed-batch-20261026 \
      --report experiment_registry=data/research/evidence/eurusd_registry.json \
      --report validation_workflow=data/research/evidence/eurusd_validation_workflow.json \
      --report release_readiness_run_a=data/research/runs/<run-a>_release_readiness.json \
      --report release_readiness_run_b=data/research/runs/<run-b>_release_readiness.json \
      --required-role release_readiness_run_a \
      --required-role release_readiness_run_b \
      --output data/research/evidence/eurusd-m15-reviewed-batch-20261026.json

The manifest and its audit receipt are written beside the requested output. The default audit path is <manifest-stem>_audit.json; use --audit-output to choose a different new path. All referenced reports must exist beneath the manifest directory (including after symlink resolution), roles and paths must be unique, and the required roles must be supplied. Core roles experiment_registry and validation_workflow are always required. The builder computes exact SHA-256 values from report bytes and immediately runs the existing evidence-pack audit.

The builder never overwrites existing outputs. It returns exit code 0 for PASS or PASS_WITH_WARNINGS, 1 when the completed audit finds integrity/coverage failures, and 2 for invalid input or unsafe output handling. When audit findings exist, the manifest and audit receipt are retained for inspection; a failed audit is not a release approval.

A passing pack verifies only supplied artifact hashes, declared coverage, dataset fingerprints, run registry links, and safety flags. It does not establish predictive validity, profitability, statistical significance, or deployment suitability.

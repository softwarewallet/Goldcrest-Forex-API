# Offline research evidence-pack audit

The evidence-pack auditor validates an already-produced manifest and its referenced JSON
reports. It does not run training or upstream audits, select models, infer statistical
significance, connect to a broker, or authorize trading.

## Manifest contract

The manifest must use `schema_version: 1`, a non-empty `pack_id`, and a non-empty
`reports` list. Every report entry declares a unique `role`, a pack-relative `path`,
and the exact lowercase or uppercase SHA-256 digest in `expected_sha256`. Paths must
remain inside the manifest's directory, including after symlink resolution.

The required roles always include:

- `experiment_registry`: all declared runs must be represented and validated.
- `validation_workflow`: required audit coverage and dataset fingerprints must be present.

A manifest may require additional roles, but it cannot remove either core role. Numeric
coverage counts must be actual integers; JSON booleans are not accepted as counts. Dataset
fingerprints are reconciled across reports, and mismatches fail the audit.

## Run the audit

```bash
python -m app.research_evidence_pack_audit \
  --input data/research/evidence/<pack-id>.json \
  --output data/research/evidence/<pack-id>_audit.json
```

The input manifest and report files are read-only. The output path must be different from
the input and must not already exist. Publication uses a temporary file and an atomic
no-clobber operation so a concurrent process cannot have its output silently replaced.
Exit codes are `0` for `PASS` or `PASS_WITH_WARNINGS`, `1` for a completed audit with
integrity/coverage failures, and `2` for an invalid invocation or unsafe output path.

A passing pack means only that the supplied reports' hashes, declared coverage, dataset
fingerprints, and safety flags are consistent. It is not proof of predictive validity,
profitability, absence of all leakage, or suitability for deployment.


## Release-readiness receipts across a run registry

A pack can additionally require the `release_readiness` role:

```json
{
  "schema_version": 1,
  "pack_id": "reviewed-release-batch-20261026",
  "required_roles": [
    "experiment_registry",
    "validation_workflow",
    "release_readiness"
  ],
  "reports": [
    {
      "role": "experiment_registry",
      "path": "experiment_registry.json",
      "expected_sha256": "<sha256>"
    },
    {
      "role": "validation_workflow",
      "path": "validation_workflow.json",
      "expected_sha256": "<sha256>"
    },
    {
      "role": "release_readiness",
      "path": "run-a_release_readiness.json",
      "expected_sha256": "<sha256>"
    }
  ]
}
```

For this role, the report must have `status: PASS`, readiness
`RELEASE_READY_FOR_HUMAN_REVIEW`, explicit false model-qualification/deployment flags, a
dataset fingerprint, and a run ID present in the experiment registry. A warning-status or
unregistered run fails the pack. Dataset fingerprints are reconciled across the supplied
reports. Use one release-readiness report per run when a batch has multiple runs. Roles must remain unique,
so name additional entries `release_readiness_run_a`, `release_readiness_run_b`, and so on; all
roles beginning `release_readiness_` receive the same qualification checks. A pack PASS still does not authorize deployment.

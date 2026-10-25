# Research certification integrity workflow

This is an offline audit workflow. It does not run experiments, train models, connect to a broker, place trades, or authorize deployment.

## Inputs that must already exist

- A readiness report and evidence bundle.
- A captured provenance manifest and its detached seal.
- An independently curated inventory specification and its detached seal.
- The artifact root, current config, and dataset paths.
- A local Git checkout whose working tree is clean.

Keep the provenance manifest seal and inventory specification seal in a separately controlled location from their corresponding inputs. Detached SHA-256 seals detect changes; they are not digital signatures and do not prove that the inputs are truthful or complete.

## Create integrity seals

Create a provenance seal after reviewing the captured manifest:

```bash
python -m app.research_provenance_manifest_seal \
  --mode create \
  --manifest /path/to/provenance_manifest.json \
  --output /separate/trusted/location/provenance_manifest.seal.json
```

Create an inventory specification seal after the inventory has been independently curated and reviewed:

```bash
python -m app.research_artifact_inventory_spec_seal \
  --mode create \
  --inventory-spec /path/to/inventory_spec.json \
  --output /separate/trusted/location/inventory_spec.seal.json
```

The output seal file must not already exist. The `--seal` argument is only needed in verify mode.

## Run unified certification

```bash
python -m app.research_unified_provenance_certification \
  --readiness-report /path/to/readiness.json \
  --evidence-bundle /path/to/evidence_bundle.json \
  --evidence-root /path/to/evidence \
  --provenance-manifest /path/to/provenance_manifest.json \
  --provenance-seal /separate/trusted/location/provenance_manifest.seal.json \
  --inventory-spec /path/to/inventory_spec.json \
  --inventory-seal /separate/trusted/location/inventory_spec.seal.json \
  --artifact-root /path/to/research_artifacts \
  --repo-path /path/to/clean/git/checkout \
  --config-path /path/to/current/config.json \
  --dataset-path /path/to/current/dataset \
  --output /path/to/output/unified_certification.json \
  --report-seal-output /separate/trusted/location/unified_certification.seal.json
```

The report seal is optional, but recommended for retaining tamper-evident certification evidence. Keep it separate from the report, or anchor its digest in a separately controlled location.

## Verify a saved report later

```bash
python -m app.research_certification_report_seal \
  --mode verify \
  --report /path/to/unified_certification.json \
  --seal /separate/trusted/location/unified_certification.seal.json
```

A valid seal proves canonical content consistency only. Certification passing does not prove strategy profitability, predictive validity, statistical significance, or suitability for deployment.


## Input and output preflight

Before audit execution, the unified CLI checks that each JSON input is an existing file, the evidence/artifact roots and Git checkout are existing directories, and the config/dataset inputs are existing files or directories. It rejects missing or incorrectly typed inputs with a path-specific JSON error rather than beginning partial certification work. Report and report-seal paths must be distinct from input paths and from each other, and must not already exist.

Report JSON and its optional detached seal are staged in temporary files in their respective output directories before publication. If publication fails partway through, files published by that attempt are removed. This reduces partial-output risk but is not a filesystem-wide atomic transaction across two different directories; retain and verify the seal independently.

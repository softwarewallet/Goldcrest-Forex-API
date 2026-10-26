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


## Final-chain schema validation

The final certification-chain audit requires non-empty plan and registry identifiers, lowercase hexadecimal SHA-256 digests, and integer report counts (booleans are rejected even though Python treats them as integers). It also rejects malformed diagnostic fields and requires declared and verified report counts to reconcile exactly. These checks prevent malformed JSON values from passing through Python's permissive equality and truthiness rules. They validate report consistency only; they do not independently prove the source artifacts' scientific validity or profitability.


## Final-chain report publication safety

The final certification-chain CLI rejects an output path that resolves to any supplied input and refuses to overwrite an existing report. It stages and fsyncs JSON in the destination directory, then publishes through a no-clobber filesystem link so a concurrent destination creation cannot silently replace another file. If publication fails, the staged temporary file is cleaned up. Use a fresh output filename for each audit attempt.

Completion and final-certification audits reject duplicate JSON object keys in their input reports. Ambiguous inputs fail closed instead of allowing a later duplicate field to shadow the value first written.

## Strict JSON input syntax

The completion, run-integrity, and final-certification-chain readers reject duplicate object keys and non-standard numeric constants (NaN, Infinity, and -Infinity). These tokens are accepted by Python's JSON decoder by default but are not valid RFC 8259 JSON. A report containing one is treated as malformed and cannot pass an integrity gate. This syntax check complements, rather than replaces, exact-byte SHA-256 binding and schema/status validation. The audits remain offline and report-only; passing them does not authorize model selection, training, deployment, or trading.

## Strict JSON parsing across certification readers

The evidence-bundle, evaluation-integrity, readiness, artifact-bundle, unified-provenance, final-certification, report-seal, and provenance-manifest readers reject duplicate object keys and non-standard numeric constants (`NaN`, `Infinity`, `-Infinity`). This prevents Python's permissive JSON decoder from silently normalizing ambiguous or non-RFC-8259 inputs before audit checks. Malformed input must fail closed; digest checks continue to bind the exact source bytes where defined. These offline integrity checks do not establish model validity or authorize deployment or trading.


## Strict JSON parsing for provenance and evaluation readers

The artifact freshness audit, artifact-inventory specification seal, backtest-result integrity audit, evaluation-output integrity audit, and evaluation-window lineage audit reject duplicate object keys and non-standard JSON numeric constants (`NaN`, `Infinity`, and `-Infinity`) when reading manifests and report artifacts. These inputs are treated as invalid rather than allowing parser-specific ambiguity to influence provenance, integrity, or readiness conclusions. Regression tests cover malformed top-level and nested manifest values. This hardening is limited to offline research validation; it does not change model selection, model training, or trading/execution behavior.


## Strict JSON parsing in certification workflow gates

The freshness certification gate, reproducibility readiness audit, consolidated validation workflow audit, research report audit, and experiment-matrix provenance auditor now reject duplicate JSON object keys and non-standard numeric constants (`NaN`, `Infinity`, `-Infinity`) in JSON artifacts they read. The experiment-matrix provenance auditor also calculates the source SHA-256 from the exact byte snapshot it parsed, preventing a separate second read from binding the receipt to different bytes. This is offline provenance/integrity hardening only; no model training, automatic model selection, broker access, or trading execution behavior changes.

# End-to-end research release readiness

The release-readiness audit is a strict, offline qualification gate over an already completed
research run. It is intentionally stricter than the run-integrity audit: any unresolved
cross-artifact warning, non-clean required audit, missing completion receipt, or unverified
provider-terms evidence keeps the run **NOT RELEASE READY**.

## Run the gate

Create a provider-terms review evidence JSON and preserve the underlying source document or
captured evidence file. The evidence JSON contract is:

- `schema_version`: `1`
- `evidence_type`: `provider_terms_review`
- `provider`: provider name
- `terms_reference`: exact HTTP(S) URI recorded by the pipeline manifest
- `terms_reference_sha256`: SHA-256 of the UTF-8 encoded, trimmed reference
- `source_evidence_path`: path to the preserved source evidence file, absolute or relative to the evidence JSON
- `source_evidence_sha256`: SHA-256 of the exact source evidence file bytes
- `review_status`: `VERIFIED`
- `applicability`: `CONFIRMED`
- `reviewed_by`: reviewer identity
- `reviewed_at_utc`: timezone-aware ISO-8601 timestamp

The evidence JSON records an explicit human review. The script verifies its consistency and
source-file hash; it cannot independently determine whether the terms legally apply to the
provider, account, instrument, geography, or intended data use. Do not mark the review as
verified unless an authorized reviewer has actually checked the source.

Run:

```bash
python -m app.research_release_readiness \
  --manifest data/research/runs/<run-id>_pipeline_manifest.json \
  --provider-terms-evidence data/research/runs/<run-id>_provider_terms_review.json \
  --output data/research/runs/<run-id>_release_readiness.json \
  --json
```

The command refuses to overwrite an existing output and returns exit code 0 only for a clean
PASS; exit code 3 means NOT RELEASE READY. It does not modify source artifacts, rerun model
training, connect to cTrader or a broker, choose a winner, qualify a model, or authorize
deployment.

## Release checklist covered

- Existing cross-artifact integrity audit and recorded safety flags.
- Pipeline and dataset-manifest provider-terms reference/hash consistency.
- Explicit human-review record and hash verification of preserved terms evidence.
- Required raw-data, feature-source, point-in-time invariance, label-maturity, provenance,
  temporal-calibration, model-evaluation, and reproducibility statuses must each be exactly
  `PASS` (not `PASS_WITH_WARNINGS`).
- Completion receipt must exist, match the run and pipeline manifest, be a clean PASS, and
  retain non-actionable/no-execution/no-model-qualification flags.
- Any unresolved warning or missing evidence fails closed.

A PASS is an artifact/evidence checklist result only. It is not evidence of predictive validity,
profitability, statistical significance, complete leakage exclusion, or readiness for live
trading. The existing raw OHLC audit has unresolved long-gap warnings; until those are
resolved against an authoritative instrument/provider trading calendar and a fresh run is
audited, the current research run should not qualify for release.

# Provenance-bound run manifests

`build_run_manifest` can now accept `dataset_provenance` and
`actual_snapshot_sha256`. When supplied, both are mandatory as a pair: the function
checks the original snapshot digest against the provenance record and checks the parsed
bars' row count, instrument set, and single timeframe against the declared metadata.
The complete provenance record and its canonical fingerprint are stored in the manifest.

The provenance fingerprint is included in the run identity. Changing the source version,
retrieval metadata, policy declarations, terms reference, or any other provenance field
therefore creates a different run ID even when the parsed OHLC rows happen to be equal.
This prevents an experiment registry retry from silently treating different declared
lineage as the same run.

Legacy callers that do not yet supply provenance remain supported and retain the
previous run-identity calculation. For new audited experiments, pass provenance and the
SHA-256 digest computed from the exact original snapshot bytes. The helper cannot
independently read or authenticate that source file, and a matching digest does not prove
provider authenticity, legal permission, point-in-time availability, or market-data
correctness. All manifests remain offline research artifacts and non-actionable.

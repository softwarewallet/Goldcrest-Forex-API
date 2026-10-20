# Offline dataset provenance contract

The evaluation workflow requires explicit, auditable market-data lineage before any
result is interpreted. `app/dataset_provenance.py` provides a small contract for
recording and checking that lineage; it does not download market data or infer missing
metadata.

## Required metadata

A record includes source name and version, retrieval timestamp normalized to UTC, the
original snapshot's SHA-256 digest, source timestamp timezone, duplicate handling policy,
corporate-action/adjustment policy (use an explicit value such as `none` only when
accurate), license/terms reference, instrument identifiers, timeframe, and parsed row
count. Empty placeholders are rejected.

Compute `snapshot_sha256` over the exact immutable source file bytes before parsing or
transformation. Do not hash a normalized export and call it the original source snapshot.
Record transformations separately in the experiment configuration and code revision.
The license/terms reference is operator-supplied; the helper does not validate legal
rights or authenticate the source.

## Verification

Call `verify_dataset_provenance` with the provenance record and independently derived
facts from the actual snapshot and parsed rows. It checks snapshot digest, row count,
instrument set, timeframe, and schema version, then returns a canonical provenance
fingerprint. A successful result means the supplied values agree; it does not prove
source authenticity, data correctness, point-in-time availability, absence of revisions,
or licensing compliance.

All verification results are `actionable=false`. This contract does not fetch data,
train a model, mutate a registry, serve predictions, connect to a broker, or authorize
trading. Synthetic fixtures test consistency mechanics only; they provide no evidence
of market performance.

## Operator sequence

1. Preserve the original provider snapshot as an immutable file.
2. Compute SHA-256 from its raw bytes and record the provider/version and retrieval time.
3. Declare timezone, duplicate policy, adjustment policy, instruments, timeframe, and
   licensing/terms reference explicitly.
4. Parse the snapshot and independently derive row count, instrument set, and timeframe.
5. Run provenance verification before using the dataset in the offline evaluation flow.
6. Store the provenance record and its fingerprint alongside the run manifest/report.

# Provenance-aware research-run preparation

The prepare_research_run function now accepts optional dataset_provenance and
actual_snapshot_sha256 inputs. They must be supplied together. When present, the
pipeline verifies the SHA-256 digest of the exact original snapshot bytes against the
declared provenance and reconciles parsed row count, instrument set, and timeframe
before registering the run.

The provenance record and its canonical digest are stored in the run manifest; its
digest participates in the run ID. Changing declared source/version/retrieval/policy
metadata creates a distinct registered run even when parsed bars are identical.
Malformed or mismatched provenance fails closed before registry insertion.

For a real dataset, compute the snapshot hash over the immutable source artifact before
parsing or transforming it, and construct provenance from the actual artifact and its
documented provider terms. Do not use a hash of normalized rows as a substitute for the
original-file hash. This code validates consistency, not source authenticity, provider
quality, licensing, point-in-time availability, or data correctness. It does not fetch
market data.

Callers that omit provenance retain legacy behavior for compatibility. New audited
runs should supply it. Synthetic fixtures in tests demonstrate validation mechanics only;
they do not establish predictive performance. All research outputs remain offline and
non-actionable; no serving, broker, or trading integration is added.

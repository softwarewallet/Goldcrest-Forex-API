# Research artifact audit

`app/research_artifact_audit.py` adds a read-only, end-to-end consistency check for a
completed offline experiment. It loads a report from disk, verifies the canonical report
SHA-256 against the completed SQLite run record, validates that the report and every fold
remain `actionable=false`, derives expected counts for each model name, fitted model version, fold, and validation/test
partition, then reconciles those groups against SQLite. This catches misplaced rows even
when the overall ledger row count happens to match the report.

The audit fails closed if the report is missing/malformed/tampered, its completed-run
binding does not match, a fold metric count is missing/invalid, the persisted row count
differs, or any ledger row is actionable. The database query is read-only. It does not
repair files, mutate run state, re-run models, or infer missing market-data/target/cost
assumptions.

Example:

```python
from app.prediction_ledger import PredictionLedger
from app.research_artifact_audit import verify_research_artifacts

ledger = PredictionLedger("artifacts/research.sqlite")
audit = verify_research_artifacts("artifacts/report.json", ledger=ledger)
print(audit)
```

A passing audit establishes internal artifact consistency, not source-data authenticity,
model validity, calibration, profitability, or future performance. Synthetic fixtures test
only software mechanics. All outputs remain research-only and non-actionable; no serving,
broker, trading, or deployment integration is added.

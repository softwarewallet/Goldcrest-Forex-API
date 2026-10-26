# Authoritative market-calendar gap adjudication

This is an additional, evidence-bound review layer. It does not change the raw OHLC snapshot or the existing raw-data audit.

## Required evidence

Create a JSON record with schema_version 1, evidence_type market_calendar_gap_adjudication, provider, symbol, timeframe, authority, source_reference, retrieved_at_utc, source_evidence_sha256, coverage_start_utc, coverage_end_utc, and a closed_intervals array. Each interval requires start_utc, end_utc, reason, and source_locator. Use the actual applicable provider/instrument calendar and an archived source document; do not use a placeholder or infer closures from holiday names.

Each interval must cover the entire missing-bar interval from the first expected bar after the last observed bar up to (but not including) the next observed timestamp. Partial overlap does not adjudicate a gap. The archived source SHA-256 must match the evidence record. A hash proves file identity, not that the source is authoritative; that requires human review.

## Run

```bash
python -m app.research_calendar_gap_adjudication \\
  --ohlc-csv data/research/runs/<run-id>_EURUSD_M15_ctrader_ohlc.csv \\
  --snapshot-manifest data/research/runs/<run-id>_EURUSD_M15_ctrader_ohlc.csv.manifest.json \\
  --calendar-evidence data/research/evidence/<calendar-evidence>.json \\
  --source-evidence data/research/evidence/<archived-calendar-document> \\
  --output data/research/evidence/<run-id>_calendar_gap_adjudication.json \\
  --json
```

The report returns PASS only when evidence checks pass and every gap over the threshold is fully covered by a declared closed interval. PASS_WITH_WARNINGS means gaps remain unresolved; FAIL means evidence or snapshot integrity checks failed. The tool does not infer holidays from dates, repair candles, or authorize deployment.

Release qualification must separately verify provider-terms review evidence. Calendar evidence cannot substitute for provider terms, and provider terms cannot substitute for calendar evidence.

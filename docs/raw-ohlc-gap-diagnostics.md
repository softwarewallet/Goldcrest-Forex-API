# Raw OHLC timestamp and coverage diagnostics

The local raw OHLC integrity audit provides three complementary descriptive views:

- `gap_diagnostics_between_one_frame_and_8_hours` summarizes timestamp intervals longer than one nominal timeframe and no longer than eight hours. It groups intervals by elapsed timeframe multiples, records repeated UTC weekday/hour boundaries, and lists the largest examples.
- `gap_diagnostics_over_8_hours` summarizes longer gaps by the audit's existing categories, duration bands, UTC boundary distributions, and repeated boundary patterns.
- `timestamp_coverage_profile` reports observed rows by UTC calendar date and hour, daily row-count summary statistics, the full observed-date profile, and examples/counts of calendar dates with no rows between the first and last timestamp.

Run the audit from the repository root with the project environment active. For a CSV already used by a completed research run, prefer its pipeline manifest so the audit resolves the exact recorded raw input instead of requiring the CSV path to be supplied again:

```bash
python -m app.research_data_audit \
  --pipeline-manifest data/research/runs/<run-id>_pipeline_manifest.json \
  --json \
  --output data/research/runs/<run-id>_raw_ohlc_reaudit.json
```

The command verifies the pipeline-manifest fingerprint, obtains `raw_ohlc_path`, defaults the expected symbol and timeframe from that manifest, then compares the file's SHA-256 with `raw_ohlc_sha256`. A missing, edited, or mismatched raw file produces a hard audit failure rather than silently auditing a different snapshot. If the original file has been moved, supply the replacement location with `--csv <actual-file-path>` as well; its hash must still match the pipeline manifest. The adjacent cTrader snapshot manifest is used by default as `<csv>.manifest.json`, or can be supplied explicitly with `--manifest`.

A direct-file audit is still supported for standalone snapshots:

```bash
python -m app.research_data_audit \
  --csv data/research/<symbol-timeframe>.csv \
  --manifest data/research/<symbol-timeframe>.csv.manifest.json \
  --symbol EURUSD \
  --timeframe M15 \
  --json \
  --output data/research/runs/<run-id>_raw_ohlc_audit.json
```

In both modes, the audit fingerprints the CSV and does not modify the source snapshot.

The interval and calendar-date profiles describe observed timestamp spacing only. An interval spanning two or more timeframe units or a UTC calendar date with no rows is not automatically a missing-candle finding. Unobserved dates can reflect weekends, instrument sessions, holidays, provider coverage, collection choices, or other factors. The report does not assume that each UTC day should contain a fixed number of rows. The greater-than-eight-hour report uses a conservative UTC weekend heuristic and informational fixed-date holiday tags; neither is authoritative evidence of an expected market closure.

Use repeated boundary patterns and daily/hourly coverage profiles to prioritize manual comparison against the relevant instrument/provider session calendar, cTrader pagination evidence, and source export. Non-monotonic timestamps prevent interval coverage summaries. These diagnostics do not reclassify gaps, suppress warnings, repair or impute bars, select or promote models, or authorize execution. Overall status and existing warnings remain governed by the integrity checks; the coverage profiles do not independently alter them.


## Gap-policy audit diagnostic-schema integrity

The gap-policy equivalence audit treats a passing input report as structurally inconsistent if its
optional `errors` or `warnings` field is not a list of non-empty strings, or if a report marked
`PASS` / `PASS_WITH_WARNINGS` still contains errors. Gap-event affected-label row identities are
validated as dataset row numbers before comparing exact membership; malformed members are not
silently filtered out. These checks fail closed and do not alter the underlying OHLC, dataset,
eligibility policies, or model results.

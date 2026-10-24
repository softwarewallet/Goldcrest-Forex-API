# Raw OHLC timestamp-gap diagnostics

The local raw OHLC integrity audit provides two complementary descriptive views:

- `gap_diagnostics_between_one_frame_and_8_hours` summarizes timestamp intervals longer than one nominal timeframe and no longer than eight hours. It groups intervals by elapsed timeframe multiples, records repeated UTC weekday/hour boundaries, and lists the largest examples.
- `gap_diagnostics_over_8_hours` summarizes longer gaps by the audit's existing categories, duration bands, UTC boundary distributions, and repeated boundary patterns.

Run the audit from the repository root with the project environment active:

```bash
python -m app.research_data_audit \
  --csv data/research/<symbol-timeframe>.csv \
  --manifest data/research/<symbol-timeframe>.csv.manifest.json \
  --symbol EURUSD \
  --timeframe M15 \
  --json \
  --output data/research/runs/<run-id>_raw_ohlc_audit.json
```

Replace the placeholder paths with the exact raw OHLC CSV and its integrity manifest for the research run. The audit fingerprints the CSV and validates its manifest; the source snapshot is never modified. If the manifest is next to the CSV under the default `<csv>.manifest.json` name, `--manifest` may be omitted.

The sub-eight-hour report describes timestamp spacing only. An interval spanning two or more timeframe units is not automatically a missing-candle finding: instrument sessions, holidays, provider coverage, and expected market closures are not inferred. The greater-than-eight-hour report uses a conservative UTC weekend heuristic and informational fixed-date holiday tags; neither is authoritative evidence of an expected market closure.

Use repeated boundary patterns to prioritize manual comparison against the relevant instrument/provider session calendar, cTrader pagination evidence, and source export. These diagnostics do not reclassify gaps, suppress warnings, repair or impute bars, select or promote models, or authorize execution. Overall status and existing warnings remain governed by the integrity checks; the new sub-eight-hour summaries do not independently alter them.

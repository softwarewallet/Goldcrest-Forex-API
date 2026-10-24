# Raw OHLC long-gap diagnostics

The local raw OHLC integrity audit now includes `gap_diagnostics_over_8_hours` in its JSON report.
It adds descriptive summaries for gaps longer than eight hours: counts by existing category, duration
bands, UTC weekday/hour distributions at each boundary, and boundary patterns repeated within the
snapshot.

Run it from the repository root with the project environment active:

```bash
python -m app.research_data_audit \
  --csv data/research/<symbol-timeframe>.csv \
  --json \
  --output data/research/runs/<run-id>_raw_ohlc_audit.json
```

The path above is a placeholder; use the exact raw OHLC CSV for the research run. The report remains
read-only and fingerprints its input. Repeated boundary patterns can help prioritize manual inspection
for recurring provider gaps or session breaks, but they do not prove that a gap is expected or that
candles are missing. These diagnostics do not use an external holiday/session calendar, reclassify
gaps, suppress warnings, alter source rows, impute candles, select models, or authorize execution.
Keep reviewing non-weekend long gaps against the relevant instrument/provider session calendar and
the original snapshot's pagination and provenance evidence.

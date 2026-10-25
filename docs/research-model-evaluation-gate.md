# Offline model evaluation integrity gate

The evaluation gate checks a persisted experiment-matrix JSON without retraining models or modifying the source report.

## Run

```bash
python -m app.research_model_evaluation_gate \
  --experiment-matrix data/research/runs/<run-id>_experiment_matrix.json \
  --output data/research/runs/<run-id>_model_evaluation_gate.json \
  --json
```

The output path must not already exist. Exit code `0` means the checked report passed; exit code `3` means it failed. The source matrix is read-only. The offline research pipeline generates this receipt automatically as `<run-id>_model_evaluation_gate.json` and records its SHA-256 and status in the pipeline manifest. Add `--strict-model-evaluation-gate` to `app.research_pipeline` to return exit code `6` when this gate fails.

## Checks

- Explicit non-actionable and no-test-selection/no-test-tuning flags.
- Unique experiment IDs and unique fold indices.
- Cross-experiment common-fold consistency: shared fold indices must use identical timezone-normalized train, validation, and test boundaries and the same held-out sample count.
- Declared fold count agrees with each experiment's recorded folds.
- Complete timezone-aware train, validation, and test boundaries, with each window's start preceding its end.
- Finite metric values and expected ranges for sample counts, prevalence, Brier score, log loss, fixed-threshold accuracy, mean probability, and expected calibration error.
- Positive and negative counts sum to sample count; prevalence agrees with positive-count/sample-count within a small floating-point tolerance.

## Interpretation and limits

A `PASS` means the listed structure and arithmetic checks are internally consistent. It does **not** prove that the features are point-in-time correct, that all data leakage is absent, that model results are statistically significant, or that any strategy is profitable. The gate does not compare models, select a winner, fit a calibrator, alter persisted artifacts, connect to a broker, or authorize execution. Treat it as one independent integrity layer alongside provenance, temporal calibration, and research-run audits.


The common-fold check detects mismatched candidate windows before paired or comparative summaries are trusted. It is a structural consistency check only; identical windows do not establish independent observations, statistical significance, or a fair experimental design by themselves.


## Chronological out-of-sample window identity

The early/middle/late OOS period diagnostic cross-checks every available train, validation, and test boundary across candidates sharing a fold index, in addition to checking chronological test-start order. Where all candidates record held-out sample counts, those counts must also match. A mismatch withholds the period summaries. Legacy matrices with partial or absent boundary metadata are explicitly reported as partial/unknown; they are not silently described as having fully verified windows. This is a structural audit, not evidence that overlapping folds are statistically independent.


## Dependence-qualified baseline uncertainty

The consolidated evidence interpretation does not classify paired bootstrap intervals as positive or negative interval evidence when the baseline-analysis receipt reports repeated held-out timestamps across walk-forward folds. It preserves the point estimates and reports how many intervals were withheld from classification. The numeric intervals remain in the source receipt for auditability, but overlapping/repeated held-out observations can make fold-level forecast errors dependent; this guard prevents the summary from implying that the interval evidence is independent or stronger than the diagnostics support. This is a conservative interpretation guard, not a new significance test and not proof of independence when no repeated timestamps are detected.


## Phase 9: Multiplicity context ledger

The consolidated evidence report now inventories the number of comparisons represented by baseline probability-loss intervals, temporal calibration changes, volatility-regime intervals, and chronological out-of-sample candidate summaries. It explicitly records that no multiplicity adjustment has been applied and that confirmatory inference is not supported by this report. Counts across these diagnostic families are an audit inventory, not a claim that all comparisons form one statistical testing family. Any confirmatory analysis requires a pre-specified hypothesis family and an appropriate method; the ledger does not perform that correction.


## Phase 10: Manifest-to-receipt consistency

For hash-bound provenance, model-evaluation-gate, and temporal-calibration receipts, the consolidated evidence report checks any status/error/warning summary fields copied into the pipeline manifest against the receipt contents. A mismatch is an integrity error even if the receipt itself is readable. This catches stale or contradictory manifest metadata; it supplements, and does not replace, the artifact hash, scope, run, and source-matrix checks. Legacy manifests that do not record a summary field are not failed solely because that field is absent.


## Phase 11: Artifact identity metadata completeness

Each artifact record now reports whether symbol, timeframe, and manifest-path identity metadata were present and matched the current run context. Missing identity fields generate explicit warnings and a `PARTIAL` metadata status rather than being silently treated as verified. A reported-but-mismatched identity remains an error. This is backward-compatible with older receipts and makes provenance limitations visible without overstating what legacy artifacts establish.


## Phase 12: Canonical experiment-matrix payload integrity

The consolidated evidence report independently verifies the canonical `report_payload_sha256` embedded in the experiment matrix. A mismatch is an integrity error and suppresses chronological out-of-sample period summaries derived from that matrix. Legacy matrices without the self-hash are explicitly marked `LEGACY_UNVERIFIED` and produce a warning rather than being described as verified. This self-hash complements the manifest's raw-file SHA-256: the two checks answer different questions and neither establishes data leakage absence or model validity.


## Phase 13: Diagnostic declared-count consistency

For temporal calibration, baseline, volatility-regime, and temporal-stability receipts, the consolidated evidence report compares declared comparison/experiment counts with the corresponding list lengths. Contradictions fail the affected artifact record; missing legacy count fields are surfaced as warnings when the rows exist. This prevents internally inconsistent summaries from appearing verified while preserving visibility for older report schemas.


## Phase 14: Consolidated integrity inventory

The final report includes an `integrity_summary` that aggregates artifact-section statuses, complete versus partial identity metadata, declared-count mismatches, run-integrity status, and matrix payload-hash status. This gives downstream audit consumers one concise inventory without collapsing warnings into a pass claim or implying predictive quality. The summary is descriptive only and does not authorize deployment or execution.

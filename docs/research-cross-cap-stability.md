# Cross-cap label-duration stability audit

The cross-cap stability audit consolidates the five PASS common-held-out-cohort audits for
seeds 20261021–20261025. It reads each audit and its three source comparison reports, checks
source status, non-actionable safety flags, per-seed dataset-hash equivalence, cap coverage,
metric availability, and the common sample count before summarizing any deltas.

Run from the repository root after all five common-cohort audits have passed:

```bash
python -m app.research_cross_cap_stability \
  --audit-report data/research/runs/common_cohort_audit_1h_1p5h_8h_seed_20261021.json \
  --audit-report data/research/runs/common_cohort_audit_1h_1p5h_8h_seed_20261022.json \
  --audit-report data/research/runs/common_cohort_audit_1h_1p5h_8h_seed_20261023.json \
  --audit-report data/research/runs/common_cohort_audit_1h_1p5h_8h_seed_20261024.json \
  --audit-report data/research/runs/common_cohort_audit_1h_1p5h_8h_seed_20261025.json \
  --output data/research/runs/cross_cap_stability_1h_1p5h_8h_five_seed.json
```

The command refuses to overwrite an existing output. To deliberately replace a prior result,
add `--overwrite` after reviewing the target path.

## Reading the report

- `per_seed_results` retains model, fold, common timestamp count, timestamp fingerprint,
  metrics by cap, and metric deltas relative to the 1-hour cap.
- Each fold also retains the eligible timestamp count for each cap and the number
  excluded when forming the shared evaluation cohort. `eligibility_accounting_by_cap`
  reconciles these counts per seed and model/feature group. Counts summed across folds
  are explicitly marked as observations, not unique timestamps, because fold windows may overlap.
- A passing audit with malformed diagnostic fields or a non-empty `errors` list is rejected.
  Fold-window boundaries are checked during cap-equivalence diagnostics as well as during
  common-cohort recomputation.
- `cross_seed_summary` first averages fold deltas within each seed, then summarizes those
  five seed-level values using mean, median, population standard deviation, minimum, and maximum.
  This avoids treating overlapping fold windows as independent seeds.
- Negative log-loss and Brier-score deltas favor the comparison cap; positive accuracy deltas
  favor the comparison cap. Deltas are always comparison-cap metric minus 1-hour metric.
- `seed_favorable_count`, `seed_adverse_count`, and `seed_neutral_count` describe the direction
  across seeds. They are not significance tests or a model-selection rule.
- `baseline_dataset_sha256` and baseline run IDs are checked within each seed. Dataset hashes
  are deliberately not required to match across seeds.
- `PASS` means structural/provenance checks passed; it does not establish statistical
  significance, causal improvement, profitability, or deployment readiness.

The audit is descriptive and fail-closed. It does not modify source reports, promote models,
connect to execution, or enable trading. If a source metric or report is missing or inconsistent,
the audit returns `FAIL` rather than substituting a metric from a different evaluation cohort.


## Common-cohort input integrity

The common-held-out-cohort auditor requires each source comparison report to explicitly
set `inferential: false`, `automatic_model_selection: false`, `actionable: false`,
and `execution_connected: false`. When supplied, `errors` and `warnings` must be
lists of non-empty strings. A report marked `PASS` or `PASS_WITH_WARNINGS` with a
non-empty `errors` list is rejected. Missing or malformed safety/diagnostic metadata
is not silently treated as safe. These checks validate report integrity; they do not
remove legitimate provenance warnings or establish model quality.

import { useState } from "react";

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord
    : null;
}

function valueAt(source: JsonRecord | null, key: string): unknown {
  return source ? source[key] : undefined;
}

function textValue(value: unknown): string {
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  return "—";
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatNumber(value: unknown, digits = 4): string {
  const numeric = numberValue(value);
  return numeric === null ? "—" : numeric.toLocaleString(undefined, { maximumFractionDigits: digits });
}


function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function stabilityMetric(values: number[]): JsonRecord | null {
  if (!values.length) return null;
  const wins = values.filter(value => value < 0).length;
  const losses = values.filter(value => value > 0).length;
  return {
    folds_with_metric: values.length,
    candidate_wins: wins,
    ties: values.length - wins - losses,
    candidate_losses: losses,
    median_candidate_minus_baseline: median(values),
  };
}

function buildFoldStability(foldResults: JsonRecord | null, comparisons: unknown[]): JsonRecord[] {
  if (!foldResults || !comparisons.length) return [];
  const models = new Map<string, Map<number, JsonRecord>>();
  for (const [name, rows] of Object.entries(foldResults)) {
    const folds = new Map<number, JsonRecord>();
    if (Array.isArray(rows)) {
      for (const item of rows) {
        const fold = record(item);
        const index = numberValue(valueAt(fold, "fold_index"));
        if (fold && index !== null && Number.isInteger(index) && index >= 0) folds.set(index, fold);
      }
    }
    models.set(name, folds);
  }

  const pairs = new Map<string, { candidate: string; baseline: string; partition: string }>();
  for (const item of comparisons) {
    const row = record(item);
    const candidate = valueAt(row, "candidate_name");
    const baseline = valueAt(row, "baseline_name");
    const partition = valueAt(row, "partition");
    if (typeof candidate !== "string" || typeof baseline !== "string" ||
        (partition !== "validation" && partition !== "test")) continue;
    pairs.set(JSON.stringify([candidate, baseline, partition]), { candidate, baseline, partition });
  }

  const output: JsonRecord[] = [];
  for (const pair of pairs.values()) {
    const candidateFolds = models.get(pair.candidate);
    const baselineFolds = models.get(pair.baseline);
    if (!candidateFolds || !baselineFolds) continue;
    const common = [...candidateFolds.keys()].filter(index => baselineFolds.has(index)).sort((a, b) => a - b);
    const brier: number[] = [];
    const logLoss: number[] = [];
    let sampleMismatch = false;
    const included: number[] = [];
    for (const index of common) {
      const candidateMetrics = record(valueAt(candidateFolds.get(index) || null, pair.partition));
      const baselineMetrics = record(valueAt(baselineFolds.get(index) || null, pair.partition));
      if (!candidateMetrics || !baselineMetrics) continue;
      const candidateCount = numberValue(valueAt(candidateMetrics, "sample_count"));
      const baselineCount = numberValue(valueAt(baselineMetrics, "sample_count"));
      if (candidateCount === null || baselineCount === null || candidateCount !== baselineCount) {
        sampleMismatch = true;
        break;
      }
      included.push(index);
      for (const [key, values] of [["brier_score", brier], ["log_loss", logLoss]] as const) {
        const candidateValue = numberValue(valueAt(candidateMetrics, key));
        const baselineValue = numberValue(valueAt(baselineMetrics, key));
        if (candidateValue !== null && baselineValue !== null) values.push(candidateValue - baselineValue);
      }
    }
    output.push({
      candidate_name: pair.candidate,
      baseline_name: pair.baseline,
      partition: pair.partition,
      fold_indices: included,
      fold_count: included.length,
      metrics: {
        brier_score: stabilityMetric(brier),
        log_loss: stabilityMetric(logLoss),
      },
      sample_count_mismatch: sampleMismatch,
      actionable: false,
    });
  }
  return output;
}

function Metric({ label, value, note }: { label: string; value: string; note?: string }) {
  return <div className="research-report-metric"><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div>;
}

export default function ResearchReportPanel() {
  const [report, setReport] = useState<JsonRecord | null>(null);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");

  const loadReport = async (file?: File) => {
    if (!file) return;
    setError("");
    try {
      const parsed: unknown = JSON.parse(await file.text());
      const root = record(parsed);
      if (!root) throw new Error("The file must contain a JSON object.");
      const hasAudit = Boolean(record(root.data_quality_audit) || record(root.feature_preparation) || record(root.label_preparation));
      const hasEvaluation = Boolean(record(root.evaluation) || record(root.fold_results));
      if (!hasAudit && !hasEvaluation) throw new Error("This JSON does not look like a GoldcrestAI research report.");
      if (root.actionable === true || root.scope === "LIVE_EXECUTION") {
        throw new Error("Actionable or execution-scoped reports are not accepted by this research-only viewer.");
      }
      setReport(root);
      setFileName(file.name);
    } catch (e) {
      setReport(null);
      setFileName("");
      setError(e instanceof Error ? e.message : "Unable to read report.");
    }
  };

  const evaluation = record(valueAt(report, "evaluation")) || report;
  const evaluationContract = record(valueAt(evaluation, "evaluation_contract"));
  const manifest = record(valueAt(evaluation, "manifest"));
  const provenance = record(valueAt(evaluationContract, "dataset_provenance"))
    || record(valueAt(manifest, "dataset_provenance"));
  const quality = record(valueAt(report, "data_quality_audit"));
  const lineage = record(valueAt(report, "artifact_lineage"));
  const temporalAudit = record(valueAt(manifest, "temporal_integrity_audit"));
  const reproducibility = record(valueAt(manifest, "reproducibility"));
  const features = record(valueAt(report, "feature_preparation"));
  const labels = record(valueAt(report, "label_preparation"));
  const foldResults = record(valueAt(evaluation, "fold_results"));
  const pairedComparisons = Array.isArray(valueAt(evaluation, "paired_comparisons")) ? valueAt(evaluation, "paired_comparisons") as unknown[] : [];
  const foldStability = buildFoldStability(foldResults, pairedComparisons);
  const selectionAudit = record(valueAt(evaluation, "model_selection_audit"));
  const oosRobustness = record(valueAt(evaluation, "oos_period_robustness_audit"));
  const regimeUncertainty = record(valueAt(evaluation, "regime_uncertainty_audit"));
  const blockLengthAudit = record(valueAt(evaluation, "block_length_sensitivity_audit"));
  const temporalCalibration = record(valueAt(evaluation, "temporal_calibration_stability_audit"));
  const temporalCalibrationRows = Array.isArray(valueAt(temporalCalibration, "comparisons")) ? valueAt(temporalCalibration, "comparisons") as unknown[] : [];
  const blockLengthRows = Array.isArray(valueAt(blockLengthAudit, "comparisons")) ? valueAt(blockLengthAudit, "comparisons") as unknown[] : [];
  const modelRows = foldResults ? Object.entries(foldResults).map(([name, rows]) => ({
    name,
    folds: Array.isArray(rows) ? rows : [],
  })) : [];
  const warnings = Array.isArray(valueAt(quality, "warnings")) ? valueAt(quality, "warnings") as unknown[] : [];
  const actionable = report?.actionable === false || valueAt(evaluation, "actionable") === false;

  return <section className="research-report-panel">
    <div className="section-heading">
      <div><div className="eyebrow">OFFLINE EVIDENCE / DATA LINEAGE</div><h2>Historical research report review</h2></div>
      <span className="research-tag">LOCAL FILE · READ-ONLY</span>
    </div>
    <p className="research-report-intro">Load a pipeline JSON report to inspect data-quality evidence, feature/label counts, and walk-forward model results. The file is parsed in this browser and is not uploaded to the API.</p>
    <label className="research-report-upload">SELECT JSON REPORT<input type="file" accept=".json,application/json" onChange={event => void loadReport(event.target.files?.[0])} /></label>
    {error && <div className="research-report-error" role="alert">{error}</div>}
    {!report && !error && <div className="research-report-empty">No report loaded. Use a combined report produced by <code>app.historical_research_pipeline</code>.</div>}
    {report && <>
      <div className="research-report-file"><span>LOADED ARTIFACT</span><b>{fileName}</b><span>RUN ID</span><b>{textValue(valueAt(record(valueAt(report, "evaluation")), "run_id") ?? valueAt(report, "run_id"))}</b><span>STATUS</span><b>{actionable ? "NON-ACTIONABLE RESEARCH" : "REVIEW REQUIRED"}</b></div>
      {provenance && <div className="research-provenance">
        <div className="research-provenance-heading">DATASET PROVENANCE <span>{valueAt(provenance, "provenance_complete") === true || valueAt(evaluationContract, "dataset_provenance_verification") !== undefined ? "DECLARED / HASH-CHECKED" : "DECLARATIONS"}</span></div>
        <div className="research-provenance-grid">
          <div><small>SOURCE</small><b>{textValue(valueAt(provenance, "source_name"))}</b></div>
          <div><small>VERSION</small><b>{textValue(valueAt(provenance, "source_version"))}</b></div>
          <div><small>RETRIEVED (UTC)</small><b>{textValue(valueAt(provenance, "retrieved_at"))}</b></div>
          <div><small>TIMEZONE</small><b>{textValue(valueAt(provenance, "timestamp_timezone"))}</b></div>
          <div><small>ADJUSTMENTS</small><b>{textValue(valueAt(provenance, "adjustment_policy"))}</b></div>
          <div><small>LICENSE / TERMS</small><b>{textValue(valueAt(provenance, "license_or_terms_reference"))}</b></div>
          <div><small>SNAPSHOT SHA-256</small><b>{textValue(valueAt(provenance, "snapshot_sha256"))}</b></div>
          <div><small>ROWS / SYMBOL / TF</small><b>{textValue(valueAt(provenance, "row_count"))} / {textValue(Array.isArray(valueAt(provenance, "instruments")) ? (valueAt(provenance, "instruments") as unknown[]).join(", ") : "—")} / {textValue(valueAt(provenance, "timeframe"))}</b></div>
        </div>
        <p className="research-report-note">Hash verification checks artifact consistency, not provider authenticity, legal license validity, or upstream feature causality. Confirm source declarations against the original provider export.</p>
      </div>}
      {lineage && <div className="research-lineage">
        <div><b>ARTIFACT LINEAGE</b><span>{valueAt(lineage, "verified") === true ? "HASH CHAIN VERIFIED" : "VERIFICATION REQUIRED"}</span></div>
        <p>Raw OHLC → causal features → explicit labels → evaluation dataset</p>
        <div className="research-lineage-hashes">
          <small>RAW</small><code>{textValue(valueAt(lineage, "raw_ohlc_sha256"))}</code>
          <small>FEATURE CSV</small><code>{textValue(valueAt(lineage, "feature_csv_sha256"))}</code>
          <small>LABELED CSV</small><code>{textValue(valueAt(lineage, "labeled_csv_sha256"))}</code>
          <small>EVALUATION</small><code>{textValue(valueAt(lineage, "evaluation_dataset_sha256"))}</code>
        </div>
        <p className="research-report-note">The pipeline checks byte-level hash continuity between generated artifacts and the evaluated CSV. This proves artifact linkage, not the authenticity of upstream data or the absence of leakage in externally supplied features.</p>
      </div>}
      {reproducibility && <div className="research-lineage">
        <div><b>REPRODUCIBILITY MANIFEST</b><span>DECLARED CONFIGURATION · NON-ACTIONABLE</span></div>
        <p>Runtime versions, model factory configuration, model source fingerprints, and configuration hashes captured for this experiment.</p>
        <div className="research-lineage-hashes">
          <small>MANIFEST SHA-256</small><code>{textValue(valueAt(reproducibility, "manifest_sha256"))}</code>
          <small>PYTHON</small><code>{textValue(valueAt(record(valueAt(reproducibility, "environment")), "python_version"))} · {textValue(valueAt(record(valueAt(reproducibility, "environment")), "python_implementation"))}</code>
          <small>PLATFORM</small><code>{textValue(valueAt(record(valueAt(reproducibility, "environment")), "platform"))}</code>
          <small>NUMPY</small><code>{textValue(valueAt(record(valueAt(record(valueAt(reproducibility, "environment")), "packages")), "numpy"))}</code>
          <small>SCIKIT-LEARN</small><code>{textValue(valueAt(record(valueAt(record(valueAt(reproducibility, "environment")), "packages")), "scikit-learn"))}</code>
          <small>LIGHTGBM</small><code>{textValue(valueAt(record(valueAt(record(valueAt(reproducibility, "environment")), "packages")), "lightgbm"))}</code>
          <small>XGBOOST</small><code>{textValue(valueAt(record(valueAt(record(valueAt(reproducibility, "environment")), "packages")), "xgboost"))}</code>
        </div>
        {record(valueAt(reproducibility, "initial_model_configuration")) && <details><summary>Inspect model configuration fingerprints</summary><div className="research-model-table-wrap"><table className="research-model-table"><thead><tr><th>MODEL</th><th>CLASS</th><th>VERSION</th><th>CONFIG SHA-256</th><th>SOURCE SHA-256</th></tr></thead><tbody>{Object.entries(record(valueAt(reproducibility, "initial_model_configuration")) || {}).map(([name, item]) => {
          const config = record(item);
          return <tr key={name}><td>{name}</td><td>{textValue(valueAt(config, "model_class"))}</td><td>{textValue(valueAt(config, "model_version"))}</td><td><code>{textValue(valueAt(config, "configuration_sha256"))}</code></td><td><code>{textValue(valueAt(config, "source_sha256"))}</code></td></tr>;
        })}</tbody></table></div></details>}
        <p className="research-report-note">Fingerprints help detect configuration drift and support auditability; they do not guarantee bitwise reproducibility across hardware or authenticate market data.</p>
      </div>}
      {temporalAudit && <div className="research-lineage">
        <div><b>TEMPORAL INTEGRITY AUDIT</b><span>{valueAt(temporalAudit, "test_windows_overlap") === true ? "OVERLAPPING TEST WINDOWS" : "NO REPEATED TEST TIMESTAMPS"}</span></div>
        <p>Fold boundaries, label maturity at each training cutoff, and repeated out-of-sample timestamps.</p>
        <div className="research-report-metrics">
          <Metric label="FOLDS" value={textValue(valueAt(temporalAudit, "fold_count"))} note="Chronological partitions" />
          <Metric label="TEST OBSERVATIONS" value={textValue(valueAt(temporalAudit, "test_observations_across_folds"))} note="Counted across folds" />
          <Metric label="UNIQUE TEST TIMES" value={textValue(valueAt(temporalAudit, "unique_test_timestamps"))} note="Distinct timestamps" />
          <Metric label="REPEATED OBSERVATIONS" value={textValue(valueAt(temporalAudit, "repeated_test_timestamp_observations"))} note="Do not treat as independent" />
          <Metric label="UNMATURED TRAIN LABELS" value={textValue(valueAt(temporalAudit, "training_rows_excluded_unmatured_across_folds"))} note="Excluded at training cutoffs" />
        </div>
        <p className="research-report-note">{textValue(valueAt(temporalAudit, "interpretation"))}</p>
        <p className="research-report-note">Feature causality verified: {valueAt(temporalAudit, "feature_causality_verified") === true ? "yes" : "no"} · Provider authenticity verified: {valueAt(temporalAudit, "provider_authenticity_verified") === true ? "yes" : "no"}. Hash lineage does not prove either.</p>
        {Array.isArray(valueAt(temporalAudit, "folds")) && <details><summary>Inspect fold boundaries and purge counts</summary><div className="research-model-table-wrap"><table className="research-model-table"><thead><tr><th>FOLD</th><th>TRAIN START → END (UTC)</th><th>TRAIN ROWS</th><th>UNMATURED EXCLUDED</th><th>VALIDATION START → END</th><th>TEST START → END</th><th>PURGED</th></tr></thead><tbody>{(valueAt(temporalAudit, "folds") as unknown[]).map(item => {
          const fold = record(item);
          const train = record(valueAt(fold, "train"));
          const validation = record(valueAt(fold, "validation"));
          const test = record(valueAt(fold, "test"));
          return <tr key={textValue(valueAt(fold, "fold_index"))}><td>{textValue(valueAt(fold, "fold_index"))}</td><td>{textValue(valueAt(train, "start_utc"))}<br />{textValue(valueAt(train, "end_utc"))}</td><td>{textValue(valueAt(train, "rows_label_matured_at_cutoff"))} / {textValue(valueAt(train, "rows_available"))}</td><td>{textValue(valueAt(train, "rows_excluded_unmatured"))}</td><td>{textValue(valueAt(validation, "start_utc"))}<br />{textValue(valueAt(validation, "end_utc"))}</td><td>{textValue(valueAt(test, "start_utc"))}<br />{textValue(valueAt(test, "end_utc"))}</td><td>{textValue(valueAt(fold, "purged_rows"))}</td></tr>;
        })}</tbody></table></div></details>}
      </div>}
      <div className="research-report-metrics">
        <Metric label="SOURCE ROWS" value={textValue(valueAt(quality, "row_count") ?? valueAt(features, "input_rows"))} note="Raw OHLC rows" />
        <Metric label="INVALID ROWS" value={textValue(valueAt(quality, "invalid_row_count") ?? 0)} note="Structural validation" />
        <Metric label="DUPLICATE TIMES" value={textValue(valueAt(quality, "duplicate_timestamp_count") ?? 0)} note="Exact timestamp duplicates" />
        <Metric label="LONG INTERVALS" value={textValue(valueAt(quality, "long_interval_count") ?? 0)} note={quality ? `Nominal ${textValue(valueAt(quality, "expected_interval_minutes"))} min cadence` : "Quality audit unavailable"} />
        <Metric label="FEATURE ROWS" value={textValue(valueAt(features, "output_rows"))} note="After causal warm-up" />
        <Metric label="LABELED ROWS" value={textValue(valueAt(labels, "output_rows") ?? valueAt(labels, "labeled_rows"))} note="Matured outcomes only" />
      </div>
      {quality && <div className="research-report-note"><b>Cadence interpretation:</b> {warnings.map(textValue).join(" ") || "Cadence gaps require provider/session-aware review."}</div>}
      {selectionAudit && <div className="research-model-table-wrap">
        <h3>Validation-only model-selection audit</h3>
        <p className="research-report-note">Ranking uses median validation log-loss, then median validation Brier score, across complete common validation folds. Final-test metrics are excluded from ranking. This is a research audit, not a promotion rule or evidence of profitability.</p>
        <div className="research-report-metrics">
          <Metric label="PROTOCOL" value={textValue(valueAt(selectionAudit, "protocol"))} note="Validation partitions only" />
          <Metric label="STATUS" value={textValue(valueAt(selectionAudit, "status"))} note={`${textValue(valueAt(selectionAudit, "common_validation_fold_indices") instanceof Array ? (valueAt(selectionAudit, "common_validation_fold_indices") as unknown[]).length : 0)} complete common folds`} />
          <Metric label="SELECTED BY VALIDATION" value={textValue(valueAt(selectionAudit, "selected_candidate"))} note="Not an execution authorization" />
          <Metric label="TEST METRICS USED" value={valueAt(selectionAudit, "test_metrics_used_for_selection") === false ? "NO" : "REVIEW REQUIRED"} note="Must remain untouched" />
        </div>
        {Array.isArray(valueAt(selectionAudit, "ranking")) && (valueAt(selectionAudit, "ranking") as unknown[]).length > 0
          ? <table className="research-model-table"><thead><tr><th>RANK</th><th>CANDIDATE</th><th>VALIDATION FOLDS</th><th>MEDIAN LOG-LOSS</th><th>MEDIAN BRIER</th></tr></thead><tbody>{(valueAt(selectionAudit, "ranking") as unknown[]).map((item, index) => {
            const row = record(item);
            return <tr key={`selection-${index}`}><td>{textValue(valueAt(row, "rank"))}</td><td>{textValue(valueAt(row, "candidate_name"))}</td><td>{textValue(valueAt(row, "validation_fold_count"))}</td><td>{formatNumber(valueAt(row, "median_validation_log_loss"))}</td><td>{formatNumber(valueAt(row, "median_validation_brier_score"))}</td></tr>;
          })}</tbody></table>
          : <div className="research-report-empty">Selection withheld: insufficient complete common validation folds. At least two are required.</div>}
        <p className="research-report-note">{textValue(valueAt(selectionAudit, "interpretation"))}</p>
      </div>}
      {oosRobustness && <div className="research-model-table-wrap">
        <h3>Out-of-sample period robustness · post-selection diagnostic</h3>
        <p className="research-report-note">This read-only diagnostic divides common chronological test-fold indices into EARLY / MIDDLE / LATE groups and shows per-period medians plus worst-fold losses. It does not select or rank models. Fold groups are not proof of market regimes or independent samples; overlapping test windows may reuse observations. Do not tune against final-test results.</p>
        <div className="research-report-metrics">
          <Metric label="STATUS" value={textValue(valueAt(oosRobustness, "status"))} note="Three or more common test folds required" />
          <Metric label="COMMON TEST FOLDS" value={textValue(Array.isArray(valueAt(oosRobustness, "common_test_fold_indices")) ? (valueAt(oosRobustness, "common_test_fold_indices") as unknown[]).length : 0)} note="Fold indices, not raw independent samples" />
          <Metric label="MODEL SELECTION" value={valueAt(oosRobustness, "model_selection_performed") === false ? "NO" : "REVIEW REQUIRED"} note="Diagnostic only" />
        </div>
        {Array.isArray(valueAt(oosRobustness, "candidates")) && (valueAt(oosRobustness, "candidates") as unknown[]).length > 0
          ? (valueAt(oosRobustness, "candidates") as unknown[]).map((item, index) => {
            const candidate = record(item);
            const periods = Array.isArray(valueAt(candidate, "periods")) ? valueAt(candidate, "periods") as unknown[] : [];
            return <div key={"oos-period-" + index}>
              <h4>{textValue(valueAt(candidate, "candidate_name"))} · Brier range {formatNumber(valueAt(candidate, "test_brier_range"))} · log-loss range {formatNumber(valueAt(candidate, "test_log_loss_range"))}</h4>
              <table className="research-model-table"><thead><tr><th>PERIOD</th><th>FOLDS</th><th>MEDIAN TEST BRIER</th><th>WORST-FOLD BRIER</th><th>MEDIAN TEST LOG-LOSS</th><th>WORST-FOLD LOG-LOSS</th></tr></thead><tbody>{periods.map((periodItem, periodIndex) => {
                const period = record(periodItem);
                return <tr key={"oos-period-" + index + "-" + periodIndex}><td>{textValue(valueAt(period, "period"))}</td><td>{textValue(valueAt(period, "fold_count"))}</td><td>{formatNumber(valueAt(period, "median_test_brier_score"))}</td><td>{formatNumber(valueAt(period, "worst_fold_brier_score"))}</td><td>{formatNumber(valueAt(period, "median_test_log_loss"))}</td><td>{formatNumber(valueAt(period, "worst_fold_log_loss"))}</td></tr>;
              })}</tbody></table>
            </div>;
          })
          : <div className="research-report-empty">Period diagnostic withheld: at least three complete common test folds are required.</div>}
        <p className="research-report-note">{textValue(valueAt(oosRobustness, "interpretation"))}</p>
      </div>}
      {regimeUncertainty && <div className="research-model-table-wrap">
        <h3>Fold uncertainty &amp; declared regime stratification</h3>
        <p className="research-report-note">Quartiles and ranges describe fold-to-fold dispersion, not confidence intervals. Regime stratification appears only when each label has an explicit availability timestamp, a validity interval covering the actual test window, declared source/method, and SHA-256 source-snapshot fingerprint. Legacy string-only labels remain visible for audit but are withheld from stratification.</p>
        <div className="research-report-metrics">
          <Metric label="COMMON TEST FOLDS" value={textValue(Array.isArray(valueAt(regimeUncertainty, "common_complete_test_fold_indices")) ? (valueAt(regimeUncertainty, "common_complete_test_fold_indices") as unknown[]).length : 0)} note="Complete Brier/log-loss folds" />
          <Metric label="CONFIDENCE INTERVALS" value={valueAt(regimeUncertainty, "confidence_intervals_produced") === false ? "NOT PRODUCED" : "REVIEW REQUIRED"} note="IQR/range only" />
          <Metric label="REGIME LABELS" value={textValue(valueAt(record(valueAt(regimeUncertainty, "regime_annotations")), "status"))} note={textValue(valueAt(record(valueAt(regimeUncertainty, "regime_annotations")), "method"))} />
        </div>
        {Array.isArray(valueAt(regimeUncertainty, "uncertainty_envelopes")) && (valueAt(regimeUncertainty, "uncertainty_envelopes") as unknown[]).map((item, index) => {
          const row = record(item);
          const metrics = record(valueAt(row, "metrics"));
          return <div key={"uncertainty-" + index}>
            <h4>{textValue(valueAt(row, "candidate_name"))} · fold-level test metric dispersion</h4>
            <table className="research-model-table"><thead><tr><th>METRIC</th><th>FOLDS</th><th>MIN</th><th>Q1</th><th>MEDIAN</th><th>Q3</th><th>MAX</th><th>RANGE</th></tr></thead><tbody>{["brier_score", "log_loss", "accuracy_at_fixed_half", "expected_calibration_error"].map(metricName => {
              const metric = record(valueAt(metrics, metricName));
              return metric ? <tr key={metricName}><td>{metricName}</td><td>{textValue(valueAt(metric, "fold_count"))}</td><td>{formatNumber(valueAt(metric, "minimum"))}</td><td>{formatNumber(valueAt(metric, "q1"))}</td><td>{formatNumber(valueAt(metric, "median"))}</td><td>{formatNumber(valueAt(metric, "q3"))}</td><td>{formatNumber(valueAt(metric, "maximum"))}</td><td>{formatNumber(valueAt(metric, "range"))}</td></tr> : null;
            })}</tbody></table>
          </div>;
        })}
        {Array.isArray(valueAt(record(valueAt(regimeUncertainty, "regime_annotations")), "stratified_results")) && (valueAt(record(valueAt(regimeUncertainty, "regime_annotations")), "stratified_results") as unknown[]).length > 0
          ? <table className="research-model-table"><thead><tr><th>CANDIDATE</th><th>DECLARED REGIME</th><th>FOLDS</th><th>METRIC</th><th>MEDIAN</th><th>MIN</th><th>MAX</th></tr></thead><tbody>{(valueAt(record(valueAt(regimeUncertainty, "regime_annotations")), "stratified_results") as unknown[]).flatMap((item, index) => {
            const row = record(item);
            const metrics = record(valueAt(row, "metrics"));
            return ["brier_score", "log_loss"].map(metricName => {
              const metric = record(valueAt(metrics, metricName));
              return <tr key={"regime-" + index + "-" + metricName}><td>{textValue(valueAt(row, "candidate_name"))}</td><td>{textValue(valueAt(row, "regime"))}</td><td>{textValue(valueAt(row, "fold_count"))}</td><td>{metricName}</td><td>{formatNumber(valueAt(metric, "median"))}</td><td>{formatNumber(valueAt(metric, "minimum"))}</td><td>{formatNumber(valueAt(metric, "maximum"))}</td></tr>;
            });
          })}</tbody></table>
          : <div className="research-report-empty">No point-in-time-valid regime stratification available. Provide source/method, source_snapshot_sha256, and per-fold label/available_at_utc/valid_from_utc/valid_through_utc in manifest.fold_regime_annotations, with actual test boundaries. Groups with fewer than two common folds are withheld.</div>}
        <p className="research-report-note">{textValue(valueAt(regimeUncertainty, "interpretation"))}</p>
      </div>}
      {foldStability.length > 0 && <div className="research-model-table-wrap">
        <h3>Fold stability · direction consistency</h3>
        <p className="research-report-note">Counts and median candidate-minus-baseline differences are descriptive summaries across matching folds, not significance tests. Negative Brier/log-loss deltas favor the candidate. Overlapping test windows are not independent, and these summaries do not establish profitability.</p>
        <table className="research-model-table"><thead><tr><th>CANDIDATE</th><th>BASELINE</th><th>PARTITION</th><th>FOLDS</th><th>BRIER W/T/L</th><th>MEDIAN BRIER Δ</th><th>LOG-LOSS W/T/L</th><th>MEDIAN LOG-LOSS Δ</th></tr></thead><tbody>{foldStability.map((item, index) => {
          const metrics = record(valueAt(item, "metrics"));
          const brier = record(valueAt(metrics, "brier_score"));
          const logLoss = record(valueAt(metrics, "log_loss"));
          const mismatch = valueAt(item, "sample_count_mismatch") === true;
          return <tr key={`stability-${index}`}>
            <td>{textValue(valueAt(item, "candidate_name"))}</td><td>{textValue(valueAt(item, "baseline_name"))}</td>
            <td>{textValue(valueAt(item, "partition"))}</td><td>{textValue(valueAt(item, "fold_count"))}</td>
            <td>{mismatch ? "INCONSISTENT N" : `${textValue(valueAt(brier, "candidate_wins"))}/${textValue(valueAt(brier, "ties"))}/${textValue(valueAt(brier, "candidate_losses"))}`}</td>
            <td>{mismatch ? "—" : formatNumber(valueAt(brier, "median_candidate_minus_baseline"))}</td>
            <td>{mismatch ? "INCONSISTENT N" : `${textValue(valueAt(logLoss, "candidate_wins"))}/${textValue(valueAt(logLoss, "ties"))}/${textValue(valueAt(logLoss, "candidate_losses"))}`}</td>
            <td>{mismatch ? "—" : formatNumber(valueAt(logLoss, "median_candidate_minus_baseline"))}</td>
          </tr>;
        })}</tbody></table>
      </div>}
      {pairedComparisons.length > 0 && <div className="research-model-table-wrap">
        <h3>Paired out-of-sample comparison · candidate vs training-prevalence baseline</h3>
        <p className="research-report-note">Negative candidate-minus-baseline Brier/log-loss differences favor the candidate. An interval that crosses zero is not clear evidence of a difference. These are per-fold comparisons; overlapping test windows are not independent observations. Validation comparisons are diagnostic, and final-test comparisons must not be used for tuning.</p>
        <table className="research-model-table"><thead><tr><th>CANDIDATE</th><th>BASELINE</th><th>FOLD</th><th>PARTITION</th><th>N</th><th>BRIER Δ</th><th>BRIER CI</th><th>LOG LOSS Δ</th><th>LOG LOSS CI</th><th>INTERPRETATION</th></tr></thead><tbody>{pairedComparisons.map((item, index) => {
          const row = record(item);
          const comparison = record(valueAt(row, "comparison"));
          const brierLow = valueAt(comparison, "brier_difference_ci_low");
          const brierHigh = valueAt(comparison, "brier_difference_ci_high");
          const logLow = valueAt(comparison, "log_loss_difference_ci_low");
          const logHigh = valueAt(comparison, "log_loss_difference_ci_high");
          return <tr key={`${textValue(valueAt(row, "candidate_name"))}-${textValue(valueAt(row, "fold_index"))}-${textValue(valueAt(row, "partition"))}-${index}`}>
            <td>{textValue(valueAt(row, "candidate_name"))}</td><td>{textValue(valueAt(row, "baseline_name"))}</td>
            <td>{numberValue(valueAt(row, "fold_index")) === null || numberValue(valueAt(row, "fold_index"))! < 0 ? "—" : textValue(valueAt(row, "fold_index"))}</td>
            <td>{textValue(valueAt(row, "partition"))}</td><td>{textValue(valueAt(row, "sample_count"))}</td>
            <td>{formatNumber(valueAt(comparison, "brier_difference_candidate_minus_baseline"))}</td>
            <td>{formatNumber(brierLow)} … {formatNumber(brierHigh)}</td>
            <td>{formatNumber(valueAt(comparison, "log_loss_difference_candidate_minus_baseline"))}</td>
            <td>{formatNumber(logLow)} … {formatNumber(logHigh)}</td>
            <td>{textValue(valueAt(comparison, "interpretation"))}</td>
          </tr>;
        })}</tbody></table>
      </div>}
      {temporalCalibrationRows.length > 0 && <div className="research-model-table-wrap">
        <h3>Temporal calibration stability · chronological ECE</h3>
        <p className="research-report-note">Uses declared test-window boundaries to order each fold's validation/test ECE. Endpoint interval separation is descriptive only, not a significance test. Overlapping folds may not be independent; missing boundaries and small temporal coverage are not guessed.</p>
        <table className="research-model-table"><thead><tr><th>MODEL</th><th>PARTITION</th><th>PERIODS</th><th>FIRST ECE</th><th>LAST ECE</th><th>FIRST → LAST Δ</th><th>ECE RANGE</th><th>ADJACENT INTERVAL OVERLAP</th><th>STATUS</th></tr></thead><tbody>{temporalCalibrationRows.map((item, index) => {
          const row = record(item);
          return <tr key={"temporal-calibration-" + index}>
            <td>{textValue(valueAt(row, "model_name"))}</td><td>{textValue(valueAt(row, "partition"))}</td>
            <td>{textValue(valueAt(row, "period_count"))}</td>
            <td>{formatNumber(valueAt(row, "first_period_ece"))}</td><td>{formatNumber(valueAt(row, "last_period_ece"))}</td>
            <td>{formatNumber(valueAt(row, "first_to_last_ece_delta"))}</td><td>{formatNumber(valueAt(row, "ece_range"))}</td>
            <td>{textValue(valueAt(row, "adjacent_interval_overlap_count"))} / {textValue(valueAt(row, "adjacent_interval_comparison_count"))}</td>
            <td>{textValue(valueAt(row, "status"))}</td>
          </tr>;
        })}</tbody></table>
        <p className="research-report-note">{textValue(valueAt(temporalCalibration, "interpretation"))}</p>
      </div>}
      {blockLengthRows.length > 0 && <div className="research-model-table-wrap">
        <h3>Bootstrap uncertainty · block-length sensitivity</h3>
        <p className="research-report-note">Intervals are recomputed on the same paired loss differences using shorter, configured, and longer block lengths. Stable means interval sign classifications match only across the lengths shown; this is not proof of significance or robustness to every dependence structure. This diagnostic never selects a block length or model.</p>
        <table className="research-model-table"><thead><tr><th>CANDIDATE</th><th>BASELINE</th><th>FOLD</th><th>PARTITION</th><th>N</th><th>TESTED LENGTHS</th><th>SENSITIVITY</th><th>INTERVALS</th></tr></thead><tbody>{blockLengthRows.map((item, index) => {
          const row = record(item);
          const sensitivity = Array.isArray(valueAt(row, "sensitivity")) ? valueAt(row, "sensitivity") as unknown[] : [];
          return <tr key={"block-sensitivity-" + index}>
            <td>{textValue(valueAt(row, "candidate_name"))}</td><td>{textValue(valueAt(row, "baseline_name"))}</td>
            <td>{textValue(valueAt(row, "fold_index"))}</td><td>{textValue(valueAt(row, "partition"))}</td>
            <td>{textValue(valueAt(row, "sample_count"))}</td>
            <td>{Array.isArray(valueAt(row, "tested_block_lengths")) ? (valueAt(row, "tested_block_lengths") as unknown[]).join(", ") : "—"}</td>
            <td>{textValue(valueAt(row, "status"))}</td>
            <td><details><summary>View intervals</summary>{sensitivity.map((entry, entryIndex) => {
              const result = record(entry);
              return <div key={"block-interval-" + entryIndex}><b>Block {textValue(valueAt(result, "block_length"))}</b>
                <div>Brier: {formatNumber(valueAt(result, "brier_difference_ci_low"))} … {formatNumber(valueAt(result, "brier_difference_ci_high"))} ({textValue(valueAt(result, "brier_direction"))})</div>
                <div>Log loss: {formatNumber(valueAt(result, "log_loss_difference_ci_low"))} … {formatNumber(valueAt(result, "log_loss_difference_ci_high"))} ({textValue(valueAt(result, "log_loss_direction"))})</div>
              </div>;
            })}</details></td>
          </tr>;
        })}</tbody></table>
        <p className="research-report-note">{textValue(valueAt(blockLengthAudit, "interpretation"))}</p>
      </div>}
      {modelRows.length > 0 ? <div className="research-model-table-wrap"><h3>Walk-forward results · per-fold metrics</h3><table className="research-model-table"><thead><tr><th>MODEL</th><th>FOLDS</th><th>TEST SAMPLES</th><th>MEAN BRIER</th><th>MEAN LOG LOSS</th><th>MEAN ACCURACY</th><th>MEAN ECE</th><th>FOLD ECE CI ENVELOPE</th></tr></thead><tbody>{modelRows.map(model => {
        const tests = model.folds.map(fold => record(record(fold)?.test)).filter((item): item is JsonRecord => item !== null);
        const average = (key: string) => {
          const values = tests.map(item => numberValue(item[key])).filter((item): item is number => item !== null);
          return values.length ? values.reduce((sum, item) => sum + item, 0) / values.length : null;
        };
        const samples = tests.reduce((sum, item) => sum + (numberValue(item.sample_count) ?? 0), 0);
        return <tr key={model.name}><td>{model.name}</td><td>{model.folds.length}</td><td>{samples || "—"}</td><td>{formatNumber(average("brier_score"))}</td><td>{formatNumber(average("log_loss"))}</td><td>{formatNumber(average("accuracy_at_fixed_half"))}</td><td>{formatNumber(average("expected_calibration_error"))}</td><td>{(() => {
          const intervals = tests.map(item => record(item.calibration_uncertainty))
            .filter((item): item is JsonRecord => item !== null)
            .filter(item => numberValue(item.interval_low) !== null && numberValue(item.interval_high) !== null);
          if (!intervals.length) return "WITHHELD / LEGACY";
          const lows = intervals.map(item => numberValue(item.interval_low)!).filter(Number.isFinite);
          const highs = intervals.map(item => numberValue(item.interval_high)!).filter(Number.isFinite);
          return lows.length && highs.length ? `${formatNumber(Math.min(...lows))} … ${formatNumber(Math.max(...highs))}` : "WITHHELD / LEGACY";
        })()}</td></tr>;
      })}</tbody></table><p className="research-report-note">Means are descriptive summaries of fold-level metrics, not pooled out-of-sample scores. ECE depends on bin count and sample size and is not a calibration guarantee. The fold ECE CI envelope is the minimum lower bound through maximum upper bound across available per-fold moving-block bootstrap intervals; it is not a pooled confidence interval. Small folds and legacy reports may withhold intervals. Review fold definitions, baseline comparisons, label semantics and data provenance before interpreting results.</p></div> : <div className="research-report-empty">No per-fold model results were found in this file. Load the combined pipeline JSON or the evaluation JSON produced by the historical research CLI.</div>}
      <div className="research-report-safety"><b>RESEARCH ONLY · NO EXECUTION AUTHORITY</b><span>Predictive metrics do not establish profitability. No model is promoted or connected to order execution by this viewer.</span></div>
    </>}
  </section>;
}

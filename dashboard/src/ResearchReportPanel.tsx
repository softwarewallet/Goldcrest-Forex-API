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

  const quality = record(valueAt(report, "data_quality_audit"));
  const evaluationContract = record(valueAt(evaluation, "evaluation_contract"));
  const manifest = record(valueAt(evaluation, "manifest"));
  const provenance = record(valueAt(evaluationContract, "dataset_provenance"))
    || record(valueAt(manifest, "dataset_provenance"));
  const features = record(valueAt(report, "feature_preparation"));
  const labels = record(valueAt(report, "label_preparation"));
  const evaluation = record(valueAt(report, "evaluation")) || report;
  const foldResults = record(valueAt(evaluation, "fold_results"));
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
      <div className="research-report-metrics">
        <Metric label="SOURCE ROWS" value={textValue(valueAt(quality, "row_count") ?? valueAt(features, "input_rows"))} note="Raw OHLC rows" />
        <Metric label="INVALID ROWS" value={textValue(valueAt(quality, "invalid_row_count") ?? 0)} note="Structural validation" />
        <Metric label="DUPLICATE TIMES" value={textValue(valueAt(quality, "duplicate_timestamp_count") ?? 0)} note="Exact timestamp duplicates" />
        <Metric label="LONG INTERVALS" value={textValue(valueAt(quality, "long_interval_count") ?? 0)} note={quality ? `Nominal ${textValue(valueAt(quality, "expected_interval_minutes"))} min cadence` : "Quality audit unavailable"} />
        <Metric label="FEATURE ROWS" value={textValue(valueAt(features, "output_rows"))} note="After causal warm-up" />
        <Metric label="LABELED ROWS" value={textValue(valueAt(labels, "output_rows") ?? valueAt(labels, "labeled_rows"))} note="Matured outcomes only" />
      </div>
      {quality && <div className="research-report-note"><b>Cadence interpretation:</b> {warnings.map(textValue).join(" ") || "Cadence gaps require provider/session-aware review."}</div>}
      {modelRows.length > 0 ? <div className="research-model-table-wrap"><h3>Walk-forward results · per-fold metrics</h3><table className="research-model-table"><thead><tr><th>MODEL</th><th>FOLDS</th><th>TEST SAMPLES</th><th>MEAN BRIER</th><th>MEAN LOG LOSS</th><th>MEAN ACCURACY</th><th>MEAN ECE</th></tr></thead><tbody>{modelRows.map(model => {
        const tests = model.folds.map(fold => record(record(fold)?.test)).filter((item): item is JsonRecord => item !== null);
        const average = (key: string) => {
          const values = tests.map(item => numberValue(item[key])).filter((item): item is number => item !== null);
          return values.length ? values.reduce((sum, item) => sum + item, 0) / values.length : null;
        };
        const samples = tests.reduce((sum, item) => sum + (numberValue(item.sample_count) ?? 0), 0);
        return <tr key={model.name}><td>{model.name}</td><td>{model.folds.length}</td><td>{samples || "—"}</td><td>{formatNumber(average("brier_score"))}</td><td>{formatNumber(average("log_loss"))}</td><td>{formatNumber(average("accuracy_at_fixed_half"))}</td><td>{formatNumber(average("expected_calibration_error"))}</td></tr>;
      })}</tbody></table><p className="research-report-note">Means are descriptive summaries of fold-level metrics, not pooled out-of-sample scores. ECE depends on bin count and sample size and is not a calibration guarantee. Review fold definitions, baseline comparisons, label semantics and data provenance before interpreting results.</p></div> : <div className="research-report-empty">No per-fold model results were found in this file. Load the combined pipeline JSON or the evaluation JSON produced by the historical research CLI.</div>}
      <div className="research-report-safety"><b>RESEARCH ONLY · NO EXECUTION AUTHORITY</b><span>Predictive metrics do not establish profitability. No model is promoted or connected to order execution by this viewer.</span></div>
    </>}
  </section>;
}

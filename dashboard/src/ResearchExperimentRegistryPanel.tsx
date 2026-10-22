import { useCallback, useEffect, useState } from "react";

type FoldResult = Record<string, unknown>;
type Experiment = {
  run_id: string;
  fingerprint_sha256: string;
  dataset_sha256: string;
  symbol: string;
  timeframe: string;
  protocol_version: string;
  manifest: Record<string, unknown>;
  fold_results: Array<{ experiment_id: string; model_type: string; feature_variant: string; fold_results: FoldResult[] }>;
  created_at: string;
  actionable: false;
  automatic_winner_selection: false;
};
type RegistryResponse = { count: number; runs: Experiment[]; actionable: false; note?: string };

const shortHash = (value: string) => value ? value.slice(0, 12) + "…" : "—";
const formatDate = (value: string) => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};
const json = (value: unknown) => JSON.stringify(value, null, 2);

export default function ResearchExperimentRegistryPanel({ base, apiKey }: { base: string; apiKey: string }) {
  const [runs, setRuns] = useState<Experiment[]>([]);
  const [selected, setSelected] = useState<Experiment | null>(null);
  const [symbol, setSymbol] = useState("");
  const [timeframe, setTimeframe] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    const query = new URLSearchParams({ limit: "100" });
    if (symbol.trim()) query.set("symbol", symbol.trim().toUpperCase());
    if (timeframe.trim()) query.set("timeframe", timeframe.trim().toUpperCase());
    try {
      const response = await fetch(`${base.replace(/\/$/, "")}/api/v1/research/experiments?${query}`, {
        headers: apiKey ? { "X-API-Key": apiKey } : {},
        signal: AbortSignal.timeout(9000),
      });
      if (!response.ok) {
        const body = await response.text().catch(() => "");
        throw new Error(`${response.status} ${response.statusText}${body ? `: ${body.slice(0, 160)}` : ""}`);
      }
      const data = await response.json() as RegistryResponse;
      setRuns(Array.isArray(data.runs) ? data.runs : []);
      setSelected(current => current ? (data.runs || []).find(run => run.run_id === current.run_id) || null : null);
      setLoaded(true);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Experiment registry could not be loaded.");
      setRuns([]);
      setSelected(null);
      setLoaded(false);
    } finally {
      setLoading(false);
    }
  }, [base, apiKey, symbol, timeframe]);

  useEffect(() => { void load(); }, [load]);

  const inspect = async (runId: string) => {
    setError("");
    try {
      const response = await fetch(`${base.replace(/\/$/, "")}/api/v1/research/experiments/${encodeURIComponent(runId)}`, {
        headers: apiKey ? { "X-API-Key": apiKey } : {},
        signal: AbortSignal.timeout(9000),
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      setSelected(await response.json() as Experiment);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Experiment details could not be loaded.");
    }
  };

  return <section className="experiment-registry-panel">
    <div className="section-heading">
      <div><div className="eyebrow">PROVENANCE / REPRODUCIBILITY</div><h2>Experiment registry</h2></div>
      <span className="research-tag">OFFLINE · READ-ONLY</span>
    </div>
    <p className="registry-intro">Inspect persisted research runs, dataset identity, protocol versions and fold-level artifacts. This registry does not select or promote models.</p>
    <div className="registry-controls">
      <label>Symbol<input value={symbol} maxLength={32} placeholder="All symbols" onChange={event => setSymbol(event.target.value)} /></label>
      <label>Timeframe<input value={timeframe} maxLength={8} placeholder="All timeframes" onChange={event => setTimeframe(event.target.value)} /></label>
      <button onClick={() => void load()} disabled={loading}>{loading ? "LOADING…" : "REFRESH REGISTRY"}</button>
    </div>
    {error && <div className="comparison-message error" role="alert">Registry request failed: {error}</div>}
    {!error && loading && !loaded && <div className="comparison-message">Connecting to the authenticated experiment registry…</div>}
    {!error && loaded && runs.length === 0 && <div className="comparison-message">No registered experiments match these filters. No sample runs are fabricated.</div>}
    {runs.length > 0 && <div className="registry-layout">
      <div className="registry-run-list" aria-label="Registered experiments">
        <div className="registry-list-count">{runs.length} RUNS RETURNED · MAX 100</div>
        {runs.map(run => <button key={run.run_id} className={selected?.run_id === run.run_id ? "registry-run selected" : "registry-run"} onClick={() => void inspect(run.run_id)}>
          <span className="registry-run-top"><b>{run.symbol} <small>{run.timeframe}</small></b><i>{formatDate(run.created_at)}</i></span>
          <span className="registry-run-id">{run.run_id}</span>
          <span className="registry-run-meta"><span>PROTOCOL {run.protocol_version}</span><span>FINGERPRINT {shortHash(run.fingerprint_sha256)}</span></span>
        </button>)}
      </div>
      <div className="registry-detail">
        {!selected ? <div className="comparison-message">Select a run to inspect its manifest and fold results.</div> : <>
          <div className="registry-detail-head"><div><span className="eyebrow">SELECTED RUN</span><h3>{selected.run_id}</h3></div><span className="research-tag">NON-ACTIONABLE</span></div>
          <div className="registry-hashes"><div><small>DATASET SHA-256</small><code>{selected.dataset_sha256}</code></div><div><small>MANIFEST FINGERPRINT</small><code>{selected.fingerprint_sha256}</code></div></div>
          <div className="registry-fold-summary"><span><b>{selected.fold_results.reduce((sum, item) => sum + item.fold_results.length, 0)}</b><small>FOLD RECORDS</small></span><span><b>{selected.fold_results.length}</b><small>EXPERIMENTS</small></span><span><b>{selected.protocol_version}</b><small>PROTOCOL</small></span></div>
          <h4>Fold-by-fold artifacts</h4>
          {selected.fold_results.length === 0 ? <div className="comparison-message">This run has no experiment fold records.</div> : selected.fold_results.map((experiment, index) => <details className="registry-experiment" key={`${experiment.experiment_id}-${index}`} open={index === 0}><summary><b>{experiment.model_type || "Unspecified model"}</b><span>{experiment.experiment_id || `Experiment ${index + 1}`}</span><small>{experiment.feature_variant || "Feature variant not recorded"} · {experiment.fold_results.length} folds</small></summary><pre>{json(experiment.fold_results)}</pre></details>)}
          <details className="registry-manifest"><summary>Full manifest JSON</summary><pre>{json(selected.manifest)}</pre></details>
          <p className="comparison-disclaimer">Content hashes establish identity, not correctness, absence of leakage, statistical validity or profitability. Fold metrics are submitted research artifacts and are not independently recomputed here.</p>
        </>}
      </div>
    </div>}
    <p className="comparison-disclaimer">All registry access is read-only in this panel. No broker, Auto Live, order execution or risk-control routes are called.</p>
  </section>;
}

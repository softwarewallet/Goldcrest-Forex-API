import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import ResearchReportPanel from "./ResearchReportPanel";

type Health = { status: string; application: string; inference_status: string; database_status: string };
type Performance = { predictions_total: number; outcomes_recorded: number; note?: string };
type Prediction = {
  prediction_id: string; request_id: string; symbol: string; timeframe: string;
  decision: "BUY" | "SELL" | "NO_TRADE"; direction?: string | null; probability?: number | null;
  market_regime: string; reason_codes: string[]; model_id?: string | null; generated_at: string; expires_at: string;
};
type AuditEvent = { id: string; event_type: string; request_id?: string | null; symbol?: string | null; outcome?: string | null; duration_ms?: number | null; details?: Record<string, unknown>; created_at: string };
type LoadState = "loading" | "connected" | "degraded" | "offline";
type ProbeState = "not-run" | "running" | "passed" | "failed";
type EndpointStatus = "unknown" | "ok" | "failed";
type EndpointId = "health" | "performance" | "predictions" | "audit" | "models";
type EndpointStatuses = Record<EndpointId, EndpointStatus>;
type EndpointLastSuccess = Record<EndpointId, string | undefined>;
type ModelInventory = { inference_server?: string; configured_model_id?: string | null; models: Array<{ id: string; owned_by?: string | null }> };

const initialBase = localStorage.getItem("goldcrest-api-base") || "http://192.168.29.80:8090";
const initialKey = sessionStorage.getItem("goldcrest-api-key") || "";
const stamp = (value?: string) => value ? new Date(value).toLocaleString() : "—";
const shortId = (value?: string) => value ? value.slice(0, 8).toUpperCase() : "—";

async function request<T>(base: string, key: string, path: string): Promise<T> {
  const response = await fetch(`${base.replace(/\/$/, "")}${path}`, {
    headers: key ? { "X-API-Key": key } : {},
    signal: AbortSignal.timeout(9000),
  });
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText}${body ? `: ${body.slice(0, 140)}` : ""}`);
  }
  return response.json() as Promise<T>;
}

function Icon({ name }: { name: string }) {
  const paths: Record<string, string> = {
    pulse: "M2 12h4l3-8 5 16 3-8h5", network: "M12 3v6m0 6v6M3 12h6m6 0h6M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6ZM12 3a1 1 0 1 0 0 .01ZM3 12a1 1 0 1 0 0 .01ZM21 12a1 1 0 1 0 0 .01ZM12 21a1 1 0 1 0 0 .01Z",
    database: "M4 5c0-1.1 3.6-2 8-2s8 .9 8 2-3.6 2-8 2-8-.9-8-2Zm0 0v14c0 1.1 3.6 2 8 2s8-.9 8-2V5M4 12c0 1.1 3.6 2 8 2s8-.9 8-2",
    activity: "M3 12h4l3-7 4 14 3-7h4",
    arrow: "M5 12h14m-6-6 6 6-6 6",
    shield: "M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Zm-4-11 3 3 5-5",
    clock: "M12 8v4l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  };
  return <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name] || paths.pulse} /></svg>;
}

function StatusPill({ state, children }: { state: "good" | "warn" | "bad" | "muted"; children: ReactNode }) {
  return <span className={`status-pill ${state}`}><i />{children}</span>;
}

export default function App() {
  const [base, setBase] = useState(initialBase);
  const [key, setKey] = useState(initialKey);
  const [baseInput, setBaseInput] = useState(initialBase);
  const [keyInput, setKeyInput] = useState(initialKey);
  const [showSettings, setShowSettings] = useState(!initialKey);
  const [health, setHealth] = useState<Health | null>(null);
  const [performance, setPerformance] = useState<Performance | null>(null);
  const [modelInventory, setModelInventory] = useState<ModelInventory | null>(null);
  const [modelInventoryError, setModelInventoryError] = useState("");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [endpointStatuses, setEndpointStatuses] = useState<EndpointStatuses>({ health: "unknown", performance: "unknown", predictions: "unknown", audit: "unknown", models: "unknown" });
  const [endpointLastSuccess, setEndpointLastSuccess] = useState<EndpointLastSuccess>({ health: undefined, performance: undefined, predictions: undefined, audit: undefined, models: undefined });
  const [busy, setBusy] = useState(false);
  const refreshInFlight = useRef(false);
  const [probeState, setProbeState] = useState<ProbeState>("not-run");
  const [probeModel, setProbeModel] = useState("");
  const [probeError, setProbeError] = useState("");
  const [lastSync, setLastSync] = useState<string>();
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<Prediction | null>(null);
  const [activePanel, setActivePanel] = useState<"signals" | "audit" | "models" | "research" | null>(null);

  const refresh = useCallback(async () => {
    if (refreshInFlight.current) return;
    refreshInFlight.current = true;
    setBusy(true);
    try {
      const endpoints = await Promise.allSettled([
        request<Health>(base, key, "/api/v1/health"),
        request<Performance>(base, key, "/api/v1/performance"),
        request<Prediction[]>(base, key, "/api/v1/predictions?limit=30"),
        request<AuditEvent[]>(base, key, "/api/v1/audit?limit=12"),
        request<ModelInventory>(base, key, "/api/v1/models"),
      ]);
      const [h, p, signals, audit, inventory] = endpoints;
      const checkedAt = new Date().toISOString();
      setEndpointStatuses({ health: h.status === "fulfilled" ? "ok" : "failed", performance: p.status === "fulfilled" ? "ok" : "failed", predictions: signals.status === "fulfilled" ? "ok" : "failed", audit: audit.status === "fulfilled" ? "ok" : "failed", models: inventory.status === "fulfilled" ? "ok" : "failed" });
      setEndpointLastSuccess(previous => ({ ...previous, ...(h.status === "fulfilled" ? { health: checkedAt } : {}), ...(p.status === "fulfilled" ? { performance: checkedAt } : {}), ...(signals.status === "fulfilled" ? { predictions: checkedAt } : {}), ...(audit.status === "fulfilled" ? { audit: checkedAt } : {}), ...(inventory.status === "fulfilled" ? { models: checkedAt } : {}) }));
      const failures: string[] = [];
      if (h.status === "fulfilled") {
        setHealth(h.value);
        setState(h.value.status === "ok" && h.value.inference_status === "ok" && h.value.database_status === "ok" ? "connected" : "degraded");
      } else {
        failures.push(`health: ${h.reason instanceof Error ? h.reason.message : "request failed"}`);
      }
      if (p.status === "fulfilled") setPerformance(p.value);
      else failures.push(`performance: ${p.reason instanceof Error ? p.reason.message : "request failed"}`);
      if (signals.status === "fulfilled") setPredictions(signals.value);
      else failures.push(`predictions: ${signals.reason instanceof Error ? signals.reason.message : "request failed"}`);
      if (audit.status === "fulfilled") setEvents(audit.value);
      else failures.push(`audit: ${audit.reason instanceof Error ? audit.reason.message : "request failed"}`);
      if (inventory.status === "fulfilled") {
        setModelInventory(inventory.value);
        setModelInventoryError("");
      } else {
        setModelInventoryError(inventory.reason instanceof Error ? inventory.reason.message : "Model inventory unavailable");
        failures.push(`model inventory: ${inventory.reason instanceof Error ? inventory.reason.message : "request failed"}`);
      }
      if (failures.length) {
        setError(failures.join(" · "));
        if (h.status === "rejected") setState("offline");
        else setState("degraded");
      } else {
        setError("");
        setLastSync(new Date().toISOString());
      }
    } catch (e) {
      setState("offline");
      setError(e instanceof Error ? e.message : "API request failed");
    } finally {
      refreshInFlight.current = false;
      setBusy(false);
    }
  }, [base, key]);

  useEffect(() => {
    let active = true;
    let timer = 0;
    const poll = async () => {
      await refresh();
      if (active) timer = window.setTimeout(() => void poll(), 15000);
    };
    void poll();
    return () => { active = false; window.clearTimeout(timer); };
  }, [refresh]);

  const latestSignals = useMemo(() => predictions.slice(0, 6), [predictions]);
  const probeInference = async () => {
    setProbeState("running"); setProbeError(""); setProbeModel("");
    try {
      const result = await request<{ status: string; model_id: string; completion_valid: boolean }>(base, key, "/api/v1/readiness/inference");
      if (result.status !== "ok" || !result.completion_valid) throw new Error("Inference readiness response was invalid");
      setProbeState("passed"); setProbeModel(result.model_id || "Model ID not reported");
    } catch (e) {
      setProbeState("failed"); setProbeError(e instanceof Error ? e.message : "Inference probe failed");
    }
  };
  const saveSettings = () => {
    const normalized = baseInput.trim().replace(/\/$/, "");
    const nextBase = normalized || "http://192.168.29.80:8090";
    try {
      const parsed = new URL(nextBase);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") throw new Error("Only HTTP or HTTPS API URLs are supported.");
      if (parsed.username || parsed.password) throw new Error("Do not embed credentials in the API URL.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Enter a valid API URL.");
      setShowSettings(true);
      return;
    }
    setError("");
    setBase(nextBase);
    setKey(keyInput);
    localStorage.setItem("goldcrest-api-base", nextBase);
    sessionStorage.setItem("goldcrest-api-key", keyInput);
    setShowSettings(false);
  };
  const signalTone = (decision: string) => decision === "BUY" ? "buy" : decision === "SELL" ? "sell" : "neutral";
  const stateLabel = state === "connected" ? "SYSTEM ONLINE" : state === "degraded" ? "DEGRADED" : state === "loading" ? "CONNECTING" : "DISCONNECTED";

  return <div className="app-shell">
    <main className="main">
      <header className="topbar"><div className="brand compact-brand"><div className="brand-mark"><span/><span/><span/></div><div><b>GOLDCREST<span>AI</span></b><small>NEURAL INTELLIGENCE</small></div></div><nav className="command-nav" aria-label="Dashboard panels"><button className={activePanel === "signals" ? "nav-selected" : ""} onClick={() => setActivePanel("signals")}>SIGNAL LEDGER <span>{performance?.predictions_total ?? "—"}</span></button><button className={activePanel === "audit" ? "nav-selected" : ""} onClick={() => setActivePanel("audit")}>RESEARCH AUDIT</button><button className={activePanel === "models" ? "nav-selected" : ""} onClick={() => setActivePanel("models")}>MODEL CORE</button><button className={activePanel === "research" ? "nav-selected" : ""} onClick={() => setActivePanel("research")}>DATA LAB</button></nav><div className="top-actions"><span className="clock-label"><Icon name="clock" /> {lastSync ? new Date(lastSync).toLocaleTimeString() : "WAITING FOR SYNC"}</span><StatusPill state={state === "connected" ? "good" : state === "degraded" ? "warn" : state === "offline" ? "bad" : "muted"}>{stateLabel}</StatusPill><button className="icon-button" onClick={() => setShowSettings(true)} aria-label="Connection settings">⚙</button><button className="refresh-button compact-refresh" onClick={() => void refresh()} disabled={busy}>{busy ? "SYNCING" : "↻ SYNC"}</button></div></header>

      <div className="command-title"><div><span className="eyebrow"><span className="eyebrow-line"/> AI RESEARCH ENVIRONMENT / LIVE NEURAL FIELD</span><h1>Neural <span>Intelligence Core</span></h1></div><div className="command-actions"><button className="probe-button" onClick={() => void probeInference()} disabled={probeState === "running"}><Icon name="pulse"/>{probeState === "running" ? "INFERENCE RUNNING" : "TEST INFERENCE"}</button><span className="research-only-badge">RESEARCH ONLY · NO ORDER ROUTES</span></div></div>
      {showSettings && <div className="overlay-backdrop" role="presentation" onClick={() => setShowSettings(false)}><section className="overlay-panel settings-overlay" role="dialog" aria-modal="true" aria-label="Connection settings" onClick={e => e.stopPropagation()}><div className="overlay-heading"><div><span className="eyebrow">SYSTEM CONFIGURATION</span><h2>Connect to GoldcrestAI API</h2></div><button className="modal-close" onClick={() => setShowSettings(false)} aria-label="Close settings">×</button></div><p className="overlay-description">Connect only to an API endpoint you trust. The API key is held in session storage and sent only to the configured endpoint.</p><label>API BASE URL<input value={baseInput} onChange={e => setBaseInput(e.target.value)} placeholder="http://127.0.0.1:8090"/></label><label>API KEY<input value={keyInput} onChange={e => setKeyInput(e.target.value)} type="password" placeholder="Enter X-API-Key" autoComplete="off"/></label>{error && <div className="overlay-error">{error}</div>}<div className="overlay-actions"><button className="refresh-button" onClick={() => setShowSettings(false)}>CANCEL</button><button className="probe-button" onClick={saveSettings}>SAVE & CONNECT</button></div></section></div>}
      {probeState !== "not-run" && <div className={`probe-banner ${probeState}`}><span>{probeState === "passed" ? "✓" : probeState === "failed" ? "!" : "…"}</span><div><b>{probeState === "passed" ? "Actual model inference probe passed" : probeState === "failed" ? "Actual model inference probe failed" : "Running a real model completion…"}</b><p>{probeState === "passed" ? `Validated JSON completion from ${probeModel}. This confirms connectivity and response format only—not prediction quality.` : probeState === "failed" ? probeError : "Waiting for the inference endpoint to complete."}</p></div><button onClick={() => { setProbeState("not-run"); setProbeError(""); }}>DISMISS</button></div>}
      {error && <div className="error-banner" role="alert"><span>!</span><div><b>{state === "degraded" ? "Some research endpoints are unavailable" : "Research API unavailable"}</b><p>{error}</p><small>{state === "degraded" ? "Successful endpoints are refreshed independently. Failed sections may still show the last retrieved data." : "Check the API address, network access, and key. Existing signal data is not presented as live."}</small></div><button onClick={() => setShowSettings(true)}>CONNECTION SETTINGS</button></div>}{state === "offline" && lastSync && <div className="stale-banner" role="status"><Icon name="clock" /><div><b>Showing last successful snapshot</b><p>Live refresh failed. The records and counters below may be stale; last successful sync was {stamp(lastSync)}. Refresh or restore the API connection to update them.</p></div></div>}

      <section className={`neural-stage ${busy || probeState === "running" ? "neural-stage-active" : ""}`} aria-label="AI research data flow">
        <div className="neural-stage-head">
          <div><div className="eyebrow"><span className="eyebrow-line" /> COGNITIVE PROCESSING FIELD</div><h2>From market data <span>to research intelligence</span></h2><p>Live system state visualized across the research pipeline. No synthetic market signals are generated.</p></div>
          <div className="neural-stage-state"><i className={state === "connected" ? "state-on" : ""}/><span>{state === "connected" ? "API LINK ESTABLISHED" : state === "degraded" ? "PARTIAL API LINK" : state === "loading" ? "ESTABLISHING LINK" : "AWAITING API CONNECTION"}</span></div>
        </div>
        <div className="neural-flow">
          <div className="neural-column inputs-column">
            <div className="neural-column-label"><span>01 / INPUT LAYER</span><b>Market context</b></div>
            <div className="neural-module"><span className="module-symbol">⌁</span><div><b>Price & candles</b><small>OHLC history</small></div><em className={state === "connected" ? "module-ok" : ""}>{state === "connected" ? "LINKED" : "WAITING"}</em></div>
            <div className="neural-module"><span className="module-symbol">∿</span><div><b>Technical features</b><small>Chronological features</small></div><em>RESEARCH</em></div>
            <div className="neural-module"><span className="module-symbol">◌</span><div><b>News & sentiment</b><small>Only when provided</small></div><em>OPTIONAL</em></div>
          </div>
          <div className="neural-bridge bridge-in"><span/><span/><span/><span/><span/></div>
          <div className="neural-core-column">
            <div className="core-network-backdrop" aria-hidden="true">
              <svg viewBox="0 0 520 350" preserveAspectRatio="xMidYMid meet">
                <defs><linearGradient id="traceGradient"><stop offset="0" stopColor="#42d9ed" stopOpacity=".08"/><stop offset=".5" stopColor="#8c79ff" stopOpacity=".8"/><stop offset="1" stopColor="#49e3cb" stopOpacity=".12"/></linearGradient><radialGradient id="coreGlow"><stop offset="0" stopColor="#3addeb" stopOpacity=".2"/><stop offset="1" stopColor="#3addeb" stopOpacity="0"/></radialGradient></defs>
                <ellipse cx="260" cy="166" rx="230" ry="142" fill="url(#coreGlow)"/>
                <g className="network-traces" fill="none" stroke="url(#traceGradient)" strokeWidth="1">
                  <path d="M18 60 L100 105 L164 78 L225 128 L260 166 L310 115 L370 80 L500 45"/><path d="M8 150 L75 150 L140 185 L210 168 L260 166 L326 200 L408 165 L510 180"/><path d="M38 280 L112 230 L168 256 L222 205 L260 166 L308 242 L375 260 L488 292"/><path d="M100 20 L130 92 L112 230"/><path d="M420 28 L370 80 L408 165 L375 260"/><path d="M75 150 L100 105 L164 78 L310 115 L370 80"/><path d="M140 185 L168 256 L308 242 L326 200 L408 165"/>
                </g>
                <g className="network-nodes" fill="#55e8ef">
                  <circle cx="18" cy="60" r="3"/><circle cx="100" cy="105" r="4"/><circle cx="164" cy="78" r="3"/><circle cx="225" cy="128" r="4"/><circle cx="310" cy="115" r="4"/><circle cx="370" cy="80" r="3"/><circle cx="500" cy="45" r="3"/><circle cx="8" cy="150" r="3"/><circle cx="75" cy="150" r="4"/><circle cx="140" cy="185" r="3"/><circle cx="326" cy="200" r="4"/><circle cx="408" cy="165" r="3"/><circle cx="510" cy="180" r="3"/><circle cx="38" cy="280" r="3"/><circle cx="112" cy="230" r="4"/><circle cx="168" cy="256" r="3"/><circle cx="308" cy="242" r="4"/><circle cx="375" cy="260" r="3"/><circle cx="488" cy="292" r="3"/><circle cx="100" cy="20" r="3"/><circle cx="420" cy="28" r="3"/>
                </g>
              </svg>
            </div>
            <div className="core-orbit orbit-outer"><i/><i/><i/></div><div className="core-orbit orbit-middle"><i/><i/></div><div className="core-orbit orbit-inner"><div className="core-glyph"><span/><span/><span/><span/><span/></div></div>
            <div className="core-label">GOLDCREST <b>NEURAL CORE</b></div>
            <div className="core-status"><i className={busy || probeState === "running" ? "state-on" : ""}/>{probeState === "running" ? "INFERENCE REQUEST IN FLIGHT" : busy ? "POLLING LIVE API STATE" : probeState === "passed" ? "INFERENCE PROBE PASSED" : probeState === "failed" ? "INFERENCE PROBE FAILED" : "IDLE · AWAITING REAL INPUT"}</div>
            <div className="core-subsystems"><span>FEATURE EXTRACTION</span><span>MODEL INFERENCE</span><span>VALIDATION</span></div>
          </div>
          <div className="neural-bridge bridge-out"><span/><span/><span/><span/><span/></div>
          <div className="neural-column outputs-column">
            <div className="neural-column-label"><span>03 / OUTPUT LAYER</span><b>Research intelligence</b></div>
            <div className="output-summary"><small>STORED PREDICTIONS</small><strong>{performance?.predictions_total ?? "—"}</strong><span>Recorded by API</span></div>
            <div className="output-summary"><small>OBSERVED OUTCOMES</small><strong>{performance?.outcomes_recorded ?? "—"}</strong><span>Labels received</span></div>
            <div className="output-signal"><span className="output-pulse"/><div><b>{predictions.length ? "Signal ledger populated" : "No stored signals yet"}</b><small>{predictions.length ? `${predictions.length} records retrieved from API` : "Outputs appear after API records them"}</small></div></div>
          </div>
        </div>
        <div className="neural-stage-foot"><span><i className="legend-cyan"/> INPUTS</span><span><i className="legend-violet"/> INFERENCE LAYER</span><span><i className="legend-green"/> STORED OUTPUTS</span><b>RESEARCH-ONLY · NO ORDER ROUTES</b></div>
      </section>

      <div className="hidden-detail-panels" aria-hidden="true">
      <section className="endpoint-health-panel" aria-label="API endpoint status">
        <div className="endpoint-health-heading"><div><span className="eyebrow">CONNECTIVITY MATRIX</span><h2>Endpoint diagnostics</h2></div><span className="endpoint-health-note">Independent read checks · 15s polling</span></div>
        <div className="endpoint-health-grid">
          {([
            ["health", "Health", "/api/v1/health"],
            ["performance", "Performance", "/api/v1/performance"],
            ["predictions", "Prediction ledger", "/api/v1/predictions"],
            ["audit", "Audit trail", "/api/v1/audit"],
            ["models", "Model inventory", "/api/v1/models"],
          ] as const).map(([id, label, path]) => {
            const status = endpointStatuses[id];
            return <div className="endpoint-health-item" key={id}><div><b>{label}</b><small>{path}</small><small className="endpoint-last-success">{endpointLastSuccess[id] ? "LAST OK " + new Date(endpointLastSuccess[id]!).toLocaleTimeString() : "NO SUCCESS YET"}</small></div><StatusPill state={status === "ok" ? "good" : status === "failed" ? "bad" : "muted"}>{status === "ok" ? "RESPONDING" : status === "failed" ? "FAILED" : "WAITING"}</StatusPill></div>;
          })}
        </div>
      </section>

      <section className="metrics-grid">
        <article className="metric-card"><div className="metric-head"><span>MODEL SERVER</span><Icon name="pulse" /></div><strong className={health?.inference_status === "ok" ? "metric-good" : ""}>{health?.inference_status?.toUpperCase() ?? "—"}</strong><div className="metric-foot"><i className={health?.inference_status === "ok" ? "live-dot" : "idle-dot"} /> Model listing / server connectivity</div></article>
        <article className="metric-card"><div className="metric-head"><span>DATA PERSISTENCE</span><Icon name="database" /></div><strong className={health?.database_status === "ok" ? "metric-good" : ""}>{health?.database_status?.toUpperCase() ?? "—"}</strong><div className="metric-foot"><i className={health?.database_status === "ok" ? "live-dot" : "idle-dot"} /> SQLite / SQLAlchemy</div></article>
        <article className="metric-card"><div className="metric-head"><span>PREDICTIONS RECORDED</span><Icon name="activity" /></div><strong>{performance?.predictions_total ?? "—"}</strong><div className="metric-foot">Durable prediction records</div></article>
        <article className="metric-card"><div className="metric-head"><span>OUTCOMES OBSERVED</span><Icon name="shield" /></div><strong>{performance?.outcomes_recorded ?? "—"}</strong><div className="metric-foot">Outcome labels received</div></article>
      </section>

      <section className="model-inventory-panel"><div className="section-heading"><div><div className="eyebrow">INFERENCE SERVER</div><h2>Available model inventory</h2></div><span className="research-tag">READ-ONLY</span></div>{modelInventoryError ? <div className="inventory-message">Model inventory unavailable: {modelInventoryError}. Other dashboard data can still refresh independently.</div> : !modelInventory ? <div className="inventory-message">Waiting for a successful authenticated model inventory request.</div> : <><div className="inventory-meta"><span>SERVER <b>{modelInventory.inference_server || "Not reported"}</b></span><span>CONFIGURED MODEL <b>{modelInventory.configured_model_id || "Not specified"}</b></span><span>MODELS REPORTED <b>{modelInventory.models.length}</b></span></div>{modelInventory.models.length === 0 ? <div className="inventory-message">The inference server returned no model entries.</div> : <div className="model-list">{modelInventory.models.map(model => <div className="model-row" key={model.id}><span className="model-indicator" /><div><b>{model.id}</b><small>{model.owned_by || "Owner not reported"}</small></div><span className={model.id === modelInventory.configured_model_id ? "configured-tag" : "available-tag"}>{model.id === modelInventory.configured_model_id ? "CONFIGURED" : "AVAILABLE"}</span></div>)}</div>}<p className="inventory-disclaimer">Model listing confirms server visibility only. It does not prove inference readiness or model quality; use Test Inference for a real completion probe.</p></>}</section>

      <section className="flow-panel">
        <div className="section-heading"><div><div className="eyebrow">LIVE SYSTEM MAP</div><h2>Intelligence pipeline</h2></div><div className="panel-status"><span className={busy || probeState === "running" ? "processing-dot" : "idle-dot"} /> {probeState === "running" ? "INFERENCE PROBE IN FLIGHT" : busy ? "API REQUEST IN FLIGHT" : state === "connected" ? "AWAITING NEXT POLL" : "STREAM NOT CONNECTED"}</div></div>
        <div className={`flow-canvas ${busy || probeState === "running" ? "flow-active" : ""}`}>
          <div className="flow-grid" />
          <div className="flow-node input-node"><div className="node-icon"><Icon name="database" /></div><div className="node-title">DATA INPUT</div><div className="node-sub">OHLC · context</div><div className="node-state">{state === "connected" ? "API REACHABLE" : "NO CONNECTION"}</div></div>
          <div className="flow-track track-one"><span /><span /><span /></div>
          <div className="flow-node core-node"><div className="core-rings"><div className="core-ring ring-a" /><div className="core-ring ring-b" /><div className="core-ring ring-c" /><div className="core-center"><div className="brand-mark"><span /><span /><span /></div></div></div><div className="node-title">AI INFERENCE CORE</div><div className="node-sub">{health?.application ?? "GoldcrestAI API"}</div><div className="node-state">{probeState === "passed" ? "COMPLETION VALIDATED" : probeState === "running" ? "RUNNING COMPLETION" : "LISTING ≠ INFERENCE TEST"}</div></div>
          <div className="flow-track track-two"><span /><span /><span /></div>
          <div className="flow-node output-node"><div className="node-icon output-icon"><Icon name="activity" /></div><div className="node-title">RESEARCH SIGNALS</div><div className="node-sub">BUY · SELL · NO TRADE</div><div className="node-state">{predictions.length} STORED RECORDS</div></div>
          <div className="flow-caption"><span className="legend-dot cyan" /> Input/API <span className="legend-dot violet" /> Model inference <span className="legend-dot green" /> Persisted output</div>
        </div>
        <div className="flow-note"><span>ⓘ</span> Particle movement runs only during an actual dashboard API request. This view does not fabricate market feeds or claim that inference is running continuously.</div>
      </section>

      <section className="lower-grid">
        <div className="signals-panel">
          <div className="section-heading"><div><div className="eyebrow">MODEL OUTPUT</div><h2>Recent prediction records <span className="count-chip">{predictions.length}</span></h2></div><span className="research-tag">RESEARCH ONLY</span></div>
          {latestSignals.length === 0 ? <div className="empty-state"><div className="empty-icon"><Icon name="activity" /></div><b>No prediction records available</b><p>When the API has stored predictions, they will appear here. No demo signals are injected.</p></div> : <div className="signal-list">{latestSignals.map(item => <button className="signal-row" key={item.prediction_id} onClick={() => setSelected(item)}><div className={`signal-symbol ${signalTone(item.decision)}`}>{item.symbol.slice(0, 3)}</div><div className="signal-main"><b>{item.symbol} <small>{item.timeframe}</small></b><span>{item.market_regime.replaceAll("_", " ")}</span></div><div className={`decision ${signalTone(item.decision)}`}>{item.decision.replace("_", " ")}</div><div className="signal-time">{stamp(item.generated_at)}</div><span className="row-arrow">↗</span></button>)}</div>}
          <div className="signals-foot"><span>Direction is not a calibrated probability or trading recommendation.</span><span>API: /predictions</span></div>
        </div>
        <div className="activity-panel"><div className="section-heading"><div><div className="eyebrow">TRACE / AUDIT</div><h2>System activity</h2></div><span className="tiny-count">{events.length} EVENTS</span></div>
          {events.length === 0 ? <div className="activity-empty">Audit events will appear when the API records activity.</div> : <div className="event-list">{events.slice(0, 8).map(event => <div className="event-row" key={event.id}><div className={`event-marker ${event.event_type === "prediction" ? "violet" : "cyan"}`} /><div className="event-copy"><b>{(event.event_type || "event").replaceAll("_", " ").toUpperCase()}</b><span>{event.symbol || event.request_id || "API event"}{event.outcome ? ` · ${event.outcome}` : ""}</span><small>{stamp(event.created_at)}</small></div>{typeof event.duration_ms === "number" && <span className="duration">{event.duration_ms}ms</span>}</div>)}</div>}
          <div className="activity-foot"><span className="live-dot" /> AUDIT READ-ONLY VIEW</div>
        </div>
      </section>

      <ResearchReportPanel />

      </div>
      <footer className="footer"><span>GOLDCRESTAI · NEURAL INTELLIGENCE</span><span>OFFLINE-SAFE · READ-ONLY RESEARCH · NO BROKER ACCESS</span></footer>

    </main>

    {activePanel && <div className="overlay-backdrop" role="presentation" onClick={() => setActivePanel(null)}><section className={`overlay-panel workspace-overlay ${activePanel}-overlay`} role="dialog" aria-modal="true" aria-label={activePanel === "signals" ? "Signal ledger" : activePanel === "audit" ? "Research audit" : activePanel === "models" ? "Model core" : "Data lab"} onClick={e => e.stopPropagation()}><div className="overlay-heading"><div><span className="eyebrow">GOLDCRESTAI / WORKSPACE MODULE</span><h2>{activePanel === "signals" ? "Signal ledger" : activePanel === "audit" ? "Research audit trail" : activePanel === "models" ? "Model intelligence" : "Historical research lab"}</h2></div><button className="modal-close" onClick={() => setActivePanel(null)} aria-label="Close panel">×</button></div>
      {activePanel === "signals" && <><div className="overlay-kpis"><div><small>STORED PREDICTIONS</small><b>{performance?.predictions_total ?? "—"}</b></div><div><small>RECORDS LOADED</small><b>{predictions.length}</b></div><div><small>STATUS</small><b>{endpointStatuses.predictions === "ok" ? "SYNCED" : "UNAVAILABLE"}</b></div></div>{predictions.length === 0 ? <div className="overlay-empty">No prediction records have been returned by the API. No demo signals are injected.</div> : <div className="overlay-list">{predictions.map(item => <button className="signal-row" key={item.prediction_id} onClick={() => { setSelected(item); setActivePanel(null); }}><div className={`signal-symbol ${signalTone(item.decision)}`}>{item.symbol.slice(0,3)}</div><div className="signal-main"><b>{item.symbol} <small>{item.timeframe}</small></b><span>{item.market_regime.replaceAll("_"," ")}</span></div><div className={`decision ${signalTone(item.decision)}`}>{item.decision.replace("_"," ")}</div><div className="signal-time">{stamp(item.generated_at)}</div><span className="row-arrow">↗</span></button>)}</div>}<p className="overlay-description">Research records only. Direction is not a calibrated probability or a trading recommendation.</p></>}
      {activePanel === "audit" && <><div className="overlay-kpis"><div><small>AUDIT EVENTS</small><b>{events.length}</b></div><div><small>API ENDPOINT</small><b>{endpointStatuses.audit === "ok" ? "RESPONDING" : "UNAVAILABLE"}</b></div><div><small>ACCESS</small><b>READ-ONLY</b></div></div>{events.length === 0 ? <div className="overlay-empty">No audit events have been returned by the API.</div> : <div className="overlay-list audit-list">{events.map(event => <div className="event-row" key={event.id}><div className={`event-marker ${event.event_type === "prediction" ? "violet" : "cyan"}`}/><div className="event-copy"><b>{(event.event_type || "event").replaceAll("_"," ").toUpperCase()}</b><span>{event.symbol || event.request_id || "API event"}{event.outcome ? ` · ${event.outcome}` : ""}</span><small>{stamp(event.created_at)}</small></div>{typeof event.duration_ms === "number" && <span className="duration">{event.duration_ms}ms</span>}</div>)}</div>}</>}
      {activePanel === "models" && <><div className="overlay-kpis"><div><small>INFERENCE STATUS</small><b>{health?.inference_status?.toUpperCase() ?? "UNKNOWN"}</b></div><div><small>DATABASE</small><b>{health?.database_status?.toUpperCase() ?? "UNKNOWN"}</b></div><div><small>MODEL COUNT</small><b>{modelInventory?.models.length ?? "—"}</b></div></div><div className="overlay-model-meta"><span>SERVER <b>{modelInventory?.inference_server || "Not reported"}</b></span><span>CONFIGURED MODEL <b>{modelInventory?.configured_model_id || "Not specified"}</b></span></div>{modelInventoryError ? <div className="overlay-empty">Model inventory unavailable: {modelInventoryError}</div> : !modelInventory ? <div className="overlay-empty">Waiting for a successful authenticated model inventory request.</div> : modelInventory.models.map(model => <div className="overlay-model-row" key={model.id}><span className="legend-dot cyan"/><div><b>{model.id}</b><small>{model.owned_by || "Owner not reported"}</small></div><span>{model.id === modelInventory.configured_model_id ? "CONFIGURED" : "AVAILABLE"}</span></div>)}<p className="overlay-description">Model visibility does not prove model quality. Use Test Inference to run a real completion check.</p></>}
      {activePanel === "research" && <><div className="overlay-description">Inspect a previously generated offline pipeline report. The selected file is parsed locally in your browser and is not uploaded to the API.</div><ResearchReportPanel /></>}
    </section></div>}
    {selected && <div className="modal-backdrop" role="presentation" onClick={() => setSelected(null)}><section className="signal-modal" role="dialog" aria-modal="true" aria-label="Prediction details" onClick={e => e.stopPropagation()}><button className="modal-close" onClick={() => setSelected(null)}>×</button><div className="eyebrow">PREDICTION DETAIL / {shortId(selected.prediction_id)}</div><h2>{selected.symbol} <span>{selected.timeframe}</span></h2><div className={`modal-decision ${signalTone(selected.decision)}`}>{selected.decision.replace("_", " ")}</div><div className="detail-grid"><div><small>MODEL</small><b>{selected.model_id || "Not reported"}</b></div><div><small>PROBABILITY</small><b>{selected.probability == null ? "Not reported" : `${(selected.probability * 100).toFixed(1)}%`}</b></div><div><small>MARKET REGIME</small><b>{selected.market_regime}</b></div><div><small>GENERATED</small><b>{stamp(selected.generated_at)}</b></div><div><small>EXPIRES</small><b>{stamp(selected.expires_at)}</b></div><div><small>REQUEST ID</small><b>{selected.request_id}</b></div></div><div className="reason-block"><small>REASON CODES</small>{selected.reason_codes.length ? selected.reason_codes.map(reason => <span key={reason}>{reason}</span>) : <span>None supplied</span>}</div><div className="modal-warning"><Icon name="shield" /> Research-only output. This signal is not connected to order execution.</div></section></div>}
  </div>;
}

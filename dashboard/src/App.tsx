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
  const [selectedGraphNode, setSelectedGraphNode] = useState("core");

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

      <section className={`immersive-stage ${busy || probeState === "running" ? "neural-stage-active" : ""}`} aria-label="Interactive neural intelligence graph">
        <div className="immersive-topline"><div><span className="eyebrow"><span className="eyebrow-line"/> GOLDCREST / NEURAL KNOWLEDGE GRAPH</span><h2>Market intelligence <span>in motion</span></h2><p>Explore the research architecture. Select any node to inspect its role and current system state.</p></div><div className="graph-live-state"><i className={state === "connected" ? "state-on" : ""}/>{state === "connected" ? "CONNECTED" : state === "degraded" ? "PARTIAL CONNECTION" : "OFFLINE VIEW"}</div></div>
        <div className="immersive-workspace">
          <div className="graph-viewport">
            <div className="graph-vignette"/><div className="graph-grid"/>
            <svg className="knowledge-graph" viewBox="0 0 1000 650" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Interactive conceptual map of market research processing">
              <defs>
                <radialGradient id="nodeGlow"><stop offset="0" stopColor="#b9fff8" stopOpacity=".95"/><stop offset=".35" stopColor="#4de6ee" stopOpacity=".35"/><stop offset="1" stopColor="#4de6ee" stopOpacity="0"/></radialGradient>
                <radialGradient id="coreAura"><stop offset="0" stopColor="#b7b2ff" stopOpacity=".42"/><stop offset=".55" stopColor="#3acbdc" stopOpacity=".12"/><stop offset="1" stopColor="#071321" stopOpacity="0"/></radialGradient>
                <linearGradient id="linkFlow"><stop offset="0" stopColor="#44d9e8" stopOpacity=".08"/><stop offset=".5" stopColor="#8c83ff" stopOpacity=".88"/><stop offset="1" stopColor="#f0d49a" stopOpacity=".3"/></linearGradient>
                <filter id="softGlow"><feGaussianBlur stdDeviation="5"/></filter>
              </defs>
              <circle cx="480" cy="324" r="270" fill="url(#coreAura)"/>
              <g className="graph-links" fill="none" stroke="url(#linkFlow)" strokeWidth="1.2">
                <path d="M480 324 Q360 250 270 150"/><path d="M480 324 Q330 300 180 310"/><path d="M480 324 Q345 390 250 500"/><path d="M480 324 Q480 190 500 95"/><path d="M480 324 Q610 210 700 150"/><path d="M480 324 Q660 290 820 280"/><path d="M480 324 Q640 420 740 500"/><path d="M480 324 Q490 470 500 570"/>
                <path d="M270 150 Q210 220 180 310"/><path d="M180 310 Q210 420 250 500"/><path d="M270 150 Q380 120 500 95"/><path d="M500 95 Q625 90 700 150"/><path d="M700 150 Q780 195 820 280"/><path d="M820 280 Q820 390 740 500"/><path d="M740 500 Q630 560 500 570"/><path d="M250 500 Q380 560 500 570"/>
                <path d="M270 150 Q390 245 480 324" strokeDasharray="3 8"/><path d="M700 150 Q585 250 480 324" strokeDasharray="3 8"/><path d="M740 500 Q610 410 480 324" strokeDasharray="3 8"/>
              </g>
              <g className="graph-particles" fill="#6cecf0">
                <circle r="3"><animateMotion dur="5s" repeatCount="indefinite" path="M270 150 Q360 250 480 324"/></circle>
                <circle r="2.5"><animateMotion dur="6.5s" repeatCount="indefinite" path="M180 310 Q330 300 480 324"/></circle>
                <circle r="3"><animateMotion dur="5.8s" repeatCount="indefinite" path="M250 500 Q345 390 480 324"/></circle>
                <circle r="2.5"><animateMotion dur="7s" repeatCount="indefinite" path="M700 150 Q610 210 480 324"/></circle>
                <circle r="3"><animateMotion dur="6s" repeatCount="indefinite" path="M820 280 Q660 290 480 324"/></circle>
                <circle r="2.5"><animateMotion dur="7.2s" repeatCount="indefinite" path="M740 500 Q640 420 480 324"/></circle>
              </g>
              <g className="graph-node" role="button" tabIndex="0" aria-label="Market history" onClick={() => setSelectedGraphNode("history")} onKeyDown={e => e.key === "Enter" && setSelectedGraphNode("history")}><circle className="node-halo" cx="270" cy="150" r="36"/><circle cx="270" cy="150" r="16" fill="url(#nodeGlow)"/><circle cx="270" cy="150" r="7" fill="#8ff7f4"/><text x="270" y="115" textAnchor="middle">MARKET HISTORY</text><text x="270" y="185" textAnchor="middle" className="node-sub">{state === "connected" ? "API LINKED" : "AWAITING DATA"}</text></g>
              <g className="graph-node" role="button" tabIndex="0" aria-label="Technical features" onClick={() => setSelectedGraphNode("features")} onKeyDown={e => e.key === "Enter" && setSelectedGraphNode("features")}><circle className="node-halo" cx="180" cy="310" r="31"/><circle cx="180" cy="310" r="13" fill="url(#nodeGlow)"/><circle cx="180" cy="310" r="6" fill="#8ff7f4"/><text x="180" y="278" textAnchor="middle">FEATURES</text><text x="180" y="341" textAnchor="middle" className="node-sub">CAUSAL TA</text></g>
              <g className="graph-node" role="button" tabIndex="0" aria-label="Historical research" onClick={() => setSelectedGraphNode("research")} onKeyDown={e => e.key === "Enter" && setSelectedGraphNode("research")}><circle className="node-halo" cx="250" cy="500" r="32"/><circle cx="250" cy="500" r="14" fill="url(#nodeGlow)"/><circle cx="250" cy="500" r="6" fill="#9a8cff"/><text x="250" y="465" textAnchor="middle">HISTORICAL</text><text x="250" y="485" textAnchor="middle">RESEARCH</text></g>
              <g className="graph-node" role="button" tabIndex="0" aria-label="Model inventory" onClick={() => setSelectedGraphNode("models")} onKeyDown={e => e.key === "Enter" && setSelectedGraphNode("models")}><circle className="node-halo" cx="500" cy="95" r="32"/><circle cx="500" cy="95" r="14" fill="url(#nodeGlow)"/><circle cx="500" cy="95" r="6" fill="#a89aff"/><text x="500" y="60" textAnchor="middle">MODEL LAYER</text></g>
              <g className="graph-node" role="button" tabIndex="0" aria-label="News context" onClick={() => setSelectedGraphNode("news")} onKeyDown={e => e.key === "Enter" && setSelectedGraphNode("news")}><circle className="node-halo" cx="700" cy="150" r="34"/><circle cx="700" cy="150" r="15" fill="url(#nodeGlow)"/><circle cx="700" cy="150" r="6" fill="#f0d49a"/><text x="700" y="112" textAnchor="middle">NEWS CONTEXT</text><text x="700" y="184" textAnchor="middle" className="node-sub">OPTIONAL INPUT</text></g>
              <g className="graph-node" role="button" tabIndex="0" aria-label="Validation" onClick={() => setSelectedGraphNode("validation")} onKeyDown={e => e.key === "Enter" && setSelectedGraphNode("validation")}><circle className="node-halo" cx="820" cy="280" r="36"/><circle cx="820" cy="280" r="16" fill="url(#nodeGlow)"/><circle cx="820" cy="280" r="7" fill="#f0d49a"/><text x="820" y="242" textAnchor="middle">VALIDATION</text><text x="820" y="314" textAnchor="middle" className="node-sub">RESEARCH GATE</text></g>
              <g className="graph-node" role="button" tabIndex="0" aria-label="Research outputs" onClick={() => setSelectedGraphNode("outputs")} onKeyDown={e => e.key === "Enter" && setSelectedGraphNode("outputs")}><circle className="node-halo" cx="740" cy="500" r="34"/><circle cx="740" cy="500" r="15" fill="url(#nodeGlow)"/><circle cx="740" cy="500" r="6" fill="#68e8c0"/><text x="740" y="463" textAnchor="middle">RESEARCH OUTPUT</text><text x="740" y="533" textAnchor="middle" className="node-sub">{predictions.length} STORED</text></g>
              <g className="graph-node" role="button" tabIndex="0" aria-label="Audit trace" onClick={() => setSelectedGraphNode("audit")} onKeyDown={e => e.key === "Enter" && setSelectedGraphNode("audit")}><circle className="node-halo" cx="500" cy="570" r="30"/><circle cx="500" cy="570" r="13" fill="url(#nodeGlow)"/><circle cx="500" cy="570" r="5" fill="#68e8c0"/><text x="500" y="610" textAnchor="middle">AUDIT TRACE</text></g>
              <g className="central-core"><circle cx="480" cy="324" r="116" fill="none" stroke="#4edce6" strokeOpacity=".18"/><circle cx="480" cy="324" r="88" fill="none" stroke="#a395ff" strokeOpacity=".55" strokeDasharray="2 7"/><circle cx="480" cy="324" r="63" fill="#09172a" stroke="#56e7e8" strokeOpacity=".8"/><circle cx="480" cy="324" r="59" fill="url(#coreAura)"/><g transform="translate(480 324)" fill="#7ff8f0"><rect x="-20" y="-10" width="5" height="20" rx="2"/><rect x="-10" y="-22" width="5" height="44" rx="2"/><rect x="0" y="-31" width="5" height="62" rx="2"/><rect x="10" y="-17" width="5" height="34" rx="2"/><rect x="20" y="-8" width="5" height="16" rx="2"/></g><text x="480" y="420" textAnchor="middle" className="core-caption">GOLDCREST NEURAL CORE</text><text x="480" y="440" textAnchor="middle" className="node-sub">{probeState === "running" ? "INFERENCE IN FLIGHT" : probeState === "passed" ? "PROBE PASSED" : probeState === "failed" ? "PROBE FAILED" : "RESEARCH ARCHITECTURE"}</text></g>
            </svg>
            <div className="graph-bottom-legend"><span><i className="legend-cyan"/> DATA CONTEXT</span><span><i className="legend-violet"/> MODEL LAYER</span><span><i className="legend-green"/> RESEARCH RECORDS</span></div>
          </div>
          <aside className="graph-inspector">
            <div className="inspector-kicker">NODE INSPECTOR / 0{["core","history","features","research","models","news","validation","outputs","audit"].indexOf(selectedGraphNode)+1}</div>
            <h3>{({core:"Neural processing core",history:"Market history",features:"Feature extraction",research:"Historical research",models:"Model layer",news:"News context",validation:"Validation gate",outputs:"Research outputs",audit:"Audit trace"} as Record<string,string>)[selectedGraphNode]}</h3>
            <p className="inspector-copy">{({core:"The central research architecture connects source data, causal features, model inference and validation. This conceptual map does not indicate active processing unless the API state confirms it.",history:"Historical OHLC data is the source layer for offline research. Live availability depends on the configured API and data-ingestion workflow.",features:"Causal technical features are derived from chronological bars. Feature availability and quality must be verified in the generated dataset report.",research:"Historical experiments use chronological validation and walk-forward evaluation. Reports are evidence for review, not live trading instructions.",models:"The model inventory is read from the configured API when available. A listed model is not evidence of predictive quality.",news:"News or sentiment is an optional research input only when a supported provider supplies data. Missing news is not fabricated.",validation:"Validation checks data provenance, fold consistency and out-of-sample evidence before research results are considered.",outputs:"Stored prediction records are retrieved from the API. The current count reflects records actually returned, not generated demo signals.",audit:"Audit events describe recorded system activity. No events are invented to fill this view."} as Record<string,string>)[selectedGraphNode]}</p>
            <div className="inspector-divider"/>
            <div className="inspector-metrics"><div><small>API LINK</small><b className={state === "connected" ? "metric-good" : "metric-muted"}>{state === "connected" ? "CONNECTED" : state === "degraded" ? "DEGRADED" : "OFFLINE"}</b></div><div><small>STORED SIGNALS</small><b>{performance?.predictions_total ?? "—"}</b></div><div><small>OBSERVED OUTCOMES</small><b>{performance?.outcomes_recorded ?? "—"}</b></div></div>
            <div className="inspector-actions"><button onClick={() => setActivePanel("signals")}>OPEN SIGNAL LEDGER <span>↗</span></button><button onClick={() => setActivePanel("audit")}>VIEW AUDIT TRAIL <span>↗</span></button><button onClick={() => setActivePanel("models")}>INSPECT MODELS <span>↗</span></button></div>
            <div className="inspector-status"><i className={probeState === "passed" ? "state-on" : ""}/><span>{probeState === "passed" ? "REAL INFERENCE PROBE PASSED" : probeState === "failed" ? "INFERENCE PROBE FAILED" : "NO ACTIVE INFERENCE CLAIMED"}</span></div>
          </aside>
        </div>
        <div className="immersive-footer"><span>GOLDCRESTAI / INTELLIGENCE MAP</span><span>READ-ONLY · RESEARCH ONLY · NO ORDER ROUTES</span></div>
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

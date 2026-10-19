import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

type Health = { status: string; application: string; inference_status: string; database_status: string };
type Performance = { predictions_total: number; outcomes_recorded: number; note?: string };
type Prediction = {
  prediction_id: string; request_id: string; symbol: string; timeframe: string;
  decision: "BUY" | "SELL" | "NO_TRADE"; direction?: string | null; probability?: number | null;
  market_regime: string; reason_codes: string[]; model_id?: string | null; generated_at: string; expires_at: string;
};
type AuditEvent = { id: string; event_type: string; request_id?: string | null; symbol?: string | null; outcome?: string | null; duration_ms?: number | null; details?: Record<string, unknown>; created_at: string };
type LoadState = "loading" | "connected" | "degraded" | "offline";

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
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [busy, setBusy] = useState(false);
  const [lastSync, setLastSync] = useState<string>();
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<Prediction | null>(null);

  const refresh = useCallback(async () => {
    setBusy(true);
    try {
      const [h, p, signals, audit] = await Promise.all([
        request<Health>(base, key, "/api/v1/health"),
        request<Performance>(base, key, "/api/v1/performance"),
        request<Prediction[]>(base, key, "/api/v1/predictions?limit=30"),
        request<AuditEvent[]>(base, key, "/api/v1/audit?limit=12"),
      ]);
      setHealth(h); setPerformance(p); setPredictions(signals); setEvents(audit);
      setState(h.status === "ok" && h.inference_status === "ok" && h.database_status === "ok" ? "connected" : "degraded");
      setError("");
      setLastSync(new Date().toISOString());
    } catch (e) {
      setState("offline");
      setError(e instanceof Error ? e.message : "API request failed");
    } finally {
      setBusy(false);
    }
  }, [base, key]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 15000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const latestSignals = useMemo(() => predictions.slice(0, 6), [predictions]);
  const saveSettings = () => {
    const normalized = baseInput.trim().replace(/\/$/, "");
    const nextBase = normalized || "http://192.168.29.80:8090";
    setBase(nextBase);
    setKey(keyInput);
    localStorage.setItem("goldcrest-api-base", nextBase);
    sessionStorage.setItem("goldcrest-api-key", keyInput);
    setShowSettings(false);
  };
  const signalTone = (decision: string) => decision === "BUY" ? "buy" : decision === "SELL" ? "sell" : "neutral";
  const stateLabel = state === "connected" ? "SYSTEM ONLINE" : state === "degraded" ? "DEGRADED" : state === "loading" ? "CONNECTING" : "DISCONNECTED";

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><span /><span /><span /></div><div><b>GOLDCREST<span>AI</span></b><small>INTELLIGENCE SYSTEMS</small></div></div>
      <div className="nav-label">WORKSPACE</div>
      <div className="nav-item active"><Icon name="network" /> Neural console <span className="nav-active-dot" /></div>
      <div className="nav-item"><Icon name="activity" /> Signal ledger <span className="nav-count">{performance?.predictions_total ?? "—"}</span></div>
      <div className="nav-item"><Icon name="database" /> Research audit</div>
      <div className="sidebar-bottom"><div className="mini-orbit"><span /><i /><b /></div><p>RESEARCH ENGINE</p><small>Isolated from execution systems</small><div className="safe-label"><Icon name="shield" /> NO BROKER ACCESS</div></div>
    </aside>

    <main className="main">
      <header className="topbar"><div className="breadcrumbs">WORKSPACE <span>/</span> <b>NEURAL CONSOLE</b></div><div className="top-actions"><span className="clock-label"><Icon name="clock" /> {lastSync ? `SYNC ${new Date(lastSync).toLocaleTimeString()}` : "WAITING FOR SYNC"}</span><StatusPill state={state === "connected" ? "good" : state === "degraded" ? "warn" : state === "offline" ? "bad" : "muted"}>{stateLabel}</StatusPill><button className="icon-button" onClick={() => setShowSettings(v => !v)} aria-label="Connection settings">⚙</button></div></header>

      <section className="page-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> AI RESEARCH ENVIRONMENT / 01</div><h1>Neural <span>Signal Console</span></h1><p>Observe market inputs move through inference and emerge as research signals.</p></div><button className="refresh-button" onClick={() => void refresh()} disabled={busy}><span className={busy ? "refresh-spin" : ""}>↻</span> {busy ? "SYNCING" : "REFRESH DATA"}</button></section>

      {showSettings && <section className="settings-panel"><div><b>Connect to GoldcrestAI API</b><p>Requests use your API key in this browser session. The key is not written to local storage.</p></div><label>API BASE URL<input value={baseInput} onChange={e => setBaseInput(e.target.value)} placeholder="http://192.168.29.80:8090" /></label><label>API KEY<input value={keyInput} onChange={e => setKeyInput(e.target.value)} type="password" placeholder="Enter X-API-Key" autoComplete="off" /></label><button onClick={saveSettings}>SAVE & CONNECT</button></section>}

      {error && <div className="error-banner"><span>!</span><div><b>Research API unavailable</b><p>{error}</p><small>Check the API address, network access, and key. Existing signal data is not presented as live.</small></div><button onClick={() => setShowSettings(true)}>CONNECTION SETTINGS</button></div>}

      <section className="metrics-grid">
        <article className="metric-card"><div className="metric-head"><span>INFERENCE ENGINE</span><Icon name="pulse" /></div><strong className={health?.inference_status === "ok" ? "metric-good" : ""}>{health?.inference_status === "ok" ? "READY" : health?.inference_status?.toUpperCase() ?? "—"}</strong><div className="metric-foot"><i className={health?.inference_status === "ok" ? "live-dot" : "idle-dot"} /> llama.cpp model endpoint</div></article>
        <article className="metric-card"><div className="metric-head"><span>DATA PERSISTENCE</span><Icon name="database" /></div><strong className={health?.database_status === "ok" ? "metric-good" : ""}>{health?.database_status?.toUpperCase() ?? "—"}</strong><div className="metric-foot"><i className={health?.database_status === "ok" ? "live-dot" : "idle-dot"} /> SQLite / SQLAlchemy</div></article>
        <article className="metric-card"><div className="metric-head"><span>PREDICTIONS RECORDED</span><Icon name="activity" /></div><strong>{performance?.predictions_total ?? "—"}</strong><div className="metric-foot">Durable prediction records</div></article>
        <article className="metric-card"><div className="metric-head"><span>OUTCOMES OBSERVED</span><Icon name="shield" /></div><strong>{performance?.outcomes_recorded ?? "—"}</strong><div className="metric-foot">Outcome labels received</div></article>
      </section>

      <section className="flow-panel">
        <div className="section-heading"><div><div className="eyebrow">LIVE SYSTEM MAP</div><h2>Intelligence pipeline</h2></div><div className="panel-status"><span className={busy ? "processing-dot" : "idle-dot"} /> {busy ? "API REQUEST IN FLIGHT" : state === "connected" ? "AWAITING NEXT POLL" : "STREAM NOT CONNECTED"}</div></div>
        <div className={`flow-canvas ${busy ? "flow-active" : ""}`}>
          <div className="flow-grid" />
          <div className="flow-node input-node"><div className="node-icon"><Icon name="database" /></div><div className="node-title">DATA INPUT</div><div className="node-sub">OHLC · context</div><div className="node-state">{state === "connected" ? "API REACHABLE" : "NO CONNECTION"}</div></div>
          <div className="flow-track track-one"><span /><span /><span /></div>
          <div className="flow-node core-node"><div className="core-rings"><div className="core-ring ring-a" /><div className="core-ring ring-b" /><div className="core-ring ring-c" /><div className="core-center"><div className="brand-mark"><span /><span /><span /></div></div></div><div className="node-title">AI INFERENCE CORE</div><div className="node-sub">{health?.application ?? "GoldcrestAI API"}</div><div className="node-state">{health?.inference_status === "ok" ? "MODEL ENDPOINT READY" : "READINESS UNCONFIRMED"}</div></div>
          <div className="flow-track track-two"><span /><span /><span /></div>
          <div className="flow-node output-node"><div className="node-icon output-icon"><Icon name="activity" /></div><div className="node-title">RESEARCH SIGNALS</div><div className="node-sub">BUY · SELL · NO TRADE</div><div className="node-state">{predictions.length} STORED RECORDS</div></div>
          <div className="flow-caption"><span className="legend-dot cyan" /> Input/API <span className="legend-dot violet" /> Model inference <span className="legend-dot green" /> Persisted output</div>
        </div>
        <div className="flow-note"><span>ⓘ</span> Particle movement runs only during an actual dashboard API request. This view does not fabricate market feeds or claim that inference is running continuously.</div>
      </section>

      <section className="lower-grid">
        <div className="signals-panel">
          <div className="section-heading"><div><div className="eyebrow">OUT-OF-MODEL OUTPUT</div><h2>Recent prediction records <span className="count-chip">{predictions.length}</span></h2></div><span className="research-tag">RESEARCH ONLY</span></div>
          {latestSignals.length === 0 ? <div className="empty-state"><div className="empty-icon"><Icon name="activity" /></div><b>No prediction records available</b><p>When the API has stored predictions, they will appear here. No demo signals are injected.</p></div> : <div className="signal-list">{latestSignals.map(item => <button className="signal-row" key={item.prediction_id} onClick={() => setSelected(item)}><div className={`signal-symbol ${signalTone(item.decision)}`}>{item.symbol.slice(0, 3)}</div><div className="signal-main"><b>{item.symbol} <small>{item.timeframe}</small></b><span>{item.market_regime.replaceAll("_", " ")}</span></div><div className={`decision ${signalTone(item.decision)}`}>{item.decision.replace("_", " ")}</div><div className="signal-time">{stamp(item.generated_at)}</div><span className="row-arrow">↗</span></button>)}</div>}
          <div className="signals-foot"><span>Direction is not a calibrated probability or trading recommendation.</span><span>API: /predictions</span></div>
        </div>
        <div className="activity-panel"><div className="section-heading"><div><div className="eyebrow">TRACE / AUDIT</div><h2>System activity</h2></div><span className="tiny-count">{events.length} EVENTS</span></div>
          {events.length === 0 ? <div className="activity-empty">Audit events will appear when the API records activity.</div> : <div className="event-list">{events.slice(0, 8).map(event => <div className="event-row" key={event.id}><div className={`event-marker ${event.event_type === "prediction" ? "violet" : "cyan"}`} /><div className="event-copy"><b>{(event.event_type || "event").replaceAll("_", " ").toUpperCase()}</b><span>{event.symbol || event.request_id || "API event"}{event.outcome ? ` · ${event.outcome}` : ""}</span><small>{stamp(event.created_at)}</small></div>{typeof event.duration_ms === "number" && <span className="duration">{event.duration_ms}ms</span>}</div>)}</div>}
          <div className="activity-foot"><span className="live-dot" /> AUDIT READ-ONLY VIEW</div>
        </div>
      </section>

      <footer className="footer"><span>GOLDCRESTAI <i>•</i> NEURAL RESEARCH CONSOLE</span><span>NO ORDER ROUTES <i>•</i> NO BROKER CREDENTIALS <i>•</i> NON-ACTIONABLE</span></footer>
    </main>

    {selected && <div className="modal-backdrop" role="presentation" onClick={() => setSelected(null)}><section className="signal-modal" role="dialog" aria-modal="true" aria-label="Prediction details" onClick={e => e.stopPropagation()}><button className="modal-close" onClick={() => setSelected(null)}>×</button><div className="eyebrow">PREDICTION DETAIL / {shortId(selected.prediction_id)}</div><h2>{selected.symbol} <span>{selected.timeframe}</span></h2><div className={`modal-decision ${signalTone(selected.decision)}`}>{selected.decision.replace("_", " ")}</div><div className="detail-grid"><div><small>MODEL</small><b>{selected.model_id || "Not reported"}</b></div><div><small>PROBABILITY</small><b>{selected.probability == null ? "Not reported" : `${(selected.probability * 100).toFixed(1)}%`}</b></div><div><small>MARKET REGIME</small><b>{selected.market_regime}</b></div><div><small>GENERATED</small><b>{stamp(selected.generated_at)}</b></div><div><small>EXPIRES</small><b>{stamp(selected.expires_at)}</b></div><div><small>REQUEST ID</small><b>{selected.request_id}</b></div></div><div className="reason-block"><small>REASON CODES</small>{selected.reason_codes.length ? selected.reason_codes.map(reason => <span key={reason}>{reason}</span>) : <span>None supplied</span>}</div><div className="modal-warning"><Icon name="shield" /> Research-only output. This signal is not connected to order execution.</div></section></div>}
  </div>;
}

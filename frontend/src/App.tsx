import { useCallback, useEffect, useMemo, useState } from "react";

import { deleteJson, getJson, patchJson, postJson, putJson } from "./api";
import type {
  CompatibilityTest, EventDefinition, EventRecord, EventRevision, EventValidation,
  GatewayStatus, HardwareProbe, LiveState, ManagementHealth, ModelEntry,
  ProtocolContract, RuntimeTemplate, Telemetry, Worker, WorkerLog,
} from "./types";

type View = "live" | "events" | "workers" | "models" | "advanced";
type WorkerOperation = "start" | "stop" | "restart" | "smoke";
type WorkerSort = "name-asc" | "name-desc" | "model-asc" | "runtime-asc" | "state";
type ModelSort = "name-asc" | "name-desc" | "size-desc" | "size-asc" | "readiness" | "workers";

const NAVIGATION: Array<{ view: View; label: string; path: string }> = [
  { view: "live", label: "Live", path: "/" },
  { view: "events", label: "Events", path: "/events" },
  { view: "workers", label: "Workers", path: "/workers" },
  { view: "models", label: "Models", path: "/models" },
  { view: "advanced", label: "Advanced", path: "/advanced" },
];

function viewFromPath(path: string): View {
  return NAVIGATION.find((item) => item.path === path)?.view ?? "live";
}

export default function App() {
  const [view, setView] = useState<View>(() => viewFromPath(window.location.pathname));
  const [health, setHealth] = useState<ManagementHealth | null>(null);
  const [gateway, setGateway] = useState<GatewayStatus | null>(null);
  const [hardware, setHardware] = useState<HardwareProbe | null>(null);
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [live, setLive] = useState<LiveState>({ active_event: null, routes: [] });
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [contracts, setContracts] = useState<ProtocolContract[]>([]);
  const [templates, setTemplates] = useState<RuntimeTemplate[]>([]);
  const [compatibility, setCompatibility] = useState<CompatibilityTest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [nextHealth, nextGateway, nextHardware, nextTelemetry, nextLive, nextWorkers,
      nextEvents, catalogue, contractResponse, templateResponse, tests] = await Promise.all([
      getJson<ManagementHealth>("/api/health"),
      getJson<GatewayStatus>("/api/gateway/status"),
      getJson<HardwareProbe>("/api/hardware"),
      getJson<Telemetry>("/api/telemetry"),
      getJson<LiveState>("/api/live"),
      getJson<Worker[]>("/api/workers"),
      getJson<{ events: EventRecord[] }>("/api/events"),
      getJson<{ models: ModelEntry[] }>("/api/catalogue"),
      getJson<{ contracts: ProtocolContract[] }>("/api/protocol-contracts"),
      getJson<{ templates: RuntimeTemplate[] }>("/api/runtime-templates"),
      getJson<{ tests: CompatibilityTest[] }>("/api/compatibility"),
    ]);
    setHealth(nextHealth); setGateway(nextGateway); setHardware(nextHardware);
    setTelemetry(nextTelemetry); setLive(nextLive); setWorkers(nextWorkers);
    setEvents(nextEvents.events); setModels(catalogue.models);
    setContracts(contractResponse.contracts); setTemplates(templateResponse.templates);
    setCompatibility(tests.tests);
  }, []);

  useEffect(() => {
    setLoading(true);
    refresh().catch((reason) => setError(messageFrom(reason))).finally(() => setLoading(false));
  }, [refresh]);

  useEffect(() => {
    const onPop = () => setView(viewFromPath(window.location.pathname));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (document.hidden) return;
      Promise.all([
        getJson<Worker[]>("/api/workers").then(setWorkers),
        getJson<LiveState>("/api/live").then(setLive),
        getJson<GatewayStatus>("/api/gateway/status").then(setGateway),
      ]).catch(() => undefined);
    }, 5000);
    return () => window.clearInterval(timer);
  }, []);

  const operate = async (worker: Worker, operation: WorkerOperation) => {
    const key = `${worker.id}:${operation}`;
    setPending(key); setError(null);
    try {
      const result = await postJson<{ ok?: boolean; test?: { evidence?: { error_summary?: string } } }>(`/api/workers/${worker.id}/${operation}`);
      await refresh();
      if (operation === "smoke" && result.ok === false) {
        throw new Error(result.test?.evidence?.error_summary ?? "Worker generation smoke test failed.");
      }
    } catch (reason) { setError(messageFrom(reason)); }
    finally { setPending(null); }
  };

  const navigate = (next: View, path: string) => {
    window.history.pushState({}, "", path); setView(next);
  };

  if (loading) return <Loading />;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">MD</span><div><strong>ModelDeck</strong><small>Operator console</small></div></div>
        <nav aria-label="Primary navigation">{NAVIGATION.map((item) => (
          <a className={view === item.view ? "nav-link active" : "nav-link"} href={item.path}
            key={item.view} onClick={(event) => { event.preventDefault(); navigate(item.view, item.path); }}>
            {item.label}
          </a>
        ))}</nav>
        <div className="sidebar-policy"><StatusDot state={gateway?.available ? "good" : "warn"} /><span>Local gateway only</span></div>
      </aside>
      <main className="main-content">
        <header className="topbar"><div><p className="eyebrow">Framework Desktop · local control plane</p><h1>{NAVIGATION.find((item) => item.view === view)?.label}</h1></div>
          <div className="topbar-status">
            {health && <div className={`mode-badge ${health.open_day ? "locked" : "unlocked"}`} aria-label="Configuration status"><StatusDot state={health.open_day ? "warn" : "good"} /><span>{health.open_day ? "Open Day · configuration locked" : "Configuration unlocked"}</span></div>}
            <div className={`gateway-badge ${gateway?.available ? "ready" : "unavailable"}`}><StatusDot state={gateway?.available ? "good" : "bad"} /><span>{gateway?.available ? "Gateway available" : "Gateway unavailable"}</span></div>
          </div>
        </header>
        {error && <div className="alert error" role="alert"><strong>Action failed</strong><span>{error}</span><button className="icon-button" onClick={() => setError(null)}>×</button></div>}
        {!health || !hardware || !telemetry || !gateway ? <Unavailable retry={refresh} />
          : view === "live" ? <LiveView live={live} workers={workers} models={models} operate={operate} pending={pending} />
          : view === "events" ? <EventsView events={events} workers={workers} contracts={contracts} openDay={health.open_day} refresh={refresh} />
          : view === "workers" ? <WorkersView workers={workers} pending={pending} operate={operate} refresh={refresh} openDay={health.open_day} />
          : view === "models" ? <ModelsView models={models} templates={templates} refresh={refresh} openDay={health.open_day} />
          : <AdvancedView hardware={hardware} telemetry={telemetry} contracts={contracts} templates={templates} compatibility={compatibility} workers={workers} />}
      </main>
    </div>
  );
}

function LiveView({ live, workers, models, operate, pending }: {
  live: LiveState; workers: Worker[]; models: ModelEntry[];
  operate: (worker: Worker, operation: WorkerOperation) => Promise<void>; pending: string | null;
}) {
  const [routeFeedback, setRouteFeedback] = useState<string | null>(null);
  const [smokingRoute, setSmokingRoute] = useState<string | null>(null);
  const smokeRoute = async (routeId: string) => {
    if (!live.active_event) return;
    setSmokingRoute(routeId); setRouteFeedback(null);
    try {
      await postJson(`/api/events/${live.active_event.id}/routes/${routeId}/smoke`);
      setRouteFeedback("The published Route responded through the gateway.");
    } catch (reason) { setRouteFeedback(messageFrom(reason)); }
    finally { setSmokingRoute(null); }
  };
  if (!workers.length || !live.active_event) return (
    <div className="view-stack">
      <section className="hero-panel"><div><p className="eyebrow">Initial setup</p><h2>Build your first local route</h2><p>ModelDeck starts empty: create a Worker from a discovered Model, create an Event and Route, then publish it.</p></div></section>
      <section className="panel"><PanelHeading title="Setup checklist" detail={`${models.length} cached Models discovered`} />
        <ol className="setup-list"><li className={models.length ? "done" : ""}>Discover a cached Model</li><li className={workers.length ? "done" : ""}>Create a Worker</li><li className={live.active_event ? "done" : ""}>Create and publish an Event</li><li>Start and smoke-test the Route’s Worker</li></ol>
      </section>
    </div>
  );
  return <div className="view-stack">
    <section className="hero-panel"><div><p className="eyebrow">Published Event · revision {live.active_event.revision}</p><h2>{live.active_event.name}</h2><p>Publishing controls routing only. Worker processes remain under explicit operator control.</p></div><div className="hero-status"><StatusDot state={live.routes.every((route) => route.ready) ? "good" : "warn"} /><span>{live.routes.filter((route) => route.ready).length} of {live.routes.length} Routes ready</span></div></section>
    <section className="panel table-panel"><PanelHeading title="Live Routes" detail={`${live.routes.length} published`} />
      {routeFeedback && <div className="configuration-feedback">{routeFeedback}</div>}
      {live.routes.length ? <div className="active-route-table-wrap"><table className="active-route-table"><thead><tr><th>Public route</th><th>Protocol</th><th>Worker order</th><th>Effective Worker</th><th>Actions</th></tr></thead><tbody>
        {live.routes.map((route) => <tr key={route.id}><td><strong>{route.display_name}</strong><code>{route.public_name}</code></td><td>{route.protocol_contract}</td><td><div className="active-worker-chain">{route.workers.map((worker, index) => <span key={worker.id}>{index === 0 ? "Primary" : `Backup ${index}`}: {worker.name} <StateBadge state={worker.state} /></span>)}</div></td><td>{route.effective_worker?.name ?? "No ready Worker"}</td><td>{route.workers[0] && <div className="button-row"><button disabled={pending !== null || route.workers[0].state === "ready"} onClick={() => void operate(route.workers[0], "start")}>Start primary</button><button className="secondary" disabled={pending !== null || smokingRoute !== null || !route.ready} onClick={() => void smokeRoute(route.id)}>Rehearse Route</button></div>}</td></tr>)}
      </tbody></table></div> : <p className="muted">This Event publishes no Routes.</p>}
    </section>
  </div>;
}

function EventsView({ events, workers, contracts, openDay, refresh }: {
  events: EventRecord[]; workers: Worker[]; contracts: ProtocolContract[]; openDay: boolean; refresh: () => Promise<void>;
}) {
  const [selectedId, setSelectedId] = useState(events[0]?.definition.id ?? "");
  const selected = events.find((event) => event.definition.id === selectedId) ?? events[0];
  const [draft, setDraft] = useState<EventDefinition | null>(selected?.definition ?? null);
  const [saveState, setSaveState] = useState("Saved");
  const [validation, setValidation] = useState<EventValidation | null>(null);
  const [revisions, setRevisions] = useState<EventRevision[]>([]);
  const [feedback, setFeedback] = useState<string | null>(null);

  useEffect(() => { setDraft(selected?.definition ?? null); setSaveState("Saved"); setValidation(null); }, [selected?.definition]);
  useEffect(() => {
    if (!selectedId && events[0]) setSelectedId(events[0].definition.id);
  }, [events, selectedId]);
  useEffect(() => {
    if (!draft || !selected || JSON.stringify(draft) === JSON.stringify(selected.definition) || openDay) return;
    setSaveState("Saving…");
    const timer = window.setTimeout(() => {
      putJson(`/api/events/${draft.id}/draft`, draft).then(() => setSaveState("Saved"))
        .catch((reason) => { setSaveState("Save failed"); setFeedback(messageFrom(reason)); });
    }, 500);
    return () => window.clearTimeout(timer);
  }, [draft, selected, openDay]);

  const createEvent = async () => {
    const definition: EventDefinition = { id: crypto.randomUUID(), name: "New Event", description: "", qualification: "compatible", demos: [], routes: [] };
    const record = await postJson<EventRecord>("/api/events", definition); await refresh(); setSelectedId(record.definition.id);
  };
  const validate = async () => { if (!draft) return; setValidation(await postJson(`/api/events/${draft.id}/validate`)); };
  const publish = async () => { if (!draft) return; await putJson(`/api/events/${draft.id}/draft`, draft); await postJson(`/api/events/${draft.id}/publish`); setFeedback("Routing published. No Workers were started or stopped."); await refresh(); };
  const discard = async () => { if (!draft) return; await deleteJson(`/api/events/${draft.id}/draft`); await refresh(); };
  const deleteEvent = async () => { if (!draft || selected?.latest_revision || !window.confirm(`Delete draft-only Event “${draft.name}”?`)) return; await deleteJson(`/api/events/${draft.id}`); setSelectedId(""); await refresh(); };
  const loadRevisions = async () => { if (!draft) return; const result = await getJson<{ revisions: EventRevision[] }>(`/api/events/${draft.id}/revisions`); setRevisions(result.revisions); };
  const updateRoute = (id: string, change: Partial<EventDefinition["routes"][number]>) => setDraft((current) => current && ({ ...current, routes: current.routes.map((route) => route.id === id ? { ...route, ...change } : route) }));
  const removeRoute = (id: string) => setDraft((current) => current && ({ ...current, routes: current.routes.filter((route) => route.id !== id), demos: current.demos.map((demo) => ({ ...demo, route_ids: demo.route_ids.filter((routeId) => routeId !== id) })) }));

  return <div className="view-stack">
    <div className="view-actions"><p>Events describe what demos expect. Their Routes are shared and publish independently of Worker processes.</p><button disabled={openDay} onClick={() => void createEvent().catch((reason) => setFeedback(messageFrom(reason)))}>Create Event</button></div>
    {!selected || !draft ? <section className="empty-state"><h2>No Events yet</h2><p>Create an Event after configuring at least one Worker.</p></section> : <div className="event-layout">
      <aside className="panel event-list">{events.map((event) => <button className={`event-select ${event.definition.id === draft.id ? "active" : ""}`} key={event.definition.id} onClick={() => setSelectedId(event.definition.id)}><span><strong>{event.definition.name}</strong><small>{event.active ? `Live revision ${event.active_revision}` : event.latest_revision ? `Published revision ${event.latest_revision}` : "Draft only"}</small></span></button>)}</aside>
      <section className="panel event-detail"><div className="event-heading"><div><p className="eyebrow">{selected.active ? `Live revision ${selected.active_revision}` : "Draft"} · {saveState}</p><h2>{draft.name}</h2></div><StateBadge state={selected.active ? "ready" : "stopped"} /></div>
        {feedback && <div className="configuration-feedback">{feedback}</div>}
        <div className="button-row event-actions"><button className="secondary" onClick={() => void validate()}>Validate</button><button disabled={openDay || saveState === "Saving…"} onClick={() => void publish().catch((reason) => setFeedback(messageFrom(reason)))}>Publish routing</button><button className="secondary" disabled={openDay || !selected.latest_revision} onClick={() => void discard().catch((reason) => setFeedback(messageFrom(reason)))}>Discard draft</button><button className="secondary" onClick={() => void loadRevisions()}>History</button><button className="secondary danger" disabled={openDay || Boolean(selected.latest_revision)} onClick={() => void deleteEvent().catch((reason) => setFeedback(messageFrom(reason)))}>Delete Event</button></div>
        {validation && <div className={`validation-summary ${validation.valid ? "good" : "bad"}`}><strong>{validation.valid ? "Ready to publish" : "Validation needs attention"}</strong>{validation.errors.length > 0 && <ul>{validation.errors.map((error, index) => <li key={index}>{error.message}</li>)}</ul>}{validation.warnings.length > 0 && <ul>{validation.warnings.map((warning, index) => <li key={`warning-${index}`}>Note: {warning.message}</li>)}</ul>}</div>}
        {revisions.length > 0 && <details className="revision-history" open><summary>Published revisions</summary><div>{revisions.map((revision) => <article key={revision.revision}><span><strong>Revision {revision.revision}</strong><small>{new Date(revision.published_at).toLocaleString()}</small></span><button className="secondary" disabled={revision.active || openDay} onClick={() => void postJson(`/api/events/${draft.id}/revisions/${revision.revision}/publish`).then(refresh)}>Make live</button></article>)}</div></details>}
        <div className="event-editor">
          <div className="field-grid"><label>Event name<input value={draft.name} disabled={openDay} onChange={(event) => setDraft({ ...draft, name: event.target.value })} /></label><label>Qualification<select value={draft.qualification} disabled={openDay} onChange={(event) => setDraft({ ...draft, qualification: event.target.value as EventDefinition["qualification"] })}><option value="compatible">Protocol compatible</option><option value="tested-working">Tested working (Open Day)</option></select></label></div>
          <label>Description<textarea value={draft.description} disabled={openDay} onChange={(event) => setDraft({ ...draft, description: event.target.value })} /></label>
          <div className="editor-section-heading"><div><h3>Routes</h3><p className="muted">The first Worker is primary; backups follow in the exact displayed order.</p></div><button disabled={openDay || !workers.length} onClick={() => setDraft({ ...draft, routes: [...draft.routes, { id: crypto.randomUUID(), display_name: "New Route", public_name: `route-${draft.routes.length + 1}`, protocol_contract: contracts[0]?.id ?? "openai-chat-v1", worker_ids: [workers[0].id] }] })}>Add Route</button></div>
          <div className="route-editor-list">{draft.routes.map((route) => <article className="route-editor" key={route.id}><div className="route-editor-title"><h4>{route.display_name}</h4><button className="secondary danger" disabled={openDay} onClick={() => removeRoute(route.id)}>Remove</button></div><div className="field-grid"><label>Display name<input value={route.display_name} disabled={openDay} onChange={(event) => updateRoute(route.id, { display_name: event.target.value })} /></label><label>Public model name<input value={route.public_name} disabled={openDay} onChange={(event) => updateRoute(route.id, { public_name: event.target.value })} /></label><label>Protocol contract<select value={route.protocol_contract} disabled={openDay} onChange={(event) => updateRoute(route.id, { protocol_contract: event.target.value })}>{contracts.map((contract) => <option value={contract.id} key={contract.id}>{contract.display_name}</option>)}</select></label></div>
            <h4>Worker order</h4><div className="worker-order-list">{route.worker_ids.map((workerId, index) => <div key={`${workerId}-${index}`}><span className="order-label">{index === 0 ? "Primary" : `Backup ${index}`}</span><select value={workerId} disabled={openDay} onChange={(event) => { const next = [...route.worker_ids]; next[index] = event.target.value; updateRoute(route.id, { worker_ids: next }); }}>{workers.map((worker) => <option key={worker.id} value={worker.id} disabled={route.worker_ids.includes(worker.id) && worker.id !== workerId}>{worker.name} · {worker.model_id}</option>)}</select><button className="secondary" disabled={openDay || index === 0} onClick={() => { const next = [...route.worker_ids]; [next[index - 1], next[index]] = [next[index], next[index - 1]]; updateRoute(route.id, { worker_ids: next }); }}>↑</button><button className="secondary" disabled={openDay || index === route.worker_ids.length - 1} onClick={() => { const next = [...route.worker_ids]; [next[index], next[index + 1]] = [next[index + 1], next[index]]; updateRoute(route.id, { worker_ids: next }); }}>↓</button><button className="secondary danger" disabled={openDay || index === 0} onClick={() => updateRoute(route.id, { worker_ids: route.worker_ids.filter((_, item) => item !== index) })}>Remove</button></div>)}</div>
            <button className="secondary" disabled={openDay || workers.every((worker) => route.worker_ids.includes(worker.id))} onClick={() => { const worker = workers.find((item) => !route.worker_ids.includes(item.id)); if (worker) updateRoute(route.id, { worker_ids: [...route.worker_ids, worker.id] }); }}>Add backup</button>
          </article>)}</div>
          <div className="editor-section-heading"><div><h3>Demos</h3><p className="muted">A Demo can reference any number of the Event’s shared Routes.</p></div><button disabled={openDay} onClick={() => setDraft({ ...draft, demos: [...draft.demos, { id: crypto.randomUUID(), name: "New Demo", route_ids: [] }] })}>Add Demo</button></div>
          <div className="demo-editor-list">{draft.demos.map((demo) => <article className="route-editor" key={demo.id}><div className="route-editor-title"><label>Demo name<input value={demo.name} disabled={openDay} onChange={(event) => setDraft({ ...draft, demos: draft.demos.map((item) => item.id === demo.id ? { ...item, name: event.target.value } : item) })} /></label><button className="secondary danger" disabled={openDay} onClick={() => setDraft({ ...draft, demos: draft.demos.filter((item) => item.id !== demo.id) })}>Remove</button></div><div className="route-membership">{draft.routes.map((route) => <label key={route.id}><input type="checkbox" checked={demo.route_ids.includes(route.id)} disabled={openDay} onChange={(event) => setDraft({ ...draft, demos: draft.demos.map((item) => item.id === demo.id ? { ...item, route_ids: event.target.checked ? [...item.route_ids, route.id] : item.route_ids.filter((id) => id !== route.id) } : item) })} /> {route.display_name}</label>)}</div></article>)}</div>
        </div>
      </section>
    </div>}
  </div>;
}

function WorkersView({ workers, pending, operate, refresh, openDay }: { workers: Worker[]; pending: string | null; operate: (worker: Worker, operation: WorkerOperation) => Promise<void>; refresh: () => Promise<void>; openDay: boolean }) {
  const [sort, setSort] = useState<WorkerSort>("name-asc");
  const [feedback, setFeedback] = useState<string | null>(null);
  const sorted = useMemo(() => [...workers].sort((a, b) => sort === "name-desc" ? b.name.localeCompare(a.name) : sort === "model-asc" ? a.model_id.localeCompare(b.model_id) : sort === "runtime-asc" ? a.runtime.localeCompare(b.runtime) : sort === "state" ? a.state.localeCompare(b.state) : a.name.localeCompare(b.name)), [workers, sort]);
  const rename = async (worker: Worker) => { const name = window.prompt("Worker name", worker.name)?.trim(); if (!name || name === worker.name) return; await patchJson(`/api/workers/${worker.id}`, { name }); await refresh(); };
  const archive = async (worker: Worker) => { if (!window.confirm(`Archive Worker “${worker.name}”? Cached Model files will be kept.`)) return; await deleteJson(`/api/workers/${worker.id}`); setFeedback(`Archived Worker “${worker.name}”; its cached Model was kept.`); await refresh(); };
  return <div className="view-stack"><div className="view-actions"><p>A Worker is one configured, startable service. Its name is editable; its execution identity is not.</p><div className="worker-toolbar"><label>Sort workers<select value={sort} onChange={(event) => setSort(event.target.value as WorkerSort)}><option value="name-asc">Name A–Z</option><option value="name-desc">Name Z–A</option><option value="model-asc">Model</option><option value="runtime-asc">Runtime</option><option value="state">State</option></select></label></div></div>
    {feedback && <div className="configuration-feedback">{feedback}</div>}
    {!workers.length ? <section className="empty-state"><h2>No Workers configured</h2><p>Create one from the Models view. ModelDeck does not create packaged Worker cards.</p></section> : <div className="worker-grid">{sorted.map((worker) => <article className={`worker-card state-${worker.state}`} key={worker.id}><div className="worker-card-heading"><div><p className="worker-id">{worker.generation_family}</p><h3>{worker.name}</h3></div><StateBadge state={worker.state} /></div><p className="worker-summary">{worker.model_id} · {worker.runtime}</p>{worker.last_error && <p className="inline-error">{worker.last_error}</p>}<details><summary>Immutable execution details</summary><DefinitionList rows={[["Internal ID", worker.id], ["Revision", worker.revision], ["Runtime", worker.runtime], ["Template", worker.runtime_template_id ?? "Built in"], ["Port", String(worker.port)], ["Lifecycle", worker.lifecycle], ["Data type", worker.dtype]]} /></details><div className="button-row"><button className="secondary" disabled={openDay} onClick={() => void rename(worker).catch((reason) => setFeedback(messageFrom(reason)))}>Rename</button><button disabled={pending !== null || worker.state === "ready"} onClick={() => void operate(worker, "start")}>Start</button><button className="secondary" disabled={pending !== null || worker.state !== "ready"} onClick={() => void operate(worker, "smoke")}>Smoke</button><button className="secondary" disabled={pending !== null || worker.state === "stopped"} onClick={() => void operate(worker, "stop")}>Stop</button><button className="secondary danger" disabled={openDay || pending !== null || worker.state !== "stopped"} onClick={() => void archive(worker).catch((reason) => setFeedback(messageFrom(reason)))}>Archive</button></div></article>)}</div>}
  </div>;
}

function ModelsView({ models, templates, refresh, openDay }: { models: ModelEntry[]; templates: RuntimeTemplate[]; refresh: () => Promise<void>; openDay: boolean }) {
  const [sort, setSort] = useState<ModelSort>("name-asc");
  const [query, setQuery] = useState("");
  const [configuring, setConfiguring] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [runtime, setRuntime] = useState("");
  const [artifact, setArtifact] = useState("");
  const [feedback, setFeedback] = useState<string | null>(null);
  const sorted = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    return models.filter((model) => !needle || [model.model_id, model.generation_family_hint, model.runnable_reason, ...model.capability_hints].some((value) => value?.toLocaleLowerCase().includes(needle))).sort((a, b) => sort === "name-desc" ? b.model_id.localeCompare(a.model_id) : sort === "size-desc" ? b.physical_size_bytes - a.physical_size_bytes : sort === "size-asc" ? a.physical_size_bytes - b.physical_size_bytes : sort === "readiness" ? Number(b.runnable) - Number(a.runnable) : sort === "workers" ? b.worker_count - a.worker_count : a.model_id.localeCompare(b.model_id));
  }, [models, query, sort]);
  const begin = (model: ModelEntry) => { setConfiguring(`${model.model_id}@${model.revision}`); setName(model.model_id.split("/").at(-1)?.replaceAll("-", " ") ?? "New Worker"); setRuntime(model.configuration_support ?? ""); setArtifact(model.artifacts?.[0]?.artifact_id ?? ""); setFeedback(null); };
  const create = async (model: ModelEntry) => { await postJson("/api/workers", { name, model_id: model.model_id, revision: model.revision, dtype: "float16", lifecycle: "on-demand", context_length: 2048, maximum_new_tokens: 128, maximum_denoising_steps: 24, runtime_template_id: runtime || undefined, artifact_id: artifact || undefined }); setConfiguring(null); setFeedback(`Created Worker “${name}”.`); await refresh(); };
  return <div className="view-stack"><div className="view-actions"><p>Models are read-only discoveries from the local Hugging Face cache. Create as many Workers as a Model needs.</p><div className="model-library-toolbar"><label>Search models<input type="search" value={query} placeholder="Name or capability" onChange={(event) => setQuery(event.target.value)} /></label><label>Sort models<select value={sort} onChange={(event) => setSort(event.target.value as ModelSort)}><option value="name-asc">Name A–Z</option><option value="name-desc">Name Z–A</option><option value="readiness">Runnable first</option><option value="workers">Most Workers</option><option value="size-desc">Largest</option><option value="size-asc">Smallest</option></select></label></div></div>{openDay && <div className="configuration-feedback">Open Day mode locks configuration. Restart ModelDeck without <code>-OpenDay</code> to create Workers.</div>}{feedback && <div className="configuration-feedback good">{feedback}</div>}
    <section className="panel"><PanelHeading title="Discovered Models" detail={query.trim() ? `${sorted.length} of ${models.length} cached` : `${models.length} cached`} />{sorted.length ? <div className="model-list">{sorted.map((model) => { const key = `${model.model_id}@${model.revision}`; const baseline = templates.find((item) => item.id === model.configuration_support); const availableTemplates = templates.filter((item) => baseline && item.generation_family === baseline.generation_family && item.cache_setting === baseline.cache_setting && item.uses_base_model_identity === baseline.uses_base_model_identity); return <article className="model-row" key={key}><div className="model-main"><div><h3>{model.model_id}</h3><p>{model.generation_family_hint ?? "Unclassified"} · {formatBytes(model.physical_size_bytes)} · {model.worker_count} Worker{model.worker_count === 1 ? "" : "s"}</p><div className="tag-list">{model.capability_hints.map((hint) => <span className="tag" key={hint}>{humanise(hint)}</span>)}</div></div><StateBadge state={model.runnable ? "recognised" : model.download_state} /></div><p className="model-stage">{model.runnable_reason}</p>{configuring !== key ? <div className="model-actions"><button disabled={openDay || !model.runnable || !model.modeldeck_allowed || !model.revision} onClick={() => begin(model)}>Create Worker</button></div> : <div className="runtime-form"><div className="runtime-form-heading"><strong>Create a Worker</strong><small>The trusted runtime determines the immutable execution identity.</small></div><div className="runtime-fields"><label>Worker name<input value={name} onChange={(event) => setName(event.target.value)} /></label><label>Runtime<select value={runtime} onChange={(event) => setRuntime(event.target.value)}>{availableTemplates.map((template) => <option key={template.id} value={template.id}>{template.display_name}</option>)}</select></label>{model.artifacts && model.artifacts.length > 0 && <label>Model artefact<select value={artifact} onChange={(event) => setArtifact(event.target.value)}>{model.artifacts.map((item) => <option key={item.artifact_id} value={item.artifact_id}>{item.artifact_id} · {item.filenames.join(", ")}</option>)}</select></label>}</div><div className="runtime-form-actions"><button disabled={openDay} onClick={() => void create(model).catch((reason) => setFeedback(messageFrom(reason)))}>Create Worker</button><button className="secondary" onClick={() => setConfiguring(null)}>Cancel</button></div></div>}</article>; })}</div> : <div className="empty-state compact"><h3>No Models match “{query.trim()}”</h3><p>Try a model name, generation family or capability.</p></div>}</section>
  </div>;
}

function AdvancedView({ hardware, telemetry, contracts, templates, compatibility, workers }: { hardware: HardwareProbe; telemetry: Telemetry; contracts: ProtocolContract[]; templates: RuntimeTemplate[]; compatibility: CompatibilityTest[]; workers: Worker[] }) {
  return <div className="view-stack"><section className="panel"><PanelHeading title="Detected hardware" detail="Reported, never assumed" /><DefinitionList rows={[["Configured target", `${hardware.configured.gpu} (${hardware.configured.gpu_architecture})`], ["Detected Fedora", hardware.detected.fedora_release ?? "Not detected"], ["Kernel", hardware.detected.kernel], ["ROCm packages", hardware.detected.rocm_packages.join(", ") || "Not detected"], ["Available memory", formatBytes(telemetry.memory.available_bytes)]]} /></section>
    <div className="two-column"><section className="panel"><PanelHeading title="Trusted protocol contracts" detail={`${contracts.length} code-owned`} /><ul className="status-list">{contracts.map((contract) => <li key={contract.id}><StatusDot state="good" /><span><strong>{contract.display_name}</strong><small>{contract.id} · {contract.surfaces.join(", ")}</small></span></li>)}</ul></section><section className="panel"><PanelHeading title="Trusted runtimes" detail={`${templates.length} installed`} /><ul className="status-list">{templates.map((template) => <li key={template.id}><StatusDot state="good" /><span><strong>{template.display_name}</strong><small>{template.id} · {template.package_version}</small></span></li>)}</ul></section></div>
    <section className="panel"><PanelHeading title="Compatibility evidence" detail={`${compatibility.length} records`} /><div className="evidence-list">{compatibility.length ? compatibility.map((test) => <details className="evidence-row" key={test.id}><summary><span><StateBadge state={test.result} /><strong>{String(test.evidence.model_id ?? "Unknown Model")}</strong><small>{new Date(test.tested_at).toLocaleString()}</small></span><code>{test.fingerprint.slice(0, 12)}</code></summary><DefinitionList rows={Object.entries(test.evidence).slice(0, 16).map(([key, value]) => [humanise(key), String(value ?? "—")])} /></details>) : <p className="muted">Smoke-test a Worker to record evidence.</p>}</div></section>
    <LogsPanel workers={workers} />
  </div>;
}

function LogsPanel({ workers }: { workers: Worker[] }) {
  const [workerId, setWorkerId] = useState(workers[0]?.id ?? "");
  const [logs, setLogs] = useState<WorkerLog[]>([]);
  useEffect(() => { if (!workerId) return; getJson<{ logs: WorkerLog[] }>(`/api/workers/${workerId}/logs`).then((value) => setLogs(value.logs)).catch(() => setLogs([])); }, [workerId]);
  return <section className="panel log-panel"><div className="log-toolbar"><div><label htmlFor="log-worker">Worker logs</label><select id="log-worker" value={workerId} onChange={(event) => setWorkerId(event.target.value)}>{workers.map((worker) => <option key={worker.id} value={worker.id}>{worker.name}</option>)}</select></div></div><div className="log-view">{logs.length ? logs.map((log, index) => <div className={`log-entry ${log.level}`} key={`${log.timestamp}-${index}`}><time>{new Date(log.timestamp).toLocaleTimeString()}</time><span>{log.source}</span><code>{log.message}</code></div>) : <p>No logs for this Worker.</p>}</div></section>;
}

function Loading() { return <main className="loading-screen"><div className="brand-mark">MD</div><h1>Starting operator console</h1><p>Reading local Events, Routes, Workers and Models…</p><div className="loading-bar"><span /></div></main>; }
function Unavailable({ retry }: { retry: () => Promise<void> }) { return <section className="empty-state"><span className="empty-icon">!</span><h2>Management data is unavailable</h2><p>No cloud service was contacted.</p><button onClick={() => void retry()}>Retry local connection</button></section>; }
function PanelHeading({ title, detail }: { title: string; detail: string }) { return <div className="panel-heading"><h2>{title}</h2><span>{detail}</span></div>; }
function StatusDot({ state }: { state: "good" | "warn" | "bad" | "neutral" }) { return <span className={`status-dot ${state}`} aria-hidden="true" />; }
function StateBadge({ state }: { state: string }) { return <span className={`state-badge state-${state}`}>{humanise(state)}</span>; }
function DefinitionList({ rows }: { rows: Array<[string, string]> }) { return <dl className="definition-list compact">{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>; }
function formatBytes(value: number) { if (!Number.isFinite(value) || value <= 0) return "0 B"; const units = ["B", "KiB", "MiB", "GiB", "TiB"]; const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1); return `${(value / 1024 ** index).toFixed(index > 2 ? 1 : 0)} ${units[index]}`; }
function humanise(value: string) { return value.replaceAll("_", " ").replaceAll("-", " "); }
function messageFrom(reason: unknown) { return reason instanceof Error ? reason.message : "The operation failed."; }

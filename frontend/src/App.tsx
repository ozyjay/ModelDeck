import { createContext, useCallback, useContext, useEffect, useId, useMemo, useState } from "react";
import type { Dispatch, ReactNode, SetStateAction } from "react";

import { ApiError, deleteJson, getJson, patchJson, postJson, putJson } from "./api";
import type {
  CompatibilityTest, EventDefinition, EventRecord, EventRevision, EventValidation,
  GatewayStatus, HardwareProbe, LiveState, ManagementHealth, ModelEntry,
  MockScenario, MockWorkerTemplate, ProtocolContract, RuntimeTemplate, Telemetry, Worker, WorkerLog,
} from "./types";

type View = "live" | "events" | "workers" | "models" | "advanced";
type WorkerOperation = "start" | "stop" | "restart" | "smoke";
type WorkerSort = "name-asc" | "name-desc" | "model-asc" | "runtime-asc" | "state";
type ModelSort = "name-asc" | "name-desc" | "size-desc" | "size-asc" | "readiness" | "workers";

interface CollapsePreferences {
  allCollapsed: boolean;
  sections: Record<string, boolean>;
}

interface CollapseControls {
  preferences: CollapsePreferences;
  setAllCollapsed: (collapsed: boolean) => void;
  toggleSection: (sectionId: string) => void;
}

const COLLAPSE_STORAGE_KEY = "modeldeck-collapse-preferences-v1";
const WORKER_LIBRARY_STORAGE_KEY = "modeldeck-worker-library-preferences-v1";
const MODEL_LIBRARY_STORAGE_KEY = "modeldeck-model-library-preferences-v1";
const CollapseContext = createContext<CollapseControls | null>(null);

interface WorkerLibraryPreferences { query: string; state: string; runtime: string; sort: WorkerSort }
interface ModelLibraryPreferences { query: string; sort: ModelSort }

const WORKER_SORTS: WorkerSort[] = ["name-asc", "name-desc", "model-asc", "runtime-asc", "state"];
const MODEL_SORTS: ModelSort[] = ["name-asc", "name-desc", "size-desc", "size-asc", "readiness", "workers"];

function storedObject(key: string): Record<string, unknown> {
  try {
    const stored = window.localStorage.getItem(key);
    const parsed: unknown = stored ? JSON.parse(stored) : null;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

function loadWorkerLibraryPreferences(): WorkerLibraryPreferences {
  const stored = storedObject(WORKER_LIBRARY_STORAGE_KEY);
  return {
    query: typeof stored.query === "string" ? stored.query : "",
    state: typeof stored.state === "string" ? stored.state : "",
    runtime: typeof stored.runtime === "string" ? stored.runtime : "",
    sort: WORKER_SORTS.includes(stored.sort as WorkerSort) ? stored.sort as WorkerSort : "name-asc",
  };
}

function loadModelLibraryPreferences(): ModelLibraryPreferences {
  const stored = storedObject(MODEL_LIBRARY_STORAGE_KEY);
  return {
    query: typeof stored.query === "string" ? stored.query : "",
    sort: MODEL_SORTS.includes(stored.sort as ModelSort) ? stored.sort as ModelSort : "name-asc",
  };
}

function useStoredPreferences<T>(key: string, load: () => T): [T, Dispatch<SetStateAction<T>>] {
  const [preferences, setPreferences] = useState<T>(load);
  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(preferences));
    } catch {
      console.warn(`${key} could not be saved locally.`);
    }
  }, [key, preferences]);
  return [preferences, setPreferences];
}

function loadCollapsePreferences(): CollapsePreferences {
  try {
    const stored = window.localStorage.getItem(COLLAPSE_STORAGE_KEY);
    if (!stored) return { allCollapsed: false, sections: {} };
    const parsed = JSON.parse(stored) as Partial<CollapsePreferences>;
    return {
      allCollapsed: parsed.allCollapsed === true,
      sections: parsed.sections && typeof parsed.sections === "object" ? parsed.sections : {},
    };
  } catch {
    return { allCollapsed: false, sections: {} };
  }
}

function useCollapse(sectionId: string) {
  const controls = useContext(CollapseContext);
  if (!controls) throw new Error("Collapse controls are unavailable");
  return {
    collapsed: controls.preferences.sections[sectionId] ?? controls.preferences.allCollapsed,
    toggle: () => controls.toggleSection(sectionId),
  };
}

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
  const [mockTemplates, setMockTemplates] = useState<MockWorkerTemplate[]>([]);
  const [templates, setTemplates] = useState<RuntimeTemplate[]>([]);
  const [compatibility, setCompatibility] = useState<CompatibilityTest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<Set<string>>(() => new Set());
  const [collapsePreferences, setCollapsePreferences] = useState<CollapsePreferences>(loadCollapsePreferences);

  useEffect(() => {
    try {
      window.localStorage.setItem(COLLAPSE_STORAGE_KEY, JSON.stringify(collapsePreferences));
    } catch {
      console.warn("Collapse preferences could not be saved locally.");
    }
  }, [collapsePreferences]);

  const collapseControls = useMemo<CollapseControls>(() => ({
    preferences: collapsePreferences,
    setAllCollapsed: (collapsed) => setCollapsePreferences({ allCollapsed: collapsed, sections: {} }),
    toggleSection: (sectionId) => setCollapsePreferences((current) => {
      const collapsed = current.sections[sectionId] ?? current.allCollapsed;
      return { ...current, sections: { ...current.sections, [sectionId]: !collapsed } };
    }),
  }), [collapsePreferences]);

  const refresh = useCallback(async () => {
    const [nextHealth, nextGateway, nextHardware, nextTelemetry, nextLive, nextWorkers,
      nextEvents, catalogue, contractResponse, mockTemplateResponse, templateResponse, tests] = await Promise.all([
      getJson<ManagementHealth>("/api/health"),
      getJson<GatewayStatus>("/api/gateway/status"),
      getJson<HardwareProbe>("/api/hardware"),
      getJson<Telemetry>("/api/telemetry"),
      getJson<LiveState>("/api/live"),
      getJson<Worker[]>("/api/workers"),
      getJson<{ events: EventRecord[] }>("/api/events"),
      getJson<{ models: ModelEntry[] }>("/api/catalogue"),
      getJson<{ contracts: ProtocolContract[] }>("/api/protocol-contracts"),
      getJson<{ templates: MockWorkerTemplate[] }>("/api/mock-worker-templates"),
      getJson<{ templates: RuntimeTemplate[] }>("/api/runtime-templates"),
      getJson<{ tests: CompatibilityTest[] }>("/api/compatibility"),
    ]);
    setHealth(nextHealth); setGateway(nextGateway); setHardware(nextHardware);
    setTelemetry(nextTelemetry); setLive(nextLive); setWorkers(nextWorkers);
    setEvents(nextEvents.events); setModels(catalogue.models);
    setContracts(contractResponse.contracts); setMockTemplates(mockTemplateResponse.templates); setTemplates(templateResponse.templates);
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
    setPending((current) => new Set(current).add(key)); setError(null);
    try {
      const result = await postJson<{ ok?: boolean; test?: { evidence?: { error_summary?: string } } }>(`/api/workers/${worker.id}/${operation}`);
      await refresh();
      if (operation === "smoke" && result.ok === false) {
        throw new Error(result.test?.evidence?.error_summary ?? "Worker generation smoke test failed.");
      }
    } catch (reason) { setError(messageFrom(reason)); }
    finally { setPending((current) => { const next = new Set(current); next.delete(key); return next; }); }
  };

  const navigate = (next: View, path: string) => {
    window.history.pushState({}, "", path); setView(next);
  };

  if (loading) return <Loading />;
  return (
    <CollapseContext.Provider value={collapseControls}>
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
            <button className="secondary collapse-all-button" onClick={() => collapseControls.setAllCollapsed(!collapsePreferences.allCollapsed)}>{collapsePreferences.allCollapsed ? "Expand all" : "Collapse all"}</button>
            {health && <div className={`mode-badge ${health.open_day ? "locked" : "unlocked"}`} aria-label="Configuration status"><StatusDot state={health.open_day ? "warn" : "good"} /><span>{health.open_day ? "Open Day · configuration locked" : "Configuration unlocked"}</span></div>}
            <div className={`gateway-badge ${gateway?.available ? "ready" : "unavailable"}`}><StatusDot state={gateway?.available ? "good" : "bad"} /><span>{gateway?.available ? "Gateway available" : "Gateway unavailable"}</span></div>
          </div>
        </header>
        {error && <div className="alert error" role="alert"><strong>Action failed</strong><span>{error}</span><button className="icon-button" onClick={() => setError(null)}>×</button></div>}
        {!health || !hardware || !telemetry || !gateway ? <Unavailable retry={refresh} />
          : view === "live" ? <LiveView live={live} workers={workers} models={models} operate={operate} pending={pending} />
          : view === "events" ? <EventsView events={events} workers={workers} contracts={contracts} mockTemplates={mockTemplates} openDay={health.open_day} refresh={refresh} />
          : view === "workers" ? <WorkersView workers={workers} templates={templates} mockTemplates={mockTemplates} pending={pending} operate={operate} refresh={refresh} openDay={health.open_day} />
          : view === "models" ? <ModelsView models={models} workers={workers} templates={templates} refresh={refresh} openDay={health.open_day} />
          : <AdvancedView hardware={hardware} telemetry={telemetry} contracts={contracts} templates={templates} compatibility={compatibility} workers={workers} />}
      </main>
    </div>
    </CollapseContext.Provider>
  );
}

function LiveView({ live, workers, models, operate, pending }: {
  live: LiveState; workers: Worker[]; models: ModelEntry[];
  operate: (worker: Worker, operation: WorkerOperation) => Promise<void>; pending: ReadonlySet<string>;
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
      <CollapsiblePanel sectionId="live-setup" title="Setup checklist" detail={`${models.length} cached Models discovered`}>
        <ol className="setup-list"><li className={models.length ? "done" : ""}>Discover a cached Model</li><li className={workers.length ? "done" : ""}>Create a Worker</li><li className={live.active_event ? "done" : ""}>Create and publish an Event</li><li>Start and smoke-test the Route’s Worker</li></ol>
      </CollapsiblePanel>
    </div>
  );
  return <div className="view-stack">
    <section className="hero-panel"><div><p className="eyebrow">Published Event · revision {live.active_event.revision}</p><h2>{live.active_event.name}</h2><p>Publishing controls routing only. Worker processes remain under explicit operator control.</p></div><div className="hero-status"><StatusDot state={live.routes.every((route) => route.ready) ? "good" : "warn"} /><span>{live.routes.filter((route) => route.ready).length} of {live.routes.length} Routes ready</span></div></section>
    <CollapsiblePanel sectionId="live-routes" title="Live Routes" detail={`${live.routes.length} published`} className="table-panel">
      {routeFeedback && <div className="configuration-feedback">{routeFeedback}</div>}
      {live.routes.length ? <div className="active-route-table-wrap"><table className="active-route-table"><thead><tr><th>Public route</th><th>Route status</th><th>Protocol</th><th>Worker order</th><th>Effective Worker</th><th>Actions</th></tr></thead><tbody>
        {live.routes.map((route) => <tr className={route.ready ? "route-ready" : "route-unavailable"} key={route.id}><td><strong>{route.display_name}</strong><code>{route.public_name}</code></td><td><div className={`route-readiness ${route.ready ? "ready" : "unavailable"}`} role="status" aria-label={`${route.display_name} Route status`}><StatusDot state={route.ready ? "good" : "warn"} /><span><strong>{route.ready ? "Ready" : "Not serving"}</strong><small>{route.ready ? "Accepting requests" : "Start a Worker"}</small></span></div></td><td>{route.protocol_contract}</td><td><div className="active-worker-chain">{route.workers.map((worker, index) => { const order = index === 0 ? "Primary" : `Backup ${index}`; return <div className="active-worker-item" aria-label={`${order} Worker ${worker.name}`} key={worker.id}><span><small>{order}</small><strong>{worker.name}</strong></span><StateBadge state={worker.state} /></div>; })}</div></td><td className={route.effective_worker ? "effective-worker" : "effective-worker unavailable"}>{route.effective_worker?.name ?? "No ready Worker"}</td><td>{route.workers[0] && <div className="button-row"><button disabled={workerOperationPending(pending, route.workers[0].id) || route.workers[0].state === "ready"} onClick={() => void operate(route.workers[0], "start")}>{pending.has(`${route.workers[0].id}:start`) ? "Starting…" : "Start primary"}</button><button className="secondary" disabled={smokingRoute !== null || !route.ready} onClick={() => void smokeRoute(route.id)}>Rehearse Route</button></div>}</td></tr>)}
      </tbody></table></div> : <p className="muted">This Event publishes no Routes.</p>}
    </CollapsiblePanel>
  </div>;
}

function EventsView({ events, workers, contracts, mockTemplates, openDay, refresh }: {
  events: EventRecord[]; workers: Worker[]; contracts: ProtocolContract[]; mockTemplates: MockWorkerTemplate[]; openDay: boolean; refresh: () => Promise<void>;
}) {
  const [selectedId, setSelectedId] = useState(events[0]?.definition.id ?? "");
  const selected = events.find((event) => event.definition.id === selectedId) ?? events[0];
  const [draft, setDraft] = useState<EventDefinition | null>(selected?.definition ?? null);
  const [saveState, setSaveState] = useState("Saved");
  const [validation, setValidation] = useState<EventValidation | null>(null);
  const [revisions, setRevisions] = useState<EventRevision[]>([]);
  const [feedback, setFeedback] = useState<string | null>(null);
  const publicNameConflicts = duplicatePublicNameConflicts(draft?.routes ?? []);
  const hasPublicNameConflicts = publicNameConflicts.size > 0;

  useEffect(() => { setDraft(selected?.definition ?? null); setSaveState("Saved"); setValidation(null); }, [selected?.definition]);
  useEffect(() => {
    if (!selectedId && events[0]) setSelectedId(events[0].definition.id);
  }, [events, selectedId]);
  useEffect(() => {
    if (!draft || !selected || JSON.stringify(draft) === JSON.stringify(selected.definition) || openDay) return;
    if (hasPublicNameConflicts) {
      setSaveState("Needs attention");
      return;
    }
    setSaveState("Saving…");
    const timer = window.setTimeout(() => {
      putJson(`/api/events/${draft.id}/draft`, draft).then(() => setSaveState("Saved"))
        .catch((reason) => { setSaveState("Save failed"); setFeedback(messageFrom(reason)); });
    }, 500);
    return () => window.clearTimeout(timer);
  }, [draft, selected, openDay, hasPublicNameConflicts]);

  const createEvent = async () => {
    const definition: EventDefinition = { id: crypto.randomUUID(), name: "New Event", description: "", qualification: "compatible", demos: [], routes: [] };
    const record = await postJson<EventRecord>("/api/events", definition); await refresh(); setSelectedId(record.definition.id);
  };
  const validate = async () => {
    if (!draft) return;
    if (!openDay) {
      setSaveState("Saving…");
      await putJson(`/api/events/${draft.id}/draft`, draft);
      setSaveState("Saved");
    }
    setValidation(await postJson(`/api/events/${draft.id}/validate`));
    setFeedback(null);
  };
  const publish = async () => {
    if (!draft) return;
    await putJson(`/api/events/${draft.id}/draft`, draft);
    try {
      await postJson(`/api/events/${draft.id}/publish`);
    } catch (reason) {
      const failedValidation = validationFromApiError(reason);
      if (failedValidation) setValidation(failedValidation);
      throw reason;
    }
    setFeedback("Routing published. No Workers were started or stopped.");
    await refresh();
  };
  const discard = async () => { if (!draft) return; await deleteJson(`/api/events/${draft.id}/draft`); await refresh(); };
  const deleteEvent = async () => { if (!draft || selected?.latest_revision || !window.confirm(`Delete draft-only Event “${draft.name}”?`)) return; await deleteJson(`/api/events/${draft.id}`); setSelectedId(""); await refresh(); };
  const loadRevisions = async () => { if (!draft) return; const result = await getJson<{ revisions: EventRevision[] }>(`/api/events/${draft.id}/revisions`); setRevisions(result.revisions); };
  const updateRoute = (id: string, change: Partial<EventDefinition["routes"][number]>) => {
    setFeedback(null);
    setValidation(null);
    setDraft((current) => current && ({ ...current, routes: current.routes.map((route) => route.id === id ? { ...route, ...change } : route) }));
  };
  const removeDemo = (id: string) => {
    const demo = draft?.demos.find((item) => item.id === id);
    if (!demo || !window.confirm(`Remove Demo “${demo.name}”?\n\nIts Route assignments will be removed from this draft. Shared Routes and Workers will be kept.`)) return;
    setDraft((current) => current && ({ ...current, demos: current.demos.filter((item) => item.id !== id) }));
  };
  const removeRoute = (id: string) => {
    const route = draft?.routes.find((item) => item.id === id);
    if (!route || !window.confirm(`Remove Route “${route.display_name}”?\n\nIt will be removed from every Demo in this draft. Its Workers will be kept.`)) return;
    setDraft((current) => current && ({ ...current, routes: current.routes.filter((item) => item.id !== id), demos: current.demos.map((demo) => ({ ...demo, route_ids: demo.route_ids.filter((routeId) => routeId !== id) })) }));
  };
  const removeRouteWorker = (routeId: string, workerIndex: number) => {
    const route = draft?.routes.find((item) => item.id === routeId);
    const worker = route ? workers.find((item) => item.id === route.worker_ids[workerIndex]) : null;
    if (!route || !worker || workerIndex === 0 || !window.confirm(`Remove Worker “${worker.name}” from Route “${route.display_name}”?\n\nThe Worker itself will be kept and can still be used by other Routes.`)) return;
    updateRoute(routeId, { worker_ids: route.worker_ids.filter((_, index) => index !== workerIndex) });
  };
  const toggleDemoRoute = (demoId: string, routeId: string, included: boolean) => setDraft((current) => current && ({
    ...current,
    demos: current.demos.map((demo) => demo.id === demoId ? {
      ...demo,
      route_ids: included ? [...demo.route_ids, routeId] : demo.route_ids.filter((id) => id !== routeId),
    } : demo),
  }));
  const assignCreatedMock = async (routeId: string, worker: Worker) => {
    if (!draft) return;
    const nextDraft = {
      ...draft,
      routes: draft.routes.map((route) => route.id === routeId
        ? { ...route, worker_ids: [...route.worker_ids, worker.id] }
        : route),
    };
    try {
      setSaveState("Saving…");
      await putJson(`/api/events/${draft.id}/draft`, nextDraft);
      setDraft(nextDraft);
      setSaveState("Saved");
      setFeedback(`Created mock Worker “${worker.name}” and added it as the last Route backup. Publish the draft when ready.`);
    } catch (reason) {
      setSaveState("Save failed");
      setFeedback(`Mock Worker “${worker.name}” was created, but it could not be assigned to the Route: ${messageFrom(reason)}`);
    }
    await refresh();
  };
  const assignedRouteIds = new Set(draft?.demos.flatMap((demo) => demo.route_ids) ?? []);
  const unassignedRoutes = draft?.routes.filter((route) => !assignedRouteIds.has(route.id)) ?? [];

  return <div className="view-stack">
    <div className="view-actions"><p>Events describe what demos expect. Their Routes are shared and publish independently of Worker processes.</p><button disabled={openDay} onClick={() => void createEvent().catch((reason) => setFeedback(messageFrom(reason)))}>Create Event</button></div>
    {!selected || !draft ? <section className="empty-state"><h2>No Events yet</h2><p>Create an Event after configuring at least one Worker.</p></section> : <div className="event-layout">
      <aside className="panel event-list">{events.map((event) => <button className={`event-select ${event.definition.id === draft.id ? "active" : ""}`} key={event.definition.id} onClick={() => setSelectedId(event.definition.id)}><span><strong>{event.definition.name}</strong><small>{event.active ? `Live revision ${event.active_revision}` : event.latest_revision ? `Published revision ${event.latest_revision}` : "Draft only"}</small></span></button>)}</aside>
      <CollapsiblePanel sectionId={`event-${draft.id}`} title={draft.name} detail={`${selected.active ? `Live revision ${selected.active_revision}` : "Draft"} · ${saveState}`} className="event-detail" accessory={<StateBadge state={selected.active ? "ready" : "stopped"} />}>
        {feedback && <div className="configuration-feedback">{feedback}</div>}
        <div className="button-row event-actions"><button className="secondary" disabled={hasPublicNameConflicts} onClick={() => void validate().catch((reason) => setFeedback(messageFrom(reason)))}>Validate</button><button disabled={openDay || saveState === "Saving…" || hasPublicNameConflicts} onClick={() => void publish().catch((reason) => setFeedback(messageFrom(reason)))}>Publish routing</button><button className="secondary" disabled={openDay || !selected.latest_revision} onClick={() => void discard().catch((reason) => setFeedback(messageFrom(reason)))}>Discard draft</button><button className="secondary" onClick={() => void loadRevisions()}>History</button><button className="secondary danger" disabled={openDay || Boolean(selected.latest_revision)} onClick={() => void deleteEvent().catch((reason) => setFeedback(messageFrom(reason)))}>Delete Event</button></div>
        {validation && <div className={`validation-summary ${validation.valid ? "good" : "bad"}`}><strong>{validation.valid ? "Ready to publish" : "Validation needs attention"}</strong>{validation.errors.length > 0 && <section aria-label="Validation errors"><h3>{validation.errors.length} error{validation.errors.length === 1 ? "" : "s"}</h3><ul>{validation.errors.map((error, index) => <ValidationIssue key={index} issue={error} draft={draft} workers={workers} />)}</ul></section>}{validation.warnings.length > 0 && <section aria-label="Validation notes"><h3>{validation.warnings.length} note{validation.warnings.length === 1 ? "" : "s"}</h3><ul>{validation.warnings.map((warning, index) => <ValidationIssue key={`warning-${index}`} issue={warning} draft={draft} workers={workers} note />)}</ul></section>}</div>}
        {revisions.length > 0 && <details className="revision-history" open><summary>Published revisions</summary><div>{revisions.map((revision) => <article key={revision.revision}><span><strong>Revision {revision.revision}</strong><small>{new Date(revision.published_at).toLocaleString()}</small></span><button className="secondary" disabled={revision.active || openDay} onClick={() => void postJson(`/api/events/${draft.id}/revisions/${revision.revision}/publish`).then(refresh)}>Make live</button></article>)}</div></details>}
        <div className="event-editor">
          <div className="field-grid"><label>Event name<input value={draft.name} disabled={openDay} onChange={(event) => setDraft({ ...draft, name: event.target.value })} /></label><label>Qualification<select value={draft.qualification} disabled={openDay} onChange={(event) => setDraft({ ...draft, qualification: event.target.value as EventDefinition["qualification"] })}><option value="compatible">Protocol compatible</option><option value="tested-working">Tested working (Open Day)</option></select></label></div>
          <label>Description<textarea value={draft.description} disabled={openDay} onChange={(event) => setDraft({ ...draft, description: event.target.value })} /></label>
          <CollapsibleEditorSection sectionId={`event-demos-${draft.id}`} title="Demos" description="Each Demo contains references to the shared Routes it uses." accessory={<button disabled={openDay} onClick={() => setDraft({ ...draft, demos: [...draft.demos, { id: crypto.randomUUID(), name: "New Demo", route_ids: [] }] })}>Add Demo</button>}>
            <div className="demo-editor-list">{draft.demos.map((demo) => <CollapsibleEditorCard sectionId={`event-demo-${draft.id}-${demo.id}`} label={`Demo ${demo.name}`} className="demo-editor" heading={<label>Demo name<input value={demo.name} disabled={openDay} onChange={(event) => setDraft({ ...draft, demos: draft.demos.map((item) => item.id === demo.id ? { ...item, name: event.target.value } : item) })} /></label>} accessory={<button className="secondary danger" aria-label={`Remove Demo ${demo.name}`} disabled={openDay} onClick={() => removeDemo(demo.id)}>Remove</button>} key={demo.id}><div className="demo-route-section"><h4>Routes used by this Demo</h4>{draft.routes.length ? <div className="demo-route-reference-list">{draft.routes.map((route) => { const included = demo.route_ids.includes(route.id); return <label className={`demo-route-reference${included ? " selected" : ""}`} key={route.id}><input type="checkbox" checked={included} disabled={openDay} onChange={(event) => toggleDemoRoute(demo.id, route.id, event.target.checked)} /><span><strong>{route.display_name}</strong><code>{route.public_name}</code></span></label>; })}</div> : <p className="muted">Create a shared Route below, then include it in this Demo.</p>}</div></CollapsibleEditorCard>)}</div>
          </CollapsibleEditorSection>
          <CollapsibleEditorSection sectionId={`event-routes-${draft.id}`} title="Routes" description="Configure the Routes available to this Event. A Route can be used by multiple Demos or remain unassigned." accessory={<button disabled={openDay || !workers.length} onClick={() => setDraft({ ...draft, routes: [...draft.routes, { id: crypto.randomUUID(), display_name: "New Route", public_name: `route-${draft.routes.length + 1}`, protocol_contract: contracts[0]?.id ?? "openai-chat-v1", worker_ids: [workers[0].id] }] })}>Add Route</button>}>
            <section className="unassigned-routes" aria-labelledby="unassigned-routes-heading"><div><h4 id="unassigned-routes-heading">Unassigned Routes</h4><small>Shared Routes that are not used by any Demo.</small></div>{unassignedRoutes.length ? <div className="compact-route-reference-list">{unassignedRoutes.map((route) => <div className="compact-route-reference" key={route.id}><strong>{route.display_name}</strong><code>{route.public_name}</code></div>)}</div> : <p className="muted">{draft.routes.length ? "Every shared Route is used by at least one Demo." : "No shared Routes have been created."}</p>}</section>
            <div className="route-editor-list">{draft.routes.map((route) => { const publicNameConflict = publicNameConflicts.get(route.id); return <CollapsibleEditorCard sectionId={`event-route-${draft.id}-${route.id}`} label={`Route ${route.display_name}`} heading={<h4>{route.display_name}</h4>} accessory={<button className="secondary danger" aria-label={`Remove Route ${route.display_name}`} disabled={openDay} onClick={() => removeRoute(route.id)}>Remove</button>} key={route.id}><div className="field-grid"><label>Route Label<input value={route.display_name} disabled={openDay} onChange={(event) => updateRoute(route.id, { display_name: event.target.value })} /></label><label>API Model ID<small className="field-help">Sent by clients in the <code>model</code> field and must be unique within this Event.</small><input aria-label="API Model ID" aria-invalid={Boolean(publicNameConflict)} aria-describedby={publicNameConflict ? `route-public-name-error-${route.id}` : undefined} value={route.public_name} disabled={openDay} onChange={(event) => updateRoute(route.id, { public_name: event.target.value })} />{publicNameConflict && <small className="field-error" id={`route-public-name-error-${route.id}`}>“{route.public_name}” is already used by {publicNameConflict}. Choose a unique API Model ID.</small>}</label><label>Protocol contract<select value={route.protocol_contract} disabled={openDay} onChange={(event) => updateRoute(route.id, { protocol_contract: event.target.value })}>{contracts.map((contract) => <option value={contract.id} key={contract.id}>{contract.display_name}</option>)}</select></label></div>
            <h4>Worker order</h4><p className="provider-priority-help">{contractRequirement(route.protocol_contract, contracts)}</p><div className="worker-order-list">{route.worker_ids.map((workerId, index) => { const assignedWorker = workers.find((worker) => worker.id === workerId); return <div key={`${workerId}-${index}`}><span className="order-label">{index === 0 ? "Primary" : `Backup ${index}`}</span><select value={workerId} disabled={openDay} onChange={(event) => { const next = [...route.worker_ids]; next[index] = event.target.value; updateRoute(route.id, { worker_ids: next }); }}>{workers.map((worker) => { const compatible = workerSupportsContract(worker, route.protocol_contract, contracts); return <option key={worker.id} value={worker.id} disabled={(route.worker_ids.includes(worker.id) && worker.id !== workerId) || (!compatible && worker.id !== workerId)}>{worker.name} · {worker.model_id}{compatible ? "" : " · incompatible"}</option>; })}</select><button className="secondary" disabled={openDay || index === 0} onClick={() => { const next = [...route.worker_ids]; [next[index - 1], next[index]] = [next[index], next[index - 1]]; updateRoute(route.id, { worker_ids: next }); }}>↑</button><button className="secondary" disabled={openDay || index === route.worker_ids.length - 1} onClick={() => { const next = [...route.worker_ids]; [next[index], next[index + 1]] = [next[index + 1], next[index]]; updateRoute(route.id, { worker_ids: next }); }}>↓</button><button className="secondary danger" aria-label={`Remove ${index === 0 ? "Primary" : `Backup ${index}`} Worker ${assignedWorker?.name ?? workerId} from Route ${route.display_name}`} disabled={openDay || index === 0} onClick={() => removeRouteWorker(route.id, index)}>Remove</button></div>; })}</div>
            <div className="route-backup-actions"><button className="secondary" disabled={openDay || !workers.some((worker) => !route.worker_ids.includes(worker.id) && workerSupportsContract(worker, route.protocol_contract, contracts))} onClick={() => { const worker = workers.find((item) => !route.worker_ids.includes(item.id) && workerSupportsContract(item, route.protocol_contract, contracts)); if (worker) updateRoute(route.id, { worker_ids: [...route.worker_ids, worker.id] }); }}>Add compatible backup</button><MockWorkerCreator templates={mockTemplates} fixedContract={route.protocol_contract} disabled={openDay} buttonLabel="Create mock backup" onCreated={(worker) => assignCreatedMock(route.id, worker)} /></div>
            </CollapsibleEditorCard>; })}</div>
          </CollapsibleEditorSection>
        </div>
      </CollapsiblePanel>
    </div>}
  </div>;
}

function validationFromApiError(reason: unknown): EventValidation | null {
  if (!(reason instanceof ApiError) || !reason.detail || typeof reason.detail !== "object" || !("validation" in reason.detail)) return null;
  const validation = (reason.detail as { validation: unknown }).validation;
  if (!validation || typeof validation !== "object") return null;
  const candidate = validation as Partial<EventValidation>;
  return typeof candidate.valid === "boolean" && Array.isArray(candidate.errors) && Array.isArray(candidate.warnings)
    ? candidate as EventValidation
    : null;
}

function duplicatePublicNameConflicts(routes: EventDefinition["routes"]) {
  const grouped = new Map<string, EventDefinition["routes"]>();
  for (const route of routes) {
    const key = route.public_name.toLocaleLowerCase();
    grouped.set(key, [...(grouped.get(key) ?? []), route]);
  }
  const conflicts = new Map<string, string>();
  for (const duplicates of grouped.values()) {
    if (duplicates.length < 2) continue;
    for (const route of duplicates) {
      const others = duplicates.filter((item) => item.id !== route.id).map((item) => `Route “${item.display_name}”`);
      conflicts.set(route.id, others.join(" and "));
    }
  }
  return conflicts;
}

function workerSupportsContract(worker: Worker, contractId: string, contracts: ProtocolContract[]) {
  const contract = contracts.find((item) => item.id === contractId);
  return Boolean(contract
    && worker.generation_family === contract.generation_family
    && contract.required_capabilities.every((capability) => worker.capabilities[capability] === true));
}

function contractRequirement(contractId: string, contracts: ProtocolContract[]) {
  const contract = contracts.find((item) => item.id === contractId);
  if (!contract) return "Select a trusted protocol contract.";
  const capabilities = contract.required_capabilities.length
    ? ` with ${contract.required_capabilities.map(humanise).join(" and ")}`
    : "";
  const family = humanise(contract.generation_family);
  const article = /^[aeiou]/i.test(family) ? "an" : "a";
  return `${contract.display_name} requires ${article} ${family} Worker${capabilities}. Incompatible alternatives are disabled; an existing mismatch stays visible and labelled.`;
}

function ValidationIssue({ issue, draft, workers, note = false }: { issue: EventValidation["errors"][number] | EventValidation["warnings"][number]; draft: EventDefinition; workers: Worker[]; note?: boolean }) {
  const route = issue.route_id ? draft.routes.find((item) => item.id === issue.route_id) : undefined;
  const demo = "demo_id" in issue && issue.demo_id ? draft.demos.find((item) => item.id === issue.demo_id) : undefined;
  const worker = "worker_id" in issue && issue.worker_id ? workers.find((item) => item.id === issue.worker_id) : undefined;
  const workerIndex = route && "worker_id" in issue ? route.worker_ids.indexOf(issue.worker_id ?? "") : -1;
  const workerRole = workerIndex === 0 ? "Primary Worker" : workerIndex > 0 ? `Backup ${workerIndex} Worker` : "Worker";
  const location = demo
    ? `Demos → ${demo.name}`
    : route && worker
      ? `Routes → ${route.display_name} → Worker order → ${workerRole}`
      : route
        ? `Routes → ${route.display_name}`
        : "Event";
  return <li className="validation-issue"><div className="validation-location"><strong>{location}</strong>{route && <code>{route.public_name}</code>}</div><p>{note ? "Note: " : ""}{issue.message}</p>{worker && <small>{worker.name} · {worker.model_id} · {worker.runtime}</small>}{!worker && "worker_id" in issue && issue.worker_id && <small>Worker ID: {issue.worker_id}</small>}</li>;
}

function CollapsibleEditorSection({ sectionId, title, description, accessory, children }: { sectionId: string; title: string; description: string; accessory: ReactNode; children: ReactNode }) {
  const { collapsed, toggle } = useCollapse(sectionId);
  return <section className="collapsible-editor-section" aria-label={title}><div className="editor-section-heading"><div><h3>{title}</h3><p className="muted">{description}</p></div><div className="editor-collapse-actions">{accessory}<button className="secondary compact-button" aria-expanded={!collapsed} aria-label={`${collapsed ? "Expand" : "Collapse"} ${title}`} onClick={toggle}>{collapsed ? "Expand" : "Collapse"}</button></div></div><div className="collapsible-editor-content" hidden={collapsed}>{children}</div></section>;
}

function CollapsibleEditorCard({ sectionId, label, className = "", heading, accessory, children }: { sectionId: string; label: string; className?: string; heading: ReactNode; accessory: ReactNode; children: ReactNode }) {
  const { collapsed, toggle } = useCollapse(sectionId);
  return <article className={`route-editor ${className}${collapsed ? " collapsed" : ""}`} aria-label={label}><div className="route-editor-title">{heading}<div className="editor-collapse-actions"><button className="secondary compact-button" aria-expanded={!collapsed} aria-label={`${collapsed ? "Expand" : "Collapse"} ${label}`} onClick={toggle}>{collapsed ? "Expand" : "Collapse"}</button>{accessory}</div></div><div className="collapsible-editor-content" hidden={collapsed}>{children}</div></article>;
}

interface WorkerParameterValues {
  dtype: "float16" | "bfloat16" | "float32";
  lifecycle: "resident" | "on-demand" | "exclusive";
  contextLength: number;
  maximumNewTokens: number;
  maximumDenoisingSteps: number;
  visualTokenBudget: number;
}

function integerSetting(settings: RuntimeTemplate["settings"] | Worker["settings"] | undefined, name: string, fallback: number) {
  const value = settings?.[name];
  return typeof value === "number" && Number.isInteger(value) ? value : fallback;
}

function runtimeParameterDefaults(template?: RuntimeTemplate, worker?: Worker): WorkerParameterValues {
  return {
    dtype: (["float16", "bfloat16", "float32"] as const).includes(
      worker?.dtype as "float16" | "bfloat16" | "float32",
    ) ? worker!.dtype as WorkerParameterValues["dtype"] : template?.dtype ?? "float16",
    lifecycle: worker?.lifecycle ?? template?.lifecycle ?? "on-demand",
    contextLength: integerSetting(worker?.settings, "context_length", integerSetting(template?.settings, "context_length", 2048)),
    maximumNewTokens: integerSetting(worker?.settings, "maximum_new_tokens", integerSetting(template?.settings, "maximum_new_tokens", 128)),
    maximumDenoisingSteps: integerSetting(worker?.settings, "maximum_denoising_steps", integerSetting(template?.settings, "maximum_denoising_steps", 24)),
    visualTokenBudget: integerSetting(worker?.settings, "visual_token_budget", integerSetting(template?.settings, "visual_token_budget", 280)),
  };
}

function workerParameterPayload(template: RuntimeTemplate, values: WorkerParameterValues) {
  return {
    dtype: values.dtype,
    lifecycle: values.lifecycle,
    ...(["autoregressive", "vision-language"].includes(template.generation_family) ? { context_length: values.contextLength } : {}),
    ...(!["speech-conversation", "text-translation", "speech-synthesis"].includes(template.generation_family)
      ? { maximum_new_tokens: values.maximumNewTokens }
      : {}),
    ...(template.generation_family === "text-diffusion" ? { maximum_denoising_steps: values.maximumDenoisingSteps } : {}),
    ...(template.generation_family === "vision-language" ? { visual_token_budget: values.visualTokenBudget } : {}),
  };
}

function parametersAreValid(template: RuntimeTemplate, values: WorkerParameterValues) {
  const validContext = !["autoregressive", "vision-language"].includes(template.generation_family)
    || (values.contextLength >= 256 && values.contextLength <= 32768);
  const validOutput = ["speech-conversation", "text-translation", "speech-synthesis"].includes(template.generation_family)
    || (values.maximumNewTokens >= 1 && values.maximumNewTokens <= 512);
  const validDenoising = template.generation_family !== "text-diffusion"
    || (values.maximumDenoisingSteps >= 1 && values.maximumDenoisingSteps <= 48);
  const validVisualBudget = template.generation_family !== "vision-language"
    || [70, 140, 280, 560, 1120].includes(values.visualTokenBudget);
  return validContext && validOutput && validDenoising && validVisualBudget;
}

function WorkerParameterFields({ template, values, onChange }: {
  template: RuntimeTemplate;
  values: WorkerParameterValues;
  onChange: (values: WorkerParameterValues) => void;
}) {
  const update = (change: Partial<WorkerParameterValues>) => onChange({ ...values, ...change });
  const hasContext = ["autoregressive", "vision-language"].includes(template.generation_family);
  const hasOutput = !["speech-conversation", "text-translation", "speech-synthesis"].includes(template.generation_family);
  const hasDenoising = template.generation_family === "text-diffusion";
  const hasVisualBudget = template.generation_family === "vision-language";
  return <>
    <div className="runtime-fields worker-parameter-fields">
      <label>Data type{template.dtype && <small>Required by trusted runtime</small>}
        <select aria-label="Data type" value={values.dtype} disabled={template.dtype !== null} onChange={(event) => update({ dtype: event.target.value as WorkerParameterValues["dtype"] })}>
          <option value="float16">Float16</option><option value="bfloat16">BFloat16</option><option value="float32">Float32</option>
        </select>
      </label>
      <label>Lifecycle{template.lifecycle && <small>Required by trusted runtime</small>}
        <select aria-label="Lifecycle" value={values.lifecycle} disabled={template.lifecycle !== null} onChange={(event) => update({ lifecycle: event.target.value as WorkerParameterValues["lifecycle"] })}>
          <option value="on-demand">On demand</option><option value="resident">Resident</option><option value="exclusive">Exclusive</option>
        </select>
      </label>
      {hasContext && <label>Context length<small>256–32,768 tokens</small><input aria-label="Context length" type="number" min={256} max={32768} value={values.contextLength} onChange={(event) => update({ contextLength: event.target.valueAsNumber })} /></label>}
      {hasOutput && <label>Maximum output<small>1–512 tokens</small><input aria-label="Maximum output" type="number" min={1} max={512} value={values.maximumNewTokens} onChange={(event) => update({ maximumNewTokens: event.target.valueAsNumber })} /></label>}
      {hasDenoising && <label>Maximum denoising steps<small>1–48 refinement steps</small><input aria-label="Maximum denoising steps" type="number" min={1} max={48} value={values.maximumDenoisingSteps} onChange={(event) => update({ maximumDenoisingSteps: event.target.valueAsNumber })} /></label>}
      {hasVisualBudget && <label>Visual token budget<small>Trusted Gemma 4 image detail limit</small><select aria-label="Visual token budget" value={values.visualTokenBudget} onChange={(event) => update({ visualTokenBudget: Number(event.target.value) })}>{[70, 140, 280, 560, 1120].map((budget) => <option key={budget} value={budget}>{budget} tokens</option>)}</select></label>}
    </div>
    <p className="manifest-note">These limits become part of the immutable Worker definition. Sampling controls such as temperature, seed and top-k remain per-request parameters.</p>
  </>;
}

function replacementName(name: string) {
  const suffix = " replacement";
  return `${name.slice(0, 80 - suffix.length)}${suffix}`;
}

function confirmArchiveWorker(worker: Worker) {
  return window.confirm(
    `Archive Worker “${worker.name}”?\n\n` +
    "It will disappear from configured Workers and cannot be restored in ModelDeck. " +
    "Historical Event revisions and cached Model files will be kept.\n\n" +
    "Cancel leaves the Worker unchanged.",
  );
}

function MockWorkerCreator({ templates, fixedContract, disabled, buttonLabel = "Create mock Worker", onCreated }: {
  templates: MockWorkerTemplate[];
  fixedContract?: string;
  disabled: boolean;
  buttonLabel?: string;
  onCreated: (worker: Worker) => Promise<void>;
}) {
  const [contract, setContract] = useState(fixedContract ?? templates[0]?.protocol_contract ?? "");
  const [scenario, setScenario] = useState<MockScenario>("success");
  const [name, setName] = useState("");
  const [delayMs, setDelayMs] = useState(1000);
  const [visualTokenBudget, setVisualTokenBudget] = useState(70);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const selected = templates.find((template) => template.protocol_contract === (fixedContract ?? contract));
  useEffect(() => { if (fixedContract) setContract(fixedContract); }, [fixedContract]);
  useEffect(() => {
    const option = selected?.options.find((item) => item.id === "visual_token_budget");
    if (option) setVisualTokenBudget(option.default);
  }, [selected]);
  const create = async () => {
    if (!selected) throw new Error("This Route contract has no trusted mock implementation.");
    setCreating(true); setError(null);
    try {
      const worker = await postJson<Worker>("/api/workers/mocks", {
        protocol_contract: selected.protocol_contract,
        scenario,
        ...(name.trim() ? { name: name.trim() } : {}),
        ...(scenario === "delayed" ? { delay_ms: delayMs } : {}),
        ...(selected.options.some((item) => item.id === "visual_token_budget") ? { visual_token_budget: visualTokenBudget } : {}),
      });
      await onCreated(worker);
      setName("");
    } catch (reason) {
      const message = reason instanceof ApiError && reason.status === 405
        ? "The running ModelDeck management service does not support generic mock Workers yet. Restart ModelDeck, then try again."
        : messageFrom(reason);
      setError(message);
    } finally { setCreating(false); }
  };
  return <section className={`mock-worker-creator${fixedContract ? " compact" : ""}`} aria-label={fixedContract ? `Mock backup for ${fixedContract}` : "Create mock Worker"}>
    {!fixedContract && <div><strong>Mock Worker</strong><small>Deterministic CPU fallback for Route rehearsal; it never performs physical model inference.</small></div>}
    <label>Contract<select aria-label="Mock contract" value={fixedContract ?? contract} disabled={disabled || creating || Boolean(fixedContract)} onChange={(event) => setContract(event.target.value)}>{templates.map((template) => <option value={template.protocol_contract} key={template.id}>{template.display_name}</option>)}</select></label>
    <label>Scenario<select aria-label={fixedContract ? `Mock scenario for ${fixedContract}` : "Mock scenario"} value={scenario} disabled={disabled || creating} onChange={(event) => setScenario(event.target.value as MockScenario)}><option value="success">Success</option><option value="delayed">Delayed success</option><option value="request-error">Request failure</option></select></label>
    {!fixedContract && <label>Worker name<input aria-label="Mock Worker name" value={name} maxLength={80} placeholder={selected?.default_name ?? "Mock Worker"} disabled={disabled || creating} onChange={(event) => setName(event.target.value)} /></label>}
    {scenario === "delayed" && <label>Delay (ms)<input aria-label={fixedContract ? `Mock delay for ${fixedContract}` : "Mock delay"} type="number" min={1} max={120000} value={delayMs} disabled={disabled || creating} onChange={(event) => setDelayMs(event.target.valueAsNumber)} /></label>}
    {selected?.options.some((item) => item.id === "visual_token_budget") && <label>Visual tokens<select aria-label={fixedContract ? `Mock visual tokens for ${fixedContract}` : "Mock visual tokens"} value={visualTokenBudget} disabled={disabled || creating} onChange={(event) => setVisualTokenBudget(Number(event.target.value))}>{selected.options.find((item) => item.id === "visual_token_budget")?.choices.map((choice) => <option value={choice} key={choice}>{choice}</option>)}</select></label>}
    <button disabled={disabled || creating || !selected || (scenario === "delayed" && (!Number.isInteger(delayMs) || delayMs < 1 || delayMs > 120000))} onClick={() => void create()}>{creating ? "Creating…" : buttonLabel}</button>
    {error && <small className="inline-error">{error}</small>}
  </section>;
}

function WorkersView({ workers, templates, mockTemplates, pending, operate, refresh, openDay }: { workers: Worker[]; templates: RuntimeTemplate[]; mockTemplates: MockWorkerTemplate[]; pending: ReadonlySet<string>; operate: (worker: Worker, operation: WorkerOperation) => Promise<void>; refresh: () => Promise<void>; openDay: boolean }) {
  const collapseControls = useContext(CollapseContext);
  if (!collapseControls) throw new Error("Collapse controls are unavailable");
  const [libraryPreferences, setLibraryPreferences] = useStoredPreferences(WORKER_LIBRARY_STORAGE_KEY, loadWorkerLibraryPreferences);
  const { query, state: stateFilter, runtime: runtimeFilter, sort } = libraryPreferences;
  const [feedback, setFeedback] = useState<string | null>(null);
  const [replacing, setReplacing] = useState<string | null>(null);
  const [replacementWorkerName, setReplacementWorkerName] = useState("");
  const [replacementParameters, setReplacementParameters] = useState<WorkerParameterValues>(() => runtimeParameterDefaults());
  const [rebindDrafts, setRebindDrafts] = useState(true);
  const states = useMemo(() => [...new Set(workers.map((worker) => worker.state))].sort(), [workers]);
  const runtimes = useMemo(() => [...new Set(workers.map((worker) => worker.runtime))].sort(), [workers]);
  const filteredWorkers = useMemo(() => {
    const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    return workers.filter((worker) => {
      if (stateFilter && worker.state !== stateFilter) return false;
      if (runtimeFilter && worker.runtime !== runtimeFilter) return false;
      const searchable = [
        worker.name, worker.model_id, worker.artifact_model_id, worker.generation_family,
        worker.runtime, worker.runtime_template_id, worker.state, worker.lifecycle, worker.dtype,
        worker.id, ...Object.keys(worker.capabilities).filter((capability) => worker.capabilities[capability]),
      ].filter(Boolean).join(" ").toLocaleLowerCase();
      return terms.every((term) => searchable.includes(term));
    });
  }, [workers, query, stateFilter, runtimeFilter]);
  const sorted = useMemo(() => [...filteredWorkers].sort((a, b) => sort === "name-desc" ? b.name.localeCompare(a.name) : sort === "model-asc" ? a.model_id.localeCompare(b.model_id) : sort === "runtime-asc" ? a.runtime.localeCompare(b.runtime) : sort === "state" ? a.state.localeCompare(b.state) : a.name.localeCompare(b.name)), [filteredWorkers, sort]);
  const filtersActive = Boolean(query.trim() || stateFilter || runtimeFilter);
  const clearFilters = () => setLibraryPreferences((current) => ({ ...current, query: "", state: "", runtime: "" }));
  const rename = async (worker: Worker) => { const name = window.prompt("Worker name", worker.name)?.trim(); if (!name || name === worker.name) return; await patchJson(`/api/workers/${worker.id}`, { name }); await refresh(); };
  const archive = async (worker: Worker) => {
    if (!confirmArchiveWorker(worker)) return;
    await deleteJson(`/api/workers/${worker.id}`);
    setFeedback(`Archived Worker “${worker.name}”; its cached Model was kept.`);
    await refresh();
  };
  const beginReplacement = (worker: Worker, template: RuntimeTemplate) => { setReplacing(worker.id); setReplacementWorkerName(replacementName(worker.name)); setReplacementParameters(runtimeParameterDefaults(template, worker)); setRebindDrafts(true); setFeedback(null); };
  const replace = async (worker: Worker, template: RuntimeTemplate) => {
    const result = await postJson<{ replacement: Worker; rebound_event_drafts: string[] }>(`/api/workers/${worker.id}/replacement`, {
      name: replacementWorkerName,
      ...workerParameterPayload(template, replacementParameters),
      rebind_drafts: rebindDrafts,
    });
    setReplacing(null);
    const rebound = result.rebound_event_drafts.length;
    setFeedback(`Created replacement Worker “${result.replacement.name}”. ${rebound} draft Event${rebound === 1 ? " was" : "s were"} updated; published routing is unchanged until you publish a draft.`);
    await refresh();
  };
  return <div className="view-stack"><div className="view-actions worker-view-heading"><p>A Worker is one configured, startable service. Its name is editable; its execution identity is not. Use Replace to change safe model limits without mutating the original Worker.</p></div>
    <MockWorkerCreator templates={mockTemplates} disabled={openDay} onCreated={async (worker) => { setFeedback(`Created ${worker.name}. Mock output is deterministic and is labelled as fallback traffic.`); await refresh(); }} />
    {!!workers.length && <div className="worker-toolbar" aria-label="Worker search and filters"><label>Search workers<input type="search" value={query} placeholder="Name, model or capability" onChange={(event) => setLibraryPreferences((current) => ({ ...current, query: event.target.value }))} /></label><label>State<select value={stateFilter} onChange={(event) => setLibraryPreferences((current) => ({ ...current, state: event.target.value }))}><option value="">All states</option>{stateFilter && !states.includes(stateFilter as Worker["state"]) && <option value={stateFilter}>{stateFilter.replaceAll("-", " ")} (not currently present)</option>}{states.map((state) => <option key={state} value={state}>{state.replaceAll("-", " ")}</option>)}</select></label><label>Runtime<select value={runtimeFilter} onChange={(event) => setLibraryPreferences((current) => ({ ...current, runtime: event.target.value }))}><option value="">All runtimes</option>{runtimeFilter && !runtimes.includes(runtimeFilter) && <option value={runtimeFilter}>{runtimeFilter} (not currently present)</option>}{runtimes.map((runtime) => <option key={runtime} value={runtime}>{runtime}</option>)}</select></label><label>Sort workers<select value={sort} onChange={(event) => setLibraryPreferences((current) => ({ ...current, sort: event.target.value as WorkerSort }))}><option value="name-asc">Name A–Z</option><option value="name-desc">Name Z–A</option><option value="model-asc">Model</option><option value="runtime-asc">Runtime</option><option value="state">State</option></select></label><div className="worker-filter-summary" role="status"><span>{sorted.length} of {workers.length} Worker{workers.length === 1 ? "" : "s"}</span><button className="secondary compact-button" disabled={!filtersActive} onClick={clearFilters}>Clear filters</button></div></div>}
    {feedback && <div className="configuration-feedback">{feedback}</div>}
    {!workers.length ? <section className="empty-state"><h2>No Workers configured</h2><p>Create one from the Models view. ModelDeck does not create packaged Worker cards.</p></section> : !sorted.length ? <section className="empty-state compact"><h2>No Workers match these filters</h2><p>Try a different name, model, capability, state or runtime.</p><button className="secondary" onClick={clearFilters}>Clear filters</button></section> : <div className="worker-grid">{sorted.map((worker) => {
      const workerPending = workerOperationPending(pending, worker.id);
      const template = templates.find((item) => item.id === worker.runtime_template_id);
      const sectionId = `worker-${worker.id}`;
      const collapsed = collapseControls.preferences.sections[sectionId] ?? collapseControls.preferences.allCollapsed;
      return <article className={`worker-card state-${worker.state}${collapsed ? " collapsed" : ""}`} key={worker.id}><div className="worker-card-heading"><div><p className="worker-id">{worker.runtime === "mock" ? `${worker.generation_family} · mock` : worker.generation_family}</p><h3>{worker.name}</h3></div><div className="worker-card-heading-actions"><StateBadge state={worker.state} /><button className="secondary compact-button" aria-expanded={!collapsed} aria-label={`${collapsed ? "Expand" : "Collapse"} Worker ${worker.name}`} onClick={() => collapseControls.toggleSection(sectionId)}>{collapsed ? "Expand" : "Collapse"}</button></div></div><div className="worker-card-body" hidden={collapsed}><p className="worker-summary">{worker.model_id} · {worker.runtime}{worker.runtime === "mock" ? ` · ${humanise(String(worker.settings.mock_contract_id ?? "legacy contract"))} · ${humanise(String(worker.settings.mock_scenario ?? "success"))}` : ""}</p>{worker.last_error && <p className="inline-error">{worker.last_error}</p>}<details><summary>Immutable execution details</summary><DefinitionList rows={[["Internal ID", worker.id], ["Revision", worker.revision], ["Runtime", worker.runtime], ["Template", worker.runtime_template_id ?? "Built in"], ...(worker.runtime === "mock" ? [["Mock contract", String(worker.settings.mock_contract_id ?? "Legacy family mock")], ["Mock scenario", String(worker.settings.mock_scenario ?? "success")], ["Mock delay", worker.settings.mock_delay_ms ? `${worker.settings.mock_delay_ms} ms` : "None"]] as Array<[string, string]> : []), ["Port", String(worker.port)], ["Lifecycle", worker.lifecycle], ["Data type", worker.dtype], ["Context length", String(worker.settings.context_length ?? "Not applicable")], ["Maximum output", String(worker.settings.maximum_new_tokens ?? "Not applicable")], ["Visual token budget", String(worker.settings.visual_token_budget ?? "Not applicable")], ["Maximum denoising steps", String(worker.settings.maximum_denoising_steps ?? "Not applicable")]]} /></details><div className="button-row"><button className="secondary" disabled={openDay || workerPending} onClick={() => void rename(worker).catch((reason) => setFeedback(messageFrom(reason)))}>Rename</button><button className="secondary" disabled={openDay || workerPending || !template} title={template ? "Create a new Worker with revised parameters" : "The trusted runtime is no longer installed"} onClick={() => template && beginReplacement(worker, template)}>Replace</button><button disabled={workerPending || worker.state === "ready"} onClick={() => void operate(worker, "start")}>{pending.has(`${worker.id}:start`) ? "Starting…" : "Start"}</button><button className="secondary" disabled={workerPending || worker.state !== "ready"} onClick={() => void operate(worker, "smoke")}>{pending.has(`${worker.id}:smoke`) ? "Smoking…" : "Smoke"}</button><button className="secondary" disabled={workerPending || worker.state === "stopped"} onClick={() => void operate(worker, "stop")}>{pending.has(`${worker.id}:stop`) ? "Stopping…" : "Stop"}</button><button className="secondary danger" disabled={openDay || workerPending || !["stopped", "failed"].includes(worker.state)} onClick={() => void archive(worker).catch((reason) => setFeedback(messageFrom(reason)))}>Archive</button></div>
        {replacing === worker.id && template && <div className="runtime-form worker-replacement-form"><div className="runtime-form-heading"><strong>Replace this Worker</strong><small>The Model, revision and trusted runtime stay fixed. The original Worker is kept.</small></div><div className="runtime-fields"><label>Replacement name<input value={replacementWorkerName} maxLength={80} onChange={(event) => setReplacementWorkerName(event.target.value)} /></label><label>Model<input value={worker.artifact_model_id ?? worker.model_id} disabled /></label><label>Runtime<input value={template.display_name} disabled /></label></div><WorkerParameterFields template={template} values={replacementParameters} onChange={setReplacementParameters} /><label className="rebind-option"><input type="checkbox" checked={rebindDrafts} onChange={(event) => setRebindDrafts(event.target.checked)} /> Rebind draft Event routes to the replacement</label><p className="manifest-note">Published Event revisions always keep the original Worker until you explicitly publish an updated draft.</p><div className="runtime-form-actions"><button disabled={!replacementWorkerName.trim() || !parametersAreValid(template, replacementParameters)} onClick={() => void replace(worker, template).catch((reason) => setFeedback(messageFrom(reason)))}>Create replacement</button><button className="secondary" onClick={() => setReplacing(null)}>Cancel</button></div></div>}</div>
      </article>;
    })}</div>}
  </div>;
}

function workersForModel(model: ModelEntry, workers: Worker[]) {
  return workers.filter((worker) =>
    (worker.artifact_model_id ?? worker.model_id) === model.model_id
    && (worker.artifact_revision ?? worker.revision) === model.revision
  ).sort((a, b) => a.name.localeCompare(b.name));
}

function ModelWorkerSummary({ model, workers, openDay, removingWorker, onRemove }: { model: ModelEntry; workers: Worker[]; openDay: boolean; removingWorker: string | null; onRemove: (worker: Worker) => Promise<void> }) {
  const configured = workersForModel(model, workers);
  const { collapsed, toggle } = useCollapse(`model-workers-${model.model_id}@${model.revision}`);
  return <section className={`model-worker-summary${configured.length ? " has-workers" : ""}`} aria-label={`Workers for ${model.model_id}`}>
    <div className="model-worker-summary-heading"><strong>Configured Workers</strong><div><span>{configured.length} configured</span><button className="secondary compact-button" aria-expanded={!collapsed} aria-label={`${collapsed ? "Expand" : "Collapse"} configured Workers for ${model.model_id}`} onClick={toggle}>{collapsed ? "Expand" : "Collapse"}</button></div></div>
    <div hidden={collapsed}>{configured.length ? <div className="model-worker-list">{configured.map((worker) => <div className="model-worker-item" key={worker.id}>
      <div><strong>{worker.name}</strong><small>{humanise(worker.runtime)} · {humanise(worker.lifecycle)}{typeof worker.settings.visual_token_budget === "number" ? ` · ${worker.settings.visual_token_budget} visual tokens` : ""}</small></div>
      <div className="model-worker-item-actions"><StateBadge state={worker.state} /><button className="secondary danger compact-button" disabled={openDay || removingWorker !== null || !["stopped", "failed"].includes(worker.state)} title={["stopped", "failed"].includes(worker.state) ? "Archive this Worker without removing its cached Model" : "Stop this Worker before removing it"} aria-label={`Remove Worker ${worker.name}`} onClick={() => void onRemove(worker)}>{removingWorker === worker.id ? "Removing…" : "Remove"}</button></div>
    </div>)}</div> : <p>No Workers have been configured from this Model.</p>}</div>
  </section>;
}

function ModelCardShell({ model, children }: { model: ModelEntry; children: ReactNode }) {
  const { collapsed, toggle } = useCollapse(`model-${model.model_id}@${model.revision}`);
  return <article className={`model-row${collapsed ? " collapsed" : ""}`}>
    <div className="model-main"><div><h3>{model.model_id}</h3><p>{model.generation_family_hint ?? "Unclassified"} · {formatBytes(model.physical_size_bytes)}</p></div><div className="model-card-heading-actions"><StateBadge state={model.runnable ? "recognised" : model.download_state} /><button className="secondary compact-button" aria-expanded={!collapsed} aria-label={`${collapsed ? "Expand" : "Collapse"} Model ${model.model_id}`} onClick={toggle}>{collapsed ? "Expand" : "Collapse"}</button></div></div>
    <div className="model-card-body" hidden={collapsed}>{children}</div>
  </article>;
}

function ModelsView({ models, workers, templates, refresh, openDay }: { models: ModelEntry[]; workers: Worker[]; templates: RuntimeTemplate[]; refresh: () => Promise<void>; openDay: boolean }) {
  const [libraryPreferences, setLibraryPreferences] = useStoredPreferences(MODEL_LIBRARY_STORAGE_KEY, loadModelLibraryPreferences);
  const { query, sort } = libraryPreferences;
  const [configuring, setConfiguring] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [runtime, setRuntime] = useState("");
  const [artifact, setArtifact] = useState("");
  const [parameters, setParameters] = useState<WorkerParameterValues>(() => runtimeParameterDefaults());
  const [feedback, setFeedback] = useState<string | null>(null);
  const [removingWorker, setRemovingWorker] = useState<string | null>(null);
  const sorted = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    return models.filter((model) => !needle || [model.model_id, model.generation_family_hint, model.runnable_reason, ...model.capability_hints].some((value) => value?.toLocaleLowerCase().includes(needle))).sort((a, b) => sort === "name-desc" ? b.model_id.localeCompare(a.model_id) : sort === "size-desc" ? b.physical_size_bytes - a.physical_size_bytes : sort === "size-asc" ? a.physical_size_bytes - b.physical_size_bytes : sort === "readiness" ? Number(b.runnable) - Number(a.runnable) : sort === "workers" ? b.worker_count - a.worker_count : a.model_id.localeCompare(b.model_id));
  }, [models, query, sort]);
  const begin = (model: ModelEntry) => {
    const template = templates.find((item) => item.id === model.configuration_support);
    setConfiguring(`${model.model_id}@${model.revision}`);
    setName(model.model_id.split("/").at(-1)?.replaceAll("-", " ") ?? "New Worker");
    setRuntime(model.configuration_support ?? "");
    setArtifact(model.artifacts?.[0]?.artifact_id ?? "");
    setParameters(runtimeParameterDefaults(template));
    setFeedback(null);
  };
  const create = async (model: ModelEntry, selectedTemplate?: RuntimeTemplate) => {
    const template = selectedTemplate ?? templates.find((item) => item.id === runtime);
    if (!template) throw new Error("Select an installed trusted runtime.");
    await postJson("/api/workers", {
      name,
      model_id: model.model_id,
      revision: model.revision,
      runtime_template_id: template.id,
      artifact_id: artifact || undefined,
      ...workerParameterPayload(template, parameters),
    });
    setConfiguring(null);
    setFeedback(`Created Worker “${name}”.`);
    await refresh();
  };
  const removeWorker = async (worker: Worker) => {
    if (!confirmArchiveWorker(worker)) return;
    setRemovingWorker(worker.id);
    try {
      await deleteJson(`/api/workers/${worker.id}`);
      setFeedback(`Removed Worker “${worker.name}” from ModelDeck; its cached Model was kept.`);
      await refresh();
    } catch (reason) {
      setFeedback(messageFrom(reason));
    } finally {
      setRemovingWorker(null);
    }
  };
  if (configuring) {
    const model = models.find((item) => `${item.model_id}@${item.revision}` === configuring);
    const baseline = templates.find((item) => item.id === model?.configuration_support);
    const availableTemplates = templates.filter((item) => baseline && item.generation_family === baseline.generation_family && item.cache_setting === baseline.cache_setting && item.uses_base_model_identity === baseline.uses_base_model_identity);
    const selectedTemplate = availableTemplates.find((template) => template.id === runtime);
    if (model) return <div className="view-stack"><section className="panel model-configuration"><div className="runtime-form-heading"><p className="eyebrow">{model.generation_family_hint ?? "Model"}</p><h2>Create a Worker</h2><small>{model.model_id} at pinned revision {model.revision}</small></div><div className="runtime-fields"><label>Worker name<input value={name} maxLength={80} onChange={(event) => setName(event.target.value)} /></label><label>Runtime<select value={runtime} onChange={(event) => { const nextRuntime = event.target.value; setRuntime(nextRuntime); setParameters(runtimeParameterDefaults(availableTemplates.find((item) => item.id === nextRuntime))); }}>{availableTemplates.map((template) => <option key={template.id} value={template.id}>{template.display_name}</option>)}</select></label>{model.artifacts && model.artifacts.length > 0 && <label>Model artefact<select value={artifact} onChange={(event) => setArtifact(event.target.value)}>{model.artifacts.map((item) => <option key={item.artifact_id} value={item.artifact_id}>{item.artifact_id} · {item.filenames.join(", ")}</option>)}</select></label>}</div>{selectedTemplate ? <WorkerParameterFields template={selectedTemplate} values={parameters} onChange={setParameters} /> : <div className="configuration-feedback bad">No compatible trusted runtime is installed for this Model.</div>}<div className="runtime-form-actions"><button disabled={openDay || !name.trim() || !selectedTemplate || (selectedTemplate ? !parametersAreValid(selectedTemplate, parameters) : true)} onClick={() => selectedTemplate && void create(model, selectedTemplate).catch((reason) => setFeedback(messageFrom(reason)))}>Create Worker</button><button className="secondary" onClick={() => setConfiguring(null)}>Cancel</button></div>{feedback && <div className="configuration-feedback bad">{feedback}</div>}</section></div>;
  }
  return <div className="view-stack"><div className="view-actions"><p>Models are read-only discoveries from the local Hugging Face cache. Create as many Workers as a Model needs.</p><div className="model-library-toolbar"><label>Search models<input type="search" value={query} placeholder="Name or capability" onChange={(event) => setLibraryPreferences((current) => ({ ...current, query: event.target.value }))} /></label><label>Sort models<select value={sort} onChange={(event) => setLibraryPreferences((current) => ({ ...current, sort: event.target.value as ModelSort }))}><option value="name-asc">Name A–Z</option><option value="name-desc">Name Z–A</option><option value="readiness">Runnable first</option><option value="workers">Most Workers</option><option value="size-desc">Largest</option><option value="size-asc">Smallest</option></select></label></div></div>{openDay && <div className="configuration-feedback">Open Day mode locks configuration. Restart ModelDeck without <code>-OpenDay</code> to create Workers.</div>}{feedback && <div className="configuration-feedback good">{feedback}</div>}
    <section className="panel"><StaticPanelHeading title="Discovered Models" detail={query.trim() ? `${sorted.length} of ${models.length} cached` : `${models.length} cached`} />{sorted.length ? <div className="model-list">{sorted.map((model) => {
      const key = `${model.model_id}@${model.revision}`;
      const baseline = templates.find((item) => item.id === model.configuration_support);
      const availableTemplates = templates.filter((item) => baseline && item.generation_family === baseline.generation_family && item.cache_setting === baseline.cache_setting && item.uses_base_model_identity === baseline.uses_base_model_identity);
      const selectedTemplate = availableTemplates.find((template) => template.id === runtime);
      return <ModelCardShell model={model} key={key}>
        <div className="tag-list">{model.capability_hints.map((hint) => <span className="tag" key={hint}>{humanise(hint)}</span>)}</div>
        <ModelWorkerSummary model={model} workers={workers} openDay={openDay} removingWorker={removingWorker} onRemove={removeWorker} />
        <p className="model-stage">{model.runnable_reason}</p>
        {configuring !== key ? <div className="model-actions"><button disabled={openDay || !model.runnable || !model.modeldeck_allowed || !model.revision} onClick={() => begin(model)}>Create Worker</button></div> : <div className="runtime-form"><div className="runtime-form-heading"><strong>Create a Worker</strong><small>The trusted runtime determines the immutable execution identity.</small></div><div className="runtime-fields"><label>Worker name<input value={name} onChange={(event) => setName(event.target.value)} /></label><label>Runtime<select value={runtime} onChange={(event) => setRuntime(event.target.value)}>{availableTemplates.map((template) => <option key={template.id} value={template.id}>{template.display_name}</option>)}</select></label>{model.artifacts && model.artifacts.length > 0 && <label>Model artefact<select value={artifact} onChange={(event) => setArtifact(event.target.value)}>{model.artifacts.map((item) => <option key={item.artifact_id} value={item.artifact_id}>{item.artifact_id} · {item.filenames.join(", ")}</option>)}</select></label>}</div>{selectedTemplate && <p className="manifest-note">Runtime defaults: {selectedTemplate.dtype ?? "float16"} · {String(selectedTemplate.settings.context_length ?? 2048)} context · {String(selectedTemplate.settings.maximum_new_tokens ?? 128)} max output · {selectedTemplate.lifecycle ?? "on-demand"}</p>}<div className="runtime-form-actions"><button disabled={openDay} onClick={() => void create(model).catch((reason) => setFeedback(messageFrom(reason)))}>Create Worker</button><button className="secondary" onClick={() => setConfiguring(null)}>Cancel</button></div></div>}
      </ModelCardShell>;
    })}</div> : <div className="empty-state compact"><h3>No Models match “{query.trim()}”</h3><p>Try a model name, generation family or capability.</p></div>}</section>
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
  return <CollapsiblePanel sectionId="advanced-worker-logs" title="Worker logs" detail={`${logs.length} entries`} className="log-panel"><div className="log-toolbar"><div><label htmlFor="log-worker">Worker</label><select id="log-worker" value={workerId} onChange={(event) => setWorkerId(event.target.value)}>{workers.map((worker) => <option key={worker.id} value={worker.id}>{worker.name}</option>)}</select></div></div><div className="log-view">{logs.length ? logs.map((log, index) => <div className={`log-entry ${log.level}`} key={`${log.timestamp}-${index}`}><time>{new Date(log.timestamp).toLocaleTimeString()}</time><span>{log.source}</span><code>{log.message}</code></div>) : <p>No logs for this Worker.</p>}</div></CollapsiblePanel>;
}

function Loading() { return <main className="loading-screen"><div className="brand-mark">MD</div><h1>Starting operator console</h1><p>Reading local Events, Routes, Workers and Models…</p><div className="loading-bar"><span /></div></main>; }
function Unavailable({ retry }: { retry: () => Promise<void> }) { return <section className="empty-state"><span className="empty-icon">!</span><h2>Management data is unavailable</h2><p>No cloud service was contacted.</p><button onClick={() => void retry()}>Retry local connection</button></section>; }
function CollapsiblePanel({ sectionId, title, detail, className = "", accessory, children }: { sectionId: string; title: string; detail: string; className?: string; accessory?: ReactNode; children: ReactNode }) {
  const { collapsed, toggle } = useCollapse(sectionId);
  const contentId = useId();
  return <section className={`panel collapsible-panel${collapsed ? " collapsed" : ""}${className ? ` ${className}` : ""}`}>
    <div className="panel-heading"><div><h2>{title}</h2><span>{detail}</span></div><div className="panel-heading-actions">{accessory}<button className="secondary compact-button" aria-controls={contentId} aria-expanded={!collapsed} aria-label={`${collapsed ? "Expand" : "Collapse"} ${title}`} onClick={toggle}>{collapsed ? "Expand" : "Collapse"}</button></div></div>
    <div className="collapsible-panel-content" id={contentId} hidden={collapsed}>{children}</div>
  </section>;
}
function PanelHeading({ title, detail }: { title: string; detail: string }) {
  const sectionId = `panel-${title.toLocaleLowerCase().replaceAll(/[^a-z0-9]+/g, "-")}`;
  const { collapsed, toggle } = useCollapse(sectionId);
  return <div className="panel-heading" data-collapsed={collapsed}><div><h2>{title}</h2><span>{detail}</span></div><div className="panel-heading-actions"><button className="secondary compact-button" aria-expanded={!collapsed} aria-label={`${collapsed ? "Expand" : "Collapse"} ${title}`} onClick={toggle}>{collapsed ? "Expand" : "Collapse"}</button></div></div>;
}
function StaticPanelHeading({ title, detail }: { title: string; detail: string }) { return <div className="panel-heading static"><div><h2>{title}</h2><span>{detail}</span></div></div>; }
function StatusDot({ state }: { state: "good" | "warn" | "bad" | "neutral" }) { return <span className={`status-dot ${state}`} aria-hidden="true" />; }
function StateBadge({ state }: { state: string }) { return <span className={`state-badge state-${state}`}>{humanise(state)}</span>; }
function DefinitionList({ rows }: { rows: Array<[string, string]> }) { return <dl className="definition-list compact">{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>; }
function formatBytes(value: number) { if (!Number.isFinite(value) || value <= 0) return "0 B"; const units = ["B", "KiB", "MiB", "GiB", "TiB"]; const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1); return `${(value / 1024 ** index).toFixed(index > 2 ? 1 : 0)} ${units[index]}`; }
function humanise(value: string) { return value.replaceAll("_", " ").replaceAll("-", " "); }
function messageFrom(reason: unknown) { return reason instanceof Error ? reason.message : "The operation failed."; }

function workerOperationPending(pending: ReadonlySet<string>, workerId: string) { return [...pending].some((key) => key.startsWith(`${workerId}:`)); }

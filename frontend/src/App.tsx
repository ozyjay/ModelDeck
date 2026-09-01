import { createContext, useCallback, useContext, useEffect, useId, useMemo, useState } from "react";
import type { Dispatch, ReactNode, SetStateAction } from "react";

import { ApiError, deleteJson, getJson, patchJson, postJson, putJson } from "./api";
import type {
  BenchmarkHistory, BenchmarkThroughputPoint, CapabilityPublicationPreview, CapabilitySetup, CapabilitySetupPreview, CapabilitySetupSelection,
  CompatibilityTest, GatewayStatus, HardwareProbe, LiveCapability, LiveState, ManagementHealth, ModelEntry, PotentialCapability,
  ProtocolContract, RoutingProfile, RoutingProfileRecord,
  RoutingProfileRevision, RoutingProfileValidation, RuntimeInstallation, RuntimeTemplate, Telemetry, ThermalStatus,
  Worker, WorkerLog,
} from "./types";

type View = "setup" | "live" | "profiles" | "workers" | "models" | "advanced";
type WorkerOperation = "start" | "stop" | "restart" | "smoke";
type WorkerSort = "name-asc" | "name-desc" | "model-asc" | "runtime-asc" | "state";
type ModelSort = "name-asc" | "name-desc" | "size-desc" | "size-asc" | "readiness" | "workers";
type ModelStatusFilter = "" | "runtime-available" | "runtime-missing" | "workers-configured" | "workers-missing";

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
const LIVE_CAPABILITY_VISIBILITY_STORAGE_KEY = "modeldeck-live-capability-visibility-v1";
const CollapseContext = createContext<CollapseControls | null>(null);

interface WorkerLibraryPreferences { query: string; state: string; runtime: string; sort: WorkerSort }
interface ModelLibraryPreferences { query: string; status: ModelStatusFilter; sort: ModelSort }

const WORKER_SORTS: WorkerSort[] = ["name-asc", "name-desc", "model-asc", "runtime-asc", "state"];
const MODEL_SORTS: ModelSort[] = ["name-asc", "name-desc", "size-desc", "size-asc", "readiness", "workers"];
const MODEL_STATUS_FILTERS: ModelStatusFilter[] = ["", "runtime-available", "runtime-missing", "workers-configured", "workers-missing"];

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

function loadHiddenLiveCapabilities(): Set<string> {
  try {
    const parsed: unknown = JSON.parse(window.localStorage.getItem(LIVE_CAPABILITY_VISIBILITY_STORAGE_KEY) ?? "[]");
    return new Set(Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : []);
  } catch {
    return new Set();
  }
}

function liveCapabilityVisibilityKey(capability: LiveCapability): string {
  return `${capability.profile_id ?? "unknown-profile"}:${capability.id}`;
}

function loadModelLibraryPreferences(): ModelLibraryPreferences {
  const stored = storedObject(MODEL_LIBRARY_STORAGE_KEY);
  return {
    query: typeof stored.query === "string" ? stored.query : "",
    status: MODEL_STATUS_FILTERS.includes(stored.status as ModelStatusFilter) ? stored.status as ModelStatusFilter : "",
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
  { view: "setup", label: "Setup", path: "/" },
  { view: "live", label: "Live", path: "/live" },
  { view: "advanced", label: "Advanced", path: "/advanced" },
];

const ADVANCED_NAVIGATION: Array<{ view: View; label: string; path: string }> = [
  { view: "models", label: "Models", path: "/models" },
  { view: "workers", label: "Workers", path: "/workers" },
  { view: "profiles", label: "Routing profiles", path: "/profiles" },
];

function viewFromPath(path: string): View {
  return [...NAVIGATION, ...ADVANCED_NAVIGATION].find((item) => item.path === path)?.view ?? "setup";
}

export default function App() {
  const [view, setView] = useState<View>(() => viewFromPath(window.location.pathname));
  const [health, setHealth] = useState<ManagementHealth | null>(null);
  const [gateway, setGateway] = useState<GatewayStatus | null>(null);
  const [hardware, setHardware] = useState<HardwareProbe | null>(null);
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [thermal, setThermal] = useState<ThermalStatus | null>(null);
  const [live, setLive] = useState<LiveState>({ active_profile: null, active_profiles: [], capabilities: [] });
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [profiles, setProfiles] = useState<RoutingProfileRecord[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [contracts, setContracts] = useState<ProtocolContract[]>([]);
  const [templates, setTemplates] = useState<RuntimeTemplate[]>([]);
  const [runtimeInstallations, setRuntimeInstallations] = useState<RuntimeInstallation[]>([]);
  const [compatibility, setCompatibility] = useState<CompatibilityTest[]>([]);
  const [benchmarkHistory, setBenchmarkHistory] = useState<BenchmarkHistory>({ points: [], reports_scanned: 0, measurement: "median benchmark throughput" });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [workerOperationErrors, setWorkerOperationErrors] = useState<Record<string, string>>({});
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
    const [nextHealth, nextGateway, nextHardware, nextTelemetry, nextThermal, nextLive, nextWorkers,
      nextProfiles, catalogue, contractResponse, templateResponse, installationResponse, tests, history] = await Promise.all([
      getJson<ManagementHealth>("/api/health"),
      getJson<GatewayStatus>("/api/gateway/status"),
      getJson<HardwareProbe>("/api/hardware"),
      getJson<Telemetry>("/api/telemetry"),
      getJson<ThermalStatus>("/api/thermal"),
      getJson<LiveState>("/api/live"),
      getJson<Worker[]>("/api/workers"),
      getJson<{ profiles: RoutingProfileRecord[] }>("/api/routing-profiles"),
      getJson<{ models: ModelEntry[] }>("/api/catalogue"),
      getJson<{ contracts: ProtocolContract[] }>("/api/protocol-contracts"),
      getJson<{ templates: RuntimeTemplate[] }>("/api/runtime-templates"),
      getJson<{ installations: RuntimeInstallation[] }>("/api/runtime-installations"),
      getJson<{ tests: CompatibilityTest[] }>("/api/compatibility"),
      getJson<BenchmarkHistory>("/api/benchmark-history"),
    ]);
    setHealth(nextHealth); setGateway(nextGateway); setHardware(nextHardware);
    setTelemetry(nextTelemetry); setThermal(nextThermal); setLive(nextLive); setWorkers(nextWorkers);
    setProfiles(nextProfiles.profiles); setModels(catalogue.models);
    setContracts(contractResponse.contracts); setTemplates(templateResponse.templates);
    setRuntimeInstallations(installationResponse.installations);
    setCompatibility(tests.tests);
    setBenchmarkHistory(history);
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
        getJson<ThermalStatus>("/api/thermal").then(setThermal),
        getJson<BenchmarkHistory>("/api/benchmark-history").then(setBenchmarkHistory),
      ]).catch(() => undefined);
    }, 5000);
    return () => window.clearInterval(timer);
  }, []);

  const operate = async (worker: Worker, operation: WorkerOperation) => {
    if (operation === "start" && thermal?.model_loading_allowed === false) return;
    const key = `${worker.id}:${operation}`;
    setPending((current) => new Set(current).add(key)); setError(null);
    try {
      const result = await postJson<{ ok?: boolean; test?: { evidence?: { error_summary?: string } } }>(`/api/workers/${worker.id}/${operation}`);
      await refresh();
      setWorkerOperationErrors((current) => {
        const { [worker.id]: _removed, ...remaining } = current;
        return remaining;
      });
      if (operation === "smoke" && result.ok === false) {
        throw new Error(result.test?.evidence?.error_summary ?? "Worker diagnostic failed.");
      }
    } catch (reason) {
      const message = messageFrom(reason);
      setError(message);
      if (operation === "start" && reason instanceof ApiError && reason.status === 429) {
        setWorkerOperationErrors((current) => ({
          ...current,
          [worker.id]: `Start blocked (HTTP 429) — ${message}`,
        }));
      }
    }
    finally { setPending((current) => { const next = new Set(current); next.delete(key); return next; }); }
  };

  const navigate = (next: View, path: string) => {
    window.history.pushState({}, "", path); setView(next);
  };
  const advancedActive = ["profiles", "workers", "models", "advanced"].includes(view);

  if (loading) return <Loading />;
  return (
    <CollapseContext.Provider value={collapseControls}>
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">MD</span><div><strong>ModelDeck</strong><small>Operator console</small></div></div>
        <nav aria-label="Primary navigation">{NAVIGATION.map((item) => (
          <a className={view === item.view || (item.view === "advanced" && advancedActive) ? "nav-link active" : "nav-link"} href={item.path}
            key={item.view} onClick={(event) => { event.preventDefault(); navigate(item.view, item.path); }}>
            {item.label}
          </a>
        ))}{advancedActive && <div className="advanced-nav" aria-label="Advanced sections">{ADVANCED_NAVIGATION.map((item) => (
          <a className={view === item.view ? "nav-link active" : "nav-link"} href={item.path} key={item.view}
            onClick={(event) => { event.preventDefault(); navigate(item.view, item.path); }}>{item.label}</a>
        ))}</div>}</nav>
        <div className="sidebar-policy"><StatusDot state={gateway?.available ? "good" : "warn"} /><span>Local gateway only</span></div>
      </aside>
      <main className="main-content">
        <header className="topbar"><div><p className="eyebrow">Framework Desktop · local control plane</p><h1>{[...NAVIGATION, ...ADVANCED_NAVIGATION].find((item) => item.view === view)?.label}</h1></div>
          <div className="topbar-status">
            <button className="secondary collapse-all-button" onClick={() => collapseControls.setAllCollapsed(!collapsePreferences.allCollapsed)}>{collapsePreferences.allCollapsed ? "Expand all" : "Collapse all"}</button>
            {health && <div className="mode-badge state-store-badge" title={`State directory: ${health.state_store.directory}`} aria-label="State store"><StatusDot state="good" /><span>{health.state_store.label}</span></div>}
            {health && <div className={`mode-badge ${health.configuration_locked ? "locked" : "unlocked"}`} aria-label="Configuration status"><StatusDot state={health.configuration_locked ? "warn" : "good"} /><span>{health.configuration_locked ? "Configuration locked" : "Configuration unlocked"}</span></div>}
            {thermal && <div className={`mode-badge ${thermal.state === "normal" ? "unlocked" : "locked"}`} aria-label="Thermal status"><StatusDot state={thermal.state === "normal" ? "good" : thermal.state === "warm" || thermal.state === "hot" ? "warn" : "bad"} /><span>Thermal: {humanise(thermal.state)}{thermal.temperature_c == null ? "" : ` · ${thermal.temperature_c.toFixed(1)}°C`}</span></div>}
            <div className={`gateway-badge ${gateway?.available ? "ready" : "unavailable"}`}><StatusDot state={gateway?.available ? "good" : "bad"} /><span>{gateway?.available ? "Gateway available" : "Gateway unavailable"}</span></div>
          </div>
        </header>
        {error && <div className="alert error" role="alert"><strong>Action failed</strong><span>{error}</span><button className="icon-button" onClick={() => setError(null)}>×</button></div>}
        {!health || !hardware || !telemetry || !thermal || !gateway ? <Unavailable retry={refresh} />
          : view === "setup" ? <SetupView models={models} workers={workers} templates={templates} live={live} refresh={refresh} openDay={health.configuration_locked} />
          : view === "live" ? <LiveView live={live} workers={workers} models={models} thermal={thermal} operate={operate} pending={pending} refresh={refresh} />
          : view === "profiles" ? <RoutingProfilesView profiles={profiles} workers={workers} contracts={contracts} openDay={health.configuration_locked} refresh={refresh} />
          : view === "workers" ? <WorkersView workers={workers} models={models} templates={templates} thermal={thermal} operationErrors={workerOperationErrors} pending={pending} operate={operate} refresh={refresh} openDay={health.configuration_locked} />
          : view === "models" ? <ModelsView models={models} workers={workers} templates={templates} refresh={refresh} openDay={health.configuration_locked} />
          : <AdvancedView hardware={hardware} telemetry={telemetry} thermal={thermal} contracts={contracts} templates={templates} runtimeInstallations={runtimeInstallations} compatibility={compatibility} workers={workers} benchmarkHistory={benchmarkHistory} />}
      </main>
    </div>
    </CollapseContext.Provider>
  );
}

function SetupView({ models, workers, templates, live, refresh, openDay }: {
  models: ModelEntry[]; workers: Worker[]; templates: RuntimeTemplate[]; live: LiveState;
  refresh: () => Promise<void>; openDay: boolean;
}) {
  const [capabilityId, setCapabilityId] = useState("");
  const [modelKey, setModelKey] = useState("");
  const [runtimeTemplateId, setRuntimeTemplateId] = useState("");
  const [query, setQuery] = useState("");
  const [preview, setPreview] = useState<CapabilitySetupPreview | null>(null);
  const [setup, setSetup] = useState<CapabilitySetup | null>(null);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [publicName, setPublicName] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [publication, setPublication] = useState<CapabilityPublicationPreview | null>(null);
  const capabilities = useMemo(() => {
    const values = new Map<string, PotentialCapability>();
    for (const model of models) for (const capability of model.potential_capabilities) {
      if (capability.available_runtime_template_ids.length) values.set(capability.id, capability);
    }
    return [...values.values()].sort((left, right) => left.display_name.localeCompare(right.display_name));
  }, [models]);
  const compatibleModels = useMemo(() => models.filter((model) => {
    if (!capabilityId || !model.revision || model.download_state !== "installed-untested") return false;
    if (query && !model.model_id.toLocaleLowerCase().includes(query.toLocaleLowerCase())) return false;
    return model.potential_capabilities.some((item) => item.id === capabilityId && item.available_runtime_template_ids.length);
  }), [capabilityId, models, query]);
  const selectedModel = models.find((model) => `${model.model_id}@${model.revision}` === modelKey);
  const selectedCapability = selectedModel?.potential_capabilities.find((item) => item.id === capabilityId);
  const runtimeIds = selectedCapability?.available_runtime_template_ids ?? [];

  useEffect(() => {
    getJson<{ setups: CapabilitySetup[] }>("/api/capability-setups")
      .then(({ setups }) => {
        const resumable = setups.find((item) => !["succeeded", "cancelled"].includes(item.state));
        if (!resumable) return;
        setSetup(resumable);
        const planned = resumable.plan.selection;
        setDisplayName(humanise(planned.capability_id));
        setPublicName(planned.model_id.split("/").at(-1)?.toLocaleLowerCase().replace(/[^a-z0-9._-]+/g, "-") ?? "local-model");
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    setModelKey(""); setRuntimeTemplateId(""); setPreview(null); setSetup(null); setFeedback(null);
  }, [capabilityId]);
  useEffect(() => {
    setRuntimeTemplateId(runtimeIds.length === 1 ? runtimeIds[0] : ""); setPreview(null); setFeedback(null);
  }, [modelKey]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!setup || ["succeeded", "failed", "cancelled"].includes(setup.state)) return;
    const source = new EventSource(`/api/capability-setups/${setup.id}/events`);
    const update = () => getJson<CapabilitySetup>(`/api/capability-setups/${setup.id}`).then(setSetup).catch(() => undefined);
    source.addEventListener("setup", update);
    source.onerror = () => { source.close(); void update(); };
    const timer = window.setInterval(update, 2000);
    return () => { source.close(); window.clearInterval(timer); };
  }, [setup?.id, setup?.state]);

  const selection = (): CapabilitySetupSelection | null => {
    if (!selectedModel?.revision || !capabilityId || !runtimeTemplateId) return null;
    const shortName = selectedModel.model_id.split("/").at(-1) ?? selectedModel.model_id;
    return {
      capability_id: capabilityId,
      model_id: selectedModel.model_id,
      revision: selectedModel.revision,
      worker_name: `${shortName} · ${selectedCapability?.display_name ?? capabilityId}`.slice(0, 80),
      runtime_template_id: runtimeTemplateId,
      artifact_id: selectedModel.artifacts?.length === 1 ? selectedModel.artifacts[0].artifact_id : null,
      prefix_cache_enabled: false,
    };
  };
  const review = async () => {
    const selected = selection(); if (!selected) return;
    setBusy(true); setFeedback(null);
    try { setPreview(await postJson<CapabilitySetupPreview>("/api/capability-setups/preview", selected)); }
    catch (reason) { setFeedback(messageFrom(reason)); }
    finally { setBusy(false); }
  };
  const createAndTest = async () => {
    const selected = selection(); if (!selected || !preview) return;
    setBusy(true); setFeedback(null);
    try {
      const created = await postJson<CapabilitySetup>("/api/capability-setups", {
        request_id: crypto.randomUUID(), preview_fingerprint: preview.preview_fingerprint, selection: selected,
      });
      setSetup(created); setDisplayName(selectedCapability?.display_name ?? "Local capability");
      setPublicName(selected.model_id.split("/").at(-1)?.toLocaleLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+/, "") || "local-model");
    } catch (reason) { setFeedback(messageFrom(reason)); }
    finally { setBusy(false); }
  };
  const publicationBody = () => ({
    display_name: displayName, public_name: publicName, tool_calling_enabled: false,
    route_action: live.capabilities.some((item) => item.public_name.toLocaleLowerCase() === publicName.toLocaleLowerCase()) ? "replace-primary" : "add",
  });
  const reviewPublication = async () => {
    if (!setup) return; setBusy(true); setFeedback(null);
    try { setPublication(await postJson(`/api/capability-setups/${setup.id}/publication-preview`, publicationBody())); }
    catch (reason) { setFeedback(messageFrom(reason)); }
    finally { setBusy(false); }
  };
  const publish = async () => {
    if (!setup || !publication) return; setBusy(true); setFeedback(null);
    try {
      const result = await postJson<CapabilitySetup>(`/api/capability-setups/${setup.id}/publish`, {
        ...publicationBody(), publication_fingerprint: publication.publication_fingerprint,
      });
      setSetup(result); await refresh();
    } catch (reason) { setFeedback(messageFrom(reason)); }
    finally { setBusy(false); }
  };
  const reset = () => { setCapabilityId(""); setModelKey(""); setPreview(null); setSetup(null); setPublication(null); setFeedback(null); };

  if (setup?.state === "succeeded") return <div className="view-stack">
    <section className="hero-panel setup-success"><div><p className="eyebrow">Setup complete</p><h2>{setup.publication?.public_name} is serving locally</h2><p>The exact qualified Worker is published through Routing Profile revision {setup.publication?.revision}.</p></div></section>
    <section className="panel"><PanelHeading title="Serving identity" detail="Verified locally" /><DefinitionList rows={[
      ["API Model ID", setup.publication?.public_name ?? "—"], ["Worker", setup.worker_id ?? "—"],
      ["Runtime", String(setup.resolved_identity?.runtime ?? "—")], ["Backend", String(setup.resolved_identity?.backend ?? "—")],
      ["Device", String(setup.resolved_identity?.device ?? "—")], ["Configuration fingerprint", String(setup.resolved_identity?.configuration_fingerprint ?? "Not reported")],
    ]} /><button onClick={reset}>Set up another capability</button></section>
  </div>;

  return <div className="view-stack">
    <section className="hero-panel"><div><p className="eyebrow">Guided local setup</p><h2>Set up a local capability</h2><p>Choose the outcome. ModelDeck keeps the exact Model, Runtime, Worker evidence and routing decision visible.</p></div><div className="hero-status"><StatusDot state={live.capabilities.length ? "good" : "warn"} /><span>{live.capabilities.length} published capabilities</span></div></section>
    {feedback && <div className="alert error" role="alert" aria-live="assertive"><strong>Setup needs attention</strong><span>{feedback}</span></div>}
    {setup ? <section className="panel setup-progress" aria-live="polite"><PanelHeading title="Create and test" detail={humanise(setup.state)} />
      <ol className="setup-list">{["applying-policy", "creating-worker", "starting-worker", "verifying-identity", "qualifying", "awaiting-publication"].map((step) => <li className={step === setup.current_step ? "active" : ""} key={step}>{humanise(step)}</li>)}</ol>
      {setup.state === "waiting-for-thermal-capacity" && <p className="thermal-load-notice">Setup is safely paused until fresh thermal telemetry permits model loading.</p>}
      {setup.error && <div className="validation-summary bad"><strong>{setup.error.message}</strong><p>{setup.error.retryable ? "The exact setup can be retried." : "Adjust the configuration and create a new setup."}</p></div>}
      {setup.state === "failed" && <div className="button-row"><button disabled={!setup.error?.retryable} onClick={() => void postJson<CapabilitySetup>(`/api/capability-setups/${setup.id}/retry`).then(setSetup).catch((reason) => setFeedback(messageFrom(reason)))}>Try again</button><button className="secondary" onClick={reset}>Adjust configuration</button></div>}
      {setup.state === "awaiting-publication" && <div className="publication-review"><h3>Review publication</h3><p>Qualification passed. Publishing is a separate explicit action.</p><div className="field-grid"><label>Capability label<input value={displayName} onChange={(event) => { setDisplayName(event.target.value); setPublication(null); }} /></label><label>API Model ID<input value={publicName} onChange={(event) => { setPublicName(event.target.value); setPublication(null); }} /></label></div>
        {!publication ? <button disabled={busy} onClick={() => void reviewPublication()}>{busy ? "Reviewing…" : "Review routing changes"}</button> : <><div className={`validation-summary ${publication.validation.valid ? "good" : "bad"}`}><strong>{publication.validation.valid ? "Ready to publish" : "Publication is blocked"}</strong><p>{publication.before.capabilities.length} existing and {publication.after.capabilities.length} resulting capabilities. No unlisted fallback is added.</p>{publication.validation.errors.map((issue, index) => <p key={index}>{issue.message}</p>)}</div><button disabled={busy || !publication.validation.valid} onClick={() => void publish()}>{busy ? "Publishing…" : "Publish"}</button></>}
      </div>}
    </section> : <>
      <section className="panel"><PanelHeading title="1. Choose an outcome" detail={`${capabilities.length} available`} /><div className="capability-grid">{capabilities.map((capability) => <button className={`capability-choice ${capabilityId === capability.id ? "selected" : ""}`} key={capability.id} onClick={() => setCapabilityId(capability.id)}><strong>{capability.display_name}</strong><span>{capability.description}</span></button>)}</div></section>
      {capabilityId && <section className="panel"><PanelHeading title="2. Choose a cached Model" detail={`${compatibleModels.length} compatible`} /><label>Search cached Models<input type="search" value={query} onChange={(event) => setQuery(event.target.value)} /></label><div className="model-choice-list">{compatibleModels.map((model) => <button className={modelKey === `${model.model_id}@${model.revision}` ? "selected" : ""} key={`${model.model_id}@${model.revision}`} onClick={() => setModelKey(`${model.model_id}@${model.revision}`)}><strong>{model.model_id}</strong><small>Pinned revision {model.revision?.slice(0, 12)}</small></button>)}</div></section>}
      {selectedModel && <section className="panel"><PanelHeading title="3. Review configuration" detail="Exact and immutable" /><DefinitionList rows={[["Model", selectedModel.model_id], ["Revision", selectedModel.revision ?? "—"], ["Capability", selectedCapability?.display_name ?? capabilityId]]} /><details><summary>Advanced Runtime and parameters</summary><label>Trusted Runtime<select value={runtimeTemplateId} onChange={(event) => { setRuntimeTemplateId(event.target.value); setPreview(null); }}><option value="">Choose a Runtime</option>{runtimeIds.map((id) => <option key={id} value={id}>{templates.find((item) => item.id === id)?.display_name ?? id}</option>)}</select></label><p className="manifest-note">With multiple compatible Runtimes, ModelDeck requires an explicit choice unless exact matching local evidence identifies one.</p></details>
        {!preview ? <button disabled={busy || !runtimeTemplateId || openDay} onClick={() => void review()}>{busy ? "Reviewing…" : "Review Create and test"}</button> : <div className="configuration-review"><h3>Policy and execution review</h3><DefinitionList rows={[["Runtime selection", humanise(preview.selection_basis)], ["Runtime", String(preview.worker.runtime_template_id)], ["Model policy change", preview.policy_changes.model_allowed ? "Allow exact revision" : "Already allowed"], ["Capability policy change", preview.policy_changes.capability_allowed ? "Allow exact capability" : "Already allowed"], ["Resolved Backend/device", "Verified after Worker start"]]} /><button disabled={busy || openDay} onClick={() => void createAndTest()}>{busy ? "Creating…" : "Create and test"}</button></div>}
      </section>}
    </>}
  </div>;
}

function LiveView({ live, workers, models, thermal, operate, pending, refresh }: {
  live: LiveState; workers: Worker[]; models: ModelEntry[]; thermal: ThermalStatus;
  operate: (worker: Worker, operation: WorkerOperation) => Promise<void>; pending: ReadonlySet<string>; refresh: () => Promise<void>;
}) {
  const [routeFeedback, setRouteFeedback] = useState<string | null>(null);
  const [smokingRoute, setSmokingRoute] = useState<string | null>(null);
  const [hiddenCapabilities, setHiddenCapabilities] = useState<Set<string>>(loadHiddenLiveCapabilities);
  const capabilityKeys = useMemo(() => live.capabilities.map(liveCapabilityVisibilityKey), [live.capabilities]);
  const visibleCapabilities = useMemo(
    () => live.capabilities.filter((capability) => !hiddenCapabilities.has(liveCapabilityVisibilityKey(capability))),
    [hiddenCapabilities, live.capabilities],
  );
  useEffect(() => {
    try {
      window.localStorage.setItem(LIVE_CAPABILITY_VISIBILITY_STORAGE_KEY, JSON.stringify([...hiddenCapabilities].sort()));
    } catch {
      console.warn("Live capability visibility preferences could not be saved locally.");
    }
  }, [hiddenCapabilities]);
  const setCapabilityVisible = (capability: LiveCapability, visible: boolean) => {
    const key = liveCapabilityVisibilityKey(capability);
    setHiddenCapabilities((current) => {
      const next = new Set(current);
      if (visible) next.delete(key); else next.add(key);
      return next;
    });
  };
  const showAllCapabilities = () => setHiddenCapabilities((current) => {
    const next = new Set(current);
    for (const key of capabilityKeys) next.delete(key);
    return next;
  });
  const hideAllCapabilities = () => setHiddenCapabilities((current) => new Set([...current, ...capabilityKeys]));
  const smokeCapability = async (capability: LiveCapability) => {
    if (!capability.profile_id) return;
    setSmokingRoute(capability.id); setRouteFeedback(null);
    try {
      const result = await postJson<{ ok: boolean; tool_calling?: { failure_code: string | null } }>(`/api/routing-profiles/${capability.profile_id}/capabilities/${capability.id}/smoke`);
      await refresh();
      setRouteFeedback(result.ok ? "Tool calling passed the bounded public-route rehearsal." : `Tool calling was not verified: ${result.tool_calling?.failure_code ?? "probe failed"}.`);
    } catch (reason) { setRouteFeedback(messageFrom(reason)); }
    finally { setSmokingRoute(null); }
  };
  if (!workers.length || !live.active_profiles.length) return (
    <div className="view-stack">
      <section className="hero-panel"><div><p className="eyebrow">Initial setup</p><h2>Build your first local capability</h2><p>ModelDeck starts empty: create a Worker from a discovered Model, create a Routing Profile and capability, then publish it.</p></div></section>
      <CollapsiblePanel sectionId="live-setup" title="Setup checklist" detail={`${models.length} cached Models discovered`}>
        <ol className="setup-list"><li className={models.length ? "done" : ""}>Discover a cached Model</li><li className={workers.length ? "done" : ""}>Create a Worker</li><li className={live.active_profiles.length ? "done" : ""}>Create and publish a Routing Profile</li><li>Start and qualify the capability’s Worker</li></ol>
      </CollapsiblePanel>
    </div>
  );
  return <div className="view-stack">
    <section className="hero-panel"><div><p className="eyebrow">Published Routing Profiles · {live.active_profiles.length} active</p><h2>{live.active_profiles.map((profile) => `${profile.name} · revision ${profile.revision}`).join(" · ")}</h2><p>Publishing controls routing only. Worker processes remain under explicit operator control.</p></div><div className="hero-status"><StatusDot state={live.capabilities.every((capability) => capability.ready) ? "good" : "warn"} /><span>{live.capabilities.filter((capability) => capability.ready).length} of {live.capabilities.length} capabilities ready</span></div></section>
    <CollapsiblePanel sectionId="live-capabilities" title="Live capabilities" detail={`${visibleCapabilities.length} of ${live.capabilities.length} shown · ${live.capabilities.length} published`} className="table-panel" accessory={<div className="live-visibility-actions"><button className="secondary compact-button" disabled={!visibleCapabilities.length} onClick={hideAllCapabilities}>Hide all</button><button className="secondary compact-button" disabled={visibleCapabilities.length === live.capabilities.length} onClick={showAllCapabilities}>Show all</button></div>}>
      {routeFeedback && <div className="configuration-feedback">{routeFeedback}</div>}
      {live.capabilities.length ? <><details className="live-capability-visibility"><summary>Capability visibility · {visibleCapabilities.length} shown</summary><p className="manifest-note">This browser-only preference does not change published routing or create a Routing Profile draft.</p><div className="capability-visibility-list">{live.capabilities.map((capability) => <label key={liveCapabilityVisibilityKey(capability)}><input type="checkbox" checked={!hiddenCapabilities.has(liveCapabilityVisibilityKey(capability))} onChange={(event) => setCapabilityVisible(capability, event.target.checked)} /><span><strong>{capability.display_name}</strong><small>{capability.public_name}</small></span></label>)}</div></details>{visibleCapabilities.length ? <div className="active-route-table-wrap"><table className="active-route-table"><thead><tr><th>Published capability</th><th>Capability status</th><th>Protocol</th><th>Worker order and control</th><th>Effective Worker</th><th>Actions</th></tr></thead><tbody>
        {visibleCapabilities.map((capability) => <tr className={capability.ready ? "route-ready" : "route-unavailable"} key={liveCapabilityVisibilityKey(capability)}><td><strong>{capability.display_name}</strong><code>{capability.public_name}</code><small className="tool-calling-state">Tool calling: {!capability.tool_calling_enabled ? "not enabled" : capability.tool_calling?.supported ? "verified" : capability.tool_calling?.rehearsed ? `failed (${capability.tool_calling.failure_code ?? "probe"})` : "not rehearsed"}</small></td><td><div className={`route-readiness ${capability.ready ? "ready" : "unavailable"}`} role="status" aria-label={`${capability.display_name} capability status`}><StatusDot state={capability.ready ? "good" : "warn"} /><span><strong>{capability.ready ? "Ready" : "Not serving"}</strong><small>{capability.ready ? "Accepting requests" : "Start a Worker"}</small></span></div></td><td>{capability.protocol_contract}</td><td><div className="active-worker-chain">{capability.workers.map((worker, index) => {
          const order = index === 0 ? "Primary" : `Backup ${index}`;
          const workerPending = workerOperationPending(pending, worker.id);
          const canStart = ["stopped", "failed"].includes(worker.state);
          const canStop = !["stopped", "stopping", "archived"].includes(worker.state);
          return <div className="active-worker-item" aria-label={`${order} Worker ${worker.name}`} key={worker.id}><span><small>{order}</small><strong>{worker.name}</strong></span><div className="active-worker-controls"><StateBadge state={worker.state} />{canStart ? <button className="compact-button" disabled={workerPending || !thermal.model_loading_allowed} title={thermal.model_loading_allowed ? `Start ${worker.name}` : thermalLoadingNotice(thermal)} aria-label={`Start Worker ${worker.name}`} onClick={() => void operate(worker, "start")}>{pending.has(`${worker.id}:start`) ? "Starting…" : "Start"}</button> : <button className="secondary compact-button" disabled={workerPending || !canStop} aria-label={`Stop Worker ${worker.name}`} onClick={() => void operate(worker, "stop")}>{pending.has(`${worker.id}:stop`) ? "Stopping…" : "Stop"}</button>}</div></div>;
        })}</div></td><td className={capability.effective_worker ? "effective-worker" : "effective-worker unavailable"}>{capability.effective_worker?.name ?? "No ready Worker"}</td><td><div className="button-row"><button className="secondary" disabled={smokingRoute !== null || !capability.ready} onClick={() => void smokeCapability(capability)}>Rehearse capability</button><button className="secondary" aria-label={`Hide capability ${capability.display_name}`} onClick={() => setCapabilityVisible(capability, false)}>Hide</button></div></td></tr>)}
      </tbody></table></div> : <p className="muted">All published capabilities are hidden in this browser. Use Capability visibility or Show all to restore them.</p>}</> : <p className="muted">The published Routing Profiles contain no capabilities.</p>}
    </CollapsiblePanel>
  </div>;
}

function RoutingProfilesView({ profiles, workers, contracts, openDay, refresh }: {
  profiles: RoutingProfileRecord[]; workers: Worker[]; contracts: ProtocolContract[]; openDay: boolean; refresh: () => Promise<void>;
}) {
  const [selectedId, setSelectedId] = useState(profiles[0]?.definition.id ?? "");
  const selected = profiles.find((profile) => profile.definition.id === selectedId) ?? profiles[0];
  const [draft, setDraft] = useState<RoutingProfile | null>(selected?.definition ?? null);
  const [saveState, setSaveState] = useState("Saved");
  const [validation, setValidation] = useState<RoutingProfileValidation | null>(null);
  const [revisions, setRevisions] = useState<RoutingProfileRevision[]>([]);
  const [feedback, setFeedback] = useState<string | null>(null);
  const duplicateNames = capabilityNameConflicts(draft?.capabilities ?? []);

  useEffect(() => { setDraft(selected?.definition ?? null); setSaveState("Saved"); setValidation(null); }, [selected?.definition]);
  useEffect(() => { if (!selectedId && profiles[0]) setSelectedId(profiles[0].definition.id); }, [profiles, selectedId]);
  useEffect(() => {
    if (!draft || !selected || JSON.stringify(draft) === JSON.stringify(selected.definition) || openDay) return;
    if (duplicateNames.size) { setSaveState("Needs attention"); return; }
    setSaveState("Saving…");
    const timer = window.setTimeout(() => {
      putJson(`/api/routing-profiles/${draft.id}/draft`, draft)
        .then(() => setSaveState("Saved"))
        .catch((reason) => { setSaveState("Save failed"); setFeedback(messageFrom(reason)); });
    }, 500);
    return () => window.clearTimeout(timer);
  }, [draft, duplicateNames.size, openDay, selected]);

  const createProfile = async () => {
    const definition: RoutingProfile = { id: crypto.randomUUID(), name: "New Routing Profile", description: "", qualification: "compatible", capabilities: [] };
    const record = await postJson<RoutingProfileRecord>("/api/routing-profiles", definition);
    await refresh(); setSelectedId(record.definition.id);
  };
  const save = async () => {
    if (!draft) return;
    await putJson(`/api/routing-profiles/${draft.id}/draft`, draft); setSaveState("Saved");
  };
  const validate = async () => {
    if (!draft) return;
    await save(); setValidation(await postJson(`/api/routing-profiles/${draft.id}/validate`)); setFeedback(null);
  };
  const publish = async () => {
    if (!draft) return;
    await save();
    try { await postJson(`/api/routing-profiles/${draft.id}/publish`); }
    catch (reason) { const failed = routingProfileValidationFromApiError(reason); if (failed) setValidation(failed); throw reason; }
    setFeedback("Routing Profile published. No Workers were started or stopped."); await refresh();
  };
  const updateCapability = (id: string, change: Partial<RoutingProfile["capabilities"][number]>) => {
    setFeedback(null); setValidation(null);
    setDraft((current) => current && ({ ...current, capabilities: current.capabilities.map((capability) => capability.id === id ? { ...capability, ...change } : capability) }));
  };
  const addCapability = () => {
    if (!draft || !workers.length) return;
    const contractId = contracts[0]?.id ?? "openai-chat-v1";
    const worker = workers.find((candidate) => workerSupportsContract(candidate, contractId, contracts)) ?? workers[0];
    setDraft({ ...draft, capabilities: [...draft.capabilities, { id: crypto.randomUUID(), display_name: "New capability", public_name: `capability-${draft.capabilities.length + 1}`, protocol_contract: contractId, tool_calling_enabled: false, worker_ids: [worker.id] }] });
  };
  const changeContract = (capabilityId: string, protocolContract: string) => {
    const compatible = workers.filter((worker) => workerSupportsContract(worker, protocolContract, contracts));
    updateCapability(capabilityId, { protocol_contract: protocolContract, tool_calling_enabled: false, ...(compatible.length === 1 ? { worker_ids: [compatible[0].id] } : {}) });
  };
  const deleteProfile = async () => {
    if (!draft || selected?.latest_revision || !window.confirm(`Delete draft-only Routing Profile “${draft.name}”?`)) return;
    await deleteJson(`/api/routing-profiles/${draft.id}`); setSelectedId(""); await refresh();
  };

  return <div className="view-stack">
    <div className="view-actions"><p>Routing Profiles publish reusable local capabilities for any application or event.</p><button disabled={openDay} onClick={() => void createProfile().catch((reason) => setFeedback(messageFrom(reason)))}>Create Routing Profile</button></div>
    {!selected || !draft ? <section className="empty-state"><h2>No Routing Profiles yet</h2><p>Create a profile after configuring at least one Worker.</p></section> : <div className="event-layout">
      <aside className="panel event-list">{profiles.map((profile) => <button className={`event-select ${profile.definition.id === draft.id ? "active" : ""}`} key={profile.definition.id} onClick={() => setSelectedId(profile.definition.id)}><span><strong>{profile.definition.name}</strong><small>{profile.active ? `Live revision ${profile.active_revision}` : profile.latest_revision ? `Published revision ${profile.latest_revision}` : "Draft only"}</small></span></button>)}</aside>
      <CollapsiblePanel sectionId={`profile-${draft.id}`} title={draft.name} detail={`${selected.active ? `Live revision ${selected.active_revision}` : "Draft"} · ${saveState}`} className="event-detail" accessory={<StateBadge state={selected.active ? "ready" : "stopped"} />}>
        {feedback && <div className="configuration-feedback">{feedback}</div>}
        <div className="button-row event-actions"><button className="secondary" disabled={Boolean(duplicateNames.size)} onClick={() => void validate().catch((reason) => setFeedback(messageFrom(reason)))}>Validate</button><button disabled={openDay || saveState === "Saving…" || Boolean(duplicateNames.size)} onClick={() => void publish().catch((reason) => setFeedback(messageFrom(reason)))}>Publish routing</button><button className="secondary" disabled={openDay || !selected.latest_revision} onClick={() => void deleteJson(`/api/routing-profiles/${draft.id}/draft`).then(refresh)}>Discard draft</button><button className="secondary" onClick={() => void getJson<{ revisions: RoutingProfileRevision[] }>(`/api/routing-profiles/${draft.id}/revisions`).then((result) => setRevisions(result.revisions))}>History</button><button className="secondary danger" disabled={openDay || Boolean(selected.latest_revision)} onClick={() => void deleteProfile().catch((reason) => setFeedback(messageFrom(reason)))}>Delete Routing Profile</button></div>
        {validation && <div className={`validation-summary ${validation.valid ? "good" : "bad"}`}><strong>{validation.valid ? "Ready to publish" : "Validation needs attention"}</strong><ul>{[...validation.errors, ...validation.warnings].map((issue, index) => <li className="validation-issue" key={index}><p>{"capability_id" in issue && issue.capability_id ? `${draft.capabilities.find((item) => item.id === issue.capability_id)?.display_name ?? "Capability"}: ` : ""}{issue.message}</p></li>)}</ul></div>}
        {revisions.length > 0 && <details className="revision-history" open><summary>Published revisions</summary><div>{revisions.map((revision) => <article key={revision.revision}><span><strong>Revision {revision.revision}</strong><small>{new Date(revision.published_at).toLocaleString()}</small></span><button className="secondary" disabled={revision.active || openDay} onClick={() => void postJson(`/api/routing-profiles/${draft.id}/revisions/${revision.revision}/publish`).then(refresh)}>Make live</button></article>)}</div></details>}
        <div className="event-editor"><div className="field-grid"><label>Profile name<input value={draft.name} disabled={openDay} onChange={(event) => setDraft({ ...draft, name: event.target.value })} /></label><label>Qualification<select value={draft.qualification} disabled={openDay} onChange={(event) => setDraft({ ...draft, qualification: event.target.value as RoutingProfile["qualification"] })}><option value="compatible">Protocol compatible</option><option value="tested-working">Tested working</option></select></label></div><label>Description<textarea value={draft.description} disabled={openDay} onChange={(event) => setDraft({ ...draft, description: event.target.value })} /></label>
          {draft.capabilities.some((capability) => capability.protocol_contract === "openai-chat-v1") && <CollapsibleEditorSection sectionId={`profile-tool-calling-${draft.id}`} title="Tool calling" description="Enable a complete tool-call rehearsal only for a capability served by tool-capable Workers."><div className="route-editor-list">{draft.capabilities.filter((capability) => capability.protocol_contract === "openai-chat-v1").map((capability) => { const assignedWorkersSupportTools = capability.worker_ids.every((workerId) => workers.find((worker) => worker.id === workerId)?.capabilities.tool_calling === true); return <label className="field-checkbox" key={capability.id}><input type="checkbox" checked={capability.tool_calling_enabled === true} disabled={openDay || !assignedWorkersSupportTools} onChange={(event) => updateCapability(capability.id, { tool_calling_enabled: event.target.checked })} /><span><strong>{capability.display_name}</strong><small>{assignedWorkersSupportTools ? "Require tool calling and run the complete rehearsal for this capability." : "Assign only Workers that support tool calling before enabling it."}</small></span></label>; })}</div></CollapsibleEditorSection>}
          <CollapsibleEditorSection sectionId={`profile-capabilities-${draft.id}`} title="Published capabilities" description="Each capability selects one trusted protocol and its ordered local Workers." accessory={<button disabled={openDay || !workers.length} onClick={addCapability}>Add capability</button>}><div className="route-editor-list">{draft.capabilities.map((capability) => { const conflict = duplicateNames.get(capability.id); const compatible = workers.filter((worker) => workerSupportsContract(worker, capability.protocol_contract, contracts)); return <CollapsibleEditorCard sectionId={`profile-capability-${draft.id}-${capability.id}`} label={`Capability ${capability.display_name}`} heading={<h4>{capability.display_name}</h4>} accessory={<button className="secondary danger" disabled={openDay} onClick={() => setDraft({ ...draft, capabilities: draft.capabilities.filter((item) => item.id !== capability.id) })}>Remove</button>} key={capability.id}><div className="field-grid"><label>Capability label<input value={capability.display_name} disabled={openDay} onChange={(event) => updateCapability(capability.id, { display_name: event.target.value })} /></label><label>API model ID<small className="field-help">Sent by clients in the <code>model</code> field and unique across active Routing Profiles.</small><input value={capability.public_name} aria-invalid={Boolean(conflict)} disabled={openDay} onChange={(event) => updateCapability(capability.id, { public_name: event.target.value })} />{conflict && <small className="field-error">{conflict}</small>}</label><label>Protocol contract<select value={capability.protocol_contract} disabled={openDay} onChange={(event) => changeContract(capability.id, event.target.value)}>{contracts.map((contract) => <option value={contract.id} key={contract.id}>{contract.display_name}</option>)}</select></label></div><h4>Worker order</h4><p className="provider-priority-help">{contractRequirement(capability.protocol_contract, contracts)}</p><div className="worker-order-list">{capability.worker_ids.map((workerId, index) => <div key={`${workerId}-${index}`}><span className="order-label">{index === 0 ? "Primary" : `Backup ${index}`}</span><select value={compatible.some((worker) => worker.id === workerId) ? workerId : ""} disabled={openDay} onChange={(event) => { const next = [...capability.worker_ids]; next[index] = event.target.value; updateCapability(capability.id, { worker_ids: next }); }}>{compatible.map((worker) => <option value={worker.id} disabled={capability.worker_ids.includes(worker.id) && worker.id !== workerId} key={worker.id}>{worker.name} · {worker.model_id}</option>)}</select><button className="secondary" disabled={openDay || index === 0} onClick={() => { const next = [...capability.worker_ids]; [next[index - 1], next[index]] = [next[index], next[index - 1]]; updateCapability(capability.id, { worker_ids: next }); }}>↑</button><button className="secondary" disabled={openDay || index === capability.worker_ids.length - 1} onClick={() => { const next = [...capability.worker_ids]; [next[index], next[index + 1]] = [next[index + 1], next[index]]; updateCapability(capability.id, { worker_ids: next }); }}>↓</button><button className="secondary danger" disabled={openDay || index === 0} onClick={() => updateCapability(capability.id, { worker_ids: capability.worker_ids.filter((_, workerIndex) => workerIndex !== index) })}>Remove</button></div>)}</div><div className="route-backup-actions"><button className="secondary" disabled={openDay || !workers.some((worker) => !capability.worker_ids.includes(worker.id) && workerSupportsContract(worker, capability.protocol_contract, contracts))} onClick={() => { const worker = workers.find((item) => !capability.worker_ids.includes(item.id) && workerSupportsContract(item, capability.protocol_contract, contracts)); if (worker) updateCapability(capability.id, { worker_ids: [...capability.worker_ids, worker.id] }); }}>Add compatible backup</button></div></CollapsibleEditorCard>; })}</div></CollapsibleEditorSection>
        </div>
      </CollapsiblePanel>
    </div>}
  </div>;
}

function routingProfileValidationFromApiError(reason: unknown): RoutingProfileValidation | null {
  if (!(reason instanceof ApiError) || !reason.detail || typeof reason.detail !== "object" || !("validation" in reason.detail)) return null;
  const validation = (reason.detail as { validation: unknown }).validation;
  return validation && typeof validation === "object" && "valid" in validation && Array.isArray((validation as RoutingProfileValidation).errors) && Array.isArray((validation as RoutingProfileValidation).warnings) ? validation as RoutingProfileValidation : null;
}

function capabilityNameConflicts(capabilities: RoutingProfile["capabilities"]) {
  const grouped = new Map<string, RoutingProfile["capabilities"]>();
  for (const capability of capabilities) grouped.set(capability.public_name.toLocaleLowerCase(), [...(grouped.get(capability.public_name.toLocaleLowerCase()) ?? []), capability]);
  const conflicts = new Map<string, string>();
  for (const duplicates of grouped.values()) if (duplicates.length > 1) for (const capability of duplicates) conflicts.set(capability.id, `“${capability.public_name}” is already used by another capability.`);
  return conflicts;
}

function workerSupportsContract(worker: Worker, contractId: string, contracts: ProtocolContract[]) {
  const contract = contracts.find((item) => item.id === contractId);
  return Boolean(contract
    && (contract.compatible_generation_families ?? [contract.generation_family]).includes(worker.generation_family)
    && contract.required_capabilities.every((capability) => worker.capabilities[capability] === true));
}

function contractRequirement(contractId: string, contracts: ProtocolContract[]) {
  const contract = contracts.find((item) => item.id === contractId);
  if (!contract) return "Select a trusted protocol contract.";
  const capabilities = contract.required_capabilities.length
    ? ` with ${contract.required_capabilities.map(humanise).join(" and ")}`
    : "";
  const families = contract.compatible_generation_families ?? [contract.generation_family];
  const familyDescription = families.map(humanise).join(" or ");
  const article = /^[aeiou]/i.test(familyDescription) ? "an" : "a";
  return `${contract.display_name} requires ${article} ${familyDescription} Worker${capabilities}. Incompatible Workers are hidden; an existing mismatch must be replaced with a compatible Worker.`;
}

function CollapsibleEditorSection({ sectionId, title, description, accessory, children }: { sectionId: string; title: string; description: string; accessory?: ReactNode; children: ReactNode }) {
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
  prefixCacheEnabled: boolean;
}

const APPLICATION_MANAGED_PREFIX_CACHE_MODELS = new Set([
  "Qwen/Qwen2.5-0.5B-Instruct",
  "Qwen/Qwen2.5-3B-Instruct",
]);

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
    prefixCacheEnabled: worker?.settings.prefix_cache_enabled === true,
  };
}

function workerParameterPayload(template: RuntimeTemplate, values: WorkerParameterValues) {
  return {
    dtype: values.dtype,
    lifecycle: values.lifecycle,
    ...(["autoregressive", "vision-language"].includes(template.generation_family) ? { context_length: values.contextLength } : {}),
    ...(!["speech-conversation", "text-translation", "speech-synthesis", "speech-recognition"].includes(template.generation_family)
      ? { maximum_new_tokens: values.maximumNewTokens }
      : {}),
    ...(template.generation_family === "text-diffusion" ? { maximum_denoising_steps: values.maximumDenoisingSteps } : {}),
    ...(template.generation_family === "vision-language" ? { visual_token_budget: values.visualTokenBudget } : {}),
    ...(template.generation_family === "autoregressive" ? { prefix_cache_enabled: values.prefixCacheEnabled } : {}),
  };
}

function parametersAreValid(template: RuntimeTemplate, values: WorkerParameterValues) {
  const validContext = !["autoregressive", "vision-language"].includes(template.generation_family)
    || (values.contextLength >= 256 && values.contextLength <= 32768);
  const validOutput = ["speech-conversation", "text-translation", "speech-synthesis", "speech-recognition"].includes(template.generation_family)
    || (values.maximumNewTokens >= 1 && values.maximumNewTokens <= 4096);
  const validDenoising = template.generation_family !== "text-diffusion"
    || (values.maximumDenoisingSteps >= 1 && values.maximumDenoisingSteps <= 48);
  const validVisualBudget = template.generation_family !== "vision-language"
    || [70, 140, 280, 560, 1120].includes(values.visualTokenBudget);
  return validContext && validOutput && validDenoising && validVisualBudget;
}

function WorkerParameterFields({ template, values, onChange, prefixCacheAvailable = false, capabilityId }: {
  template: RuntimeTemplate;
  values: WorkerParameterValues;
  onChange: (values: WorkerParameterValues) => void;
  prefixCacheAvailable?: boolean;
  capabilityId?: string | null;
}) {
  const update = (change: Partial<WorkerParameterValues>) => onChange({ ...values, ...change });
  const hasContext = ["autoregressive", "vision-language"].includes(template.generation_family);
  const hasOutput = !["speech-conversation", "text-translation", "speech-synthesis", "speech-recognition"].includes(template.generation_family);
  const hasDenoising = template.generation_family === "text-diffusion";
  const hasVisualBudget = template.generation_family === "vision-language" && capabilityId !== "general-chat";
  const visualBudgetLabel = capabilityId === "general-image-chat" ? "Image detail limit" : "Visual token budget";
  const visualBudgetHelp = capabilityId === "general-image-chat" ? "Used only for image chat" : "Trusted Gemma 4 image detail limit";
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
      {hasOutput && <label>Maximum output<small>1–4,096 tokens</small><input aria-label="Maximum output" type="number" min={1} max={4096} value={values.maximumNewTokens} onChange={(event) => update({ maximumNewTokens: event.target.valueAsNumber })} /></label>}
      {hasDenoising && <label>Maximum denoising steps<small>1–48 refinement steps</small><input aria-label="Maximum denoising steps" type="number" min={1} max={48} value={values.maximumDenoisingSteps} onChange={(event) => update({ maximumDenoisingSteps: event.target.valueAsNumber })} /></label>}
      {hasVisualBudget && <label>{visualBudgetLabel}<small>{visualBudgetHelp}</small><select aria-label={visualBudgetLabel} value={values.visualTokenBudget} onChange={(event) => update({ visualTokenBudget: Number(event.target.value) })}>{[70, 140, 280, 560, 1120].map((budget) => <option key={budget} value={budget}>{budget} tokens</option>)}</select></label>}
      {prefixCacheAvailable && <label>Application-managed prefix cache<small>Enable only after physical qualification</small><select aria-label="Application-managed prefix cache" value={values.prefixCacheEnabled ? "enabled" : "disabled"} onChange={(event) => update({ prefixCacheEnabled: event.target.value === "enabled" })}><option value="disabled">Disabled</option><option value="enabled">Enabled</option></select></label>}
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
    "Historical Routing Profile revisions and cached Model files will be kept.\n\n" +
    "Cancel leaves the Worker unchanged.",
  );
}

function WorkersView({ workers, models, templates, thermal, operationErrors, pending, operate, refresh, openDay }: { workers: Worker[]; models: ModelEntry[]; templates: RuntimeTemplate[]; thermal: ThermalStatus; operationErrors: Readonly<Record<string, string>>; pending: ReadonlySet<string>; operate: (worker: Worker, operation: WorkerOperation) => Promise<void>; refresh: () => Promise<void>; openDay: boolean }) {
  const collapseControls = useContext(CollapseContext);
  if (!collapseControls) throw new Error("Collapse controls are unavailable");
  const [libraryPreferences, setLibraryPreferences] = useStoredPreferences(WORKER_LIBRARY_STORAGE_KEY, loadWorkerLibraryPreferences);
  const { query, state: stateFilter, runtime: runtimeFilter, sort } = libraryPreferences;
  const [feedback, setFeedback] = useState<string | null>(null);
  const [replacing, setReplacing] = useState<string | null>(null);
  const [replacementWorkerName, setReplacementWorkerName] = useState("");
  const [replacementParameters, setReplacementParameters] = useState<WorkerParameterValues>(() => runtimeParameterDefaults());
  const [rebindDrafts, setRebindDrafts] = useState(true);
  const thermalLoadNotice = thermal.model_loading_allowed ? null : thermalLoadingNotice(thermal);
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
    const result = await postJson<{ replacement: Worker; rebound_profile_drafts: string[] }>(`/api/workers/${worker.id}/replacement`, {
      name: replacementWorkerName,
      ...workerParameterPayload(template, replacementParameters),
      rebind_drafts: rebindDrafts,
    });
    setReplacing(null);
    const rebound = result.rebound_profile_drafts.length;
    setFeedback(`Created replacement Worker “${result.replacement.name}”. ${rebound} draft Routing Profile${rebound === 1 ? " was" : "s were"} updated; published routing is unchanged until you publish a draft.`);
    await refresh();
  };
  const qualify = async (worker: Worker, capabilityId: string) => {
    const result = await postJson<{ ok: boolean; test?: { evidence?: { error_summary?: string } } }>(`/api/workers/${worker.id}/capabilities/${capabilityId}/qualify`);
    if (!result.ok) throw new Error(result.test?.evidence?.error_summary ?? "Capability qualification failed.");
    setFeedback(`Qualified ${humanise(capabilityId)} for “${worker.name}”.`);
    await refresh();
  };
  return <div className="view-stack"><div className="view-actions worker-view-heading"><p>A Worker is one configured, startable service. Its name is editable; its execution identity is not. Use Replace to change safe model limits without mutating the original Worker.</p></div>
    {!!workers.length && <div className="worker-toolbar" aria-label="Worker search and filters"><label>Search workers<input type="search" value={query} placeholder="Name, model or capability" onChange={(event) => setLibraryPreferences((current) => ({ ...current, query: event.target.value }))} /></label><label>State<select value={stateFilter} onChange={(event) => setLibraryPreferences((current) => ({ ...current, state: event.target.value }))}><option value="">All states</option>{stateFilter && !states.includes(stateFilter as Worker["state"]) && <option value={stateFilter}>{stateFilter.replaceAll("-", " ")} (not currently present)</option>}{states.map((state) => <option key={state} value={state}>{state.replaceAll("-", " ")}</option>)}</select></label><label>Runtime<select value={runtimeFilter} onChange={(event) => setLibraryPreferences((current) => ({ ...current, runtime: event.target.value }))}><option value="">All runtimes</option>{runtimeFilter && !runtimes.includes(runtimeFilter) && <option value={runtimeFilter}>{runtimeFilter} (not currently present)</option>}{runtimes.map((runtime) => <option key={runtime} value={runtime}>{runtime}</option>)}</select></label><label>Sort workers<select value={sort} onChange={(event) => setLibraryPreferences((current) => ({ ...current, sort: event.target.value as WorkerSort }))}><option value="name-asc">Name A–Z</option><option value="name-desc">Name Z–A</option><option value="model-asc">Model</option><option value="runtime-asc">Runtime</option><option value="state">State</option></select></label><div className="worker-filter-summary" role="status"><span>{sorted.length} of {workers.length} Worker{workers.length === 1 ? "" : "s"}</span><button className="secondary compact-button" disabled={!filtersActive} onClick={clearFilters}>Clear filters</button></div></div>}
    {feedback && <div className="configuration-feedback">{feedback}</div>}
    {!workers.length ? <section className="empty-state"><h2>No Workers configured</h2><p>Create one from the Models view. ModelDeck does not create packaged Worker cards.</p></section> : !sorted.length ? <section className="empty-state compact"><h2>No Workers match these filters</h2><p>Try a different name, model, capability, state or runtime.</p><button className="secondary" onClick={clearFilters}>Clear filters</button></section> : <div className="worker-grid">{sorted.map((worker) => {
      const workerPending = workerOperationPending(pending, worker.id);
      const template = templates.find((item) => item.id === worker.runtime_template_id);
      const workerModel = models.find((model) => (worker.artifact_model_id ?? worker.model_id) === model.model_id && (worker.artifact_revision ?? worker.revision) === model.revision);
      const workerCapabilities = workerModel?.potential_capabilities.filter((capability) => capability.effective_allowed && capability.available_runtime_template_ids.includes(worker.runtime_template_id ?? "")) ?? [];
      const operationError = operationErrors[worker.id];
      const sectionId = `worker-${worker.id}`;
      const collapsed = collapseControls.preferences.sections[sectionId] ?? collapseControls.preferences.allCollapsed;
      return <article className={`worker-card state-${worker.state}${collapsed ? " collapsed" : ""}`} key={worker.id}><div className="worker-card-heading"><div><p className="worker-id">{worker.runtime === "mock" ? `${worker.generation_family} · mock` : worker.generation_family}</p><h3>{worker.name}</h3></div><div className="worker-card-heading-actions"><StateBadge state={worker.state} /><button className="secondary compact-button" aria-expanded={!collapsed} aria-label={`${collapsed ? "Expand" : "Collapse"} Worker ${worker.name}`} onClick={() => collapseControls.toggleSection(sectionId)}>{collapsed ? "Expand" : "Collapse"}</button></div></div><div className="worker-card-body" hidden={collapsed}><p className="worker-summary">{worker.model_id} · {worker.runtime}{worker.runtime === "mock" ? ` · ${humanise(String(worker.settings.mock_contract_id ?? "legacy contract"))} · ${humanise(String(worker.settings.mock_scenario ?? "success"))}` : ""}</p>{operationError && <div className="worker-operation-error" role="alert"><strong>Start blocked</strong><span>{operationError}</span></div>}{thermalLoadNotice && <div className="thermal-load-notice" role="status"><strong>Model loading paused</strong><span>{thermalLoadNotice}</span></div>}{worker.last_error && <p className="inline-error">{worker.last_error}</p>}<details><summary>Immutable execution details</summary><DefinitionList rows={[["Internal ID", worker.id], ["Revision", worker.revision], ["Runtime", worker.runtime], ["Template", worker.runtime_template_id ?? "Built in"], ...(worker.runtime === "mock" ? [["Mock contract", String(worker.settings.mock_contract_id ?? "Legacy family mock")], ["Mock scenario", String(worker.settings.mock_scenario ?? "success")], ["Mock delay", worker.settings.mock_delay_ms ? `${worker.settings.mock_delay_ms} ms` : "None"]] as Array<[string, string]> : []), ["Port", String(worker.port)], ["Lifecycle", worker.lifecycle], ["Data type", worker.dtype], ["Thinking mode", String(worker.settings.thinking_mode ?? "Backend default")], ["Context length", String(worker.settings.context_length ?? "Not applicable")], ["Maximum output", String(worker.settings.maximum_new_tokens ?? "Not applicable")], ["Visual token budget", String(worker.settings.visual_token_budget ?? "Not applicable")], ["Maximum denoising steps", String(worker.settings.maximum_denoising_steps ?? "Not applicable")]]} /></details>{workerCapabilities.length > 0 && <div className="worker-capability-list"><strong>Allowed capabilities</strong>{workerCapabilities.map((capability) => { const ownStatus = capability.qualifying_workers.find((item) => item.worker_id === worker.id)?.status ?? "not-tested"; return <div key={capability.id}><span><b>{capability.display_name}</b><small>{humanise(ownStatus)}</small></span><button className="secondary compact-button" disabled={worker.state !== "ready"} onClick={() => void qualify(worker, capability.id).catch((reason) => setFeedback(messageFrom(reason)))}>Qualify</button></div>; })}</div>}<div className="button-row"><button className="secondary" disabled={openDay || workerPending} onClick={() => void rename(worker).catch((reason) => setFeedback(messageFrom(reason)))}>Rename</button><button className="secondary" disabled={openDay || workerPending || !template} title={template ? "Create a new Worker with revised parameters" : "The trusted runtime is no longer installed"} onClick={() => template && beginReplacement(worker, template)}>Replace</button><button disabled={workerPending || worker.state === "ready" || !thermal.model_loading_allowed} title={thermalLoadNotice ?? undefined} onClick={() => void operate(worker, "start")}>{pending.has(`${worker.id}:start`) ? "Starting…" : "Start"}</button><button className="secondary" disabled={workerPending || worker.state !== "ready"} onClick={() => void operate(worker, "smoke")}>{pending.has(`${worker.id}:smoke`) ? "Checking…" : "Check Worker"}</button><button className="secondary" disabled={workerPending || worker.state === "stopped"} onClick={() => void operate(worker, "stop")}>{pending.has(`${worker.id}:stop`) ? "Stopping…" : "Stop"}</button><button className="secondary danger" disabled={openDay || workerPending || !["stopped", "failed"].includes(worker.state)} onClick={() => void archive(worker).catch((reason) => setFeedback(messageFrom(reason)))}>Archive</button></div>
        {replacing === worker.id && template && <div className="runtime-form worker-replacement-form"><div className="runtime-form-heading"><strong>Replace this Worker</strong><small>The Model, revision and trusted runtime stay fixed. The original Worker is kept.</small></div><div className="runtime-fields"><label>Replacement name<input value={replacementWorkerName} maxLength={80} onChange={(event) => setReplacementWorkerName(event.target.value)} /></label><label>Model<input value={worker.artifact_model_id ?? worker.model_id} disabled /></label><label>Runtime<input value={template.display_name} disabled /></label></div><WorkerParameterFields template={template} values={replacementParameters} onChange={setReplacementParameters} prefixCacheAvailable={APPLICATION_MANAGED_PREFIX_CACHE_MODELS.has(worker.model_id)} /><label className="rebind-option"><input type="checkbox" checked={rebindDrafts} onChange={(event) => setRebindDrafts(event.target.checked)} /> Rebind draft Routing Profile capabilities to the replacement</label><p className="manifest-note">Published Routing Profile revisions always keep the original Worker until you explicitly publish an updated draft.</p><div className="runtime-form-actions"><button disabled={!replacementWorkerName.trim() || !parametersAreValid(template, replacementParameters)} onClick={() => void replace(worker, template).catch((reason) => setFeedback(messageFrom(reason)))}>Create replacement</button><button className="secondary" onClick={() => setReplacing(null)}>Cancel</button></div></div>}</div>
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

function runtimeTemplateIdsForModel(model: ModelEntry): string[] {
  return [...new Set(model.potential_capabilities.flatMap((capability) => capability.available_runtime_template_ids))];
}

function ModelExecutionStatus({ model, workers, templates }: { model: ModelEntry; workers: Worker[]; templates: RuntimeTemplate[] }) {
  const configured = workersForModel(model, workers);
  const runtimeIds = runtimeTemplateIdsForModel(model);
  const runtimeNames = runtimeIds.map((runtimeId) => templates.find((template) => template.id === runtimeId)?.display_name ?? runtimeId);
  const workerIds = new Set(configured.map((worker) => worker.id));
  const qualificationStatuses = [...new Set(model.potential_capabilities.flatMap((capability) => capability.qualifying_workers)
    .filter((qualification) => workerIds.has(qualification.worker_id)).map((qualification) => qualification.status))];
  const workerStates = [...new Set(configured.map((worker) => humanise(worker.state)))];
  const cacheStatus = model.download_state === "partial" ? "Partial snapshot" : "Complete snapshot";
  const runtimeStatus = runtimeNames.length
    ? `Available · ${runtimeNames.join(", ")}`
    : model.download_state === "partial" ? "Pending · complete the local snapshot" : "Missing · runtime implementation required";
  const workerStatus = configured.length
    ? `${configured.length} configured · ${workerStates.join(", ")}`
    : "None configured";
  const qualificationStatus = configured.length
    ? qualificationStatuses.length ? qualificationStatuses.map(humanise).join(", ") : "Not tested"
    : "No configured Worker";
  return <section className={`model-execution-status${runtimeNames.length ? " has-runtime" : " missing-runtime"}`} aria-label={`Execution status for ${model.model_id}`}>
    <strong>Execution status</strong>
    <DefinitionList rows={[["Cache", cacheStatus], ["Runtime", runtimeStatus], ["Workers", workerStatus], ["Qualification", qualificationStatus]]} />
  </section>;
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
    <div className="model-main"><div><h3>{model.model_id}</h3><p>{model.generation_family_hint ?? "Unclassified"} · {formatBytes(model.physical_size_bytes)}</p></div><div className="model-card-heading-actions"><StateBadge state={model.download_state} /><button className="secondary compact-button" aria-expanded={!collapsed} aria-label={`${collapsed ? "Expand" : "Collapse"} Model ${model.model_id}`} onClick={toggle}>{collapsed ? "Expand" : "Collapse"}</button></div></div>
    <div className="model-card-body" hidden={collapsed}>{children}</div>
  </article>;
}

function ModelsView({ models, workers, templates, refresh, openDay }: { models: ModelEntry[]; workers: Worker[]; templates: RuntimeTemplate[]; refresh: () => Promise<void>; openDay: boolean }) {
  const [libraryPreferences, setLibraryPreferences] = useStoredPreferences(MODEL_LIBRARY_STORAGE_KEY, loadModelLibraryPreferences);
  const { query, status, sort } = libraryPreferences;
  const [configuring, setConfiguring] = useState<string | null>(null);
  const [configuringCapability, setConfiguringCapability] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [runtime, setRuntime] = useState("");
  const [artifact, setArtifact] = useState("");
  const [parameters, setParameters] = useState<WorkerParameterValues>(() => runtimeParameterDefaults());
  const [feedback, setFeedback] = useState<string | null>(null);
  const [removingWorker, setRemovingWorker] = useState<string | null>(null);
  const [approvingCandidate, setApprovingCandidate] = useState<string | null>(null);
  const sorted = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    return models.filter((model) => {
      const matchesQuery = !needle || [model.model_id, model.generation_family_hint, model.runnable_reason, ...model.capability_hints, ...model.potential_capabilities.flatMap((capability) => [capability.id, capability.display_name, capability.description, capability.qualification_status, capability.runtime_status])].some((value) => value?.toLocaleLowerCase().includes(needle));
      const hasRuntime = runtimeTemplateIdsForModel(model).length > 0;
      const hasWorkers = workersForModel(model, workers).length > 0;
      const matchesStatus = !status
        || (status === "runtime-available" && hasRuntime)
        || (status === "runtime-missing" && !hasRuntime)
        || (status === "workers-configured" && hasWorkers)
        || (status === "workers-missing" && !hasWorkers);
      return matchesQuery && matchesStatus;
    }).sort((a, b) => sort === "name-desc" ? b.model_id.localeCompare(a.model_id) : sort === "size-desc" ? b.physical_size_bytes - a.physical_size_bytes : sort === "size-asc" ? a.physical_size_bytes - b.physical_size_bytes : sort === "readiness" ? Number(b.runnable) - Number(a.runnable) : sort === "workers" ? b.worker_count - a.worker_count : a.model_id.localeCompare(b.model_id));
  }, [models, query, sort, status, workers]);
  const filtersActive = Boolean(query.trim() || status);
  const begin = (model: ModelEntry, capabilityId: string) => {
    const capability = model.potential_capabilities.find((item) => item.id === capabilityId);
    const template = templates.find((item) => item.id === capability?.available_runtime_template_ids[0]);
    setConfiguring(`${model.model_id}@${model.revision}`);
    setConfiguringCapability(capabilityId);
    setName(model.model_id.split("/").at(-1)?.replaceAll("-", " ") ?? "New Worker");
    setRuntime(template?.id ?? "");
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
      capability_id: configuringCapability,
      artifact_id: artifact || undefined,
      ...workerParameterPayload(template, parameters),
    });
    setConfiguring(null);
    setConfiguringCapability(null);
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
  const setModelAllowed = async (model: ModelEntry, allowed: boolean) => {
    await postJson("/api/catalogue/policy", { model_id: model.model_id, revision: model.revision, allowed });
    setFeedback(`${model.model_id} is now ${allowed ? "allowed" : "disallowed"} in ModelDeck.`);
    await refresh();
  };
  const setCapabilityAllowed = async (model: ModelEntry, capabilityId: string, allowed: boolean) => {
    await postJson("/api/catalogue/capabilities/policy", { model_id: model.model_id, revision: model.revision, capability_id: capabilityId, allowed });
    setFeedback(`${humanise(capabilityId)} is now ${allowed ? "allowed" : "disallowed"}.`);
    await refresh();
  };
  const beginSupportedWorker = async (model: ModelEntry, capability: PotentialCapability) => {
    if (!capability.policy_allowed) {
      await postJson("/api/catalogue/capabilities/policy", {
        model_id: model.model_id,
        revision: model.revision,
        capability_id: capability.id,
        allowed: true,
      });
    }
    begin(model, capability.id);
  };
  const approveCandidate = async (model: ModelEntry) => {
    const identity = `${model.model_id}@${model.revision}`;
    setApprovingCandidate(identity);
    try {
      const result = await postJson<{ candidate_id: string }>("/api/catalogue/candidates/approve", {
        model_id: model.model_id,
        revision: model.revision,
      });
      setFeedback(`Approved ${model.model_id} as trusted local candidate ${result.candidate_id}.`);
      await refresh();
    } catch (reason) {
      setFeedback(messageFrom(reason));
    } finally {
      setApprovingCandidate(null);
    }
  };
  if (configuring) {
    const model = models.find((item) => `${item.model_id}@${item.revision}` === configuring);
    const capability = model?.potential_capabilities.find((item) => item.id === configuringCapability);
    const availableTemplates = templates.filter((item) => capability?.available_runtime_template_ids.includes(item.id));
    const selectedTemplate = availableTemplates.find((template) => template.id === runtime);
    if (model) return <div className="view-stack"><section className="panel model-configuration"><div className="runtime-form-heading"><p className="eyebrow">{capability?.display_name ?? model.generation_family_hint ?? "Model"}</p><h2>Create a Worker</h2><small>{model.model_id} at pinned revision {model.revision}</small></div><div className="runtime-fields"><label>Worker name<input value={name} maxLength={80} onChange={(event) => setName(event.target.value)} /></label><label>Runtime<select value={runtime} onChange={(event) => { const nextRuntime = event.target.value; setRuntime(nextRuntime); setParameters(runtimeParameterDefaults(availableTemplates.find((item) => item.id === nextRuntime))); }}>{availableTemplates.map((template) => <option key={template.id} value={template.id}>{template.display_name}</option>)}</select></label>{model.artifacts && model.artifacts.length > 0 && <label>Model artefact<select value={artifact} onChange={(event) => setArtifact(event.target.value)}>{model.artifacts.map((item) => <option key={item.artifact_id} value={item.artifact_id}>{item.artifact_id} · {item.filenames.join(", ")}</option>)}</select></label>}</div>{selectedTemplate ? <WorkerParameterFields template={selectedTemplate} values={parameters} onChange={setParameters} prefixCacheAvailable={APPLICATION_MANAGED_PREFIX_CACHE_MODELS.has(model.model_id)} capabilityId={configuringCapability} /> : <div className="configuration-feedback bad">No compatible trusted runtime is installed for this capability.</div>}<div className="runtime-form-actions"><button disabled={openDay || !name.trim() || !selectedTemplate || (selectedTemplate ? !parametersAreValid(selectedTemplate, parameters) : true)} onClick={() => selectedTemplate && void create(model, selectedTemplate).catch((reason) => setFeedback(messageFrom(reason)))}>Create Worker</button><button className="secondary" onClick={() => { setConfiguring(null); setConfiguringCapability(null); }}>Cancel</button></div>{feedback && <div className="configuration-feedback bad">{feedback}</div>}</section></div>;
  }
  const clearFilters = () => setLibraryPreferences((current) => ({ ...current, query: "", status: "" }));
  return <div className="view-stack"><div className="view-actions"><p>Models are read-only discoveries from the local Hugging Face cache. Runtime availability, configured Workers and qualification are reported separately.</p><div className="model-library-toolbar"><label>Search models<input type="search" value={query} placeholder="Name or capability" onChange={(event) => setLibraryPreferences((current) => ({ ...current, query: event.target.value }))} /></label><label>Status<select value={status} onChange={(event) => setLibraryPreferences((current) => ({ ...current, status: event.target.value as ModelStatusFilter }))}><option value="">All models</option><option value="runtime-available">Runtime available</option><option value="runtime-missing">Runtime missing</option><option value="workers-configured">Workers configured</option><option value="workers-missing">No Workers configured</option></select></label><label>Sort models<select value={sort} onChange={(event) => setLibraryPreferences((current) => ({ ...current, sort: event.target.value as ModelSort }))}><option value="name-asc">Name A–Z</option><option value="name-desc">Name Z–A</option><option value="readiness">Runnable first</option><option value="workers">Most Workers</option><option value="size-desc">Largest</option><option value="size-asc">Smallest</option></select></label><div className="model-filter-summary" role="status"><span>{sorted.length} of {models.length} Model{models.length === 1 ? "" : "s"}</span><button className="secondary compact-button" disabled={!filtersActive} onClick={clearFilters}>Clear filters</button></div></div></div>{openDay && <div className="configuration-feedback">Local deployment policy locks configuration. Restart ModelDeck without <code>-LockConfiguration</code> to create Workers.</div>}{feedback && <div className="configuration-feedback good">{feedback}</div>}
    <section className="panel"><StaticPanelHeading title="Discovered Models" detail={filtersActive ? `${sorted.length} of ${models.length} cached` : `${models.length} cached`} />{sorted.length ? <div className="model-list">{sorted.map((model) => {
      return <ModelCardShell model={model} key={`${model.model_id}@${model.revision}`}>
        <ModelExecutionStatus model={model} workers={workers} templates={templates} />
        <div className="tag-list">{model.capability_hints.map((hint) => <span className="tag" key={hint}>{humanise(hint)}</span>)}</div>
        <div className="model-policy"><span><strong>ModelDeck master policy</strong><small>{model.modeldeck_allowed ? "Allowed" : "Disallowed · capability choices retained"}</small></span><button className="secondary compact-button" disabled={openDay || !model.revision} onClick={() => void setModelAllowed(model, !model.modeldeck_allowed).catch((reason) => setFeedback(messageFrom(reason)))}>{model.modeldeck_allowed ? "Disallow model" : "Allow model"}</button></div>
        {model.candidate_registration?.eligible && <div className="model-policy"><span><strong>Local Qwen3.5 candidate</strong><small>{model.candidate_registration.reason}{model.candidate_registration.filename ? ` · ${model.candidate_registration.filename}` : ""}</small></span>{model.candidate_registration.approved ? <StateBadge state="qualified" /> : <button className="secondary compact-button" disabled={openDay || !model.revision || approvingCandidate === `${model.model_id}@${model.revision}`} onClick={() => void approveCandidate(model)}>{approvingCandidate === `${model.model_id}@${model.revision}` ? "Verifying SHA-256…" : "Verify and approve"}</button>}</div>}
        <div className="potential-capability-list">{model.potential_capabilities.length ? model.potential_capabilities.map((capability) => {
          const hasRuntime = capability.available_runtime_template_ids.length > 0;
          const readyToConfigure = hasRuntime && model.modeldeck_allowed;
          return <article key={capability.id} className={`potential-capability ${capability.effective_allowed ? "allowed" : ""} ${hasRuntime ? "runnable" : "unavailable"}`}><div className="potential-capability-heading"><span><strong>{capability.display_name}</strong><small>{capability.description}</small></span><div><StateBadge state={hasRuntime ? capability.policy_allowed ? capability.qualification_status : "disallowed" : "runtime-unavailable"} /></div></div><div className="tag-list">{capability.traits.map((trait) => <span className="tag" key={trait}>{humanise(trait)}</span>)}</div>{hasRuntime ? <><p className="capability-reason"><strong>Compatible runtime available.</strong> {capability.policy_allowed ? "Choose its runtime settings and create a local Worker." : "Set it up to allow this capability and choose its runtime settings."}</p><div className="model-actions"><button disabled={openDay || !readyToConfigure || !model.revision} onClick={() => void beginSupportedWorker(model, capability).catch((reason) => setFeedback(messageFrom(reason)))}>{capability.policy_allowed ? "Create Worker" : "Set up Worker"}</button>{capability.policy_allowed && <button className="secondary compact-button" disabled={openDay || !model.revision} onClick={() => void setCapabilityAllowed(model, capability.id, false).catch((reason) => setFeedback(messageFrom(reason)))}>Disallow</button>}</div></> : <><p className="capability-reason"><strong>Runtime implementation required.</strong> This Model’s local metadata recognises the capability, but ModelDeck has no compatible trusted runtime for it.</p><details className="capability-policy"><summary>{capability.policy_allowed ? "Allowed for a future runtime" : "Advanced: allow for a future runtime"}</summary><p>Allowing this records your permission only. It will not create or start a Worker until a compatible trusted runtime is installed.</p><button className="secondary compact-button" disabled={openDay || !model.revision} onClick={() => void setCapabilityAllowed(model, capability.id, !capability.policy_allowed).catch((reason) => setFeedback(messageFrom(reason)))}>{capability.policy_allowed ? "Disallow" : "Allow for future runtime"}</button></details></>}<details><summary>Evidence and provenance</summary>{capability.evidence.map((evidence, index) => <p className="manifest-note" key={`${evidence.source}-${index}`}><strong>{humanise(evidence.kind)} · {humanise(evidence.confidence)}</strong> — {evidence.detail}{evidence.reference && <> · <a href={evidence.reference} target="_blank" rel="noreferrer">Reviewed source</a></>}</p>)}</details></article>;
        }) : <p className="model-stage">No capability can be supported by the available local metadata yet.</p>}</div>
        <ModelWorkerSummary model={model} workers={workers} openDay={openDay} removingWorker={removingWorker} onRemove={removeWorker} />
        <p className="model-stage"><strong>Next action:</strong> {model.runnable_reason}</p>
      </ModelCardShell>;
    })}</div> : <div className="empty-state compact"><h3>No Models match these filters</h3><p>Try a different name, capability or status.</p><button className="secondary" onClick={clearFilters}>Clear filters</button></div>}</section>
  </div>;
}

function AdvancedView({ hardware, telemetry, thermal, contracts, templates, runtimeInstallations, compatibility, workers, benchmarkHistory }: { hardware: HardwareProbe; telemetry: Telemetry; thermal: ThermalStatus; contracts: ProtocolContract[]; templates: RuntimeTemplate[]; runtimeInstallations: RuntimeInstallation[]; compatibility: CompatibilityTest[]; workers: Worker[]; benchmarkHistory: BenchmarkHistory }) {
  return <div className="view-stack">
    <ThroughputHistory history={benchmarkHistory} />
    <section className="panel"><PanelHeading title="Detected hardware" detail="Reported, never assumed" /><DefinitionList rows={[["Configured target", `${hardware.configured.gpu} (${hardware.configured.gpu_architecture})`], ["Detected Fedora", hardware.detected.fedora_release ?? "Not detected"], ["Kernel", hardware.detected.kernel], ["ROCm packages", hardware.detected.rocm_packages.join(", ") || "Not detected"], ["Available memory", formatBytes(telemetry.memory.available_bytes)]]} /></section>
    <section className="panel"><PanelHeading title="Thermal workload policy" detail={thermal.enabled ? "ModelDeck self-throttling active" : "Disabled by configuration"} /><DefinitionList rows={[["ModelDeck thermal state", humanise(thermal.state)], ["APU temperature", thermal.temperature_c == null ? "Unavailable" : `${thermal.temperature_c.toFixed(1)}°C`], ["Control sensor", thermal.sensor_id ?? "Awaiting selection"], ["Heavy concurrency", `${thermal.active_heavy_concurrency ?? "Unknown"} active · limit ${thermal.heavy_concurrency_limit}`], ["Background benchmarks", thermal.background_paused ? "Paused" : "Permitted"], ["Model loading", thermal.model_loading_allowed ? "Permitted" : "Blocked"], ["Scene refresh interval", `${thermal.scenechat_degradation.minimum_frame_interval_seconds} seconds`], ["Reason", humanise(thermal.reason_code)], ["Host power policy", thermal.host_power_policy.service_active === true ? "Active · external read-only" : "Unavailable or inactive · external read-only"], ["TuneD profile", thermal.host_power_policy.tuned_profile ?? "Unavailable"]]} /></section>
    <div className="two-column"><section className="panel"><PanelHeading title="Trusted protocol contracts" detail={`${contracts.length} code-owned`} /><ul className="status-list">{contracts.map((contract) => <li key={contract.id}><StatusDot state="good" /><span><strong>{contract.display_name}</strong><small>{contract.id} · {contract.surfaces.join(", ")}</small></span></li>)}</ul></section><section className="panel"><PanelHeading title="Trusted runtime templates" detail={`${templates.length} registered`} /><ul className="status-list">{templates.map((template) => <li key={template.id}><StatusDot state="good" /><span><strong>{template.display_name}</strong><small>{template.id} · {template.package_version}</small></span></li>)}</ul></section></div>
    <section className="panel"><PanelHeading title="Runtime installations" detail={`${runtimeInstallations.filter((installation) => installation.start_allowed).length} of ${runtimeInstallations.length} ready to start`} /><div className="evidence-list">{runtimeInstallations.map((installation) => <details className="evidence-row" key={installation.installation_id}><summary><span><StatusDot state={installation.start_allowed ? "good" : installation.integrity_status === "missing" ? "neutral" : "bad"} /><strong>{installation.display_name}</strong><small>{humanise(installation.integrity_status)} · {humanise(installation.currency_status)}</small></span><code>{installation.detected.source_revision?.slice(0, 12) ?? "not detected"}</code></summary><DefinitionList rows={[
      ["Start policy", installation.start_allowed ? "Allowed" : "Blocked"],
      ["Detected revision", installation.detected.source_revision ?? "Unavailable"],
      ["Recommended revision", installation.recommended_source_revision],
      ["Executable SHA-256", installation.detected.executable_sha256 ?? "Unavailable"],
      ["Build receipt SHA-256", installation.detected.receipt_sha256 ?? "Unavailable"],
      ["Backend", installation.detected.backend ?? "Unavailable"],
      ["Platform", `${installation.detected.operating_system} · ${installation.detected.architecture}`],
      ["Consumers", `${installation.runtime_template_ids.length} templates · ${installation.worker_ids.length} Workers`],
      ["Missing features", installation.missing_features.join(", ") || "None"],
      ["Reason codes", installation.reason_codes.map(humanise).join(", ") || "None"],
      ["Inspected", new Date(installation.inspected_at).toLocaleString()],
    ]} /></details>)}</div></section>
    <section className="panel"><PanelHeading title="Compatibility evidence" detail={`${compatibility.length} records`} /><div className="evidence-list">{compatibility.length ? compatibility.map((test) => <details className="evidence-row" key={test.id}><summary><span><StateBadge state={test.result} /><strong>{String(test.evidence.model_id ?? "Unknown Model")}</strong><small>{new Date(test.tested_at).toLocaleString()}</small></span><code>{test.fingerprint.slice(0, 12)}</code></summary><DefinitionList rows={Object.entries(test.evidence).slice(0, 16).map(([key, value]) => [humanise(key), String(value ?? "—")])} /></details>) : <p className="muted">Qualify a Worker capability to record evidence.</p>}</div></section>
    <LogsPanel workers={workers} />
  </div>;
}

const CHART_COLOURS = ["#68e0b8", "#70a7ff", "#f3ba67", "#ff7b86", "#c89cff", "#74d6e7"];

function ThroughputHistory({ history }: { history: BenchmarkHistory }) {
  const series = useMemo(() => {
    const grouped = new Map<string, BenchmarkThroughputPoint[]>();
    for (const point of history.points) grouped.set(point.series_key, [...(grouped.get(point.series_key) ?? []), point]);
    return [...grouped.entries()].map(([key, points], index) => ({ key, points, colour: CHART_COLOURS[index % CHART_COLOURS.length] }));
  }, [history.points]);
  if (!history.points.length) return <section className="panel throughput-panel" aria-label="Model throughput over time"><StaticPanelHeading title="Model throughput over time" detail="Median tokens/second from comparable benchmark runs" /><div className="empty-state compact"><h3>No throughput history yet</h3><p>Run the model benchmark suite to add privacy-safe, workload-specific samples.</p></div></section>;
  const width = 760, height = 260, left = 54, right = 18, top = 18, bottom = 42;
  const times = history.points.map((point) => new Date(point.observed_at).getTime());
  const minimumTime = Math.min(...times), maximumTime = Math.max(...times);
  const maximumThroughput = Math.max(...history.points.map((point) => point.tokens_per_second));
  const yMaximum = Math.max(10, Math.ceil(maximumThroughput / 10) * 10);
  const x = (point: BenchmarkThroughputPoint) => left + ((new Date(point.observed_at).getTime() - minimumTime) / Math.max(maximumTime - minimumTime, 1)) * (width - left - right);
  const y = (point: BenchmarkThroughputPoint) => top + (1 - point.tokens_per_second / yMaximum) * (height - top - bottom);
  return <section className="panel throughput-panel" aria-label="Model throughput over time"><StaticPanelHeading title="Model throughput over time" detail={`${history.points.length} median samples from ${history.reports_scanned} benchmark reports`} />
    <div className="throughput-chart-wrap"><svg className="throughput-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Tokens per second benchmark history">
      {[0, 0.25, 0.5, 0.75, 1].map((fraction) => { const chartY = top + fraction * (height - top - bottom); const label = yMaximum * (1 - fraction); return <g key={fraction}><line x1={left} x2={width - right} y1={chartY} y2={chartY} /><text x={left - 9} y={chartY + 4}>{label.toFixed(0)}</text></g>; })}
      {series.map((item) => <g key={item.key}>{item.points.length > 1 && <polyline points={item.points.map((point) => `${x(point)},${y(point)}`).join(" ")} style={{ stroke: item.colour }} />}{item.points.map((point) => <circle key={`${point.observed_at}-${point.tokens_per_second}`} cx={x(point)} cy={y(point)} r="5" style={{ fill: item.colour }}><title>{`${point.model_id} · ${point.workload} · ${point.tokens_per_second.toFixed(2)} tok/s · ${new Date(point.observed_at).toLocaleString()}`}</title></circle>)}</g>)}
      <text className="axis-title" x="14" y="14">tok/s</text><text className="axis-time" x={left} y={height - 12}>{new Date(minimumTime).toLocaleDateString()}</text><text className="axis-time end" x={width - right} y={height - 12}>{new Date(maximumTime).toLocaleDateString()}</text>
    </svg></div>
    <div className="throughput-legend">{series.map((item) => { const latest = item.points.at(-1)!; return <article key={item.key}><span className="throughput-swatch" style={{ background: item.colour }} /><div><strong>{latest.worker_name ?? latest.model_id.split("/").at(-1)}</strong><small>{latest.runtime} · {latest.workload}</small><span>{latest.tokens_per_second.toFixed(2)} tok/s latest · revision {latest.model_revision.slice(0, 12)}</span></div></article>; })}</div>
    <p className="manifest-note">Each line keeps the Model revision, Runtime, data type and workload fixed. Points are benchmark medians, not hardware telemetry or a universal model score.</p>
  </section>;
}

function LogsPanel({ workers }: { workers: Worker[] }) {
  const [workerId, setWorkerId] = useState(workers[0]?.id ?? "");
  const [logs, setLogs] = useState<WorkerLog[]>([]);
  useEffect(() => { if (!workerId) return; getJson<{ logs: WorkerLog[] }>(`/api/workers/${workerId}/logs`).then((value) => setLogs(value.logs)).catch(() => setLogs([])); }, [workerId]);
  return <CollapsiblePanel sectionId="advanced-worker-logs" title="Worker logs" detail={`${logs.length} entries`} className="log-panel"><div className="log-toolbar"><div><label htmlFor="log-worker">Worker</label><select id="log-worker" value={workerId} onChange={(event) => setWorkerId(event.target.value)}>{workers.map((worker) => <option key={worker.id} value={worker.id}>{worker.name}</option>)}</select></div></div><div className="log-view">{logs.length ? logs.map((log, index) => <div className={`log-entry ${log.level}`} key={`${log.timestamp}-${index}`}><time>{new Date(log.timestamp).toLocaleTimeString()}</time><span>{log.source}</span><code>{log.message}</code></div>) : <p>No logs for this Worker.</p>}</div></CollapsiblePanel>;
}

function Loading() { return <main className="loading-screen"><div className="brand-mark">MD</div><h1>Starting operator console</h1><p>Reading local Routing Profiles, capabilities, Workers and Models…</p><div className="loading-bar"><span /></div></main>; }
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
function thermalLoadingNotice(thermal: ThermalStatus) {
  const temperature = thermal.temperature_c == null ? "" : ` (${thermal.temperature_c.toFixed(1)}°C)`;
  if (thermal.state === "telemetry_degraded") {
    return `Fresh thermal telemetry is stabilising${temperature}. Start is disabled until it is current.`;
  }
  return `Thermal protection is active${temperature}. Start is disabled until safe loading capacity returns.`;
}
function messageFrom(reason: unknown) { return reason instanceof Error ? reason.message : "The operation failed."; }

function workerOperationPending(pending: ReadonlySet<string>, workerId: string) { return [...pending].some((key) => key.startsWith(`${workerId}:`)); }

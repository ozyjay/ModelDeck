import { useCallback, useEffect, useMemo, useState } from "react";

import { deleteJson, getJson, postJson, putJson } from "./api";
import type {
  CompatibilityTest,
  DemoAdapter,
  DemoRouteSmokeResult,
  DemoRouteStatus,
  DemoSet,
  DemoSetPlan,
  DemoSetValidation,
  Deployment,
  DeploymentUsage,
  GatewayStatus,
  HardwareProbe,
  LocalProfileRequest,
  ManagementHealth,
  ModelEntry,
  Profile,
  ProviderSelection,
  Telemetry,
  Worker,
  WorkerEvent,
  WorkerLog,
} from "./types";

type View = "overview" | "demo-routes" | "workers" | "models" | "compatibility" | "logs";
type WorkerOperation = "start" | "stop" | "restart" | "smoke";
type WorkerSort = "name-asc" | "name-desc" | "model-asc" | "runtime-asc" | "state";
type WorkerGrouping = "family" | "runtime" | "lifecycle" | "none";
type ModelSort = "name-asc" | "name-desc" | "size-desc" | "size-asc";

const NAVIGATION: Array<{ view: View; label: string; path: string }> = [
  { view: "overview", label: "Overview", path: "/" },
  { view: "demo-routes", label: "Demo routes", path: "/demo-routes" },
  { view: "workers", label: "Workers", path: "/workers" },
  { view: "models", label: "Model library", path: "/models" },
  { view: "compatibility", label: "Compatibility", path: "/compatibility" },
  { view: "logs", label: "Logs", path: "/logs" },
];

function viewFromPath(pathname: string): View {
  return NAVIGATION.find((item) => item.path === pathname)?.view ?? "overview";
}

export default function App() {
  const [view, setView] = useState<View>(() => viewFromPath(window.location.pathname));
  const [health, setHealth] = useState<ManagementHealth | null>(null);
  const [gateway, setGateway] = useState<GatewayStatus | null>(null);
  const [hardware, setHardware] = useState<HardwareProbe | null>(null);
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [compatibility, setCompatibility] = useState<CompatibilityTest[]>([]);
  const [providerSelections, setProviderSelections] = useState<ProviderSelection[]>([]);
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [deploymentUsage, setDeploymentUsage] = useState<DeploymentUsage[]>([]);
  const [demoSets, setDemoSets] = useState<DemoSet[]>([]);
  const [demoAdapters, setDemoAdapters] = useState<DemoAdapter[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [eventStreamConnected, setEventStreamConnected] = useState(false);
  const [pending, setPending] = useState<string | null>(null);

  const refreshWorkers = useCallback(async () => {
    setWorkers(await getJson<Worker[]>("/api/workers"));
  }, []);

  const refreshCompatibility = useCallback(async () => {
    const response = await getJson<{ tests: CompatibilityTest[] }>("/api/compatibility");
    setCompatibility(response.tests);
  }, []);

  const refreshDeploymentUsage = useCallback(async () => {
    const response = await getJson<{ deployments: DeploymentUsage[] }>("/api/deployments/usage");
    setDeploymentUsage(response.deployments);
  }, []);

  const refreshGateway = useCallback(async () => {
    const [nextGateway, selections] = await Promise.all([
      getJson<GatewayStatus>("/api/gateway/status"),
      getJson<{ selections: ProviderSelection[] }>("/api/gateway/provider-selections"),
    ]);
    setGateway(nextGateway);
    setProviderSelections(selections.selections);
  }, []);

  const refreshDemoConfiguration = useCallback(async () => {
    const [deploymentResponse, demoSetResponse, usageResponse, selections] = await Promise.all([
      getJson<Deployment[]>("/api/deployments"),
      getJson<{ demo_sets: DemoSet[] }>("/api/demo-sets"),
      getJson<{ deployments: DeploymentUsage[] }>("/api/deployments/usage"),
      getJson<{ selections: ProviderSelection[] }>("/api/gateway/provider-selections"),
    ]);
    setDeployments(deploymentResponse);
    setDemoSets(demoSetResponse.demo_sets);
    setDeploymentUsage(usageResponse.deployments);
    setProviderSelections(selections.selections);
  }, []);

  const refreshConfiguration = useCallback(async () => {
    const [nextProfiles, nextWorkers, nextGateway, catalogue, selections, deploymentResponse, demoSetResponse, usageResponse] = await Promise.all([
      getJson<Profile[]>("/api/profiles"),
      getJson<Worker[]>("/api/workers"),
      getJson<GatewayStatus>("/api/gateway/status"),
      getJson<{ models: ModelEntry[] }>("/api/catalogue"),
      getJson<{ selections: ProviderSelection[] }>("/api/gateway/provider-selections"),
      getJson<Deployment[]>("/api/deployments"),
      getJson<{ demo_sets: DemoSet[] }>("/api/demo-sets"),
      getJson<{ deployments: DeploymentUsage[] }>("/api/deployments/usage"),
    ]);
    setProfiles(nextProfiles);
    setWorkers(nextWorkers);
    setGateway(nextGateway);
    setModels(catalogue.models);
    setProviderSelections(selections.selections);
    setDeployments(deploymentResponse);
    setDemoSets(demoSetResponse.demo_sets);
    setDeploymentUsage(usageResponse.deployments);
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nextHealth, nextGateway, nextHardware, nextTelemetry, nextWorkers, nextProfiles, catalogue, tests, selections, deploymentResponse, demoSetResponse, adapterResponse, usageResponse] =
        await Promise.all([
          getJson<ManagementHealth>("/api/health"),
          getJson<GatewayStatus>("/api/gateway/status"),
          getJson<HardwareProbe>("/api/hardware"),
          getJson<Telemetry>("/api/telemetry"),
          getJson<Worker[]>("/api/workers"),
          getJson<Profile[]>("/api/profiles"),
          getJson<{ models: ModelEntry[] }>("/api/catalogue"),
          getJson<{ tests: CompatibilityTest[] }>("/api/compatibility"),
          getJson<{ selections: ProviderSelection[] }>("/api/gateway/provider-selections"),
          getJson<Deployment[]>("/api/deployments"),
          getJson<{ demo_sets: DemoSet[] }>("/api/demo-sets"),
          getJson<{ adapters: DemoAdapter[] }>("/api/demo-adapters"),
          getJson<{ deployments: DeploymentUsage[] }>("/api/deployments/usage"),
        ]);
      setHealth(nextHealth);
      setGateway(nextGateway);
      setHardware(nextHardware);
      setTelemetry(nextTelemetry);
      setWorkers(nextWorkers);
      setProfiles(nextProfiles);
      setModels(catalogue.models);
      setCompatibility(tests.tests);
      setProviderSelections(selections.selections);
      setDeployments(deploymentResponse);
      setDemoSets(demoSetResponse.demo_sets);
      setDemoAdapters(adapterResponse.adapters);
      setDeploymentUsage(usageResponse.deployments);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const onPopState = () => setView(viewFromPath(window.location.pathname));
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    const source = new EventSource("/api/events");
    source.onopen = () => setEventStreamConnected(true);
    source.onerror = () => setEventStreamConnected(false);
    source.addEventListener("worker", (raw) => {
      try {
        const event = JSON.parse((raw as MessageEvent).data) as WorkerEvent;
        setWorkers((current) =>
          current.map((worker) =>
            worker.id === event.worker_id ? { ...worker, state: event.state } : worker,
          ),
        );
      } catch {
        setEventStreamConnected(false);
      }
    });
    return () => source.close();
  }, []);

  useEffect(() => {
    const poll = window.setInterval(() => {
      if (!document.hidden) {
        void refreshWorkers().catch(() => setEventStreamConnected(false));
      }
    }, eventStreamConnected ? 15_000 : 5_000);
    return () => window.clearInterval(poll);
  }, [eventStreamConnected, refreshWorkers]);

  useEffect(() => {
    const poll = window.setInterval(() => {
      if (!document.hidden) {
        void getJson<Telemetry>("/api/telemetry").then(setTelemetry).catch(() => undefined);
        void refreshGateway().catch(() => undefined);
      }
    }, 8_000);
    return () => window.clearInterval(poll);
  }, [refreshGateway]);

  const navigate = (next: View, path: string) => {
    window.history.pushState({}, "", path);
    setView(next);
  };

  const operate = async (worker: Worker, operation: WorkerOperation) => {
    if (!confirmOperation(worker, operation, workers)) return;
    const key = `${worker.id}:${operation}`;
    setPending(key);
    setError(null);
    try {
      await postJson(`/api/workers/${encodeURIComponent(worker.id)}/${operation}`);
      await Promise.all([refreshWorkers(), refreshGateway()]);
      if (operation === "smoke") await refreshCompatibility();
    } catch (reason) {
      setError(`${worker.id}: ${messageFrom(reason)}`);
    } finally {
      setPending(null);
    }
  };

  const stopAll = async () => {
    if (!window.confirm("Stop every managed ModelDeck worker?")) return;
    setPending("stop-all");
    setError(null);
    try {
      await postJson("/api/presets/stop-all");
      await Promise.all([refreshWorkers(), refreshGateway()]);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      setPending(null);
    }
  };

  const selectProvider = async (alias: string, profileId: string) => {
    setPending(`provider-selection:${alias}`);
    setError(null);
    try {
      const selection = await postJson<ProviderSelection>(
        `/api/gateway/provider-selections/${encodeURIComponent(alias)}`,
        { profile_id: profileId },
      );
      setProviderSelections((current) => current.map((item) => item.alias === alias ? selection : item));
      await Promise.all([refreshGateway(), refreshDeploymentUsage()]);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      setPending(null);
    }
  };

  if (loading) return <LoadingScreen />;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand" aria-label="ModelDeck">
          <span className="brand-mark" aria-hidden="true">MD</span>
          <div><strong>ModelDeck</strong><small>Operator console</small></div>
        </div>
        <nav aria-label="Primary navigation">
          {NAVIGATION.map((item) => (
            <a
              className={view === item.view ? "nav-link active" : "nav-link"}
              href={item.path}
              key={item.view}
              aria-current={view === item.view ? "page" : undefined}
              onClick={(event) => {
                event.preventDefault();
                navigate(item.view, item.path);
              }}
            >
              {item.label}
            </a>
          ))}
        </nav>
        <div className="sidebar-policy">
          <StatusDot state={eventStreamConnected ? "good" : "warn"} />
          <span>{eventStreamConnected ? "Live events connected" : "Polling worker state"}</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">Framework Desktop · local control plane</p>
            <h1>{NAVIGATION.find((item) => item.view === view)?.label}</h1>
          </div>
          <div className={`gateway-badge ${gateway?.available ? "ready" : "unavailable"}`}>
            <StatusDot state={gateway?.available ? "good" : "bad"} />
            <span>{gateway?.available ? "Gateway available" : "Gateway unavailable"}</span>
          </div>
        </header>

        {error && (
          <div className="alert error" role="alert">
            <strong>Action failed</strong><span>{error}</span>
            <button className="icon-button" aria-label="Dismiss error" onClick={() => setError(null)}>×</button>
          </div>
        )}

        {!health || !hardware || !telemetry || !gateway ? (
          <UnavailableState retry={load} />
        ) : view === "overview" ? (
          <Overview
            health={health}
            gateway={gateway}
            hardware={hardware}
            telemetry={telemetry}
            workers={workers}
            models={models}
            compatibility={compatibility}
          />
        ) : view === "demo-routes" ? (
          <DemoRoutesView
            demoSets={demoSets}
            deployments={deployments}
            adapters={demoAdapters}
            openDay={health.open_day}
            configurationChanged={refreshDemoConfiguration}
          />
        ) : view === "workers" ? (
          <WorkersView
            workers={workers}
            profiles={profiles}
            models={models}
            compatibility={compatibility}
            providerSelections={providerSelections}
            pending={pending}
            operate={operate}
            stopAll={stopAll}
            selectProvider={selectProvider}
          />
        ) : view === "models" ? (
          <ModelsView
            models={models}
            profiles={profiles}
            compatibility={compatibility}
            deploymentUsage={deploymentUsage}
            configurationChanged={refreshConfiguration}
          />
        ) : view === "compatibility" ? (
          <CompatibilityView tests={compatibility} />
        ) : (
          <LogsView workers={workers} />
        )}
      </main>
    </div>
  );
}

function LoadingScreen() {
  return (
    <main className="loading-screen" aria-busy="true">
      <div className="brand-mark">MD</div>
      <h1>Starting operator console</h1>
      <p>Reading local hardware, workers, and compatibility evidence…</p>
      <div className="loading-bar"><span /></div>
    </main>
  );
}

function UnavailableState({ retry }: { retry: () => Promise<void> }) {
  return (
    <section className="empty-state" role="status">
      <span className="empty-icon">!</span>
      <h2>Management data is unavailable</h2>
      <p>ModelDeck could not assemble the local operator view. No cloud service was contacted.</p>
      <button onClick={() => void retry()}>Retry local connection</button>
    </section>
  );
}

function Overview({
  health,
  gateway,
  hardware,
  telemetry,
  workers,
  models,
  compatibility,
}: {
  health: ManagementHealth;
  gateway: GatewayStatus;
  hardware: HardwareProbe;
  telemetry: Telemetry;
  workers: Worker[];
  models: ModelEntry[];
  compatibility: CompatibilityTest[];
}) {
  const ready = workers.filter((worker) => worker.state === "ready" || worker.state === "busy");
  const failed = workers.filter((worker) => worker.state === "failed");
  const work = telemetry.filesystems.find((filesystem) => filesystem.path === "/mnt/work");
  return (
    <div className="view-stack">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">System posture</p>
          <h2>{ready.length ? `${ready.length} local worker${ready.length === 1 ? "" : "s"} ready` : "Local runtimes are standing by"}</h2>
          <p>Model acquisition is separate. Workers load pinned local snapshots only, with no cloud fallback.</p>
        </div>
        <div className="hero-status">
          <StatusDot state={failed.length ? "bad" : ready.length ? "good" : "neutral"} />
          <span>{failed.length ? `${failed.length} worker failure${failed.length === 1 ? "" : "s"}` : "Control plane healthy"}</span>
        </div>
      </section>

      <div className="metric-grid">
        <Metric label="Available memory" value={formatBytes(telemetry.memory.available_bytes)} detail={`${telemetry.memory.percent.toFixed(0)}% used`} />
        <Metric label="Work storage free" value={work?.available ? formatBytes(work.free_bytes ?? 0) : "Unavailable"} detail={work?.available ? `${work.percent?.toFixed(0)}% used` : "/mnt/work missing"} />
        <Metric label="Cached repositories" value={String(models.length)} detail={`${models.filter((model) => model.download_state === "partial").length} partial`} />
        <Metric label="Compatibility records" value={String(compatibility.length)} detail={`${compatibility.filter((test) => test.result === "tested-working").length} tested-working`} />
      </div>

      <div className="two-column">
        <section className="panel">
          <PanelHeading title="Machine" detail="Detected, never assumed" />
          <DefinitionList rows={[
            ["Configured target", `${hardware.configured.gpu} (${hardware.configured.gpu_architecture})`],
            ["Detected Fedora", hardware.detected.fedora_release ?? "Not detected"],
            ["Kernel", hardware.detected.kernel],
            ["ROCm packages", hardware.detected.rocm_packages.length ? hardware.detected.rocm_packages.join(", ") : "Not detected"],
            ["GPU device nodes", Object.entries(hardware.detected.gpu_device_nodes).filter(([, found]) => found).map(([path]) => path).join(", ") || "Not visible"],
          ]} />
        </section>
        <section className="panel">
          <PanelHeading title="Gateway providers" detail={gateway.available ? `${gateway.health?.ready_providers ?? 0} ready` : "Unavailable"} />
          {gateway.providers?.providers.length ? (
            <ul className="status-list">
              {gateway.providers.providers.map((provider) => (
                <li key={provider.id}><StatusDot state={provider.ready ? "good" : "neutral"} /><span><strong>{provider.alias}</strong><small>{provider.id}</small></span><StateBadge state={provider.ready ? "ready" : "stopped"} /></li>
              ))}
            </ul>
          ) : <p className="muted">No ready gateway providers. Requests return a structured local-unavailable response.</p>}
        </section>
      </div>

      <div className="two-column">
        <section className="panel">
          <PanelHeading title="Thermals" detail="Live local sensors" />
          {telemetry.temperatures.length ? <div className="sensor-grid">{telemetry.temperatures.slice(0, 8).map((sensor, index) => <Metric key={`${sensor.source}-${sensor.label}-${index}`} label={sensor.label} value={`${sensor.celsius.toFixed(1)} °C`} detail={sensor.source} compact />)}</div> : <p className="muted">No temperature sensors were exposed to this process.</p>}
        </section>
        <section className="panel policy-panel">
          <PanelHeading title="Runtime policy" detail={health.open_day ? "Open Day mode" : "Normal local mode"} />
          <Policy label="Loopback binding" value="127.0.0.1 only" />
          <Policy label="Model downloads" value={health.downloads_allowed ? "Enabled by configuration" : "Disabled"} warning={health.downloads_allowed} />
          <Policy label="Cloud fallback" value="Never" />
          <Policy label="Worker inputs" value="Allowlisted manifests only" />
        </section>
      </div>

      <div className="two-column">
        <section className="panel">
          <PanelHeading title="Memory and filesystems" detail="Modest-interval telemetry" />
          <DefinitionList rows={[
            ["Memory", `${formatBytes(telemetry.memory.available_bytes)} available of ${formatBytes(telemetry.memory.total_bytes)}`],
            ["Swap", `${formatBytes(telemetry.swap.used_bytes)} used of ${formatBytes(telemetry.swap.total_bytes)} (${telemetry.swap.percent.toFixed(0)}%)`],
            ...telemetry.filesystems.map((filesystem) => [
              filesystem.path,
              filesystem.available
                ? `${formatBytes(filesystem.free_bytes ?? 0)} free of ${formatBytes(filesystem.total_bytes ?? 0)}`
                : "Unavailable",
            ] as [string, string]),
          ]} />
        </section>
        <section className="panel">
          <PanelHeading title="Active model processes" detail={`${telemetry.active_model_processes.length} detected`} />
          {telemetry.active_model_processes.length ? (
            <ul className="process-list">
              {telemetry.active_model_processes.map((process) => (
                <li key={process.pid}><strong>{process.name ?? "Unknown process"}</strong><span>PID {process.pid}</span><code>{process.command}</code></li>
              ))}
            </ul>
          ) : <p className="muted">No active model processes were detected.</p>}
        </section>
      </div>

      <section className="panel">
        <PanelHeading title="Advertised gateway models" detail={`${gateway.models?.data.length ?? 0} aliases`} />
        {gateway.models?.data.length ? (
          <ul className="status-list gateway-model-list">
            {gateway.models.data.map((model) => (
              <li key={model.id}><StatusDot state={model.ready ? "good" : "neutral"} /><span><strong>{model.id}</strong><small>{model.effective_provider ?? "No ready provider"}</small></span><StateBadge state={model.ready ? "ready" : "unavailable"} /></li>
            ))}
          </ul>
        ) : <p className="muted">The gateway is not advertising any model aliases.</p>}
      </section>
    </div>
  );
}

function DemoRoutesView({ demoSets, deployments, adapters, openDay, configurationChanged }: {
  demoSets: DemoSet[];
  deployments: Deployment[];
  adapters: DemoAdapter[];
  openDay: boolean;
  configurationChanged: () => Promise<void>;
}) {
  const [selectedId, setSelectedId] = useState(demoSets[0]?.id ?? "");
  const [draft, setDraft] = useState<DemoSet | null>(null);
  const [newSet, setNewSet] = useState({ id: "", display_name: "" });
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [validation, setValidation] = useState<DemoSetValidation | null>(null);
  const [plan, setPlan] = useState<DemoSetPlan | null>(null);
  const [revisions, setRevisions] = useState<DemoSet[]>([]);
  const [routeStatuses, setRouteStatuses] = useState<Record<string, DemoRouteStatus>>({});
  const [feedback, setFeedback] = useState<{ tone: "good" | "bad"; message: string } | null>(null);
  const selected = demoSets.find((item) => item.id === selectedId) ?? demoSets[0];

  useEffect(() => {
    if (!selectedId && demoSets[0]) setSelectedId(demoSets[0].id);
    if (selectedId && !demoSets.some((item) => item.id === selectedId)) {
      setSelectedId(demoSets[0]?.id ?? "");
    }
  }, [demoSets, selectedId]);

  useEffect(() => {
    if (!selected) {
      setRevisions([]);
      setRouteStatuses({});
      return;
    }
    let cancelled = false;
    void Promise.all([
      getJson<{ revisions: DemoSet[] }>(`/api/demo-sets/${encodeURIComponent(selected.id)}/revisions`),
      Promise.all(selected.routes.map((route) => getJson<DemoRouteStatus>(`/api/demo-sets/${encodeURIComponent(selected.id)}/routes/${encodeURIComponent(route.id)}/status`))),
    ]).then(([history, statuses]) => {
      if (cancelled) return;
      setRevisions(history.revisions);
      setRouteStatuses(Object.fromEntries(statuses.map((status) => [status.route_id, status])));
    }).catch(() => {
      if (!cancelled) setRouteStatuses({});
    });
    return () => { cancelled = true; };
  }, [selected?.active_revision, selected?.id, selected?.revision]);

  const runAction = async (key: string, action: () => Promise<void>) => {
    setPendingAction(key);
    setFeedback(null);
    try {
      await action();
    } catch (reason) {
      setFeedback({ tone: "bad", message: messageFrom(reason) });
    } finally {
      setPendingAction(null);
    }
  };

  const createSet = async () => {
    await runAction("create", async () => {
      const created = await postJson<DemoSet>("/api/demo-sets", {
        id: newSet.id,
        display_name: newSet.display_name,
        description: "",
        demos: [],
        routes: [],
      });
      await configurationChanged();
      setSelectedId(created.id);
      setNewSet({ id: "", display_name: "" });
      setFeedback({ tone: "good", message: `Created draft demo set ${created.display_name}.` });
    });
  };

  const saveDraft = async () => {
    if (!draft) return;
    await runAction("save", async () => {
      const saved = await putJson<DemoSet>(`/api/demo-sets/${encodeURIComponent(draft.id)}`, demoSetPayload(draft));
      await configurationChanged();
      setDraft(null);
      setValidation(null);
      setPlan(null);
      setFeedback({ tone: "good", message: `Saved ${saved.display_name} revision ${saved.revision}.` });
    });
  };

  const validateSet = async () => {
    if (!selected) return;
    await runAction("validate", async () => {
      const result = await postJson<DemoSetValidation>(`/api/demo-sets/${encodeURIComponent(selected.id)}/validate`);
      setValidation(result);
      setPlan(null);
      setFeedback({ tone: result.valid ? "good" : "bad", message: result.valid ? "All route bindings are valid." : "Route validation found blocking issues." });
    });
  };

  const planSet = async () => {
    if (!selected) return;
    await runAction("plan", async () => {
      const result = await postJson<DemoSetPlan & { validation: DemoSetValidation }>(`/api/demo-sets/${encodeURIComponent(selected.id)}/plan`);
      setValidation(result.validation);
      setPlan(result);
      setFeedback({ tone: result.validation.valid ? "good" : "bad", message: result.validation.valid ? "Activation plan is ready for review." : "Fix validation errors before activation." });
    });
  };

  const activateSet = async () => {
    if (!selected || !window.confirm(`Activate ${selected.display_name} revision ${selected.revision} for gateway routing? This does not start workers.`)) return;
    await runAction("activate", async () => {
      const result = await postJson<{ plan: DemoSetPlan }>(`/api/demo-sets/${encodeURIComponent(selected.id)}/activate`);
      await configurationChanged();
      setPlan(result.plan);
      setFeedback({ tone: "good", message: `${selected.display_name} is now the active gateway routing configuration.` });
    });
  };

  const removeSet = async () => {
    if (!selected || !window.confirm(`Delete draft demo set ${selected.display_name}?`)) return;
    await runAction("delete", async () => {
      await deleteJson(`/api/demo-sets/${encodeURIComponent(selected.id)}`);
      setSelectedId("");
      setDraft(null);
      await configurationChanged();
      setFeedback({ tone: "good", message: `Deleted ${selected.display_name}.` });
    });
  };

  const restoreRevision = async (revision: number) => {
    if (!selected || !window.confirm(`Restore revision ${revision} as a new draft revision?`)) return;
    await runAction(`restore:${revision}`, async () => {
      const restored = await postJson<DemoSet>(`/api/demo-sets/${encodeURIComponent(selected.id)}/revisions/${revision}/restore`);
      await configurationChanged();
      setFeedback({ tone: "good", message: `Restored revision ${revision} as revision ${restored.revision}.` });
    });
  };

  const activateRevision = async (revision: number) => {
    if (!selected || !window.confirm(`Activate historical revision ${revision}? This changes routing only.`)) return;
    await runAction(`activate-revision:${revision}`, async () => {
      await postJson(`/api/demo-sets/${encodeURIComponent(selected.id)}/revisions/${revision}/activate`);
      await configurationChanged();
      setFeedback({ tone: "good", message: `Revision ${revision} is now the active gateway routing configuration.` });
    });
  };

  const refreshRouteStatus = async (routeId: string) => {
    if (!selected) return;
    await runAction(`status:${routeId}`, async () => {
      const status = await getJson<DemoRouteStatus>(`/api/demo-sets/${encodeURIComponent(selected.id)}/routes/${encodeURIComponent(routeId)}/status`);
      setRouteStatuses((current) => ({ ...current, [routeId]: status }));
      setFeedback({ tone: status.ready ? "good" : "bad", message: status.ready ? `${status.public_model} is ready through the gateway.` : `${status.public_model} is not ready through the gateway.` });
    });
  };

  const smokeRoute = async (routeId: string) => {
    if (!selected || !window.confirm("Run a small generation request through this active gateway route?")) return;
    await runAction(`smoke:${routeId}`, async () => {
      const result = await postJson<DemoRouteSmokeResult>(`/api/demo-sets/${encodeURIComponent(selected.id)}/routes/${encodeURIComponent(routeId)}/smoke`);
      const status = await getJson<DemoRouteStatus>(`/api/demo-sets/${encodeURIComponent(selected.id)}/routes/${encodeURIComponent(routeId)}/status`);
      setRouteStatuses((current) => ({ ...current, [routeId]: status }));
      setFeedback({ tone: "good", message: `${result.public_model} passed through ${result.provider ?? "its ready provider"} in ${result.duration_seconds.toFixed(2)} seconds.` });
    });
  };

  return <div className="view-stack">
    <section className="notice-panel">
      <strong>Declarative demo compatibility</strong>
      <p>Define the stable model routes each demo expects, bind them to configured deployments, validate the contracts, then activate one versioned routing snapshot. Activation never starts a large model automatically.</p>
    </section>
    {openDay && <div className="configuration-feedback bad" role="status">Open Day mode is locked. Review the active configuration here, but edit and activate revisions before entering booth mode.</div>}
    {feedback && <div className={`configuration-feedback ${feedback.tone}`} role="status">{feedback.message}</div>}
    <div className="demo-set-layout">
      <aside className="panel demo-set-list">
        <PanelHeading title="Demo sets" detail={`${demoSets.length} configured`} />
        {demoSets.map((item) => <button className={item.id === selected?.id ? "demo-set-select active" : "demo-set-select"} key={item.id} onClick={() => { setSelectedId(item.id); setDraft(null); setValidation(null); setPlan(null); }}><span><strong>{item.display_name}</strong><small>{item.id} · revision {item.revision}{item.active_revision !== null && item.active_revision !== item.revision ? ` · active revision ${item.active_revision}` : ""}</small></span>{item.active && <StateBadge state="ready" />}</button>)}
        {!openDay && <form className="new-demo-set" onSubmit={(event) => { event.preventDefault(); void createSet(); }}>
          <label>Identifier<input required pattern="[a-z][a-z0-9-]{1,62}" value={newSet.id} onChange={(event) => setNewSet({ ...newSet, id: event.target.value })} placeholder="open-day-2027" /></label>
          <label>Display name<input required value={newSet.display_name} onChange={(event) => setNewSet({ ...newSet, display_name: event.target.value })} placeholder="Open Day 2027" /></label>
          <button disabled={pendingAction !== null}>{pendingAction === "create" ? "Creating…" : "Create demo set"}</button>
        </form>}
      </aside>
      <section className="panel demo-set-detail">
        {!selected ? <div className="empty-state"><h2>No demo set selected</h2><p>Create a draft to define demo route contracts.</p></div> : draft ? (
          <DemoSetEditor draft={draft} setDraft={setDraft} deployments={deployments} adapters={adapters} pending={pendingAction !== null} save={() => void saveDraft()} cancel={() => setDraft(null)} />
        ) : <>
          <div className="demo-set-heading"><div><p className="eyebrow">{selected.active_revision === selected.revision ? "Active routing snapshot" : selected.active ? `Draft revision; revision ${selected.active_revision} remains active` : "Draft routing configuration"}</p><h2>{selected.display_name}</h2><p>{selected.description || "No description."}</p></div><StateBadge state={selected.active_revision === selected.revision ? "ready" : "discovered"} /></div>
          <div className="button-row demo-set-actions">
            <button className="secondary" disabled={openDay || pendingAction !== null} onClick={() => setDraft(structuredClone(selected))}>Edit</button>
            <button className="secondary" disabled={pendingAction !== null} onClick={() => void validateSet()}>{pendingAction === "validate" ? "Validating…" : "Validate"}</button>
            <button className="secondary" disabled={pendingAction !== null} onClick={() => void planSet()}>{pendingAction === "plan" ? "Planning…" : "Plan activation"}</button>
            <button disabled={openDay || pendingAction !== null} onClick={() => void activateSet()}>{pendingAction === "activate" ? "Activating…" : "Activate routing"}</button>
            <button className="secondary danger" disabled={openDay || selected.active || pendingAction !== null} onClick={() => void removeSet()}>Delete draft</button>
          </div>
          {validation && <ValidationSummary validation={validation} />}
          {plan && <PlanSummary plan={plan} />}
          <details className="revision-history"><summary>Revision history ({revisions.length})</summary><div>{revisions.map((revision) => <article key={revision.revision}><span><strong>Revision {revision.revision}</strong><small>{formatDate(revision.updated_at)}</small></span>{revision.active_revision === revision.revision ? <StateBadge state="ready" /> : <div className="button-row"><button type="button" className="secondary" disabled={openDay || pendingAction !== null || revision.revision === selected.revision} onClick={() => void restoreRevision(revision.revision)}>{pendingAction === `restore:${revision.revision}` ? "Restoring…" : "Restore as draft"}</button><button type="button" className="secondary" disabled={openDay || pendingAction !== null} onClick={() => void activateRevision(revision.revision)}>{pendingAction === `activate-revision:${revision.revision}` ? "Activating…" : "Activate"}</button></div>}</article>)}</div></details>
          <div className="demo-route-list">{selected.routes.map((route) => {
            const demo = selected.demos.find((item) => item.id === route.demo_id);
            const adapter = adapters.find((item) => item.id === route.adapter_id);
            const status = routeStatuses[route.id];
            return <article className="demo-route-card" key={route.id}>
              <div className="demo-route-heading"><div><p className="worker-id">{demo?.display_name ?? route.demo_id}</p><h3>{route.display_name}</h3></div><code>{route.public_model}</code></div>
              <p className="worker-summary">{adapter?.display_name ?? route.adapter_id} · {humanise(route.qualification_policy)} · {humanise(route.fallback_policy)}</p>
              <div className="route-rehearsal"><span><StatusDot state={status?.ready ? "good" : status?.gateway_available ? "warn" : "bad"} /><strong>{status?.ready ? "Gateway ready" : status?.active ? "Gateway unavailable" : "Revision not active"}</strong><small>{status?.effective_provider ?? status?.smoke_unavailable_reason ?? "No effective provider"}</small></span><div className="button-row"><button type="button" className="secondary" disabled={pendingAction !== null} onClick={() => void refreshRouteStatus(route.id)}>{pendingAction === `status:${route.id}` ? "Checking…" : "Check readiness"}</button><button type="button" disabled={pendingAction !== null || !status?.ready || !status.smoke_supported} title={status?.smoke_unavailable_reason ?? undefined} onClick={() => void smokeRoute(route.id)}>{pendingAction === `smoke:${route.id}` ? "Testing…" : "Smoke route"}</button></div></div>
              <div className="route-provider-list">{route.providers.length ? [...route.providers].sort((left, right) => left.priority - right.priority).map((binding) => {
                const deployment = deployments.find((item) => item.id === binding.deployment_id);
                return <div key={binding.deployment_id}><span><strong>{binding.deployment_id}</strong><small>{deployment ? `${deployment.model.model_id} · ${deployment.runtime}` : "Missing deployment"}</small></span><StateBadge state={deployment?.worker?.state ?? "incompatible"} /></div>;
              }) : <p className="muted">No provider deployment; the route is structurally unavailable.</p>}</div>
            </article>;
          })}</div>
        </>}
      </section>
    </div>
  </div>;
}

function DemoSetEditor({ draft, setDraft, deployments, adapters, pending, save, cancel }: {
  draft: DemoSet;
  setDraft: (value: DemoSet) => void;
  deployments: Deployment[];
  adapters: DemoAdapter[];
  pending: boolean;
  save: () => void;
  cancel: () => void;
}) {
  const updateRoute = (index: number, updates: Partial<DemoSet["routes"][number]>) => setDraft({ ...draft, routes: draft.routes.map((route, routeIndex) => routeIndex === index ? { ...route, ...updates } : route) });
  const addDemo = () => {
    const index = nextAvailableSuffix("demo", draft.demos.map((demo) => demo.id));
    setDraft({ ...draft, demos: [...draft.demos, { id: `demo-${index}`, display_name: `Demo ${index}` }] });
  };
  const addRoute = () => {
    if (!draft.demos.length || !adapters.length) return;
    const index = nextAvailableSuffix("route", draft.routes.map((route) => route.id));
    setDraft({ ...draft, routes: [...draft.routes, { id: `route-${index}`, demo_id: draft.demos[0].id, display_name: `Route ${index}`, adapter_id: adapters[0].id, public_model: `route-${index}`, qualification_policy: "registered", fallback_policy: "structured-unavailable", providers: [] }] });
  };
  return <form className="demo-set-editor" onSubmit={(event) => { event.preventDefault(); save(); }}>
    <div className="demo-set-heading"><div><p className="eyebrow">Editing revision {draft.revision}</p><h2>{draft.display_name}</h2></div></div>
    <div className="runtime-fields">
      <label>Display name<input required value={draft.display_name} onChange={(event) => setDraft({ ...draft, display_name: event.target.value })} /></label>
      <label className="wide-field">Description<textarea value={draft.description} onChange={(event) => setDraft({ ...draft, description: event.target.value })} /></label>
    </div>
    <div className="editor-section-heading"><h3>Demos</h3><button type="button" className="secondary" onClick={addDemo}>Add demo</button></div>
    <div className="demo-editor-list">{draft.demos.map((demo, index) => <div key={demo.id}><code>{demo.id}</code><input aria-label={`Display name for ${demo.id}`} value={demo.display_name} onChange={(event) => setDraft({ ...draft, demos: draft.demos.map((item, demoIndex) => demoIndex === index ? { ...item, display_name: event.target.value } : item) })} /><button type="button" className="secondary danger" onClick={() => setDraft({ ...draft, demos: draft.demos.filter((_, demoIndex) => demoIndex !== index), routes: draft.routes.filter((route) => route.demo_id !== demo.id) })}>Remove</button></div>)}</div>
    <div className="editor-section-heading"><h3>Route contracts</h3><button type="button" className="secondary" disabled={!draft.demos.length} onClick={addRoute}>Add route</button></div>
    <div className="route-editor-list">{draft.routes.map((route, routeIndex) => <article className="route-editor" key={route.id}>
      <div className="route-editor-title"><strong>{route.id}</strong><button type="button" className="secondary danger" onClick={() => setDraft({ ...draft, routes: draft.routes.filter((_, index) => index !== routeIndex) })}>Remove route</button></div>
      <div className="runtime-fields">
        <label>Display name<input required value={route.display_name} onChange={(event) => updateRoute(routeIndex, { display_name: event.target.value })} /></label>
        <label>Demo<select value={route.demo_id} onChange={(event) => updateRoute(routeIndex, { demo_id: event.target.value })}>{draft.demos.map((demo) => <option value={demo.id} key={demo.id}>{demo.display_name}</option>)}</select></label>
        <label>Public model alias<input required pattern="[a-z][a-z0-9-]{1,62}" value={route.public_model} onChange={(event) => updateRoute(routeIndex, { public_model: event.target.value })} /></label>
        <label>Protocol adapter<select value={route.adapter_id} onChange={(event) => updateRoute(routeIndex, { adapter_id: event.target.value })}>{adapters.map((adapter) => <option value={adapter.id} key={adapter.id}>{adapter.display_name}</option>)}</select></label>
        <label>Qualification<select value={route.qualification_policy} onChange={(event) => updateRoute(routeIndex, { qualification_policy: event.target.value as DemoSet["routes"][number]["qualification_policy"] })}><option value="registered">Registered deployment</option><option value="tested-working-recorded">Recorded tested-working evidence</option></select></label>
        <label>Fallback policy<select value={route.fallback_policy} onChange={(event) => updateRoute(routeIndex, { fallback_policy: event.target.value as DemoSet["routes"][number]["fallback_policy"] })}><option value="structured-unavailable">Structured unavailable</option><option value="none">No fallback</option><option value="ordered">Ordered providers</option><option value="mock-visible">Visible mock fallback</option></select></label>
      </div>
      <div className="editor-section-heading"><h4>Provider deployments</h4><button type="button" className="secondary" disabled={!deployments.length} onClick={() => { const candidate = deployments.find((deployment) => !route.providers.some((binding) => binding.deployment_id === deployment.id)); if (candidate) updateRoute(routeIndex, { providers: [...route.providers, { deployment_id: candidate.id, priority: route.providers.length * 10 }] }); }}>Add provider</button></div>
      <div className="provider-editor-list">{route.providers.map((binding, providerIndex) => <div key={`${binding.deployment_id}-${providerIndex}`}><select aria-label={`Provider ${providerIndex + 1} for ${route.id}`} value={binding.deployment_id} onChange={(event) => updateRoute(routeIndex, { providers: route.providers.map((item, index) => index === providerIndex ? { ...item, deployment_id: event.target.value } : item) })}>{deployments.map((deployment) => <option value={deployment.id} key={deployment.id}>{deployment.id} · {deployment.model.model_id}</option>)}</select><input aria-label={`Priority for ${binding.deployment_id}`} type="number" min="0" max="10000" value={binding.priority} onChange={(event) => updateRoute(routeIndex, { providers: route.providers.map((item, index) => index === providerIndex ? { ...item, priority: Number(event.target.value) } : item) })} /><button type="button" className="secondary danger" onClick={() => updateRoute(routeIndex, { providers: route.providers.filter((_, index) => index !== providerIndex) })}>Remove</button></div>)}</div>
    </article>)}</div>
    <div className="button-row"><button disabled={pending}>Save new revision</button><button type="button" className="secondary" disabled={pending} onClick={cancel}>Cancel</button></div>
  </form>;
}

function nextAvailableSuffix(prefix: string, identifiers: string[]) {
  const used = new Set(identifiers);
  let suffix = 1;
  while (used.has(`${prefix}-${suffix}`)) suffix += 1;
  return suffix;
}

function ValidationSummary({ validation }: { validation: DemoSetValidation }) {
  return <section className={`validation-summary ${validation.valid ? "good" : "bad"}`}><strong>{validation.valid ? "Valid route configuration" : `${validation.errors.length} validation issue${validation.errors.length === 1 ? "" : "s"}`}</strong>{validation.errors.length > 0 && <ul>{validation.errors.map((error, index) => <li key={`${error.route_id}-${error.deployment_id}-${index}`}>{[error.route_id, error.deployment_id].filter(Boolean).join(" / ")}: {error.message}</li>)}</ul>}{validation.warnings.length > 0 && <ul>{validation.warnings.map((warning, index) => <li key={`${warning.route_id}-${index}`}>{warning.route_id}: {warning.message}</li>)}</ul>}</section>;
}

function PlanSummary({ plan }: { plan: DemoSetPlan }) {
  return <section className="validation-summary"><strong>Activation plan</strong><DefinitionList rows={[["Primary deployments", plan.desired_primary_deployments.join(", ") || "None"], ["Start required", plan.start_required.join(", ") || "None"], ["Stop required", plan.stop_required.join(", ") || "None"], ["Process changes", plan.applies_process_changes ? "Applied automatically" : "Operator controlled"]]} compact />{plan.warnings.length > 0 && <ul>{plan.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>}</section>;
}

function demoSetPayload(demoSet: DemoSet) {
  return {
    id: demoSet.id,
    display_name: demoSet.display_name,
    description: demoSet.description,
    demos: demoSet.demos,
    routes: demoSet.routes,
  };
}

function WorkersView({ workers, profiles, models, compatibility, providerSelections, pending, operate, stopAll, selectProvider }: {
  workers: Worker[];
  profiles: Profile[];
  models: ModelEntry[];
  compatibility: CompatibilityTest[];
  providerSelections: ProviderSelection[];
  pending: string | null;
  operate: (worker: Worker, operation: WorkerOperation) => Promise<void>;
  stopAll: () => Promise<void>;
  selectProvider: (alias: string, profileId: string) => Promise<void>;
}) {
  const [sort, setSort] = useState<WorkerSort>("name-asc");
  const [grouping, setGrouping] = useState<WorkerGrouping>("family");
  const sortedWorkers = useMemo(() => sortWorkers(workers, sort), [workers, sort]);
  const groups = useMemo(() => groupWorkers(sortedWorkers, grouping), [sortedWorkers, grouping]);
  return (
    <div className="view-stack">
      <div className="view-actions"><p>Start only the runtime you intend to use. Model loading may take several minutes.</p><button className="danger secondary" disabled={pending === "stop-all"} onClick={() => void stopAll()}>{pending === "stop-all" ? "Stopping…" : "Stop all workers"}</button></div>
      <div className="worker-toolbar" aria-label="Worker display controls">
        <label>Group workers<select value={grouping} onChange={(event) => setGrouping(event.target.value as WorkerGrouping)}><option value="family">Generation family</option><option value="runtime">Runtime</option><option value="lifecycle">Lifecycle</option><option value="none">No grouping</option></select></label>
        <label>Sort workers<select value={sort} onChange={(event) => setSort(event.target.value as WorkerSort)}><option value="name-asc">Worker name A–Z</option><option value="name-desc">Worker name Z–A</option><option value="model-asc">Model name A–Z</option><option value="runtime-asc">Runtime A–Z</option><option value="state">State, then name</option></select></label>
      </div>
      {providerSelections.map((selection) => <ProviderControl key={selection.alias} selection={selection} workers={workers} pending={pending === `provider-selection:${selection.alias}`} selectProvider={selectProvider} />)}
      {groups.length ? groups.map((group) => (
        <section className="worker-group" key={group.title}>
          <PanelHeading title={group.title} detail={`${group.workers.length} workers`} />
          <p className="section-description">{group.description}</p>
          <div className="worker-grid">
            {group.workers.map((worker) => {
              const profile = profiles.find((candidate) => candidate.id === worker.id);
              const cacheModelId = profile?.artifact_model_id ?? worker.model_id;
              const cacheRevision = profile?.artifact_revision ?? profile?.revision;
              const model = models.find((candidate) => candidate.model_id === cacheModelId && (!cacheRevision || candidate.revision === cacheRevision));
              return <WorkerCard key={worker.id} worker={worker} profile={profile} model={model} tests={compatibility} pending={pending} operate={operate} />;
            })}
          </div>
        </section>
      )) : <section className="empty-state"><h2>No managed workers</h2><p>Configure an allowlisted runtime from the model library first.</p></section>}
    </div>
  );
}

function ProviderControl({ selection, workers, pending, selectProvider }: { selection: ProviderSelection; workers: Worker[]; pending: boolean; selectProvider: (alias: string, profileId: string) => Promise<void> }) {
  const [chosen, setChosen] = useState(selection.selected_provider);
  useEffect(() => setChosen(selection.selected_provider), [selection.selected_provider]);
  const selected = selection.candidates.find((candidate) => candidate.profile_id === selection.selected_provider);
  const workerState = workers.find((worker) => worker.id === selection.selected_provider)?.state ?? selected?.worker_state ?? "stopped";
  return <section className="panel provider-selection-panel">
    <PanelHeading title={selection.display_name} detail={selection.superseded_by_active_demo_set ? "Managed by Demo routes" : `Legacy alias: ${selection.alias}`} />
    <p className="section-description">Applications keep using <code>{selection.alias}</code>. {selection.superseded_by_active_demo_set ? <>The active demo set is authoritative; change this binding in <a href="/demo-routes">Demo routes</a>.</> : "This compatibility selection changes legacy routing only; start, stop and smoke testing remain separate."}</p>
    <div className="provider-selection-controls">
      <label>Physical provider<select disabled={selection.superseded_by_active_demo_set} value={chosen ?? ""} onChange={(event) => setChosen(event.target.value)}>{selection.candidates.map((candidate) => <option key={candidate.profile_id} value={candidate.profile_id}>{candidate.profile_alias} · {candidate.model_id}</option>)}</select></label>
      <button disabled={selection.superseded_by_active_demo_set || pending || !chosen || chosen === selection.selected_provider || !selection.candidates.some((candidate) => candidate.profile_id === chosen)} onClick={() => { if (chosen) void selectProvider(selection.alias, chosen); }}>{pending ? "Selecting…" : "Select provider"}</button>
    </div>
    <DefinitionList rows={[["Stored compatibility selection", selection.selected_provider ?? "None"], ["Routing authority", selection.superseded_by_active_demo_set ? `${selection.active_demo_set_id} revision ${selection.active_demo_set_revision}` : "Legacy provider selection"], ["Worker state", humanise(workerState)], ["Gateway readiness", selection.gateway_ready ? "Ready" : "Not ready"], ["Effective provider", selection.effective_provider ?? "None — no fallback"]]} compact />
  </section>;
}

function WorkerCard({ worker, profile, model, tests, pending, operate }: {
  worker: Worker;
  profile?: Profile;
  model?: ModelEntry;
  tests: CompatibilityTest[];
  pending: string | null;
  operate: (worker: Worker, operation: WorkerOperation) => Promise<void>;
}) {
  const compatibility = compatibilityFor(worker, profile, tests, model);
  const active = ["validating", "starting", "loading", "warming", "ready", "busy", "degraded", "stopping"].includes(worker.state);
  const canStart = ["stopped", "failed", "incompatible"].includes(worker.state);
  const canStop = active && worker.state !== "stopping";
  const canRestart = ["ready", "busy", "degraded", "failed"].includes(worker.state);
  const canSmoke = worker.state === "ready";
  const busy = pending?.startsWith(`${worker.id}:`) ?? false;
  return (
    <article className={`worker-card state-${worker.state}`}>
      <div className="worker-card-heading"><div><p className="worker-id">{worker.id}</p><h3>{shortModelName(worker.model_id)}</h3></div><StateBadge state={worker.state} /></div>
      <p className="worker-summary">{worker.generation_family} · {worker.runtime}</p>
      <div className="compatibility-line"><StatusDot state={compatibility.tone} /><span>{compatibility.label}</span></div>
      {worker.last_error && <p className="inline-error" role="alert">{worker.last_error}</p>}
      <DefinitionList rows={[
        ["Alias", worker.alias], ["Configuration", profile?.source === "built-in" ? "Packaged profile" : profile?.source === "local" ? "Local profile" : "Unknown"], ["Revision", profile?.revision.slice(0, 12) ?? "Unknown"], ["Lifecycle", worker.lifecycle], ["Endpoint", `127.0.0.1:${worker.port}`], ["Dtype", profile?.dtype ?? "Unknown"], ["Cache snapshot", cacheSnapshotLabel(model)],
      ]} compact />
      <details><summary>Capabilities and manifest</summary><div className="tag-list">{capabilityLabels(worker.capabilities).map((label) => <span className="tag" key={label}>{label}</span>)}</div><p className="manifest-note">Local files only · remote code disabled · fixed argument-array launch</p></details>
      <div className="button-row" aria-label={`Actions for ${worker.id}`}>
        <button disabled={!canStart || busy} onClick={() => void operate(worker, "start")}>{pending === `${worker.id}:start` ? "Starting…" : "Start"}</button>
        <button className="secondary" disabled={!canStop || busy} onClick={() => void operate(worker, "stop")}>{pending === `${worker.id}:stop` ? "Stopping…" : "Stop"}</button>
        <button className="secondary" disabled={!canRestart || busy} onClick={() => void operate(worker, "restart")}>Restart</button>
        <button className="secondary" disabled={!canSmoke || busy} onClick={() => void operate(worker, "smoke")}>{pending === `${worker.id}:smoke` ? "Testing…" : "Smoke test"}</button>
      </div>
    </article>
  );
}

function ModelsView({
  models,
  profiles,
  compatibility,
  deploymentUsage,
  configurationChanged,
}: {
  models: ModelEntry[];
  profiles: Profile[];
  compatibility: CompatibilityTest[];
  deploymentUsage: DeploymentUsage[];
  configurationChanged: () => Promise<void>;
}) {
  const [configuring, setConfiguring] = useState<string | null>(null);
  const [pendingProfile, setPendingProfile] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<{ tone: "good" | "bad"; message: string } | null>(null);
  const [sort, setSort] = useState<ModelSort>("name-asc");
  const sortedModels = useMemo(() => {
    const nameOrder = (left: ModelEntry, right: ModelEntry) =>
      left.model_id.localeCompare(right.model_id, "en-AU", { sensitivity: "base" }) ||
      String(left.revision ?? "").localeCompare(String(right.revision ?? ""), "en-AU");
    return [...models].sort((left, right) => {
      if (sort === "name-desc") return -nameOrder(left, right);
      if (sort === "size-desc") return right.physical_size_bytes - left.physical_size_bytes || nameOrder(left, right);
      if (sort === "size-asc") return left.physical_size_bytes - right.physical_size_bytes || nameOrder(left, right);
      return nameOrder(left, right);
    });
  }, [models, sort]);

  const configure = async (payload: LocalProfileRequest) => {
    setPendingProfile(`create:${payload.alias}`);
    setFeedback(null);
    try {
      await postJson<Profile>("/api/profiles", payload);
      await configurationChanged();
      setConfiguring(null);
      setFeedback({ tone: "good", message: `Runtime ${payload.alias} is configured and ready to start from Workers.` });
    } catch (reason) {
      setFeedback({ tone: "bad", message: messageFrom(reason) });
    } finally {
      setPendingProfile(null);
    }
  };

  const remove = async (profile: Profile) => {
    if (!window.confirm(`Remove runtime configuration ${profile.alias}? Cached model files will be kept.`)) return;
    setPendingProfile(`delete:${profile.id}`);
    setFeedback(null);
    try {
      await deleteJson(`/api/profiles/${encodeURIComponent(profile.id)}`);
      await configurationChanged();
      setFeedback({ tone: "good", message: `Runtime ${profile.alias} was removed. Its cached model files were kept.` });
    } catch (reason) {
      setFeedback({ tone: "bad", message: messageFrom(reason) });
    } finally {
      setPendingProfile(null);
    }
  };

  const setModelPolicy = async (model: ModelEntry, allowed: boolean) => {
    if (!model.revision) return;
    if (!allowed && !window.confirm(`Disallow ${model.model_id} in ModelDeck? Cached files and runtime configurations will be kept.`)) return;
    setPendingProfile(`policy:${model.model_id}`);
    setFeedback(null);
    try {
      await postJson("/api/catalogue/policy", {
        model_id: model.model_id,
        revision: model.revision,
        allowed,
      });
      await configurationChanged();
      setConfiguring(null);
      setFeedback({ tone: "good", message: allowed ? `${model.model_id} is allowed in ModelDeck again.` : `${model.model_id} is disallowed in ModelDeck. Its cached files and configurations were kept.` });
    } catch (reason) {
      setFeedback({ tone: "bad", message: messageFrom(reason) });
    } finally {
      setPendingProfile(null);
    }
  };

  return (
    <div className="view-stack">
      <section className="notice-panel"><strong>Cache-backed runtime configuration</strong><p>HuggingFacePull still owns acquisition and cleanup. ModelDeck can configure recognised local snapshots, but never downloads or deletes model files.</p></section>
      {feedback && <div className={`configuration-feedback ${feedback.tone}`} role="status">{feedback.message}</div>}
      <section className="panel table-panel">
        <PanelHeading title="Model library" detail={`${models.length} cached repositories`} />
        {models.length ? <>
          <div className="model-library-toolbar">
            <label htmlFor="model-library-sort">Sort models</label>
            <select id="model-library-sort" value={sort} onChange={(event) => setSort(event.target.value as ModelSort)}>
              <option value="name-asc">Name (A–Z)</option>
              <option value="name-desc">Name (Z–A)</option>
              <option value="size-desc">Cache size (largest first)</option>
              <option value="size-asc">Cache size (smallest first)</option>
            </select>
          </div>
          <div className="model-list">{sortedModels.map((model) => {
          const matchingProfiles = profiles.filter((profile) => (profile.artifact_model_id ?? profile.model_id) === model.model_id && (profile.artifact_revision ?? profile.revision) === model.revision);
          const latest = compatibility.find((test) => matchingProfiles.some((profile) => test.evidence.model_id === profile.model_id && test.evidence.model_revision === profile.revision && profile.preferred_runtime === test.evidence.runtime));
          const state = !model.modeldeck_allowed ? "disallowed" : model.download_state === "partial" ? "partial" : latest?.result ?? (matchingProfiles.length ? "runtime-configured" : model.configuration_support ? "recognised" : "unsupported");
          const canConfigure = model.modeldeck_allowed && model.download_state !== "partial" && model.configuration_support !== null && Boolean(model.revision);
          const key = `${model.model_id}-${model.revision}`;
          return <article className="model-row" key={key}>
            <div className="model-main"><div><h3>{model.model_id}</h3><p>{model.generation_family_hint ?? "Unknown generation family"} · {formatBytes(model.physical_size_bytes)}</p></div><StateBadge state={state} /></div>
            <p className="model-stage">{modelStageDescription(state)}</p>
            <DefinitionList rows={[["Revision", model.revision ?? "No resolved snapshot"], ...(model.base_model_id ? [["Base model", `${model.base_model_id} @ ${model.base_model_revision}`] as [string, string]] : []), ["ModelDeck use", model.modeldeck_allowed ? "Allowed" : "Disallowed"], ["Runtime configurations", matchingProfiles.length ? matchingProfiles.map((profile) => profile.alias).join(", ") : "None"], ["Compatibility", latest ? String(latest.result) : "Not tested for a configured runtime"], ["Cache", model.download_state === "partial" ? "Incomplete snapshot" : "Complete local snapshot"]]} compact />
            {matchingProfiles.some((profile) => profile.source === "local") && <div className="configured-runtime-list">{matchingProfiles.filter((profile) => profile.source === "local").map((profile) => {
              const usage = deploymentUsage.find((candidate) => candidate.deployment_id === profile.id);
              return <div className="configured-runtime" key={profile.id}><div className="configured-runtime-heading"><span><strong>{profile.alias}</strong><small>{profile.dtype} · {humanise(profile.lifecycle)} · port {profile.port}</small></span><button className="secondary danger" title={usage?.blocking_dependencies.map((dependency) => dependency.remediation).join("; ") || undefined} disabled={pendingProfile !== null || !usage?.removable} onClick={() => void remove(profile)}>{pendingProfile === `delete:${profile.id}` ? "Removing…" : "Remove configuration"}</button></div><DeploymentUsageSummary usage={usage} /></div>;
            })}</div>}
            {configuring === key && model.revision ? <RuntimeConfigurationForm model={model} pending={pendingProfile?.startsWith("create:") ?? false} cancel={() => setConfiguring(null)} submit={configure} /> : <div className="model-actions"><button disabled={!canConfigure || pendingProfile !== null} onClick={() => { setConfiguring(key); setFeedback(null); }}>{matchingProfiles.length ? "Add runtime configuration" : "Configure runtime"}</button>{model.revision && <button className="secondary" disabled={pendingProfile !== null} onClick={() => void setModelPolicy(model, !model.modeldeck_allowed)}>{pendingProfile === `policy:${model.model_id}` ? "Updating…" : model.modeldeck_allowed ? "Disallow in ModelDeck" : "Allow in ModelDeck"}</button>}{!canConfigure && <span>{model.modeldeck_allowed ? model.configuration_support_reason : "This model is kept in the HF cache but excluded from ModelDeck workers and gateway routes."}</span>}</div>}
          </article>;
          })}</div>
        </> : <p className="muted">No cached models were discovered. Use HuggingFacePull to acquire models.</p>}
      </section>
    </div>
  );
}

function DeploymentUsageSummary({ usage }: { usage?: DeploymentUsage }) {
  if (!usage) return <p className="deployment-usage muted">Loading deployment usage…</p>;
  const references = [
    ...usage.route_bindings.map((route) => ({ key: `route:${route.demo_set_id}:${route.revision}:${route.route_id}`, label: `${route.demo_set_display_name} / ${route.route_display_name}`, detail: `${humanise(route.state)} route · ${route.public_model}` })),
    ...usage.legacy_aliases.map((alias) => ({ key: `alias:${alias.alias}`, label: alias.display_name, detail: alias.effective ? "Legacy routing authority" : "Stored legacy selection · superseded" })),
  ];
  return <div className="deployment-usage"><strong>Used by</strong>{references.length ? <ul>{references.map((reference) => <li key={reference.key}><span>{reference.label}</span><small>{reference.detail}</small></li>)}</ul> : <p>No demo routes or compatibility aliases reference this deployment.</p>}{usage.blocking_dependencies.length > 0 && <p className="dependency-guidance">Reassign {usage.blocking_dependencies.length} blocking dependenc{usage.blocking_dependencies.length === 1 ? "y" : "ies"} before removal. <a href={usage.blocking_dependencies.some((dependency) => dependency.kind === "demo-route") ? "/demo-routes" : "/workers"}>Open configuration</a></p>}</div>;
}

function RuntimeConfigurationForm({ model, pending, cancel, submit }: { model: ModelEntry; pending: boolean; cancel: () => void; submit: (payload: LocalProfileRequest) => Promise<void> }) {
  const support = model.configuration_support;
  const diffusion = support === "diffusiongemma-transformers" || support === "diffusiongemma-modeldeck-q4";
  const speech = support === "moshiko-speech";
  const [profileName, setProfileName] = useState(() => suggestedProfileName(model.model_id));
  const [alias, setAlias] = useState(() => suggestedAlias(model.model_id));
  const [dtype, setDtype] = useState<LocalProfileRequest["dtype"]>(support === "autoregressive-transformers" ? "float16" : "bfloat16");
  const [lifecycle, setLifecycle] = useState<LocalProfileRequest["lifecycle"]>(diffusion || support === "gpt-oss-llama-vulkan" || speech ? "exclusive" : "on-demand");
  const [contextLength, setContextLength] = useState(support === "scenechat-gemma4" || support === "gpt-oss-llama-vulkan" ? 8192 : 2048);
  const [maximumNewTokens, setMaximumNewTokens] = useState(support === "autoregressive-transformers" ? 128 : support === "scenechat-gemma4" ? 512 : 256);
  const [maximumDenoisingSteps, setMaximumDenoisingSteps] = useState(24);
  const artifact = (model.artifacts ?? [])[0];
  return <form className="runtime-form" onSubmit={(event) => { event.preventDefault(); if (!model.revision) return; void submit({ model_id: model.model_id, revision: model.revision, profile_name: profileName, alias, dtype, lifecycle, context_length: contextLength, maximum_new_tokens: maximumNewTokens, maximum_denoising_steps: maximumDenoisingSteps, ...(artifact ? { artifact_id: artifact.artifact_id } : {}) }); }}>
    <div className="runtime-form-heading"><div><strong>Configure {runtimeLabel(support)}</strong><small>Model, revision, cache path, worker implementation and port are fixed from the recognised snapshot.</small></div></div>
    <div className="runtime-fields">
      <label>Configuration name<input required pattern="[a-z][a-z0-9-]{1,62}" maxLength={63} value={profileName} onChange={(event) => setProfileName(event.target.value)} /></label>
      <label>Gateway alias<input required pattern="[a-z][a-z0-9-]{1,62}" maxLength={63} value={alias} onChange={(event) => setAlias(event.target.value)} /></label>
      <label>Data type<select disabled={support === "diffusiongemma-modeldeck-q4"} value={dtype} onChange={(event) => setDtype(event.target.value as LocalProfileRequest["dtype"])}><option value="float16">float16</option><option value="bfloat16">bfloat16</option></select></label>
      <label>Lifecycle<select disabled={diffusion} value={lifecycle} onChange={(event) => setLifecycle(event.target.value as LocalProfileRequest["lifecycle"])}><option value="on-demand">On demand</option><option value="resident">Resident</option><option value="exclusive">Exclusive</option></select></label>
      {!diffusion && !speech && <label>Context length<input type="number" min={256} max={32768} step={256} value={contextLength} onChange={(event) => setContextLength(event.currentTarget.valueAsNumber)} /></label>}
      {!speech && <label>Maximum new tokens<input type="number" min={1} max={512} value={maximumNewTokens} onChange={(event) => setMaximumNewTokens(event.currentTarget.valueAsNumber)} /></label>}
      {diffusion && <label>Maximum denoising steps<input type="number" min={1} max={48} value={maximumDenoisingSteps} onChange={(event) => setMaximumDenoisingSteps(event.currentTarget.valueAsNumber)} /></label>}
    </div>
    {artifact && <p className="manifest-note">Artefact: {artifact.format.toUpperCase()} · {artifact.filenames.length} pinned shards</p>}
    <p className="manifest-note">Local files only · remote code disabled · fixed allowlisted worker · no download · hardware verification required before demo selection</p>
    <div className="runtime-form-actions"><button type="submit" disabled={pending}>{pending ? "Configuring…" : "Save runtime configuration"}</button><button type="button" className="secondary" disabled={pending} onClick={cancel}>Cancel</button></div>
  </form>;
}

function CompatibilityView({ tests }: { tests: CompatibilityTest[] }) {
  return (
    <div className="view-stack"><section className="panel table-panel"><PanelHeading title="Compatibility evidence" detail={`${tests.length} immutable records`} />
      {tests.length ? <div className="evidence-list">{tests.map((test) => <details className="evidence-row" key={test.id}><summary><span><StateBadge state={test.result} /><strong>{String(test.evidence.model_id ?? "Unknown model")}</strong><small>{formatDate(test.tested_at)} · {String(test.evidence.runtime ?? "unknown runtime")}</small></span><code>{test.fingerprint.slice(0, 12)}</code></summary><dl className="evidence-grid">{Object.entries(test.evidence).filter(([key]) => !["result", "failure_class", "tested_at"].includes(key)).map(([key, value]) => <div key={key}><dt>{humanise(key)}</dt><dd>{formatEvidence(value)}</dd></div>)}</dl>{test.failure_class && <p className="inline-error">Failure class: {test.failure_class}</p>}</details>)}</div> : <p className="muted">No compatibility evidence has been recorded. A successful or failed smoke test will append a complete fingerprint.</p>}
    </section></div>
  );
}

function LogsView({ workers }: { workers: Worker[] }) {
  const [workerId, setWorkerId] = useState(workers[0]?.id ?? "");
  const [logs, setLogs] = useState<WorkerLog[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!workerId) return;
    setLogs([]);
    void getJson<{ logs: WorkerLog[] }>(`/api/workers/${encodeURIComponent(workerId)}/logs`).then((response) => setLogs(response.logs));
    const source = new EventSource(`/api/workers/${encodeURIComponent(workerId)}/logs/stream`);
    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    source.addEventListener("log", (raw) => {
      try {
        const entry = JSON.parse((raw as MessageEvent).data) as WorkerLog;
        setLogs((current) => [...current.slice(-499), entry]);
      } catch {
        setConnected(false);
      }
    });
    return () => source.close();
  }, [workerId]);

  return (
    <div className="view-stack"><section className="panel log-panel"><div className="log-toolbar"><div><label htmlFor="worker-log-select">Worker log</label><select id="worker-log-select" value={workerId} onChange={(event) => setWorkerId(event.target.value)}>{workers.map((worker) => <option key={worker.id} value={worker.id}>{worker.id}</option>)}</select></div><span className="stream-status"><StatusDot state={connected ? "good" : "warn"} />{connected ? "Live stream" : "Reconnecting"}</span></div>
      <div className="log-view" role="log" aria-live="polite" aria-label={`Logs for ${workerId}`}>{logs.length ? logs.map((entry, index) => <div className={`log-entry ${entry.level}`} key={`${entry.timestamp}-${index}`}><time>{formatTime(entry.timestamp)}</time><span>{entry.level}</span><code>{entry.message}</code></div>) : <p>No log records for this worker session.</p>}</div><p className="privacy-note">Logs are bounded and credential-, prompt-, and generated-content-shaped fields are redacted by the management service.</p>
    </section></div>
  );
}

function Metric({ label, value, detail, compact = false }: { label: string; value: string; detail: string; compact?: boolean }) {
  return <article className={compact ? "metric compact" : "metric"}><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>;
}

function PanelHeading({ title, detail }: { title: string; detail: string }) {
  return <div className="panel-heading"><h2>{title}</h2><span>{detail}</span></div>;
}

function DefinitionList({ rows, compact = false }: { rows: Array<[string, string]>; compact?: boolean }) {
  return <dl className={compact ? "definition-list compact" : "definition-list"}>{rows.map(([term, value]) => <div key={term}><dt>{term}</dt><dd title={value}>{value}</dd></div>)}</dl>;
}

function Policy({ label, value, warning = false }: { label: string; value: string; warning?: boolean }) {
  return <div className="policy-row"><span>{label}</span><strong className={warning ? "warning-text" : ""}>{value}</strong></div>;
}

function StatusDot({ state }: { state: "good" | "warn" | "bad" | "neutral" }) {
  return <span className={`status-dot ${state}`} aria-hidden="true" />;
}

function StateBadge({ state }: { state: string }) {
  return <span className={`state-badge state-${state}`}>{humanise(state)}</span>;
}

function compatibilityFor(worker: Worker, profile: Profile | undefined, tests: CompatibilityTest[], model: ModelEntry | undefined): { label: string; tone: "good" | "warn" | "bad" | "neutral" } {
  const latest = tests.find((test) => test.evidence.model_id === worker.model_id && (!profile || test.evidence.model_revision === profile.revision) && test.evidence.runtime === worker.runtime);
  if (latest?.result === "tested-working") return { label: "Tested working for recorded fingerprint", tone: "good" };
  if (latest) return { label: `${humanise(latest.result)} evidence recorded`, tone: "bad" };
  if (model?.download_state === "partial") return { label: "Partial cache; not runnable", tone: "bad" };
  if (model) return { label: "Installed, compatibility untested", tone: "warn" };
  if (worker.runtime === "mock") return { label: "Mock lifecycle fallback", tone: "neutral" };
  return { label: "Pinned model snapshot not discovered", tone: "bad" };
}

function sortWorkers(workers: Worker[], sort: WorkerSort): Worker[] {
  const byName = (left: Worker, right: Worker) => left.id.localeCompare(right.id, "en-AU", { sensitivity: "base" });
  const stateOrder = ["busy", "ready", "warming", "loading", "starting", "validating", "degraded", "stopping", "failed", "incompatible", "orphaned", "stopped", "discovered"];
  return [...workers].sort((left, right) => {
    if (sort === "name-desc") return -byName(left, right);
    if (sort === "model-asc") return left.model_id.localeCompare(right.model_id, "en-AU", { sensitivity: "base" }) || byName(left, right);
    if (sort === "runtime-asc") return left.runtime.localeCompare(right.runtime, "en-AU", { sensitivity: "base" }) || byName(left, right);
    if (sort === "state") return stateOrder.indexOf(left.state) - stateOrder.indexOf(right.state) || byName(left, right);
    return byName(left, right);
  });
}

function groupWorkers(workers: Worker[], grouping: WorkerGrouping): Array<{ title: string; description: string; workers: Worker[] }> {
  if (grouping === "none") return workers.length ? [{ title: "All workers", description: "Every worker registered by the management API.", workers }] : [];
  const grouped = new Map<string, Worker[]>();
  for (const worker of workers) {
    const key = grouping === "family" ? worker.generation_family : grouping === "runtime" ? worker.runtime : worker.lifecycle;
    grouped.set(key, [...(grouped.get(key) ?? []), worker]);
  }
  return [...grouped.entries()]
    .sort(([left], [right]) => left.localeCompare(right, "en-AU", { sensitivity: "base" }))
    .map(([key, members]) => ({
      title: `${humanise(key)} ${grouping === "family" ? "workers" : grouping}`,
      description: `${humanise(grouping)}: ${humanise(key)}. Cards are supplied by the management worker registry.`,
      workers: members,
    }));
}

function confirmOperation(worker: Worker, operation: WorkerOperation, workers: Worker[]): boolean {
  if (operation === "start" && worker.lifecycle === "exclusive") {
    const active = workers.find((candidate) => candidate.id !== worker.id && candidate.lifecycle === "exclusive" && !["stopped", "failed", "incompatible"].includes(candidate.state));
    if (active) return window.confirm(`Starting ${worker.id} will stop exclusive worker ${active.id}. Continue?`);
  }
  if (operation === "stop") return window.confirm(`Stop ${worker.id} and release its runtime memory?`);
  if (operation === "restart") return window.confirm(`Restart ${worker.id}? In-flight work will end.`);
  return true;
}

function capabilityLabels(capabilities: Worker["capabilities"]): string[] {
  const labels: string[] = [];
  if (capabilities.chat) labels.push("Chat");
  if (capabilities.completions) labels.push("Completions");
  if (capabilities.streaming) labels.push("Streaming");
  if (capabilities.cancellation) labels.push("Cancellation");
  if (capabilities.top_k_trace) labels.push("Top-k trace");
  if (capabilities.iterative_refinement) labels.push("Iterative refinement");
  if (capabilities.intermediate_frames) labels.push("Intermediate frames");
  if (capabilities.seeded_generation) labels.push("Seeded generation");
  if (capabilities.image_input) labels.push("Image input");
  if (capabilities.structured_output) labels.push("Structured output");
  if (capabilities.audio_input) labels.push("Audio input");
  if (capabilities.audio_output) labels.push("Audio output");
  if (capabilities.full_duplex) labels.push("Full duplex");
  return labels;
}

function shortModelName(modelId: string): string { return modelId.split("/").at(-1) ?? modelId; }
function suggestedAlias(modelId: string): string {
  if (modelId === "ggml-org/gpt-oss-120b-GGUF") return "repartee-strong";
  if (modelId === "kyutai/moshiko-pytorch-bf16") return "repartee-speech";
  const candidate = shortModelName(modelId).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48);
  return /^[a-z][a-z0-9-]+$/.test(candidate) ? candidate : "local-model";
}
function suggestedProfileName(modelId: string): string {
  if (modelId === "google/gemma-4-12B-it") return "scenechat-gemma-4-12b";
  if (modelId === "ggml-org/gpt-oss-120b-GGUF") return "repartee-gpt-oss-120b";
  if (modelId === "kyutai/moshiko-pytorch-bf16") return "repartee-moshiko";
  return suggestedAlias(modelId);
}
function runtimeLabel(support: ModelEntry["configuration_support"]): string {
  if (support === "scenechat-gemma4") return "SceneChat Gemma 4 runtime";
  if (support === "diffusiongemma-transformers") return "DiffusionGemma runtime";
  if (support === "diffusiongemma-modeldeck-q4") return "ModelDeck DiffusionGemma Q4 runtime";
  if (support === "gpt-oss-llama-vulkan") return "Repartee GPT-OSS Vulkan runtime";
  if (support === "moshiko-speech") return "Repartee Moshiko speech runtime";
  return "autoregressive ROCm runtime";
}
function modelStageDescription(state: string): string {
  if (state === "disallowed") return "Cached files are retained, but this revision is excluded from ModelDeck workers and gateway routes.";
  if (state === "partial") return "Download is incomplete. Finish or repair it in HuggingFacePull before configuring a runtime.";
  if (state === "recognised") return "Snapshot recognised. Configure a constrained local runtime to make it available to ModelDeck.";
  if (state === "runtime-configured") return "Runtime configured. Start the worker and run a smoke test to record compatibility evidence.";
  if (state === "unsupported") return "Snapshot recognised, but it does not match an allowlisted ModelDeck worker implementation.";
  if (state === "tested-working") return "Tested working for the recorded hardware, runtime and model fingerprint.";
  return `${humanise(state)} compatibility evidence is recorded for this snapshot.`;
}
function cacheSnapshotLabel(model: ModelEntry | undefined): string { return model ? model.download_state === "partial" ? "Partial" : "Installed" : "Not discovered"; }
function messageFrom(reason: unknown): string { return reason instanceof Error ? reason.message : "Unexpected local error."; }
function humanise(value: string): string { return value.replaceAll("_", " ").replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function formatBytes(bytes: number): string { if (!Number.isFinite(bytes) || bytes <= 0) return "0 B"; const units = ["B", "KiB", "MiB", "GiB", "TiB"]; const power = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1); return `${(bytes / 1024 ** power).toFixed(power > 2 ? 1 : 0)} ${units[power]}`; }
function formatDate(value: string): string { return new Intl.DateTimeFormat("en-AU", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)); }
function formatTime(value: string): string { return new Intl.DateTimeFormat("en-AU", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date(value)); }
function formatEvidence(value: unknown): string { return typeof value === "object" && value !== null ? JSON.stringify(value) : String(value ?? "Not recorded"); }

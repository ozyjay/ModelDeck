import { useCallback, useEffect, useMemo, useState } from "react";

import { getJson, postJson } from "./api";
import type {
  CompatibilityTest,
  GatewayStatus,
  HardwareProbe,
  ManagementHealth,
  ModelEntry,
  Profile,
  Telemetry,
  Worker,
  WorkerEvent,
  WorkerLog,
} from "./types";

type View = "overview" | "workers" | "models" | "compatibility" | "logs";
type WorkerOperation = "start" | "stop" | "restart" | "smoke";

const NAVIGATION: Array<{ view: View; label: string; path: string }> = [
  { view: "overview", label: "Overview", path: "/" },
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

  const refreshGateway = useCallback(async () => {
    setGateway(await getJson<GatewayStatus>("/api/gateway/status"));
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nextHealth, nextGateway, nextHardware, nextTelemetry, nextWorkers, nextProfiles, catalogue, tests] =
        await Promise.all([
          getJson<ManagementHealth>("/api/health"),
          getJson<GatewayStatus>("/api/gateway/status"),
          getJson<HardwareProbe>("/api/hardware"),
          getJson<Telemetry>("/api/telemetry"),
          getJson<Worker[]>("/api/workers"),
          getJson<Profile[]>("/api/profiles"),
          getJson<{ models: ModelEntry[] }>("/api/catalogue"),
          getJson<{ tests: CompatibilityTest[] }>("/api/compatibility"),
        ]);
      setHealth(nextHealth);
      setGateway(nextGateway);
      setHardware(nextHardware);
      setTelemetry(nextTelemetry);
      setWorkers(nextWorkers);
      setProfiles(nextProfiles);
      setModels(catalogue.models);
      setCompatibility(tests.tests);
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
      await refreshWorkers();
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
      await refreshWorkers();
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
        ) : view === "workers" ? (
          <WorkersView
            workers={workers}
            profiles={profiles}
            models={models}
            compatibility={compatibility}
            pending={pending}
            operate={operate}
            stopAll={stopAll}
          />
        ) : view === "models" ? (
          <ModelsView models={models} profiles={profiles} compatibility={compatibility} />
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

function WorkersView({ workers, profiles, models, compatibility, pending, operate, stopAll }: {
  workers: Worker[];
  profiles: Profile[];
  models: ModelEntry[];
  compatibility: CompatibilityTest[];
  pending: string | null;
  operate: (worker: Worker, operation: WorkerOperation) => Promise<void>;
  stopAll: () => Promise<void>;
}) {
  const groups = [
    { title: "Qwen autoregressive", description: "Pinned Transformers workers for chat, completions, and token traces.", workers: workers.filter((worker) => worker.model_id.startsWith("Qwen/")) },
    { title: "DiffusionGemma text diffusion", description: "Separate Q4 default and BF16 evaluation runtimes using the native refinement protocol.", workers: workers.filter((worker) => worker.generation_family === "text-diffusion" && worker.runtime !== "mock") },
    { title: "Mock and recovery", description: "Lifecycle fallbacks for development and demonstrated recovery only.", workers: workers.filter((worker) => worker.runtime === "mock") },
  ];
  return (
    <div className="view-stack">
      <div className="view-actions"><p>Start only the runtime you intend to use. Model loading may take several minutes.</p><button className="danger secondary" disabled={pending === "stop-all"} onClick={() => void stopAll()}>{pending === "stop-all" ? "Stopping…" : "Stop all workers"}</button></div>
      {groups.map((group) => (
        <section className="worker-group" key={group.title}>
          <PanelHeading title={group.title} detail={`${group.workers.length} workers`} />
          <p className="section-description">{group.description}</p>
          <div className="worker-grid">
            {group.workers.map((worker) => (
              <WorkerCard key={worker.id} worker={worker} profile={profiles.find((profile) => profile.id === worker.id)} model={models.find((model) => model.model_id === worker.model_id)} tests={compatibility} pending={pending} operate={operate} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
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
        ["Alias", worker.alias], ["Revision", profile?.revision.slice(0, 12) ?? "Unknown"], ["Lifecycle", worker.lifecycle], ["Endpoint", `127.0.0.1:${worker.port}`], ["Dtype", profile?.dtype ?? "Unknown"], ["Cache", model?.download_state ?? "Not discovered"],
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

function ModelsView({ models, profiles, compatibility }: { models: ModelEntry[]; profiles: Profile[]; compatibility: CompatibilityTest[] }) {
  return (
    <div className="view-stack">
      <section className="notice-panel"><strong>Read-only model discovery</strong><p>HuggingFacePull owns acquisition and cleanup. ModelDeck never starts a download from this console.</p></section>
      <section className="panel table-panel">
        <PanelHeading title="Cached Hugging Face repositories" detail={`${models.length} discovered`} />
        {models.length ? <div className="model-list">{models.map((model) => {
          const matchingProfiles = profiles.filter((profile) => profile.model_id === model.model_id);
          const latest = compatibility.find((test) => test.evidence.model_id === model.model_id && test.evidence.model_revision === model.revision);
          return <article className="model-row" key={`${model.model_id}-${model.revision}`}><div className="model-main"><div><h3>{model.model_id}</h3><p>{model.generation_family_hint ?? "Unknown generation family"} · {formatBytes(model.physical_size_bytes)}</p></div><StateBadge state={latest?.result ?? model.download_state} /></div><DefinitionList rows={[["Revision", model.revision ?? "No resolved snapshot"], ["Worker manifests", matchingProfiles.length ? matchingProfiles.map((profile) => profile.id).join(", ") : "None"], ["Cache location", model.cache_location], ["Runtime claim", latest ? String(latest.result) : model.runnable_reason]]} compact /></article>;
        })}</div> : <p className="muted">No cached models were discovered. Use HuggingFacePull to acquire models.</p>}
      </section>
    </div>
  );
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
  return labels;
}

function shortModelName(modelId: string): string { return modelId.split("/").at(-1) ?? modelId; }
function messageFrom(reason: unknown): string { return reason instanceof Error ? reason.message : "Unexpected local error."; }
function humanise(value: string): string { return value.replaceAll("_", " ").replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function formatBytes(bytes: number): string { if (!Number.isFinite(bytes) || bytes <= 0) return "0 B"; const units = ["B", "KiB", "MiB", "GiB", "TiB"]; const power = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1); return `${(bytes / 1024 ** power).toFixed(power > 2 ? 1 : 0)} ${units[power]}`; }
function formatDate(value: string): string { return new Intl.DateTimeFormat("en-AU", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)); }
function formatTime(value: string): string { return new Intl.DateTimeFormat("en-AU", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date(value)); }
function formatEvidence(value: unknown): string { return typeof value === "object" && value !== null ? JSON.stringify(value) : String(value ?? "Not recorded"); }

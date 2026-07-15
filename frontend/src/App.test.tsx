import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { GatewayStatus } from "./types";

const capabilities = {
  chat: true,
  completions: true,
  streaming: true,
  cancellation: true,
  logits: true,
  top_k_trace: true,
  hidden_states: "optional",
  iterative_refinement: false,
  intermediate_frames: false,
  seeded_generation: true,
  image_input: false,
  structured_output: false,
};

const worker = {
  id: "qwen-small-rocm",
  state: "stopped",
  model_id: "Qwen/Qwen2.5-0.5B-Instruct",
  generation_family: "autoregressive",
  runtime: "transformers-rocm",
  lifecycle: "resident",
  alias: "token-explainer",
  endpoint: "http://127.0.0.1:8620",
  port: 8620,
  pid: null,
  started_at: null,
  last_error: null,
  capabilities,
};

const profile = {
  id: worker.id,
  model_id: worker.model_id,
  revision: "7ae557604adf67be50417f59c2c2f167def9a775",
  alias: worker.alias,
  generation_family: worker.generation_family,
  preferred_runtime: worker.runtime,
  lifecycle: worker.lifecycle,
  port: worker.port,
  local_files_only: true,
  trust_remote_code: false,
  dtype: "float16",
  capabilities,
  settings: { cache_root: "/mnt/work/models/huggingface/hub" },
};

const completeModel = {
  model_id: worker.model_id,
  revision: profile.revision,
  cache_location: "/mnt/work/models/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
  physical_size_bytes: 999_604_710,
  download_state: "installed-untested",
  generation_family_hint: "autoregressive",
  runnable: false,
  runnable_reason: "Compatibility has not been tested for the current stack.",
};

const partialModel = {
  ...completeModel,
  model_id: "ozyjay/diffusiongemma-q4",
  revision: null,
  physical_size_bytes: 40,
  download_state: "partial",
  generation_family_hint: null,
};

const hardware = {
  configured: { profile_id: "framework-desktop-rocm72", os: "Fedora 44", gpu: "AMD Radeon 8060S Graphics", gpu_architecture: "gfx1151", rocm_family: "7.2.x", work_mount: "/mnt/work" },
  detected: {
    fedora_release: "Fedora release 44",
    kernel: "6.15-test",
    python: "3.12.13",
    rocm_packages: ["rocm-core-7.2"],
    gpu_device_nodes: { "/dev/kfd": true, "/dev/dri": true },
    memory: { total_bytes: 128 * 2 ** 30, available_bytes: 96 * 2 ** 30, percent: 25 },
    swap: { total_bytes: 8 * 2 ** 30, used_bytes: 0, percent: 0 },
    filesystems: [{ path: "/mnt/work", available: true, total_bytes: 900 * 2 ** 30, used_bytes: 300 * 2 ** 30, free_bytes: 600 * 2 ** 30, percent: 33 }],
    temperatures: [{ source: "amdgpu", label: "GPU edge", celsius: 48.5 }],
    fans: [],
    active_model_processes: [],
  },
  diagnostic_note: "ROCm uses the cuda device API.",
};

const telemetry = {
  memory: hardware.detected.memory,
  swap: hardware.detected.swap,
  filesystems: hardware.detected.filesystems,
  temperatures: hardware.detected.temperatures,
  fans: [],
  active_model_processes: [],
};

const defaultGateway = {
  available: true,
  health: { status: "ok", ready_providers: 0 },
  models: { data: [{ id: "fast-chat", ready: false, effective_provider: null }] },
  providers: { providers: [{ id: worker.id, alias: worker.alias, ready: false }] },
  error: null,
};

let gateway: GatewayStatus = defaultGateway;
let postFailure = false;
let currentWorker = worker;
let catalogueModels = [completeModel, partialModel];
let managementFailure = false;

class MockEventSource {
  static instances: MockEventSource[] = [];
  readonly url: string;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  private listeners = new Map<string, (event: MessageEvent) => void>();

  constructor(url: string | URL) {
    this.url = String(url);
    MockEventSource.instances.push(this);
  }

  addEventListener(name: string, listener: EventListenerOrEventListenerObject) {
    this.listeners.set(name, listener as (event: MessageEvent) => void);
  }

  emit(name: string, payload: unknown) {
    this.listeners.get(name)?.(new MessageEvent(name, { data: JSON.stringify(payload) }));
  }

  close() {}
}

function json(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), { status, headers: { "Content-Type": "application/json" } });
}

function mockFetch() {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    if (managementFailure && path === "/api/hardware") return json({ detail: "Probe unavailable" }, 503);
    if (init?.method === "POST") {
      if (postFailure) return json({ detail: "Pinned runtime is unavailable" }, 409);
      if (path.endsWith("/start")) currentWorker = { ...currentWorker, state: "ready" };
      if (path.endsWith("/stop")) currentWorker = { ...currentWorker, state: "stopped" };
      return json(currentWorker);
    }
    if (path === "/api/health") return json({ status: "ok", service: "modeldeck-management", open_day: false, downloads_allowed: false, gateway_url: "http://127.0.0.1:8600" });
    if (path === "/api/gateway/status") return json(gateway);
    if (path === "/api/hardware") return json(hardware);
    if (path === "/api/telemetry") return json(telemetry);
    if (path === "/api/workers") return json([currentWorker]);
    if (path === "/api/profiles") return json([profile]);
    if (path === "/api/catalogue") return json({ models: catalogueModels, downloads_started: false });
    if (path === "/api/compatibility") return json({ tests: [] });
    if (path.endsWith("/logs")) return json({ logs: [{ timestamp: "2026-07-14T10:00:00Z", source: "stderr", level: "warning", message: "prompt=[redacted]" }] });
    return json({ detail: `Unexpected request: ${path}` }, 404);
  });
}

describe("ModelDeck operator console", () => {
  beforeEach(() => {
    gateway = defaultGateway;
    currentWorker = worker;
    postFailure = false;
    catalogueModels = [completeModel, partialModel];
    managementFailure = false;
    MockEventSource.instances = [];
    window.history.replaceState({}, "", "/");
    vi.stubGlobal("EventSource", MockEventSource);
    vi.stubGlobal("fetch", mockFetch());
    vi.spyOn(window, "confirm").mockReturnValue(true);
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("shows loading, accessible navigation, local policy, and cached status", async () => {
    render(<App />);
    expect(screen.getByText("Starting operator console")).toBeInTheDocument();
    expect(await screen.findByRole("navigation", { name: "Primary navigation" })).toBeInTheDocument();
    expect(screen.getByText("Local runtimes are standing by")).toBeInTheDocument();
    expect(screen.getByText("Disabled")).toBeInTheDocument();
    expect(screen.getByText("Never")).toBeInTheDocument();
  });

  it("shows a structured gateway unavailable state without breaking management", async () => {
    gateway = { available: false, health: null, models: null, providers: null, error: "The local ModelDeck gateway is unavailable." };
    render(<App />);
    expect(await screen.findByText("Gateway unavailable")).toBeInTheDocument();
    expect(screen.getByText("No ready gateway providers. Requests return a structured local-unavailable response.")).toBeInTheDocument();
  });

  it("shows a recoverable management-unavailable state", async () => {
    managementFailure = true;
    render(<App />);
    expect(await screen.findByText("Management data is unavailable")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry local connection" })).toBeInTheDocument();
  });

  it("updates worker lifecycle state from the SSE stream", async () => {
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));
    const card = screen.getByText("Qwen2.5-0.5B-Instruct").closest("article")!;
    expect(within(card).getByText("Stopped")).toBeInTheDocument();
    MockEventSource.instances.find((source) => source.url === "/api/events")?.emit("worker", { worker_id: worker.id, state: "ready", message: "Ready", timestamp: "2026-07-14T10:00:00Z" });
    await waitFor(() => expect(within(card).getByText("Ready")).toBeInTheDocument());
  });

  it("reports SSE disconnection and falls back to worker polling", async () => {
    render(<App />);
    await screen.findByRole("navigation", { name: "Primary navigation" });
    const events = MockEventSource.instances.find((source) => source.url === "/api/events")!;
    events.onopen?.();
    expect(await screen.findByText("Live events connected")).toBeInTheDocument();
    events.onerror?.();
    expect(await screen.findByText("Polling worker state")).toBeInTheDocument();
  });

  it("disables invalid actions and reports structured API failures", async () => {
    postFailure = true;
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));
    const actions = screen.getByLabelText(`Actions for ${worker.id}`);
    expect(within(actions).getByRole("button", { name: "Stop" })).toBeDisabled();
    fireEvent.click(within(actions).getByRole("button", { name: "Start" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Pinned runtime is unavailable");
  });

  it("requires confirmation before a disruptive stop action", async () => {
    currentWorker = { ...worker, state: "ready" };
    vi.mocked(window.confirm).mockReturnValue(false);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));
    const actions = screen.getByLabelText(`Actions for ${worker.id}`);
    fireEvent.click(within(actions).getByRole("button", { name: "Stop" }));
    expect(window.confirm).toHaveBeenCalledWith(`Stop ${worker.id} and release its runtime memory?`);
    expect(vi.mocked(fetch).mock.calls.some(([, init]) => init?.method === "POST")).toBe(false);
  });

  it("classifies complete and partial cache entries without download controls", async () => {
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Model library" }));
    expect(screen.getByText("Installed Untested")).toBeInTheDocument();
    expect(screen.getByText("Partial")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /download/i })).not.toBeInTheDocument();
  });

  it("renders an explicit empty model-library state", async () => {
    catalogueModels = [];
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Model library" }));
    expect(screen.getByText("No cached models were discovered. Use HuggingFacePull to acquire models.")).toBeInTheDocument();
  });

  it("opens a selected worker log stream and displays server-redacted content", async () => {
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Logs" }));
    expect(await screen.findByRole("log", { name: `Logs for ${worker.id}` })).toHaveTextContent("prompt=[redacted]");
    expect(MockEventSource.instances.some((source) => source.url.endsWith("/logs/stream"))).toBe(true);
  });
});

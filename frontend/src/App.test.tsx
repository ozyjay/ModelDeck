import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { CompatibilityTest, GatewayStatus, ModelEntry, Profile, ProviderSelection, Worker } from "./types";

const capabilities: Worker["capabilities"] = {
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
  audio_input: false,
  audio_output: false,
  full_duplex: false,
};

const worker: Worker = {
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

const profile: Profile = {
  id: worker.id,
  model_id: worker.model_id,
  revision: "7ae557604adf67be50417f59c2c2f167def9a775",
  artifact_model_id: null,
  artifact_revision: null,
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
  source: "built-in",
  modeldeck_allowed: true,
};

const completeModel: ModelEntry = {
  model_id: worker.model_id,
  revision: profile.revision,
  cache_location: "/mnt/work/models/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
  physical_size_bytes: 999_604_710,
  download_state: "installed-untested",
  generation_family_hint: "autoregressive",
  configuration_support: "autoregressive-transformers" as const,
  configuration_support_reason: "Supported by the local Transformers ROCm worker.",
  modeldeck_allowed: true,
  snapshot_location: "/mnt/work/models/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
  base_model_id: null,
  base_model_revision: null,
  runnable: false,
  runnable_reason: "Compatibility has not been tested for the current stack.",
};

const partialModel: ModelEntry = {
  ...completeModel,
  model_id: "ozyjay/diffusiongemma-q4",
  revision: null,
  physical_size_bytes: 40,
  download_state: "partial",
  generation_family_hint: null,
  configuration_support: null,
  configuration_support_reason: "Finish the local snapshot before configuring a runtime.",
  snapshot_location: null,
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
  models: { data: [{ id: "fast-chat", ready: false, selected_provider: worker.id, effective_provider: null }] },
  providers: { providers: [{ id: worker.id, alias: worker.alias, ready: false }] },
  error: null,
};

let gateway: GatewayStatus = defaultGateway;
let postFailure = false;
let currentWorker = worker;
let catalogueModels: ModelEntry[] = [completeModel, partialModel];
let compatibilityTests: CompatibilityTest[] = [];
let managementFailure = false;
let localProfiles: Profile[] = [];
let additionalWorkers: Worker[] = [];
let scenechatSelection: ProviderSelection = {
  alias: "scenechat-vision",
  display_name: "SceneChat provider",
  default_provider: "scenechat-gemma4-e2b-rocm",
  explicit_selection: false,
  selected_provider: "scenechat-gemma4-e2b-rocm",
  effective_provider: null,
  gateway_ready: false,
  candidates: [
    {
      profile_id: "scenechat-gemma4-e2b-rocm",
      profile_alias: "scenechat-vision",
      model_id: "google/gemma-4-E2B-it",
      selected: true,
      worker_state: "stopped",
      gateway_ready: false,
    },
  ],
};

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
    if (init?.method === "DELETE") {
      const id = decodeURIComponent(path.split("/").at(-1) ?? "");
      localProfiles = localProfiles.filter((candidate) => candidate.id !== id);
      return json({ ok: true, profile_id: id, cache_removed: false });
    }
    if (init?.method === "POST") {
      if (postFailure) return json({ detail: "Pinned runtime is unavailable" }, 409);
      if (path === "/api/profiles") {
        const payload = JSON.parse(String(init.body));
        const created = {
          ...profile,
          ...payload,
          id: `local-${payload.alias}`,
          alias: payload.alias,
          preferred_runtime: "transformers-rocm",
          runtime: "transformers-rocm",
          port: 8630,
          settings: {
            context_length: payload.context_length,
            maximum_new_tokens: payload.maximum_new_tokens,
          },
          source: "local" as const,
        };
        localProfiles.push(created);
        return json(created, 201);
      }
      if (path === "/api/catalogue/policy") {
        const payload = JSON.parse(String(init.body));
        catalogueModels = catalogueModels.map((model) =>
          model.model_id === payload.model_id && model.revision === payload.revision
            ? { ...model, modeldeck_allowed: payload.allowed }
            : model,
        );
        return json({ ok: true, ...payload, cache_removed: false });
      }
      if (path === "/api/gateway/provider-selections/scenechat-vision") {
        const payload = JSON.parse(String(init.body));
        scenechatSelection = {
          ...scenechatSelection,
          explicit_selection: true,
          selected_provider: payload.profile_id,
          effective_provider: null,
          gateway_ready: false,
          candidates: scenechatSelection.candidates.map((candidate) => ({
            ...candidate,
            selected: candidate.profile_id === payload.profile_id,
            gateway_ready: false,
          })),
        };
        return json(scenechatSelection);
      }
      if (path.endsWith("/start")) currentWorker = { ...currentWorker, state: "ready" };
      if (path.endsWith("/stop")) currentWorker = { ...currentWorker, state: "stopped" };
      if (path.endsWith("/smoke")) {
        compatibilityTests = [{
          id: 1,
          fingerprint: "a".repeat(64),
          result: "tested-working",
          failure_class: null,
          evidence: {
            model_id: worker.model_id,
            model_revision: profile.revision,
            runtime: worker.runtime,
          },
          tested_at: "2026-07-17T10:00:00Z",
        }];
      }
      return json(currentWorker);
    }
    if (path === "/api/health") return json({ status: "ok", service: "modeldeck-management", open_day: false, downloads_allowed: false, gateway_url: "http://127.0.0.1:8600" });
    if (path === "/api/gateway/status") return json(gateway);
    if (path === "/api/gateway/provider-selections") return json({ selections: [scenechatSelection] });
    if (path === "/api/hardware") return json(hardware);
    if (path === "/api/telemetry") return json(telemetry);
    if (path === "/api/workers") return json([currentWorker, ...additionalWorkers]);
    if (path === "/api/profiles") return json([profile, ...localProfiles]);
    if (path === "/api/catalogue") return json({ models: catalogueModels, downloads_started: false });
    if (path === "/api/compatibility") return json({ tests: compatibilityTests });
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
    compatibilityTests = [];
    managementFailure = false;
    localProfiles = [];
    additionalWorkers = [];
    scenechatSelection = {
      alias: "scenechat-vision",
      display_name: "SceneChat provider",
      default_provider: "scenechat-gemma4-e2b-rocm",
      explicit_selection: false,
      selected_provider: "scenechat-gemma4-e2b-rocm",
      effective_provider: null,
      gateway_ready: false,
      candidates: [
        {
          profile_id: "scenechat-gemma4-e2b-rocm",
          profile_alias: "scenechat-vision",
          model_id: "google/gemma-4-E2B-it",
          selected: true,
          worker_state: "stopped",
          gateway_ready: false,
        },
      ],
    };
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

  it("selects a compatible physical provider for the stable SceneChat alias", async () => {
    scenechatSelection.candidates.push({
      profile_id: "local-gemma-4-26b",
      profile_alias: "gemma-4-26b",
      model_id: "google/gemma-4-26B-A4B-it",
      selected: false,
      worker_state: "ready",
      gateway_ready: false,
    });
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));

    expect(screen.getByText("Reserved alias: scenechat-vision")).toBeInTheDocument();
    expect(screen.getByText("None — no fallback")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Physical provider"), {
      target: { value: "local-gemma-4-26b" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Select provider" }));

    await waitFor(() => {
      expect(screen.getByText("local-gemma-4-26b")).toBeInTheDocument();
    });
    const selectionRequest = vi.mocked(fetch).mock.calls.find(
      ([path, init]) =>
        String(path) === "/api/gateway/provider-selections/scenechat-vision" &&
        init?.method === "POST",
    );
    expect(selectionRequest?.[1]?.body).toBe(
      JSON.stringify({ profile_id: "local-gemma-4-26b" }),
    );
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

  it("shows every API worker and allows data-driven grouping and sorting", async () => {
    const gptOssWorker: Worker = {
      ...worker,
      id: "local-repartee-gpt-oss-120b",
      model_id: "ggml-org/gpt-oss-120b-GGUF",
      runtime: "llama-vulkan",
      lifecycle: "exclusive",
      alias: "repartee-strong",
      port: 8630,
    };
    additionalWorkers = [gptOssWorker];
    localProfiles = [{
      ...profile,
      id: gptOssWorker.id,
      model_id: gptOssWorker.model_id,
      alias: gptOssWorker.alias,
      preferred_runtime: gptOssWorker.runtime,
      lifecycle: gptOssWorker.lifecycle,
      port: gptOssWorker.port,
      source: "local",
    }];

    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));

    expect(screen.getByRole("heading", { name: "gpt-oss-120b-GGUF" })).toBeInTheDocument();
    expect(screen.getByLabelText(`Actions for ${gptOssWorker.id}`)).toBeInTheDocument();
    expect(screen.getByText("Local profile")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Group workers"), { target: { value: "runtime" } });
    expect(screen.getByRole("heading", { name: "Llama Vulkan runtime" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Transformers Rocm runtime" })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Group workers"), { target: { value: "none" } });
    fireEvent.change(screen.getByLabelText("Sort workers"), { target: { value: "name-desc" } });
    const workerNames = [...document.querySelectorAll(".worker-card h3")].map((heading) => heading.textContent);
    expect(workerNames).toEqual(["Qwen2.5-0.5B-Instruct", "gpt-oss-120b-GGUF"]);
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

  it("updates compatibility after smoke without presenting cache acquisition as test state", async () => {
    currentWorker = { ...worker, state: "ready" };
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));
    const card = screen.getByText("Qwen2.5-0.5B-Instruct").closest("article")!;
    expect(within(card).getByText("Installed, compatibility untested")).toBeInTheDocument();
    expect(within(card).getByText("Cache snapshot").nextElementSibling).toHaveTextContent("Installed");

    fireEvent.click(within(card).getByRole("button", { name: "Smoke test" }));

    expect(await within(card).findByText("Tested working for recorded fingerprint")).toBeInTheDocument();
    expect(within(card).getByText("Cache snapshot").nextElementSibling).toHaveTextContent("Installed");
    expect(within(card).queryByText("Installed Untested")).not.toBeInTheDocument();
  });

  it("classifies complete and partial cache entries without download controls", async () => {
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Model library" }));
    expect(screen.getByText("Runtime Configured")).toBeInTheDocument();
    expect(screen.getByText("Partial")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /download/i })).not.toBeInTheDocument();
  });

  it("sorts the model library by name or cache size", async () => {
    catalogueModels = [
      { ...completeModel, model_id: "Zeta/Medium", physical_size_bytes: 20 },
      { ...completeModel, model_id: "Alpha/Small", physical_size_bytes: 10 },
      { ...completeModel, model_id: "Middle/Large", physical_size_bytes: 30 },
    ];
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Model library" }));

    const modelNames = () => screen.getAllByRole("heading", { level: 3 }).map((heading) => heading.textContent);
    expect(modelNames()).toEqual(["Alpha/Small", "Middle/Large", "Zeta/Medium"]);

    fireEvent.change(screen.getByLabelText("Sort models"), { target: { value: "size-desc" } });
    expect(modelNames()).toEqual(["Middle/Large", "Zeta/Medium", "Alpha/Small"]);
  });

  it("configures and removes a constrained cache-backed runtime", async () => {
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Model library" }));
    fireEvent.click(screen.getByRole("button", { name: "Add runtime configuration" }));
    expect(screen.getByText("Model, revision, cache path, worker implementation and port are fixed from the recognised snapshot.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Gateway alias"), { target: { value: "my-local-qwen" } });
    fireEvent.click(screen.getByRole("button", { name: "Save runtime configuration" }));

    expect(await screen.findByText("Runtime my-local-qwen is configured and ready to start from Workers.")).toBeInTheDocument();
    expect(screen.getByText("my-local-qwen")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Remove configuration" }));
    expect(await screen.findByText("Runtime my-local-qwen was removed. Its cached model files were kept.")).toBeInTheDocument();
    expect(window.confirm).toHaveBeenCalledWith("Remove runtime configuration my-local-qwen? Cached model files will be kept.");
  });

  it("offers dedicated forms for supported vision-language and diffusion models", async () => {
    catalogueModels = [
      {
        ...completeModel,
        model_id: "google/gemma-4-E2B-it",
        generation_family_hint: "vision-language",
        configuration_support: "scenechat-gemma4",
        configuration_support_reason: "Supported by the dedicated SceneChat Gemma 4 worker.",
      },
      {
        ...completeModel,
        model_id: "google/diffusiongemma-26B-A4B-it",
        generation_family_hint: "text-diffusion",
        configuration_support: "diffusiongemma-transformers",
        configuration_support_reason: "Supported by the dedicated DiffusionGemma Transformers worker.",
      },
      {
        ...completeModel,
        model_id: "ozyjay/diffusiongemma-modeldeck-q4",
        revision: "release-revision",
        generation_family_hint: "text-diffusion",
        configuration_support: "diffusiongemma-modeldeck-q4",
        configuration_support_reason: "Supported by the dedicated ModelDeck DiffusionGemma Q4 runtime.",
        base_model_id: "google/diffusiongemma-26B-A4B-it",
        base_model_revision: "52de6b914ee1749a7d4933202505ddf5b414ec43",
      },
    ];
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Model library" }));

    const scenechatCard = screen.getByRole("heading", { name: "google/gemma-4-E2B-it" }).closest("article")!;
    const diffusionCard = screen.getByRole("heading", { name: "google/diffusiongemma-26B-A4B-it" }).closest("article")!;
    const q4Card = screen.getByRole("heading", { name: "ozyjay/diffusiongemma-modeldeck-q4" }).closest("article")!;
    fireEvent.click(within(scenechatCard).getByRole("button", { name: "Configure runtime" }));
    expect(within(scenechatCard).getByText("Configure SceneChat Gemma 4 runtime")).toBeInTheDocument();
    fireEvent.click(within(scenechatCard).getByRole("button", { name: "Cancel" }));
    fireEvent.click(within(diffusionCard).getByRole("button", { name: "Configure runtime" }));
    expect(within(diffusionCard).getByText("Configure DiffusionGemma runtime")).toBeInTheDocument();
    expect(within(diffusionCard).getByLabelText("Lifecycle")).toBeDisabled();
    expect(within(diffusionCard).getByLabelText("Maximum denoising steps")).toBeInTheDocument();
    fireEvent.click(within(diffusionCard).getByRole("button", { name: "Cancel" }));
    fireEvent.click(within(q4Card).getByRole("button", { name: "Configure runtime" }));
    expect(within(q4Card).getByText("Configure ModelDeck DiffusionGemma Q4 runtime")).toBeInTheDocument();
    expect(within(q4Card).getByLabelText("Lifecycle")).toBeDisabled();
    expect(within(q4Card).getByLabelText("Data type")).toBeDisabled();
  });

  it("disallows and re-allows a cached model without deleting it", async () => {
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Model library" }));
    fireEvent.click(screen.getByRole("button", { name: "Disallow in ModelDeck" }));

    expect(await screen.findByText(`${completeModel.model_id} is disallowed in ModelDeck. Its cached files and configurations were kept.`)).toBeInTheDocument();
    expect(screen.getAllByText("Disallowed")).toHaveLength(2);
    expect(screen.getByText("This model is kept in the HF cache but excluded from ModelDeck workers and gateway routes.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Allow in ModelDeck" }));
    expect(await screen.findByText(`${completeModel.model_id} is allowed in ModelDeck again.`)).toBeInTheDocument();
    expect(window.confirm).toHaveBeenCalledWith(`Disallow ${completeModel.model_id} in ModelDeck? Cached files and runtime configurations will be kept.`);
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

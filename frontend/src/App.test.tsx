import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { CompatibilityTest, DemoAdapter, DemoSet, Deployment, DeploymentUsage, GatewayStatus, ModelEntry, Profile, ProviderSelection, Worker } from "./types";

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
  runtime_template_id: "autoregressive-transformers",
  runtime_template_version: "0.1.0",
  lifecycle: worker.lifecycle,
  port: worker.port,
  local_files_only: true,
  trust_remote_code: false,
  dtype: "float16",
  capabilities,
  settings: { cache_root: "/mnt/work/models/huggingface/hub" },
  source: "seed",
  modeldeck_allowed: true,
};

const deployment: Deployment = {
  id: profile.id,
  display_name: profile.id,
  source: "seed",
  model: {
    model_id: profile.model_id,
    revision: profile.revision,
    artifact_model_id: null,
    artifact_revision: null,
  },
  runtime: profile.preferred_runtime,
  generation_family: profile.generation_family,
  lifecycle: profile.lifecycle,
  capabilities,
  allowed: true,
  registered: true,
  worker,
};

const demoAdapters: DemoAdapter[] = [{
  id: "openai-chat-v1",
  display_name: "OpenAI-compatible chat",
  generation_family: "autoregressive",
  required_capabilities: ["chat"],
  surfaces: ["POST /v1/chat/completions"],
}];

const defaultDemoSet: DemoSet = {
  id: "open-day-demos",
  display_name: "Open Day demos",
  description: "Editable demo routes.",
  demos: [{ id: "chat-demo", display_name: "Chat demo" }],
  routes: [{
    id: "fast-chat",
    demo_id: "chat-demo",
    display_name: "Fast chat",
    adapter_id: "openai-chat-v1",
    public_model: "fast-chat",
    qualification_policy: "registered",
    fallback_policy: "structured-unavailable",
    providers: [{ deployment_id: worker.id, priority: 10 }],
  }],
  revision: 1,
  updated_at: "2026-07-19T10:00:00Z",
  active: false,
  active_revision: null,
};

const completeModel: ModelEntry = {
  model_id: worker.model_id,
  revision: profile.revision,
  cache_location: "/mnt/work/models/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
  physical_size_bytes: 999_604_710,
  download_state: "installed-untested",
  generation_family_hint: "autoregressive",
  capability_hints: ["text-generation", "chat"],
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
  capability_hints: [],
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
let demoSets: DemoSet[] = [defaultDemoSet];
let scenechatSelection: ProviderSelection = {
  alias: "scenechat-vision",
  display_name: "SceneChat provider",
  default_provider: "scenechat-gemma4-e2b-rocm",
  explicit_selection: false,
  selected_provider: "scenechat-gemma4-e2b-rocm",
  effective_provider: null,
  gateway_ready: false,
  routing_authority: "legacy-selection",
  superseded_by_active_demo_set: false,
  active_demo_set_id: null,
  active_demo_set_revision: null,
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

function deploymentUsages(): DeploymentUsage[] {
  return [deployment.id, ...localProfiles.map((profile) => profile.id)].map((deploymentId) => {
    const routeBindings = demoSets.flatMap((demoSet) => demoSet.routes.flatMap((route) => route.providers.filter((provider) => provider.deployment_id === deploymentId).map((provider) => ({ demo_set_id: demoSet.id, demo_set_display_name: demoSet.display_name, revision: demoSet.revision, route_id: route.id, route_display_name: route.display_name, public_model: route.public_model, state: demoSet.active_revision === demoSet.revision ? "active" as const : "draft" as const, priority: provider.priority }))));
    const legacyAliases = scenechatSelection.selected_provider === deploymentId ? [{ alias: scenechatSelection.alias, display_name: scenechatSelection.display_name, selected_provider: deploymentId, explicit_selection: scenechatSelection.explicit_selection, effective: !scenechatSelection.superseded_by_active_demo_set }] : [];
    const blockingDependencies: DeploymentUsage["blocking_dependencies"] = [
      ...routeBindings.map((route) => ({ kind: "demo-route" as const, id: `${route.demo_set_id}@${route.revision}:${route.route_id}`, label: `${route.demo_set_display_name} / ${route.route_display_name}`, authority: route.state, remediation: "Reassign or remove this provider in Demo routes" })),
      ...legacyAliases.filter((alias) => alias.effective).map((alias) => ({ kind: "legacy-alias" as const, id: alias.alias, label: alias.display_name, authority: "legacy-selection", remediation: "Select a different provider in Workers" })),
    ];
    return { deployment_id: deploymentId, source: deploymentId.startsWith("local-") ? "local" as const : "seed" as const, worker_state: "stopped" as const, route_bindings: routeBindings, legacy_aliases: legacyAliases, blocking_dependencies: blockingDependencies, removable: blockingDependencies.length === 0 };
  });
}

function mockFetch() {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    if (managementFailure && path === "/api/hardware") return json({ detail: "Probe unavailable" }, 503);
    if (init?.method === "DELETE") {
      if (path.includes("/revisions/")) {
        const parts = path.split("/");
        const id = decodeURIComponent(parts[3] ?? "");
        const discardedRevision = Number(parts.at(-1));
        const current = demoSets.find((candidate) => candidate.id === id)!;
        const restoredRevision = current.active_revision ?? discardedRevision - 1;
        const restored = { ...current, revision: restoredRevision, active: current.active_revision === restoredRevision };
        demoSets = demoSets.map((candidate) => candidate.id === id ? restored : candidate);
        return json({ ok: true, discarded_revision: discardedRevision, current: restored });
      }
      if (path.startsWith("/api/demo-sets/")) {
        const id = decodeURIComponent(path.split("/").at(-1) ?? "");
        demoSets = demoSets.filter((candidate) => candidate.id !== id);
        return json({ ok: true, demo_set_id: id });
      }
      const id = decodeURIComponent(path.split("/").at(-1) ?? "");
      localProfiles = localProfiles.filter((candidate) => candidate.id !== id);
      return json({ ok: true, profile_id: id, cache_removed: false });
    }
    if (init?.method === "PUT" && path.startsWith("/api/demo-sets/")) {
      const payload = JSON.parse(String(init.body));
      const current = demoSets.find((candidate) => candidate.id === payload.id);
      const updated = { ...current, ...payload, revision: (current?.revision ?? 0) + 1 };
      demoSets = demoSets.map((candidate) => candidate.id === updated.id ? updated : candidate);
      return json(updated);
    }
    if (init?.method === "PUT" && path.endsWith("/display-name")) {
      const payload = JSON.parse(String(init.body));
      deployment.display_name = payload.display_name;
      return json({ ok: true, deployment_id: deployment.id, display_name: payload.display_name });
    }
    if (init?.method === "POST") {
      if (postFailure) return json({ detail: "Pinned runtime is unavailable" }, 409);
      if (path === "/api/demo-sets") {
        const payload = JSON.parse(String(init.body));
        const created = { ...payload, revision: 1, updated_at: "2026-07-19T10:00:00Z", active: false, active_revision: null };
        demoSets.push(created);
        return json(created, 201);
      }
      if (path.endsWith("/validate") && path.startsWith("/api/demo-sets/")) {
        return json({ valid: true, errors: [], warnings: [] });
      }
      if (path.endsWith("/plan") && path.startsWith("/api/demo-sets/")) {
        return json({ validation: { valid: true, errors: [], warnings: [] }, desired_primary_deployments: [worker.id], start_required: [worker.id], stop_required: [], warnings: [], applies_process_changes: false });
      }
      if (path.includes("/routes/") && path.endsWith("/smoke")) {
        return json({ ok: true, route_id: "fast-chat", public_model: "demo-chat", adapter_id: "openai-chat-v1", provider: worker.id, evidence: "choices", duration_seconds: 0.25 });
      }
      if (path.endsWith("/activate") && path.startsWith("/api/demo-sets/")) {
        const id = path.split("/").at(-2);
        demoSets = demoSets.map((candidate) => ({ ...candidate, active: candidate.id === id, active_revision: candidate.id === id ? candidate.revision : null }));
        const active = demoSets.find((candidate) => candidate.id === id);
        scenechatSelection = {
          ...scenechatSelection,
          routing_authority: "active-demo-set",
          superseded_by_active_demo_set: true,
          active_demo_set_id: active?.id ?? null,
          active_demo_set_revision: active?.revision ?? null,
        };
        return json({ plan: { desired_primary_deployments: [worker.id], start_required: [worker.id], stop_required: [], warnings: [], applies_process_changes: false } });
      }
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
    if (path === "/api/deployments") return json([deployment]);
    if (path === "/api/deployments/usage") return json({ deployments: deploymentUsages() });
    if (path === "/api/demo-sets") return json({ demo_sets: demoSets });
    if (path.startsWith("/api/demo-sets/") && path.endsWith("/revisions")) {
      const id = decodeURIComponent(path.split("/").at(-2) ?? "");
      const current = demoSets.find((candidate) => candidate.id === id);
      const revisions = current ? [current, ...(current.active_revision !== null && current.active_revision !== current.revision ? [{ ...current, revision: current.active_revision, active: true }] : [])] : [];
      return json({ revisions });
    }
    if (path.includes("/routes/") && path.endsWith("/status")) {
      const parts = path.split("/");
      const id = decodeURIComponent(parts[3] ?? "");
      const routeId = decodeURIComponent(parts[5] ?? "");
      const demoSet = demoSets.find((candidate) => candidate.id === id);
      const route = demoSet?.routes.find((candidate) => candidate.id === routeId);
      const active = demoSet?.active_revision === demoSet?.revision;
      return json({ demo_set_id: id, revision: demoSet?.revision ?? 0, route_id: routeId, public_model: route?.public_model ?? "", adapter_id: route?.adapter_id ?? "", active, gateway_available: true, advertised: active, ready: active, selected_provider: active ? worker.id : null, effective_provider: active ? worker.id : null, providers: [{ deployment_id: worker.id, priority: 10, worker_state: active ? "ready" : "stopped" }], smoke_supported: true, smoke_unavailable_reason: null });
    }
    if (path === "/api/demo-adapters") return json({ adapters: demoAdapters });
    if (path === "/api/runtime-templates") return json({ templates: [
      { id: "autoregressive-transformers", display_name: "Autoregressive Transformers ROCm", implementation: "transformers-rocm", generation_family: "autoregressive", cache_setting: "cache_root", uses_base_model_identity: false, package_id: "modeldeck-core", package_version: "0.1.0", package_display_name: "ModelDeck core runtimes", publisher: "ModelDeck", source: "packaged", digest: "a".repeat(64) },
      { id: "operator-autoregressive", display_name: "Operator autoregressive preset", implementation: "transformers-rocm", generation_family: "autoregressive", cache_setting: "cache_root", uses_base_model_identity: false, package_id: "operator-presets", package_version: "1.0.0", package_display_name: "Operator presets", publisher: "Local operator", source: "trusted-local", digest: "b".repeat(64) },
      { id: "scenechat-gemma4", display_name: "SceneChat Gemma 4 ROCm", implementation: "vision-language-transformers-rocm", generation_family: "vision-language", cache_setting: "cache_root", uses_base_model_identity: false, package_id: "modeldeck-core", package_version: "0.1.0", package_display_name: "ModelDeck core runtimes", publisher: "ModelDeck", source: "packaged", digest: "a".repeat(64) },
      { id: "diffusiongemma-transformers", display_name: "DiffusionGemma Transformers ROCm", implementation: "text-diffusion-transformers-rocm", generation_family: "text-diffusion", cache_setting: "cache_root", uses_base_model_identity: false, package_id: "modeldeck-core", package_version: "0.1.0", package_display_name: "ModelDeck core runtimes", publisher: "ModelDeck", source: "packaged", digest: "a".repeat(64) },
      { id: "diffusiongemma-modeldeck-q4", display_name: "ModelDeck DiffusionGemma Q4 ROCm", implementation: "text-diffusion-gptq-rocm", generation_family: "text-diffusion", cache_setting: "q4_checkpoint_dir", uses_base_model_identity: true, package_id: "modeldeck-core", package_version: "0.1.0", package_display_name: "ModelDeck core runtimes", publisher: "ModelDeck", source: "packaged", digest: "a".repeat(64) },
      { id: "gpt-oss-llama-vulkan", display_name: "GPT-OSS llama.cpp Vulkan", implementation: "llama-vulkan", generation_family: "autoregressive", cache_setting: "artifact_path", uses_base_model_identity: false, package_id: "modeldeck-core", package_version: "0.1.0", package_display_name: "ModelDeck core runtimes", publisher: "ModelDeck", source: "packaged", digest: "a".repeat(64) },
      { id: "moshiko-speech", display_name: "Moshiko speech ROCm", implementation: "moshiko-rocm", generation_family: "speech-conversation", cache_setting: "cache_root", uses_base_model_identity: false, package_id: "modeldeck-core", package_version: "0.1.0", package_display_name: "ModelDeck core runtimes", publisher: "ModelDeck", source: "packaged", digest: "a".repeat(64) },
    ] });
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
    demoSets = [structuredClone(defaultDemoSet)];
    scenechatSelection = {
      alias: "scenechat-vision",
      display_name: "SceneChat provider",
      default_provider: "scenechat-gemma4-e2b-rocm",
      explicit_selection: false,
      selected_provider: "scenechat-gemma4-e2b-rocm",
      effective_provider: null,
      gateway_ready: false,
      routing_authority: "legacy-selection",
      superseded_by_active_demo_set: false,
      active_demo_set_id: null,
      active_demo_set_revision: null,
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
    deployment.display_name = deployment.id;
    window.history.replaceState({}, "", "/");
    vi.stubGlobal("EventSource", MockEventSource);
    vi.stubGlobal("fetch", mockFetch());
    vi.spyOn(window, "confirm").mockReturnValue(true);
    vi.spyOn(window, "prompt").mockReturnValue(null);
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

  it("edits, validates, plans and activates versioned demo routes", async () => {
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Demo routes" }));

    expect(screen.getByRole("heading", { name: "Fast chat" })).toBeInTheDocument();
    expect(screen.getByText(worker.id)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    expect(screen.getByText("Lower priorities are tried first. Equal priorities are ordered by deployment ID.")).toBeInTheDocument();
    expect(screen.getByLabelText("Provider 1 for fast-chat")).toHaveValue(worker.id);
    expect(screen.getByText(new RegExp(`Model: ${worker.model_id.replaceAll("/", "\\/")}`))).toBeInTheDocument();
    expect(screen.getByLabelText(`Priority for ${worker.id}`)).toHaveValue(10);
    vi.mocked(window.prompt).mockReturnValueOnce("Visitor Qwen");
    fireEvent.click(screen.getByRole("button", { name: "Rename" }));
    expect(await screen.findByText(`Renamed deployment ${worker.id} to Visitor Qwen.`)).toBeInTheDocument();
    expect(screen.getByLabelText("Provider 1 for fast-chat")).toHaveDisplayValue("Visitor Qwen");
    fireEvent.change(screen.getByLabelText("Identifier for Chat demo"), { target: { value: "visitor-chat" } });
    fireEvent.change(screen.getByLabelText("Public model alias"), { target: { value: "demo-chat" } });
    fireEvent.click(screen.getByRole("button", { name: "Save new revision" }));
    expect(await screen.findByText("Saved Open Day demos revision 2.")).toBeInTheDocument();
    const revisionRequest = vi.mocked(fetch).mock.calls.find(
      ([path, init]) => String(path) === "/api/demo-sets/open-day-demos" && init?.method === "PUT",
    );
    const revisionPayload = JSON.parse(String(revisionRequest?.[1]?.body));
    expect(revisionPayload.demos[0].id).toBe("visitor-chat");
    expect(revisionPayload.routes[0].demo_id).toBe("visitor-chat");

    fireEvent.click(screen.getByRole("button", { name: "Validate" }));
    expect(await screen.findByText("Valid route configuration")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Plan activation" }));
    expect(await screen.findByText("Activation plan")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Activate routing" }));
    expect(await screen.findByText("Open Day demos is now the active gateway routing configuration.")).toBeInTheDocument();
    expect(await screen.findByText("Revision history (1)")).toBeInTheDocument();
    expect(await screen.findByText("Gateway ready")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Smoke route" }));
    expect(await screen.findByText(/passed through qwen-small-rocm/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    fireEvent.change(screen.getByLabelText("Public model alias"), { target: { value: "draft-chat" } });
    fireEvent.click(screen.getByRole("button", { name: "Save new revision" }));
    expect(await screen.findByText("Saved Open Day demos revision 3.")).toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: "Discard latest draft" }));
    expect(window.confirm).toHaveBeenCalledWith("Discard revision 3? Revision 2 will become the latest configuration.");
    expect(await screen.findByText("Discarded draft revision 3. Revision 2 is now current.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("link", { name: "Workers" }));
    expect(await screen.findByText("Managed by Demo routes")).toBeInTheDocument();
    expect(screen.getByLabelText("Physical provider")).toBeDisabled();
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

    expect(screen.getByText("Legacy alias: scenechat-vision")).toBeInTheDocument();
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
    expect(screen.getByText("Local configuration")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Group workers"), { target: { value: "runtime" } });
    expect(screen.getByRole("heading", { name: "Llama Vulkan runtime" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Transformers Rocm runtime" })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Group workers"), { target: { value: "none" } });
    fireEvent.change(screen.getByLabelText("Sort workers"), { target: { value: "name-desc" } });
    const workerNames = [...document.querySelectorAll(".worker-card h3")].map((heading) => heading.textContent);
    expect(workerNames).toEqual([worker.id, "gpt-oss-120b-GGUF"]);
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

  it("sorts the model library by name, size, readiness, compatibility, configuration and family", async () => {
    catalogueModels = [
      { ...partialModel, model_id: "Zeta/Partial", physical_size_bytes: 20, generation_family_hint: "text-diffusion", capability_hints: ["text-generation", "iterative-refinement"] },
      { ...completeModel, model_id: "Alpha/Unsupported", physical_size_bytes: 10, generation_family_hint: "vision-language", capability_hints: ["text-generation", "chat", "image-input"], configuration_support: null },
      { ...completeModel, model_id: profile.model_id, physical_size_bytes: 30, generation_family_hint: "autoregressive" },
    ];
    compatibilityTests = [{ id: 7, fingerprint: "b".repeat(64), result: "tested-working", failure_class: null, evidence: { model_id: profile.model_id, model_revision: profile.revision, runtime: profile.preferred_runtime }, tested_at: "2026-07-19T10:00:00Z" }];
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Model library" }));

    const modelNames = () => screen.getAllByRole("heading", { level: 3 }).map((heading) => heading.textContent);
    expect(modelNames()).toEqual(["Alpha/Unsupported", profile.model_id, "Zeta/Partial"]);

    fireEvent.change(screen.getByLabelText("Sort models"), { target: { value: "size-desc" } });
    expect(modelNames()).toEqual([profile.model_id, "Zeta/Partial", "Alpha/Unsupported"]);
    fireEvent.change(screen.getByLabelText("Sort models"), { target: { value: "readiness" } });
    expect(modelNames()).toEqual([profile.model_id, "Alpha/Unsupported", "Zeta/Partial"]);
    fireEvent.change(screen.getByLabelText("Sort models"), { target: { value: "attention" } });
    expect(modelNames()).toEqual(["Zeta/Partial", "Alpha/Unsupported", profile.model_id]);
    fireEvent.change(screen.getByLabelText("Sort models"), { target: { value: "compatibility" } });
    expect(modelNames()[0]).toBe(profile.model_id);
    fireEvent.change(screen.getByLabelText("Sort models"), { target: { value: "configured-desc" } });
    expect(modelNames()[0]).toBe(profile.model_id);
    fireEvent.change(screen.getByLabelText("Sort models"), { target: { value: "family-asc" } });
    expect(modelNames()).toEqual([profile.model_id, "Zeta/Partial", "Alpha/Unsupported"]);
  });

  it("configures and removes a constrained cache-backed runtime", async () => {
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Model library" }));
    fireEvent.click(screen.getByRole("button", { name: "Add runtime configuration" }));
    expect(screen.getByText("The trusted template selects a reviewed launch implementation; commands, paths and environment remain non-editable.")).toBeInTheDocument();
    expect(screen.getByLabelText("Trusted runtime template")).toHaveValue("autoregressive-transformers");
    fireEvent.change(screen.getByLabelText("Trusted runtime template"), { target: { value: "operator-autoregressive" } });
    expect(screen.getByText("Configure Operator autoregressive preset")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Gateway alias"), { target: { value: "my-local-qwen" } });
    fireEvent.click(screen.getByRole("button", { name: "Save runtime configuration" }));

    expect(await screen.findByText("Runtime my-local-qwen is configured and ready to start from Workers.")).toBeInTheDocument();
    const configuredRuntime = screen.getByText("my-local-qwen").closest(".configured-runtime") as HTMLElement;
    fireEvent.click(within(configuredRuntime).getByRole("button", { name: "Remove configuration" }));
    expect(await screen.findByText("Runtime my-local-qwen was removed. Its cached model files were kept.")).toBeInTheDocument();
    expect(window.confirm).toHaveBeenCalledWith("Remove runtime configuration my-local-qwen? Cached model files will be kept.");
  });

  it("offers dedicated forms for supported vision-language and diffusion models", async () => {
    catalogueModels = [
      {
        ...completeModel,
        model_id: "google/gemma-4-E2B-it",
        generation_family_hint: "vision-language",
        capability_hints: ["text-generation", "chat", "image-input", "structured-output"],
        configuration_support: "scenechat-gemma4",
        configuration_support_reason: "Supported by the dedicated SceneChat Gemma 4 worker.",
      },
      {
        ...completeModel,
        model_id: "google/diffusiongemma-26B-A4B-it",
        generation_family_hint: "text-diffusion",
        capability_hints: ["text-generation", "iterative-refinement", "intermediate-frames", "seeded-generation"],
        configuration_support: "diffusiongemma-transformers",
        configuration_support_reason: "Supported by the dedicated DiffusionGemma Transformers worker.",
      },
      {
        ...completeModel,
        model_id: "ozyjay/diffusiongemma-modeldeck-q4",
        revision: "release-revision",
        generation_family_hint: "text-diffusion",
        capability_hints: ["text-generation", "iterative-refinement", "intermediate-frames", "seeded-generation"],
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
    expect(within(scenechatCard).getByText("Multimodal generative model · 953 MiB")).toBeInTheDocument();
    expect(within(scenechatCard).getByText("Text Generation")).toBeInTheDocument();
    expect(within(scenechatCard).getByText("Chat")).toBeInTheDocument();
    expect(within(scenechatCard).getByText("Image Input")).toBeInTheDocument();
    fireEvent.click(within(scenechatCard).getByRole("button", { name: "Configure runtime" }));
    expect(within(scenechatCard).getByText("Configure SceneChat Gemma 4 ROCm")).toBeInTheDocument();
    fireEvent.click(within(scenechatCard).getByRole("button", { name: "Cancel" }));
    fireEvent.click(within(diffusionCard).getByRole("button", { name: "Configure runtime" }));
    expect(within(diffusionCard).getByText("Configure DiffusionGemma Transformers ROCm")).toBeInTheDocument();
    expect(within(diffusionCard).getByLabelText("Lifecycle")).toBeDisabled();
    expect(within(diffusionCard).getByLabelText("Maximum denoising steps")).toBeInTheDocument();
    fireEvent.click(within(diffusionCard).getByRole("button", { name: "Cancel" }));
    fireEvent.click(within(q4Card).getByRole("button", { name: "Configure runtime" }));
    expect(within(q4Card).getByText("Configure ModelDeck DiffusionGemma Q4 ROCm")).toBeInTheDocument();
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

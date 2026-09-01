import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { LiveState, RoutingProfileRecord, Worker } from "./types";

const worker: Worker = {
  id: "b6a39318-6528-4448-9ec8-a2109029697f",
  name: "Qwen token trace", state: "stopped",
  model_id: "Qwen/Qwen2.5-0.5B-Instruct", revision: "revision-1",
  artifact_model_id: null, artifact_revision: null, generation_family: "autoregressive",
  runtime: "transformers-rocm", runtime_template_id: "autoregressive-transformers",
  runtime_template_version: "2", lifecycle: "on-demand", port: 8630, dtype: "float16",
  capabilities: { chat: true, top_k_trace: true }, settings: {}, endpoint: "http://127.0.0.1:8630",
  pid: null, started_at: null, last_error: null, archived: false,
  created_at: "2026-07-20T00:00:00Z", updated_at: "2026-07-20T00:00:00Z", archived_at: null,
};

const profile: RoutingProfileRecord = {
  definition: {
    id: "b5e4639a-5dbd-479e-a849-f93c04fd6311", name: "Local applications",
    description: "Token Trail and SprintBot", qualification: "tested-working",
    capabilities: [{
      id: "144d1dbf-9f46-4277-a324-e352577dbd5a", display_name: "Token trace",
      public_name: "qwen-0-5b", protocol_contract: "native-ar-trace-v1", tool_calling_enabled: false, worker_ids: [worker.id],
    }],
  },
  created_at: "2026-07-20T00:00:00Z", updated_at: "2026-07-20T00:00:00Z",
  active: true, active_revision: 1, latest_revision: 1,
};

function responses(configured = false): Record<string, unknown> {
  const workers = configured ? [worker] : [];
  return {
    "/api/health": { status: "ok", service: "modeldeck-management", schema_version: 5, configuration_locked: false, offline_only: true, gateway_url: "http://127.0.0.1:8600", state_store: { kind: "checkout-development", label: "Checkout development state", directory: "/workspace/.modeldeck" } },
    "/api/gateway/status": { available: true, health: { status: "ok", ready_workers: 0 }, models: { data: [] }, routes: { routes: [] }, error: null },
    "/api/hardware": { configured: { profile_id: "framework", os: "Fedora", gpu: "Radeon", gpu_architecture: "gfx1151", rocm_family: "7.2", work_mount: "/mnt/work" }, detected: { fedora_release: "44", kernel: "6.0", python: "3.13", rocm_packages: [], gpu_device_nodes: {}, memory: { total_bytes: 1, available_bytes: 1, percent: 0 }, swap: { total_bytes: 0, used_bytes: 0, percent: 0 }, filesystems: [], temperatures: [], fans: [], active_model_processes: [] }, diagnostic_note: "" },
    "/api/telemetry": { memory: { total_bytes: 1, available_bytes: 1, percent: 0 }, swap: { total_bytes: 0, used_bytes: 0, percent: 0 }, filesystems: [], temperatures: [], fans: [], active_model_processes: [] },
    "/api/thermal": { enabled: true, state: "normal", temperature_c: 62, sensor_id: "k10temp:Tctl", telemetry_age_seconds: 1, heavy_concurrency_limit: 2, active_heavy_concurrency: 0, model_load_concurrency_limit: 1, background_concurrency_limit: 1, background_paused: false, model_loading_allowed: true, scenechat_degradation: { active: false, minimum_frame_interval_seconds: 0, automatic_capture_allowed: true }, reason_code: "thermal_capacity_available", host_power_policy: { available: true, service_active: true, tuned_profile: "balanced", control: "external_read_only" } },
    "/api/live": configured ? {
      active_profile: { id: profile.definition.id, name: profile.definition.name, revision: 1 },
      active_profiles: [{ id: profile.definition.id, name: profile.definition.name, revision: 1 }],
      capabilities: [{ ...profile.definition.capabilities[0], profile_id: profile.definition.id, workers, effective_worker: null, ready: false }],
    } : { active_profile: null, active_profiles: [], capabilities: [] },
    "/api/workers": workers,
    "/api/routing-profiles": { profiles: configured ? [profile] : [] },
    "/api/catalogue": { models: [], downloads_started: false },
    "/api/protocol-contracts": { contracts: [{ id: "native-ar-trace-v1", display_name: "Native autoregressive trace", generation_family: "autoregressive", required_capabilities: ["top_k_trace"], required_worker_settings: {}, surfaces: ["POST /native/v1/autoregressive/traces"] }] },
    "/api/runtime-templates": { templates: [] },
    "/api/runtime-installations": { installations: [] },
    "/api/compatibility": { tests: [] },
    "/api/benchmark-history": { points: [], reports_scanned: 0, measurement: "median benchmark throughput" },
  };
}

function mockFetch(payloads: Record<string, unknown>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    const path = new URL(String(input), "http://localhost").pathname;
    const payload = payloads[path];
    return new Response(JSON.stringify(payload ?? { detail: `Unexpected request: ${path}` }), {
      status: payload === undefined ? 404 : 200,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function openAdvancedSection(name: "Models" | "Workers" | "Routing profiles") {
  fireEvent.click(await screen.findByRole("link", { name: "Advanced" }));
  fireEvent.click(await screen.findByRole("link", { name }));
}

describe("ModelDeck routing profile operator console", () => {
  beforeEach(() => { window.history.replaceState({}, "", "/"); window.localStorage.clear(); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("shows exact runtime installation identity and start policy", async () => {
    const payloads = responses();
    payloads["/api/runtime-installations"] = { installations: [{
      installation_id: "llama-cpp-vulkan", display_name: "llama.cpp Vulkan",
      integrity_status: "modified", currency_status: "recommended", start_allowed: false,
      detected: { source_revision: "9d77fa17254e1dee4b9e92504c91611a60b1359f", executable_sha256: "a".repeat(64), executable_size_bytes: 42, receipt_sha256: "b".repeat(64), receipt_version: 1, backend: "Vulkan", operating_system: "linux", architecture: "x86_64", version_output: "test" },
      recommended_source_revision: "9d77fa17254e1dee4b9e92504c91611a60b1359f",
      required_features: ["--model"], missing_features: [], reason_codes: ["executable_checksum_mismatch"],
      inspected_at: "2026-09-01T00:00:00Z", implementation_ids: ["llama-vulkan"],
      runtime_template_ids: ["gpt-oss-llama-vulkan"], worker_ids: [],
    }] };
    mockFetch(payloads);
    render(<App />);

    fireEvent.click(await screen.findByRole("link", { name: "Advanced" }));
    expect(await screen.findByText("llama.cpp Vulkan")).toBeInTheDocument();
    expect(screen.getByText("modified · recommended")).toBeInTheDocument();
    expect(screen.getByText("0 of 1 ready to start")).toBeInTheDocument();
  });

  it("starts with capability-oriented onboarding and no mock controls", async () => {
    const fetchMock = mockFetch(responses());
    render(<App />);

    expect(await screen.findByRole("heading", { name: "Set up a local capability" })).toBeInTheDocument();
    expect(screen.getByText(/keeps the exact Model, Runtime, Worker evidence/i)).toBeInTheDocument();
    expect(screen.queryByText(/Mock Worker/i)).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes("mock-worker"))).toBe(false);
    expect(fetchMock.mock.calls.some(([input]) => String(input) === "/api/events")).toBe(false);
    expect(screen.getByLabelText("State store")).toHaveTextContent("Checkout development state");
  });

  it("shows model token throughput over time from comparable benchmark reports", async () => {
    const payloads = responses();
    payloads["/api/benchmark-history"] = {
      reports_scanned: 2, measurement: "median benchmark throughput", points: [
        { series_key: "same-configuration", observed_at: "2026-07-18T00:00:00Z", model_id: "Qwen/Qwen3.5-4B", model_revision: "revision-123456789", runtime: "llama-vulkan", dtype: "q8", generation_family: "autoregressive", worker_id: "worker-1", worker_name: "Qwen 4B", tokens_per_second: 42.5, workload: "Standard · quick · 256 output tokens", configuration_fingerprint: "fingerprint-1", sample_count: 2 },
        { series_key: "same-configuration", observed_at: "2026-07-21T00:00:00Z", model_id: "Qwen/Qwen3.5-4B", model_revision: "revision-123456789", runtime: "llama-vulkan", dtype: "q8", generation_family: "autoregressive", worker_id: "worker-1", worker_name: "Qwen 4B", tokens_per_second: 48.25, workload: "Standard · quick · 256 output tokens", configuration_fingerprint: "fingerprint-2", sample_count: 2 },
      ],
    };
    mockFetch(payloads);
    render(<App />);

    fireEvent.click(await screen.findByRole("link", { name: "Advanced" }));

    expect(await screen.findByRole("heading", { name: "Model throughput over time" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Tokens per second benchmark history" })).toBeInTheDocument();
    expect(screen.getByText("48.25 tok/s latest · revision revision-123")).toBeInTheDocument();
    expect(screen.getByText(/Points are benchmark medians, not hardware telemetry/i)).toBeInTheDocument();
  });

  it("shows published capability readiness separately from Worker state", async () => {
    mockFetch(responses(true));
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Live" }));

    const status = await screen.findByRole("status", { name: "Token trace capability status" });
    expect(within(status).getByText("Not serving")).toBeInTheDocument();
    expect(screen.getByLabelText("Primary Worker Qwen token trace")).toBeInTheDocument();
    expect(screen.getByText("No ready Worker")).toHaveClass("unavailable");
  });

  it("hides published capabilities locally without changing routing state", async () => {
    mockFetch(responses(true));
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Live" }));

    fireEvent.click(await screen.findByRole("button", { name: "Hide capability Token trace" }));

    expect(screen.queryByRole("status", { name: "Token trace capability status" })).not.toBeInTheDocument();
    expect(screen.getByText("All published capabilities are hidden in this browser. Use Capability visibility or Show all to restore them.")).toBeInTheDocument();
    expect(screen.getByText("0 of 1 shown · 1 published")).toBeInTheDocument();
    expect(JSON.parse(window.localStorage.getItem("modeldeck-live-capability-visibility-v1") ?? "[]")).toEqual([
      `${profile.definition.id}:${profile.definition.capabilities[0].id}`,
    ]);

    fireEvent.click(screen.getByText("Capability visibility · 0 shown"));
    const visibility = screen.getByRole("checkbox", { name: /Token trace/ });
    expect(visibility).not.toBeChecked();
    fireEvent.click(visibility);
    expect(await screen.findByRole("status", { name: "Token trace capability status" })).toBeInTheDocument();
  });

  it("starts primary and backup Workers directly from the Live tab", async () => {
    const payloads = responses(true);
    const backup: Worker = { ...worker, id: "233d105c-9f80-4750-a2d5-7ed0fe8fe559", name: "Qwen backup", port: 8631 };
    payloads["/api/workers"] = [worker, backup];
    const live = payloads["/api/live"] as LiveState;
    live.capabilities[0].worker_ids = [worker.id, backup.id];
    live.capabilities[0].workers = [worker, backup];
    payloads[`/api/workers/${backup.id}/start`] = { state: "starting" };
    const fetchMock = mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Live" }));

    expect(await screen.findByRole("button", { name: `Start Worker ${worker.name}` })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: `Start Worker ${backup.name}` }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([input, init]) =>
      String(input) === `/api/workers/${backup.id}/start` && init?.method === "POST"
    )).toBe(true));
  });

  it("stops the effective Worker directly from the Live tab", async () => {
    const payloads = responses(true);
    const readyWorker: Worker = { ...worker, state: "ready", pid: 1234, started_at: "2026-08-29T00:00:00Z" };
    payloads["/api/workers"] = [readyWorker];
    const live = payloads["/api/live"] as LiveState;
    live.capabilities[0].workers = [readyWorker];
    live.capabilities[0].effective_worker = readyWorker;
    live.capabilities[0].ready = true;
    payloads[`/api/workers/${readyWorker.id}/stop`] = { state: "stopping" };
    const fetchMock = mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Live" }));

    fireEvent.click(await screen.findByRole("button", { name: `Stop Worker ${readyWorker.name}` }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([input, init]) =>
      String(input) === `/api/workers/${readyWorker.id}/stop` && init?.method === "POST"
    )).toBe(true));
  });

  it("disables Worker starts while thermal telemetry is stabilising", async () => {
    const payloads = responses(true);
    payloads["/api/thermal"] = {
      ...payloads["/api/thermal"] as object,
      state: "telemetry_degraded",
      temperature_c: null,
      telemetry_age_seconds: 12,
      model_loading_allowed: false,
      reason_code: "thermal_telemetry_degraded",
    };
    mockFetch(payloads);
    render(<App />);
    await openAdvancedSection("Workers");

    expect(await screen.findByText("Model loading paused")).toBeInTheDocument();
    expect(screen.getByText(/Fresh thermal telemetry is stabilising/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
  });

  it("edits routing profiles without Event or Demo terminology", async () => {
    const fetchMock = mockFetch(responses(true));
    render(<App />);
    await openAdvancedSection("Routing profiles");

    expect(await screen.findByText("Published capabilities")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Local applications")).toBeInTheDocument();
    expect(screen.queryByText(/^Demos$/)).not.toBeInTheDocument();
    expect(screen.getByText("API model ID")).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([input]) => String(input) === "/api/routing-profiles")).toBe(true);
  });

  it("explains missing runtimes without presenting capability policy as a Worker action", async () => {
    const payloads = responses();
    payloads["/api/catalogue"] = { downloads_started: false, models: [{
      model_id: "Qwen/Qwen3.5-4B", revision: "revision-1", cache_location: "/cache/model",
      snapshot_location: "/cache/model/snapshots/revision-1", physical_size_bytes: 1,
      download_state: "installed-untested", generation_family_hint: "vision-language",
      capability_hints: ["image-input", "video-input"], configuration_support: "scenechat-qwen35",
      configuration_support_reason: "SceneChat available", modeldeck_allowed: true,
      base_model_id: null, base_model_revision: null, runnable: false,
      runnable_reason: "No runnable allowed capability", worker_count: 0, artifacts: [],
      potential_capabilities: [{
        id: "video-understanding", display_name: "Video understanding",
        description: "Conversation or analysis grounded in video input.", protocol_contract_id: null,
        traits: ["video-input", "text-output"], evidence: [{ kind: "asserted", confidence: "direct",
          source: "reviewed-model-knowledge", detail: "The official family accepts video input.",
          reference: "https://huggingface.co/Qwen/Qwen3.5-0.8B", reviewed_at: "2026-08-12" }],
        runtime_template_ids: [], available_runtime_template_ids: [], policy_allowed: true,
        effective_allowed: true, runtime_status: "missing", qualification_status: "not-tested",
        qualifying_workers: [], published: false, creatable: false,
        reason: "Allowed; a trusted runtime is required.",
      }],
    }] };
    mockFetch(payloads);
    render(<App />);
    await openAdvancedSection("Models");

    expect(await screen.findByText("Video understanding")).toBeInTheDocument();
    expect(screen.getByText("runtime unavailable")).toBeInTheDocument();
    expect(screen.getByText(/Runtime implementation required/)).toBeInTheDocument();
    expect(screen.getByText("Missing · runtime implementation required")).toBeInTheDocument();
    expect(screen.getByText("None configured")).toBeInTheDocument();
    expect(screen.getByText(/Allowed for a future runtime/)).toBeInTheDocument();
    expect(screen.getByText("Evidence and provenance")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Disallow" })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Status"), { target: { value: "runtime-available" } });
    expect(screen.getByRole("heading", { name: "No Models match these filters" })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Status"), { target: { value: "runtime-missing" } });
    expect(screen.getByText("Video understanding")).toBeInTheDocument();
  });

  it("sets up a supported Worker directly while leaving unavailable capabilities non-actionable", async () => {
    const payloads = responses();
    const gemmaWorker: Worker = {
      ...worker,
      id: "f7fd3546-7a2e-47bd-b20f-5ca861ebd466",
      name: "Gemma 4 12B General Chat Worker",
      model_id: "google/gemma-4-E2B-it",
      revision: "revision-1",
      generation_family: "vision-language",
      runtime: "gemma4-general-chat-transformers-rocm",
      runtime_template_id: "gemma4-general-chat-rocm",
      capabilities: { chat: true, image_input: true },
      settings: { visual_token_budget: 280 },
    };
    payloads["/api/workers"] = [gemmaWorker];
    payloads["/api/catalogue"] = { downloads_started: false, models: [{
      model_id: "google/gemma-4-E2B-it", revision: "revision-1", cache_location: "/cache/model",
      snapshot_location: "/cache/model/snapshots/revision-1", physical_size_bytes: 1,
      download_state: "installed-untested", generation_family_hint: "vision-language",
      capability_hints: ["text-generation", "chat", "image-input", "structured-output"],
      configuration_support: "scenechat-gemma4", configuration_support_reason: "SceneChat available",
      modeldeck_allowed: true, base_model_id: null, base_model_revision: null, runnable: false,
      runnable_reason: "Allow at least one runnable capability before creating a Worker.", worker_count: 0, artifacts: [],
      potential_capabilities: [
        { id: "general-image-chat", display_name: "General image chat", description: "Open-ended image conversation.", protocol_contract_id: "openai-image-chat-v1", traits: ["image-input", "chat"], evidence: [], runtime_template_ids: [], available_runtime_template_ids: [], policy_allowed: true, effective_allowed: true, runtime_status: "missing", qualification_status: "not-tested", qualifying_workers: [], published: false, creatable: false, reason: "Allowed; a trusted runtime is required." },
        { id: "scene-analysis", display_name: "Scene analysis", description: "Bounded structured analysis.", protocol_contract_id: "scene-analysis-v1", traits: ["image-input", "structured-output"], evidence: [], runtime_template_ids: ["scenechat-gemma4"], available_runtime_template_ids: ["scenechat-gemma4"], policy_allowed: false, effective_allowed: false, runtime_status: "available", qualification_status: "not-tested", qualifying_workers: [], published: false, creatable: false, reason: "Allow this capability before creating a Worker or publishing a route." },
      ],
    }] };
    payloads["/api/runtime-templates"] = { templates: [{ id: "scenechat-gemma4", version: "1", display_name: "SceneChat Gemma 4 ROCm", runtime: "vision-language-transformers-rocm", generation_family: "vision-language", capabilities: { image_input: true, structured_output: true, cancellation: true }, settings: { context_length: 8192, maximum_new_tokens: 512, visual_token_budget: 280 }, cache_setting: "cache_root", include_cache_root: false, lifecycle: "on-demand", dtype: "bfloat16", uses_base_model_identity: false, package_id: "modeldeck-core", package_version: "1", package_display_name: "ModelDeck", publisher: "ModelDeck", source: "packaged", digest: "a".repeat(64) }] };
    payloads["/api/catalogue/capabilities/policy"] = { ok: true };
    const fetchMock = mockFetch(payloads);
    render(<App />);
    await openAdvancedSection("Models");

    expect(await screen.findByRole("button", { name: "Set up Worker" })).toBeInTheDocument();
    expect(screen.getByText(/Compatible runtime available/)).toBeInTheDocument();
    expect(screen.getByText("Available · SceneChat Gemma 4 ROCm")).toBeInTheDocument();
    expect(screen.getByText("1 configured · stopped")).toBeInTheDocument();
    expect(screen.queryByText(/280 visual tokens/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create Worker" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Set up Worker" }));

    expect(await screen.findByRole("heading", { name: "Create a Worker" })).toBeInTheDocument();
    const policyCall = fetchMock.mock.calls.find(([input]) => String(input) === "/api/catalogue/capabilities/policy");
    expect(policyCall?.[1]).toMatchObject({ method: "POST" });
  });

  it("reviews exact policy and Runtime identity before guided creation", async () => {
    const payloads = responses();
    payloads["/api/catalogue"] = { downloads_started: false, models: [{
      model_id: "example/local-chat", revision: "revision-1", cache_location: "/cache/model",
      snapshot_location: "/cache/model/snapshots/revision-1", physical_size_bytes: 1,
      download_state: "installed-untested", generation_family_hint: "autoregressive",
      capability_hints: ["chat"], configuration_support: "autoregressive-transformers",
      configuration_support_reason: "Trusted local adapter", modeldeck_allowed: true,
      base_model_id: null, base_model_revision: null, runnable: false, runnable_reason: "Not tested",
      worker_count: 0, artifacts: [], potential_capabilities: [{
        id: "general-chat", display_name: "General chat", description: "Conversational text generation.",
        protocol_contract_id: "openai-chat-v1", traits: ["chat"], evidence: [],
        runtime_template_ids: ["autoregressive-transformers"],
        available_runtime_template_ids: ["autoregressive-transformers"], policy_allowed: false,
        effective_allowed: false, runtime_status: "available", qualification_status: "not-tested",
        qualifying_workers: [], published: false, creatable: false, reason: "Approval required.",
      }],
    }] };
    payloads["/api/runtime-templates"] = { templates: [{
      id: "autoregressive-transformers", display_name: "Autoregressive Transformers ROCm",
      implementation: "transformers-rocm", generation_family: "autoregressive", cache_setting: "cache_root",
      uses_base_model_identity: false, lifecycle: "on-demand", dtype: "bfloat16", settings: {},
      package_id: "modeldeck-core", package_version: "1", package_display_name: "ModelDeck",
      publisher: "ModelDeck", source: "packaged", digest: "a".repeat(64),
    }] };
    payloads["/api/capability-setups/preview"] = {
      selection: {}, worker: { runtime_template_id: "autoregressive-transformers" },
      selection_basis: "only-compatible-runtime", runtime_registration_digest: "a".repeat(64),
      policy_changes: { model_allowed: false, capability_allowed: true }, warnings: [],
      preview_fingerprint: "b".repeat(64),
    };
    mockFetch(payloads);
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: /General chat/ }));
    fireEvent.click(await screen.findByRole("button", { name: /example\/local-chat/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Review Create and test" }));

    expect(await screen.findByText("Allow exact capability")).toBeInTheDocument();
    expect(screen.getByText("Verified after Worker start")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create and test" })).toBeInTheDocument();
  });

  it("offers explicit checksum approval for eligible local Qwen3.5 candidates", async () => {
    const payloads = responses();
    payloads["/api/catalogue"] = { downloads_started: false, models: [{
      model_id: "bartowski/Qwen_Qwen3.5-9B-GGUF", revision: "a".repeat(40),
      cache_location: "/cache/model", snapshot_location: `/cache/model/snapshots/${"a".repeat(40)}`,
      physical_size_bytes: 9_804_541_984, download_state: "installed-untested",
      generation_family_hint: "autoregressive", capability_hints: ["text-generation", "chat"],
      configuration_support: null, configuration_support_reason: "Explicit approval required.",
      modeldeck_allowed: true, base_model_id: null, base_model_revision: null, runnable: false,
      runnable_reason: "Explicit approval required.", worker_count: 0, artifacts: [],
      potential_capabilities: [],
      candidate_registration: {
        eligible: true, approved: false, candidate_id: null,
        filename: "Qwen_Qwen3.5-9B-Q8_0.gguf", expected_size: 9_804_541_984,
        expected_sha256: "b".repeat(64),
        reason: "Ready for explicit local approval and full SHA-256 verification.",
      },
    }] };
    payloads["/api/catalogue/candidates/approve"] = {
      ok: true, candidate_id: "qwen35-9b-q8-bbbbbbbbbbbb",
    };
    const fetchMock = mockFetch(payloads);
    render(<App />);
    await openAdvancedSection("Models");

    fireEvent.click(await screen.findByRole("button", { name: "Verify and approve" }));

    expect(await screen.findByText(/Approved bartowski\/Qwen_Qwen3.5-9B-GGUF/)).toBeInTheDocument();
    const approval = fetchMock.mock.calls.find(([input]) => String(input) === "/api/catalogue/candidates/approve");
    expect(approval?.[1]).toMatchObject({ method: "POST" });
  });
});

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { RoutingProfileRecord, Worker } from "./types";

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
      public_name: "qwen-0-5b", protocol_contract: "native-ar-trace-v1", worker_ids: [worker.id],
    }],
  },
  created_at: "2026-07-20T00:00:00Z", updated_at: "2026-07-20T00:00:00Z",
  active: true, active_revision: 1, latest_revision: 1,
};

function responses(configured = false): Record<string, unknown> {
  const workers = configured ? [worker] : [];
  return {
    "/api/health": { status: "ok", service: "modeldeck-management", schema_version: 4, configuration_locked: false, offline_only: true, gateway_url: "http://127.0.0.1:8600" },
    "/api/gateway/status": { available: true, health: { status: "ok", ready_workers: 0 }, models: { data: [] }, routes: { routes: [] }, error: null },
    "/api/hardware": { configured: { profile_id: "framework", os: "Fedora", gpu: "Radeon", gpu_architecture: "gfx1151", rocm_family: "7.2", work_mount: "/mnt/work" }, detected: { fedora_release: "44", kernel: "6.0", python: "3.13", rocm_packages: [], gpu_device_nodes: {}, memory: { total_bytes: 1, available_bytes: 1, percent: 0 }, swap: { total_bytes: 0, used_bytes: 0, percent: 0 }, filesystems: [], temperatures: [], fans: [], active_model_processes: [] }, diagnostic_note: "" },
    "/api/telemetry": { memory: { total_bytes: 1, available_bytes: 1, percent: 0 }, swap: { total_bytes: 0, used_bytes: 0, percent: 0 }, filesystems: [], temperatures: [], fans: [], active_model_processes: [] },
    "/api/thermal": { enabled: true, state: "normal", temperature_c: 62, sensor_id: "k10temp:Tctl", telemetry_age_seconds: 1, heavy_concurrency_limit: 2, active_heavy_concurrency: 0, model_load_concurrency_limit: 1, background_concurrency_limit: 1, background_paused: false, model_loading_allowed: true, scenechat_degradation: { active: false, minimum_frame_interval_seconds: 0, automatic_capture_allowed: true }, reason_code: "thermal_capacity_available", host_power_policy: { available: true, service_active: true, tuned_profile: "balanced", control: "external_read_only" } },
    "/api/live": configured ? {
      active_profile: { id: profile.definition.id, name: profile.definition.name, revision: 1 },
      active_profiles: [{ id: profile.definition.id, name: profile.definition.name, revision: 1 }],
      capabilities: [{ ...profile.definition.capabilities[0], workers, effective_worker: null, ready: false }],
    } : { active_profile: null, active_profiles: [], capabilities: [] },
    "/api/workers": workers,
    "/api/routing-profiles": { profiles: configured ? [profile] : [] },
    "/api/catalogue": { models: [], downloads_started: false },
    "/api/protocol-contracts": { contracts: [{ id: "native-ar-trace-v1", display_name: "Native autoregressive trace", generation_family: "autoregressive", required_capabilities: ["top_k_trace"], required_worker_settings: {}, surfaces: ["POST /native/v1/autoregressive/traces"] }] },
    "/api/runtime-templates": { templates: [] },
    "/api/compatibility": { tests: [] },
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

describe("ModelDeck routing profile operator console", () => {
  beforeEach(() => { window.history.replaceState({}, "", "/"); window.localStorage.clear(); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("starts with capability-oriented onboarding and no mock controls", async () => {
    const fetchMock = mockFetch(responses());
    render(<App />);

    expect(await screen.findByRole("heading", { name: "Build your first local capability" })).toBeInTheDocument();
    expect(screen.getByText(/create a Routing Profile and capability/i)).toBeInTheDocument();
    expect(screen.queryByText(/Mock Worker/i)).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes("mock-worker"))).toBe(false);
    expect(fetchMock.mock.calls.some(([input]) => String(input) === "/api/events")).toBe(false);
  });

  it("shows published capability readiness separately from Worker state", async () => {
    mockFetch(responses(true));
    render(<App />);

    const status = await screen.findByRole("status", { name: "Token trace capability status" });
    expect(within(status).getByText("Not serving")).toBeInTheDocument();
    expect(screen.getByLabelText("Primary Worker Qwen token trace")).toBeInTheDocument();
    expect(screen.getByText("No ready Worker")).toHaveClass("unavailable");
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
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));

    expect(await screen.findByText("Model loading paused")).toBeInTheDocument();
    expect(screen.getByText(/Fresh thermal telemetry is stabilising/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
  });

  it("edits routing profiles without Event or Demo terminology", async () => {
    const fetchMock = mockFetch(responses(true));
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Routing profiles" }));

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
    fireEvent.click(await screen.findByRole("link", { name: "Models" }));

    expect(await screen.findByText("Video understanding")).toBeInTheDocument();
    expect(screen.getByText("runtime unavailable")).toBeInTheDocument();
    expect(screen.getByText(/No Worker is available yet/)).toBeInTheDocument();
    expect(screen.getByText(/Allowed for a future runtime/)).toBeInTheDocument();
    expect(screen.getByText("Evidence and provenance")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Disallow" })).toBeInTheDocument();
  });

  it("sets up a supported Worker directly while leaving unavailable capabilities non-actionable", async () => {
    const payloads = responses();
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
    fireEvent.click(await screen.findByRole("link", { name: "Models" }));

    expect(await screen.findByRole("button", { name: "Set up Worker" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create Worker" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Set up Worker" }));

    expect(await screen.findByRole("heading", { name: "Create a Worker" })).toBeInTheDocument();
    const policyCall = fetchMock.mock.calls.find(([input]) => String(input) === "/api/catalogue/capabilities/policy");
    expect(policyCall?.[1]).toMatchObject({ method: "POST" });
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
    fireEvent.click(await screen.findByRole("link", { name: "Models" }));

    fireEvent.click(await screen.findByRole("button", { name: "Verify and approve" }));

    expect(await screen.findByText(/Approved bartowski\/Qwen_Qwen3.5-9B-GGUF/)).toBeInTheDocument();
    const approval = fetchMock.mock.calls.find(([input]) => String(input) === "/api/catalogue/candidates/approve");
    expect(approval?.[1]).toMatchObject({ method: "POST" });
  });
});

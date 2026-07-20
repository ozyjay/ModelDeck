import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { EventRecord, Worker } from "./types";

const worker: Worker = {
  id: "b6a39318-6528-4448-9ec8-a2109029697f",
  name: "Qwen token trace",
  state: "stopped",
  model_id: "Qwen/Qwen2.5-0.5B-Instruct",
  revision: "revision-1",
  artifact_model_id: null,
  artifact_revision: null,
  generation_family: "autoregressive",
  runtime: "transformers-rocm",
  runtime_template_id: "autoregressive-transformers",
  runtime_template_version: "2",
  lifecycle: "on-demand",
  port: 8630,
  dtype: "float16",
  capabilities: { chat: true, top_k_trace: true },
  settings: {},
  endpoint: "http://127.0.0.1:8630",
  pid: null,
  started_at: null,
  last_error: null,
  archived: false,
  created_at: "2026-07-20T00:00:00Z",
  updated_at: "2026-07-20T00:00:00Z",
  archived_at: null,
};

const eventRecord: EventRecord = {
  definition: {
    id: "b5e4639a-5dbd-479e-a849-f93c04fd6311",
    name: "2026 Open Day",
    description: "Token Trails",
    qualification: "tested-working",
    demos: [{ id: "0a0415e0-9055-40d0-b353-6d9fecb36edc", name: "Token Trails", route_ids: ["144d1dbf-9f46-4277-a324-e352577dbd5a"] }],
    routes: [{ id: "144d1dbf-9f46-4277-a324-e352577dbd5a", display_name: "Token trace", public_name: "qwen-0-5b", protocol_contract: "native-ar-trace-v1", worker_ids: [worker.id] }],
  },
  created_at: "2026-07-20T00:00:00Z",
  updated_at: "2026-07-20T00:00:00Z",
  active: true,
  active_revision: 1,
  latest_revision: 1,
};

function responses(includeConfiguration = false): Record<string, unknown> {
  const workers = includeConfiguration ? [worker] : [];
  const events = includeConfiguration ? [eventRecord] : [];
  return {
    "/api/health": { status: "ok", service: "modeldeck-management", schema_version: 2, open_day: false, downloads_allowed: false, gateway_url: "http://127.0.0.1:8600" },
    "/api/gateway/status": { available: true, health: { status: "ok", ready_workers: 0 }, models: { data: [] }, routes: { routes: [] }, error: null },
    "/api/hardware": { configured: { profile_id: "framework", os: "Fedora", gpu: "Radeon", gpu_architecture: "gfx1151", rocm_family: "7.2", work_mount: "/mnt/work" }, detected: { fedora_release: "44", kernel: "6.0", python: "3.13", rocm_packages: [], gpu_device_nodes: {}, memory: { total_bytes: 1, available_bytes: 1, percent: 0 }, swap: { total_bytes: 0, used_bytes: 0, percent: 0 }, filesystems: [], temperatures: [], fans: [], active_model_processes: [] }, diagnostic_note: "" },
    "/api/telemetry": { memory: { total_bytes: 1, available_bytes: 1, percent: 0 }, swap: { total_bytes: 0, used_bytes: 0, percent: 0 }, filesystems: [], temperatures: [], fans: [], active_model_processes: [] },
    "/api/live": includeConfiguration ? { active_event: { id: eventRecord.definition.id, name: eventRecord.definition.name, revision: 1 }, routes: [] } : { active_event: null, routes: [] },
    "/api/workers": workers,
    "/api/events": { events },
    "/api/catalogue": { models: [], downloads_started: false },
    "/api/protocol-contracts": { contracts: [{ id: "native-ar-trace-v1", display_name: "Native autoregressive trace", generation_family: "autoregressive", required_capabilities: ["top_k_trace"], surfaces: ["POST /native/autoregressive/trace"] }] },
    "/api/runtime-templates": { templates: [] },
    "/api/compatibility": { tests: [] },
  };
}

function mockFetch(payloads: Record<string, unknown>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(String(input), "http://localhost").pathname;
    const configured = payloads[path];
    const payload = typeof configured === "function"
      ? await (configured as (input: RequestInfo | URL, init?: RequestInit) => unknown)(input, init)
      : configured;
    return new Response(JSON.stringify(payload ?? { detail: `Unexpected request: ${path}` }), {
      status: payload === undefined ? 404 : 200,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function catalogueModel(modelId: string, capabilityHints: string[] = ["text-generation", "chat"]) {
  return {
    model_id: modelId, revision: "revision-1", cache_location: "/cache/model",
    snapshot_location: "/cache/snapshot", physical_size_bytes: 1,
    download_state: "installed-untested", generation_family_hint: "autoregressive",
    capability_hints: capabilityHints, configuration_support: "autoregressive-transformers",
    configuration_support_reason: "Supported", modeldeck_allowed: true,
    base_model_id: null, base_model_revision: null, runnable: true,
    runnable_reason: "Ready to create a Worker.", worker_count: 0, artifacts: [],
  };
}

describe("ModelDeck v2 operator console", () => {
  beforeEach(() => window.history.replaceState({}, "", "/"));
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("starts with an explicit onboarding workflow and no packaged cards", async () => {
    mockFetch(responses());
    render(<App />);
    expect(await screen.findByRole("heading", { name: "Build your first local route" })).toBeInTheDocument();
    expect(screen.getByLabelText("Configuration status")).toHaveTextContent("Configuration unlocked");
    expect(screen.getByText("ModelDeck starts empty: create a Worker from a discovered Model, create an Event and Route, then publish it.")).toBeInTheDocument();
  });

  it("shows editable Worker names without exposing an alias concept", async () => {
    mockFetch(responses(true));
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));
    expect(await screen.findByRole("heading", { name: "Qwen token trace" })).toBeInTheDocument();
    expect(screen.getByText(/execution identity is not/i)).toBeInTheDocument();
    expect(screen.queryByText(/provider/i)).not.toBeInTheDocument();
  });

  it("collapses and expands a Worker card", async () => {
    mockFetch(responses(true));
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));

    const collapse = await screen.findByRole("button", { name: "Collapse Worker Qwen token trace" });
    expect(screen.getByRole("button", { name: "Archive" })).toBeVisible();
    fireEvent.click(collapse);

    expect(screen.queryByRole("button", { name: "Archive" })).not.toBeInTheDocument();
    const expand = screen.getByRole("button", { name: "Expand Worker Qwen token trace" });
    expect(expand).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(expand);

    expect(screen.getByRole("button", { name: "Archive" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Collapse Worker Qwen token trace" })).toHaveAttribute("aria-expanded", "true");
  });

  it("explains the effect of archiving and leaves the Worker unchanged when cancelled", async () => {
    const payloads = responses(true);
    payloads[`/api/workers/${worker.id}`] = { ok: true, worker_id: worker.id, cache_removed: false };
    const fetchMock = mockFetch(payloads);
    const confirm = vi.spyOn(window, "confirm")
      .mockReturnValueOnce(false)
      .mockReturnValueOnce(true);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));

    const archiveButton = await screen.findByRole("button", { name: "Archive" });
    fireEvent.click(archiveButton);

    expect(confirm).toHaveBeenCalledWith(expect.stringMatching(/cannot be restored in ModelDeck/i));
    expect(confirm).toHaveBeenCalledWith(expect.stringMatching(/Historical Event revisions and cached Model files will be kept/i));
    expect(confirm).toHaveBeenCalledWith(expect.stringMatching(/Cancel leaves the Worker unchanged/i));
    expect(fetchMock.mock.calls.some(([input, init]) =>
      String(input) === `/api/workers/${worker.id}` && init?.method === "DELETE"
    )).toBe(false);

    fireEvent.click(archiveButton);
    await waitFor(() => expect(fetchMock.mock.calls.some(([input, init]) =>
      String(input) === `/api/workers/${worker.id}` && init?.method === "DELETE"
    )).toBe(true));
  });

  it("allows start requests for different Workers to be submitted together", async () => {
    const secondWorker = { ...worker, id: "d054e57f-b1fd-4575-8f55-9cfaf1f55380", name: "Second Qwen", port: 8632 };
    const payloads = responses(true);
    payloads["/api/workers"] = [worker, secondWorker];
    payloads[`/api/workers/${worker.id}/start`] = () => new Promise<never>(() => undefined);
    payloads[`/api/workers/${secondWorker.id}/start`] = () => new Promise<never>(() => undefined);
    const fetchMock = mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));
    const startButtons = screen.getAllByRole("button", { name: "Start" });
    fireEvent.click(startButtons[0]);
    expect(startButtons[0]).toBeDisabled();
    expect(startButtons[1]).toBeEnabled();
    fireEvent.click(startButtons[1]);
    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input) === `/api/workers/${worker.id}/start`)).toBe(true));
    expect(fetchMock.mock.calls.some(([input]) => String(input) === `/api/workers/${secondWorker.id}/start`)).toBe(true);
  });

  it("filters the Models library by model metadata", async () => {
    const payloads = responses();
    payloads["/api/catalogue"] = { models: [
      catalogueModel("Qwen/Qwen2.5-1.5B-Instruct"),
      catalogueModel("Example/Vision-Model", ["image-input", "chat"]),
    ], downloads_started: false };
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Models" }));
    fireEvent.change(screen.getByRole("searchbox", { name: "Search models" }), { target: { value: "qwen" } });
    expect(screen.getByRole("heading", { name: "Qwen/Qwen2.5-1.5B-Instruct" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Example/Vision-Model" })).not.toBeInTheDocument();
    expect(screen.getByText("1 of 2 cached")).toBeInTheDocument();
  });

  it("explains and disables Worker creation while Open Day mode is active", async () => {
    const payloads = responses();
    payloads["/api/health"] = { ...(payloads["/api/health"] as object), open_day: true };
    payloads["/api/catalogue"] = { models: [catalogueModel("Qwen/Qwen2.5-1.5B-Instruct")], downloads_started: false };
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Models" }));
    expect(screen.getByLabelText("Configuration status")).toHaveTextContent("Open Day · configuration locked");
    expect(screen.getByText(/Open Day mode locks configuration/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create Worker" })).toBeDisabled();
  });

  it("edits an Event with explicit primary and ordered backup Workers", async () => {
    mockFetch(responses(true));
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Events" }));
    expect(await screen.findByRole("heading", { name: "2026 Open Day" })).toBeInTheDocument();
    expect(screen.getByText("Primary")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Route Name" })).toHaveValue("Token trace");
    expect(screen.getByDisplayValue("qwen-0-5b")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/Saved/)).toBeInTheDocument());
  });

  it("preserves Event description input while autosaving", async () => {
    const payloads = responses(true);
    payloads[`/api/events/${eventRecord.definition.id}/draft`] = eventRecord;
    const fetchMock = mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Events" }));
    const description = screen.getByRole("textbox", { name: "Description" });
    fireEvent.change(description, { target: { value: "A description typed without interruption" } });
    expect(description).toHaveValue("A description typed without interruption");
    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input) === `/api/events/${eventRecord.definition.id}/draft`)).toBe(true), { timeout: 1500 });
    await waitFor(() => expect(screen.getByText(/· Saved/)).toBeInTheDocument(), { timeout: 1500 });
    expect(description).toHaveValue("A description typed without interruption");
    expect(fetchMock.mock.calls.filter(([input]) => String(input) === "/api/events")).toHaveLength(1);
  });

  it("uses trusted SceneChat runtime defaults when creating a Worker", async () => {
    const payloads = responses();
    payloads["/api/catalogue"] = { models: [{
      ...catalogueModel("google/gemma-4-E2B-it", ["image-input", "structured-output"]),
      generation_family_hint: "vision-language",
      configuration_support: "scenechat-gemma4",
    }], downloads_started: false };
    payloads["/api/runtime-templates"] = { templates: [{
      id: "scenechat-gemma4", display_name: "SceneChat Gemma 4 ROCm",
      implementation: "vision-language-transformers-rocm", generation_family: "vision-language",
      cache_setting: "cache_root", uses_base_model_identity: false,
      lifecycle: "on-demand", dtype: "bfloat16",
      settings: { context_length: 8192, maximum_new_tokens: 512 },
      package_id: "modeldeck-core", package_version: "1", package_display_name: "Core",
      publisher: "ModelDeck", source: "packaged", digest: "digest",
    }] };
    payloads["/api/workers"] = (_input: RequestInfo | URL, init?: RequestInit) =>
      init?.method === "POST" ? worker : [];
    const fetchMock = mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Models" }));
    fireEvent.click(await screen.findByRole("button", { name: "Create Worker" }));
    expect(screen.getByRole("combobox", { name: "Data type" })).toHaveValue("bfloat16");
    expect(screen.getByRole("combobox", { name: "Data type" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Lifecycle" })).toHaveValue("on-demand");
    expect(screen.getByRole("combobox", { name: "Lifecycle" })).toBeDisabled();
    expect(screen.getByRole("spinbutton", { name: "Context length" })).toHaveValue(8192);
    expect(screen.getByRole("spinbutton", { name: "Maximum output" })).toHaveValue(512);
    fireEvent.change(screen.getByRole("spinbutton", { name: "Context length" }), { target: { value: "4096" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Maximum output" }), { target: { value: "256" } });
    fireEvent.click(screen.getByRole("button", { name: "Create Worker" }));

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([input, init]) =>
        String(input) === "/api/workers" && init?.method === "POST"
      );
      expect(call).toBeDefined();
      const body = JSON.parse(String(call?.[1]?.body));
      expect(body).toMatchObject({
        model_id: "google/gemma-4-E2B-it",
        runtime_template_id: "scenechat-gemma4",
        dtype: "bfloat16",
        lifecycle: "on-demand",
        context_length: 4096,
        maximum_new_tokens: 256,
      });
    });
  });

  it("creates an immutable replacement and can rebind draft Event routes", async () => {
    const payloads = responses(true);
    payloads["/api/runtime-templates"] = { templates: [{
      id: "autoregressive-transformers", display_name: "Autoregressive Transformers ROCm",
      implementation: "autoregressive-transformers-rocm", generation_family: "autoregressive",
      cache_setting: "cache_root", uses_base_model_identity: false,
      lifecycle: null, dtype: null,
      settings: { context_length: 2048, maximum_new_tokens: 128 },
      package_id: "modeldeck-core", package_version: "1", package_display_name: "Core",
      publisher: "ModelDeck", source: "packaged", digest: "digest",
    }] };
    payloads[`/api/workers/${worker.id}/replacement`] = {
      replacement: { ...worker, id: "cf50c7e3-14fa-43b7-a073-24d103f624a8", name: "Qwen revised" },
      rebound_event_drafts: [eventRecord.definition.id],
    };
    const fetchMock = mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));
    fireEvent.click(await screen.findByRole("button", { name: "Replace" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Replacement name" }), { target: { value: "Qwen revised" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Context length" }), { target: { value: "4096" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Maximum output" }), { target: { value: "256" } });
    fireEvent.click(screen.getByRole("button", { name: "Create replacement" }));

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([input, init]) =>
        String(input) === `/api/workers/${worker.id}/replacement` && init?.method === "POST"
      );
      expect(call).toBeDefined();
      expect(JSON.parse(String(call?.[1]?.body))).toEqual({
        name: "Qwen revised",
        dtype: "float16",
        lifecycle: "on-demand",
        context_length: 4096,
        maximum_new_tokens: 256,
        rebind_drafts: true,
      });
    });
    expect(await screen.findByText(/1 draft Event was updated/)).toHaveTextContent("published routing is unchanged");
  });

  it("offers a discovered GPT-OSS artefact when creating its Worker", async () => {
    const payloads = responses();
    payloads["/api/catalogue"] = { models: [{
      model_id: "ggml-org/gpt-oss-120b-GGUF", revision: "revision-1",
      cache_location: "/cache/model", snapshot_location: "/cache/snapshot",
      physical_size_bytes: 1, download_state: "installed-untested",
      generation_family_hint: "autoregressive", capability_hints: ["text-generation", "chat"],
      configuration_support: "gpt-oss-llama-vulkan", configuration_support_reason: "Supported",
      modeldeck_allowed: true, base_model_id: null, base_model_revision: null,
      runnable: true, runnable_reason: "Ready to create a Worker.", worker_count: 0,
      artifacts: [{ artifact_id: "gpt-oss-120b-mxfp4", kind: "gguf", format: "mxfp4", filenames: ["gpt-oss-120b-MXFP4.gguf"] }],
    }], downloads_started: false };
    payloads["/api/runtime-templates"] = { templates: [{
      id: "gpt-oss-llama-vulkan", display_name: "GPT-OSS llama.cpp Vulkan",
      implementation: "llama-vulkan", generation_family: "autoregressive",
      cache_setting: "artifact_path", uses_base_model_identity: false,
      lifecycle: "exclusive", dtype: null, settings: {},
      package_id: "modeldeck-core", package_version: "1", package_display_name: "Core",
      publisher: "ModelDeck", source: "packaged", digest: "digest",
    }] };
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Models" }));
    fireEvent.click(await screen.findByRole("button", { name: "Create Worker" }));
    expect(screen.getByLabelText("Model artefact")).toHaveValue("gpt-oss-120b-mxfp4");
  });
});

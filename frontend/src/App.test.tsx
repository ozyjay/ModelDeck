import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
    "/api/mock-worker-templates": { templates: [
      { id: "native-ar-trace-v1", protocol_contract: "native-ar-trace-v1", display_name: "Native autoregressive trace", generation_family: "autoregressive", default_name: "Autoregressive trace mock", scenarios: ["success", "delayed", "request-error"], options: [] },
      { id: "scene-analysis-v1", protocol_contract: "scene-analysis-v1", display_name: "Scene analysis", generation_family: "vision-language", default_name: "Scene analysis mock", scenarios: ["success", "delayed", "request-error"], options: [{ id: "visual_token_budget", label: "Visual tokens", type: "select", default: 70, choices: [70, 140, 280, 560, 1120] }] },
    ] },
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
    if (payload instanceof Response) return payload;
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
  beforeEach(() => { window.history.replaceState({}, "", "/"); window.localStorage.clear(); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("starts with an explicit onboarding workflow and no packaged cards", async () => {
    mockFetch(responses());
    render(<App />);
    expect(await screen.findByRole("heading", { name: "Build your first local route" })).toBeInTheDocument();
    expect(screen.getByLabelText("Configuration status")).toHaveTextContent("Configuration unlocked");
    expect(screen.getByText("ModelDeck starts empty: create a Worker from a discovered Model, create an Event and Route, then publish it.")).toBeInTheDocument();
  });

  it("makes Live Route and Worker readiness independently scannable", async () => {
    const payloads = responses(true);
    payloads["/api/live"] = {
      active_event: { id: eventRecord.definition.id, name: eventRecord.definition.name, revision: 1 },
      routes: [{
        ...eventRecord.definition.routes[0],
        workers: [worker],
        effective_worker: null,
        ready: false,
      }],
    };
    mockFetch(payloads);
    render(<App />);

    const routeStatus = await screen.findByRole("status", { name: "Token trace Route status" });
    expect(within(routeStatus).getByText("Not serving")).toBeInTheDocument();
    expect(within(routeStatus).getByText("Start a Worker")).toBeInTheDocument();
    const primary = screen.getByLabelText("Primary Worker Qwen token trace");
    expect(within(primary).getByText("Primary")).toBeInTheDocument();
    expect(within(primary).getByText("stopped")).toBeInTheDocument();
    expect(screen.getByText("No ready Worker")).toHaveClass("unavailable");
  });

  it("shows editable Worker names without exposing an alias concept", async () => {
    mockFetch(responses(true));
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));
    expect(await screen.findByRole("heading", { name: "Qwen token trace" })).toBeInTheDocument();
    expect(screen.getByText(/execution identity is not/i)).toBeInTheDocument();
    expect(screen.queryByText(/provider/i)).not.toBeInTheDocument();
  });

  it("creates contract-driven mock Workers with scenario-specific options", async () => {
    const mockWorker: Worker = {
      ...worker,
      id: "d054e57f-b1fd-4575-8f55-9cfaf1f55380",
      name: "SceneChat mock 280",
      model_id: "modeldeck/mock-scenechat-vision",
      revision: "fixture-v1",
      generation_family: "vision-language",
      runtime: "mock",
      capabilities: { chat: "compatibility-only", image_input: true, structured_output: true },
      settings: { visual_token_budget: 280 },
      port: 8632,
    };
    const payloads = responses(true);
    payloads["/api/workers/mocks"] = mockWorker;
    const fetchMock = mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));
    fireEvent.change(screen.getByRole("combobox", { name: "Mock contract" }), { target: { value: "scene-analysis-v1" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Mock scenario" }), { target: { value: "delayed" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Mock delay" }), { target: { value: "2500" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Mock visual tokens" }), { target: { value: "280" } });
    fireEvent.click(screen.getByRole("button", { name: "Create mock Worker" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([input, init]) =>
      String(input).endsWith("/api/workers/mocks")
      && init?.method === "POST"
      && init.body === JSON.stringify({ protocol_contract: "scene-analysis-v1", scenario: "delayed", delay_ms: 2500, visual_token_budget: 280 })
    )).toBe(true));
    expect(await screen.findByText(/Created SceneChat mock 280/)).toBeInTheDocument();
    expect(screen.getByText(/never performs physical model inference/)).toBeInTheDocument();
  });

  it("explains that a 405 creating a generic mock requires a service restart", async () => {
    const payloads = responses(true);
    payloads["/api/workers/mocks"] = new Response(JSON.stringify({ detail: "Method Not Allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    });
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));
    fireEvent.click(screen.getByRole("button", { name: "Create mock Worker" }));

    expect(await screen.findByText(/Restart ModelDeck, then try again/)).toBeInTheDocument();
  });

  it("creates and saves a contract-matched mock as the last Route backup", async () => {
    const mockWorker: Worker = {
      ...worker,
      id: "d054e57f-b1fd-4575-8f55-9cfaf1f55380",
      name: "Autoregressive trace mock",
      model_id: "modeldeck/mock-autoregressive-trace",
      runtime: "mock",
      capabilities: { top_k_trace: true, cancellation: true },
      settings: { mock_contract_id: "native-ar-trace-v1", mock_scenario: "success" },
      port: 8632,
    };
    const payloads = responses(true);
    payloads["/api/workers/mocks"] = mockWorker;
    payloads[`/api/events/${eventRecord.definition.id}/draft`] = {};
    const fetchMock = mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Events" }));
    fireEvent.click(await screen.findByRole("button", { name: "Create mock backup" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([input, init]) => {
      if (!String(input).endsWith(`/api/events/${eventRecord.definition.id}/draft`) || init?.method !== "PUT") return false;
      const body = JSON.parse(String(init.body));
      return body.routes[0].worker_ids.at(-1) === mockWorker.id;
    })).toBe(true));
    expect(await screen.findByText(/added it as the last Route backup/)).toBeInTheDocument();
  });

  it("reports when a Route mock is created but draft assignment fails", async () => {
    const mockWorker: Worker = {
      ...worker,
      id: "d054e57f-b1fd-4575-8f55-9cfaf1f55380",
      name: "Autoregressive trace mock",
      model_id: "modeldeck/mock-autoregressive-trace",
      runtime: "mock",
      capabilities: { top_k_trace: true },
      settings: { mock_contract_id: "native-ar-trace-v1", mock_scenario: "success" },
    };
    const payloads = responses(true);
    payloads["/api/workers/mocks"] = mockWorker;
    payloads[`/api/events/${eventRecord.definition.id}/draft`] = new Response(
      JSON.stringify({ detail: "Draft storage unavailable" }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    );
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Events" }));
    fireEvent.click(await screen.findByRole("button", { name: "Create mock backup" }));

    expect(await screen.findByText(/was created, but it could not be assigned/)).toHaveTextContent("Draft storage unavailable");
  });

  it("searches and filters Workers, reports the result count and clears the filters", async () => {
    const visionWorker: Worker = {
      ...worker,
      id: "d054e57f-b1fd-4575-8f55-9cfaf1f55380",
      name: "Qwen vision",
      state: "ready",
      model_id: "Qwen/Qwen3.5-4B",
      generation_family: "vision-language",
      runtime: "qwen35-rocm",
      capabilities: { chat: true, image_input: true },
      port: 8632,
    };
    const payloads = responses(true);
    payloads["/api/workers"] = [worker, visionWorker];
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));

    expect(await screen.findByRole("status")).toHaveTextContent("2 of 2 Workers");
    fireEvent.change(screen.getByRole("searchbox", { name: "Search workers" }), { target: { value: "qwen image_input" } });
    expect(screen.getByRole("heading", { name: "Qwen vision" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Qwen token trace" })).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("1 of 2 Workers");

    fireEvent.change(screen.getByRole("combobox", { name: "State" }), { target: { value: "stopped" } });
    expect(screen.getByRole("heading", { name: "No Workers match these filters" })).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "Clear filters" })[1]);

    expect(screen.getByRole("heading", { name: "Qwen token trace" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Qwen vision" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("2 of 2 Workers");
  });

  it("remembers Worker search, filters and sorting after a reload", async () => {
    const visionWorker: Worker = {
      ...worker,
      id: "d054e57f-b1fd-4575-8f55-9cfaf1f55380",
      name: "Qwen vision",
      state: "ready",
      model_id: "Qwen/Qwen3.5-4B",
      runtime: "qwen35-rocm",
      port: 8632,
    };
    const payloads = responses(true);
    payloads["/api/workers"] = [worker, visionWorker];
    mockFetch(payloads);
    const first = render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));
    fireEvent.change(screen.getByRole("searchbox", { name: "Search workers" }), { target: { value: "vision" } });
    fireEvent.change(screen.getByRole("combobox", { name: "State" }), { target: { value: "ready" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Runtime" }), { target: { value: "qwen35-rocm" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Sort workers" }), { target: { value: "runtime-asc" } });
    await waitFor(() => expect(window.localStorage.getItem("modeldeck-worker-library-preferences-v1")).toContain('"runtime":"qwen35-rocm"'));

    first.unmount();
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Workers" }));

    expect(await screen.findByRole("searchbox", { name: "Search workers" })).toHaveValue("vision");
    expect(screen.getByRole("combobox", { name: "State" })).toHaveValue("ready");
    expect(screen.getByRole("combobox", { name: "Runtime" })).toHaveValue("qwen35-rocm");
    expect(screen.getByRole("combobox", { name: "Sort workers" })).toHaveValue("runtime-asc");
    expect(screen.getByRole("status")).toHaveTextContent("1 of 2 Workers");
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

  it("collapses every view and remembers that preference", async () => {
    mockFetch(responses(true));
    const first = render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Collapse all" }));
    expect(screen.getByRole("button", { name: "Expand all" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Expand Live Routes" })).toHaveAttribute("aria-expanded", "false");
    await waitFor(() => expect(window.localStorage.getItem("modeldeck-collapse-preferences-v1")).toContain('"allCollapsed":true'));

    first.unmount();
    mockFetch(responses(true));
    render(<App />);

    expect(await screen.findByRole("button", { name: "Expand all" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("link", { name: "Workers" }));
    expect(await screen.findByRole("button", { name: "Expand Worker Qwen token trace" })).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(screen.getByRole("link", { name: "Advanced" }));
    expect(await screen.findByRole("button", { name: "Expand Detected hardware" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("button", { name: "Expand Worker logs" })).toHaveAttribute("aria-expanded", "false");
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

  it("remembers Model search and sorting after a reload", async () => {
    const payloads = responses();
    payloads["/api/catalogue"] = { models: [
      catalogueModel("Qwen/Qwen3.5-4B"),
      catalogueModel("Example/Vision-Model", ["image-input"]),
    ], downloads_started: false };
    mockFetch(payloads);
    const first = render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Models" }));
    fireEvent.change(screen.getByRole("searchbox", { name: "Search models" }), { target: { value: "qwen" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Sort models" }), { target: { value: "size-desc" } });
    await waitFor(() => expect(window.localStorage.getItem("modeldeck-model-library-preferences-v1")).toContain('"sort":"size-desc"'));

    first.unmount();
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Models" }));

    expect(await screen.findByRole("searchbox", { name: "Search models" })).toHaveValue("qwen");
    expect(screen.getByRole("combobox", { name: "Sort models" })).toHaveValue("size-desc");
    expect(screen.getByRole("heading", { name: "Qwen/Qwen3.5-4B" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Example/Vision-Model" })).not.toBeInTheDocument();
  });

  it("keeps the Models catalogue open while collapsing individual Model cards", async () => {
    const payloads = responses();
    payloads["/api/catalogue"] = { models: [catalogueModel("Qwen/Qwen2.5-1.5B-Instruct")], downloads_started: false };
    mockFetch(payloads);
    render(<App />);

    fireEvent.click(await screen.findByRole("link", { name: "Models" }));
    expect(screen.queryByRole("button", { name: "Collapse Discovered Models" })).not.toBeInTheDocument();
    const collapse = screen.getByRole("button", { name: "Collapse Model Qwen/Qwen2.5-1.5B-Instruct" });
    fireEvent.click(collapse);

    expect(screen.getByRole("heading", { name: "Qwen/Qwen2.5-1.5B-Instruct" })).toBeVisible();
    expect(screen.getByText("recognised")).toBeVisible();
    expect(screen.getByText("Configured Workers")).not.toBeVisible();
    expect(screen.getByRole("button", { name: "Expand Model Qwen/Qwen2.5-1.5B-Instruct" })).toHaveAttribute("aria-expanded", "false");
  });

  it("shows configured Worker identities and states prominently on Model cards", async () => {
    const sceneWorker: Worker = {
      ...worker,
      id: "f5eaeff6-7142-4bbb-b1ba-869549602cd2",
      name: "SceneChat Gemma visual 140",
      model_id: "google/gemma-4-12B-it",
      generation_family: "vision-language",
      runtime: "vision-language-transformers-rocm",
      settings: { visual_token_budget: 140 },
    };
    const model = {
      ...catalogueModel("google/gemma-4-12B-it", ["image-input", "structured-output"]),
      generation_family_hint: "vision-language",
      configuration_support: "scenechat-gemma4",
      worker_count: 1,
    };
    const payloads = responses();
    payloads["/api/workers"] = [sceneWorker];
    payloads["/api/catalogue"] = { models: [model], downloads_started: false };
    mockFetch(payloads);
    render(<App />);

    fireEvent.click(await screen.findByRole("link", { name: "Models" }));
    const summary = screen.getByRole("region", { name: "Workers for google/gemma-4-12B-it" });
    expect(within(summary).getByText("Configured Workers")).toBeInTheDocument();
    expect(within(summary).getByText("1 configured")).toBeInTheDocument();
    expect(within(summary).getByText("SceneChat Gemma visual 140")).toBeInTheDocument();
    expect(within(summary).getByText("stopped")).toBeInTheDocument();
    expect(within(summary).getByText(/140 visual tokens/)).toBeInTheDocument();
  });

  it("removes a stopped Worker from its Model card without deleting the cached Model", async () => {
    const model = catalogueModel(worker.model_id);
    const payloads = responses(true);
    payloads["/api/catalogue"] = { models: [{ ...model, worker_count: 1 }], downloads_started: false };
    payloads[`/api/workers/${worker.id}`] = { ok: true, worker_id: worker.id, cache_removed: false };
    const fetchMock = mockFetch(payloads);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<App />);

    fireEvent.click(await screen.findByRole("link", { name: "Models" }));
    fireEvent.click(screen.getByRole("button", { name: `Remove Worker ${worker.name}` }));

    expect(confirm).toHaveBeenCalledWith(expect.stringMatching(/cached Model files will be kept/i));
    await waitFor(() => expect(fetchMock.mock.calls.some(([input, init]) =>
      String(input) === `/api/workers/${worker.id}` && init?.method === "DELETE"
    )).toBe(true));
    expect(await screen.findByText(/its cached Model was kept/i)).toBeInTheDocument();
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
    expect(screen.getByRole("textbox", { name: "Route Label" })).toHaveValue("Token trace");
    expect(screen.getByRole("textbox", { name: "API Model ID" })).toHaveValue("qwen-0-5b");
    expect(screen.getByText(/Sent by clients in the/)).toHaveTextContent("Sent by clients in the model field and must be unique within this Event.");
    const demo = screen.getByRole("article", { name: "Demo Token Trails" });
    expect(within(demo).getByRole("checkbox", { name: /Token trace/ })).toBeChecked();
    expect(screen.getByRole("heading", { name: "Routes" })).toBeInTheDocument();
    expect(screen.getByText("Configure the Routes available to this Event. A Route can be used by multiple Demos or remain unassigned.")).toBeInTheDocument();
    expect(screen.getByText("Every shared Route is used by at least one Demo.")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/Saved/)).toBeInTheDocument());
  });

  it("confirms every Event editor removal before changing the draft", async () => {
    const backup: Worker = {
      ...worker,
      id: "d054e57f-b1fd-4575-8f55-9cfaf1f55380",
      name: "Trace backup",
      port: 8632,
    };
    const eventWithBackup: EventRecord = {
      ...eventRecord,
      definition: {
        ...eventRecord.definition,
        routes: [{ ...eventRecord.definition.routes[0], worker_ids: [worker.id, backup.id] }],
      },
    };
    const payloads = responses(true);
    payloads["/api/workers"] = [worker, backup];
    payloads["/api/events"] = { events: [eventWithBackup] };
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Events" }));

    fireEvent.click(screen.getByRole("button", { name: "Remove Demo Token Trails" }));
    fireEvent.click(screen.getByRole("button", { name: "Remove Route Token trace" }));
    fireEvent.click(screen.getByRole("button", { name: "Remove Backup 1 Worker Trace backup from Route Token trace" }));

    expect(confirm).toHaveBeenNthCalledWith(1, expect.stringMatching(/Remove Demo “Token Trails”/));
    expect(confirm).toHaveBeenNthCalledWith(2, expect.stringMatching(/removed from every Demo/));
    expect(confirm).toHaveBeenNthCalledWith(3, expect.stringMatching(/Worker itself will be kept/));
    expect(screen.getByRole("article", { name: "Demo Token Trails" })).toBeInTheDocument();
    expect(screen.getByRole("article", { name: "Route Token trace" })).toBeInTheDocument();
    expect(screen.getByText("Backup 1")).toBeInTheDocument();
  });

  it("identifies the Route, Worker role and Worker details for validation issues", async () => {
    const payloads = responses(true);
    payloads[`/api/events/${eventRecord.definition.id}/draft`] = eventRecord;
    payloads[`/api/events/${eventRecord.definition.id}/validate`] = {
      valid: false,
      errors: [
        { route_id: eventRecord.definition.routes[0].id, worker_id: worker.id, message: "Requires text-diffusion, got autoregressive" },
        { route_id: eventRecord.definition.routes[0].id, worker_id: worker.id, message: "Missing capabilities: image_input" },
      ],
      warnings: [
        { route_id: eventRecord.definition.routes[0].id, message: "Route 'Token trace' is not used by a Demo" },
      ],
      routes: [],
    };
    const fetchMock = mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Events" }));
    fireEvent.change(await screen.findByRole("textbox", { name: "Route Label" }), { target: { value: "Updated token trace" } });
    fireEvent.click(await screen.findByRole("button", { name: "Validate" }));

    const errors = await screen.findByRole("region", { name: "Validation errors" });
    expect(within(errors).getAllByText("Routes → Updated token trace → Worker order → Primary Worker")).toHaveLength(2);
    expect(within(errors).getAllByText("qwen-0-5b")).toHaveLength(2);
    expect(within(errors).getAllByText(/Qwen token trace · Qwen\/Qwen2.5-0.5B-Instruct · transformers-rocm/)).toHaveLength(2);
    expect(within(errors).getByText("Missing capabilities: image_input")).toBeInTheDocument();

    const notes = screen.getByRole("region", { name: "Validation notes" });
    expect(within(notes).getByText("Routes → Updated token trace")).toBeInTheDocument();
    expect(within(notes).getByText("Note: Route 'Token trace' is not used by a Demo")).toBeInTheDocument();

    const draftWriteIndex = fetchMock.mock.calls.findIndex(([input, init]) => String(input).endsWith("/draft") && init?.method === "PUT");
    const validationIndex = fetchMock.mock.calls.findIndex(([input]) => String(input).endsWith("/validate"));
    expect(draftWriteIndex).toBeGreaterThanOrEqual(0);
    expect(validationIndex).toBeGreaterThan(draftWriteIndex);
    expect(JSON.parse(String(fetchMock.mock.calls[draftWriteIndex][1]?.body)).routes[0].display_name).toBe("Updated token trace");
  });

  it("marks and disables incompatible Workers for a Route contract", async () => {
    const diffusionWorker: Worker = {
      ...worker,
      id: "7a19b667-8efc-4440-a60f-b3b17b6ece55",
      name: "DiffusionGemma Q4",
      model_id: "google/diffusiongemma-26B-A4B-it",
      generation_family: "text-diffusion",
      runtime: "text-diffusion-gptq-rocm",
      capabilities: { iterative_refinement: true, intermediate_frames: true },
      port: 8632,
    };
    const payloads = responses(true);
    payloads["/api/workers"] = [worker, diffusionWorker];
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Events" }));

    const route = await screen.findByRole("article", { name: "Route Token trace" });
    expect(within(route).getByText(/Native autoregressive trace requires an autoregressive Worker with top k trace/)).toBeInTheDocument();
    expect(within(route).getByRole("option", { name: /DiffusionGemma Q4.*incompatible/ })).toBeDisabled();
  });

  it("identifies duplicate API Model IDs before autosave", async () => {
    const qwenRoute = {
      ...eventRecord.definition.routes[0],
      id: "f50dcfc1-b0b5-460c-94bd-bfc0933145fd",
      display_name: "Qwen3.5 0.8B 70",
      public_name: "qwen3-5-0-8b-70",
    };
    const record = {
      ...eventRecord,
      definition: {
        ...eventRecord.definition,
        routes: [...eventRecord.definition.routes, qwenRoute],
      },
    };
    const payloads = responses(true);
    payloads["/api/events"] = { events: [record] };
    const fetchMock = mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Events" }));

    const qwenCard = await screen.findByRole("article", { name: "Route Qwen3.5 0.8B 70" });
    fireEvent.change(within(qwenCard).getByRole("textbox", { name: "API Model ID" }), { target: { value: "qwen-0-5b" } });

    expect(within(qwenCard).getByText("“qwen-0-5b” is already used by Route “Token trace”. Choose a unique API Model ID.")).toBeInTheDocument();
    expect(within(qwenCard).getByRole("textbox", { name: "API Model ID" })).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("button", { name: "Validate" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Publish routing" })).toBeDisabled();
    expect(screen.getByText(/Needs attention/)).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([input, init]) => String(input).endsWith("/draft") && init?.method === "PUT")).toBe(false);
  });

  it("shows structured validation details when publishing fails", async () => {
    const validation = {
      valid: false,
      errors: [{ route_id: eventRecord.definition.routes[0].id, worker_id: worker.id, message: "Missing capabilities: chat" }],
      warnings: [],
      routes: [],
    };
    const payloads = responses(true);
    payloads[`/api/events/${eventRecord.definition.id}/draft`] = eventRecord;
    payloads[`/api/events/${eventRecord.definition.id}/publish`] = () => new Response(JSON.stringify({
      detail: { message: "Event validation failed", validation },
    }), { status: 409, headers: { "Content-Type": "application/json" } });
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Events" }));
    fireEvent.click(await screen.findByRole("button", { name: "Publish routing" }));

    const errors = await screen.findByRole("region", { name: "Validation errors" });
    expect(within(errors).getByText("Routes → Token trace → Worker order → Primary Worker")).toBeInTheDocument();
    expect(within(errors).getByText("Missing capabilities: chat")).toBeInTheDocument();
  });

  it("shows Routes outside every Demo as unassigned until they are included", async () => {
    const unassignedRoute = {
      ...eventRecord.definition.routes[0],
      id: "f50dcfc1-b0b5-460c-94bd-bfc0933145fd",
      display_name: "Unassigned experiment",
      public_name: "experimental-model",
    };
    const record = {
      ...eventRecord,
      definition: {
        ...eventRecord.definition,
        routes: [...eventRecord.definition.routes, unassignedRoute],
      },
    };
    const payloads = responses(true);
    payloads["/api/events"] = { events: [record] };
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Events" }));

    const unassigned = await screen.findByRole("region", { name: "Unassigned Routes" });
    expect(within(unassigned).getByText("Unassigned experiment")).toBeInTheDocument();
    const demo = screen.getByRole("article", { name: "Demo Token Trails" });
    fireEvent.click(within(demo).getByRole("checkbox", { name: /Unassigned experiment/ }));

    expect(within(unassigned).queryByText("Unassigned experiment")).not.toBeInTheDocument();
    expect(within(unassigned).getByText("Every shared Route is used by at least one Demo.")).toBeInTheDocument();
  });

  it("collapses Event Demo and Route levels and remembers individual cards", async () => {
    mockFetch(responses(true));
    const first = render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Events" }));

    const demo = await screen.findByRole("article", { name: "Demo Token Trails" });
    fireEvent.click(within(demo).getByRole("button", { name: "Collapse Demo Token Trails" }));
    expect(within(demo).getByText("Routes used by this Demo")).not.toBeVisible();

    const route = screen.getByRole("article", { name: "Route Token trace" });
    fireEvent.click(within(route).getByRole("button", { name: "Collapse Route Token trace" }));
    expect(within(route).getByRole("textbox", { name: "Route Label", hidden: true })).not.toBeVisible();
    await waitFor(() => expect(window.localStorage.getItem("modeldeck-collapse-preferences-v1")).toContain(`event-route-${eventRecord.definition.id}-${eventRecord.definition.routes[0].id}`));

    fireEvent.click(screen.getByRole("button", { name: "Collapse Demos" }));
    expect(screen.getByRole("article", { name: "Demo Token Trails", hidden: true })).not.toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Expand Demos" }));
    expect(screen.getByRole("button", { name: "Expand Demo Token Trails" })).toBeInTheDocument();

    first.unmount();
    mockFetch(responses(true));
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Events" }));
    expect(await screen.findByRole("button", { name: "Expand Demo Token Trails" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Expand Route Token trace" })).toBeInTheDocument();
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
      settings: { context_length: 8192, maximum_new_tokens: 512, visual_token_budget: 280 },
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
    expect(screen.getByRole("combobox", { name: "Visual token budget" })).toHaveValue("280");
    fireEvent.change(screen.getByRole("spinbutton", { name: "Context length" }), { target: { value: "4096" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Maximum output" }), { target: { value: "256" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Visual token budget" }), { target: { value: "140" } });
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
        visual_token_budget: 140,
      });
    });
  });

  it("uses the pinned OPUS CPU runtime without irrelevant generation controls", async () => {
    const payloads = responses();
    payloads["/api/catalogue"] = { models: [{
      ...catalogueModel("Helsinki-NLP/opus-mt-en-fr", ["translation"]),
      generation_family_hint: "text-translation",
      configuration_support: "opus-translation-cpu",
    }], downloads_started: false };
    payloads["/api/runtime-templates"] = { templates: [{
      id: "opus-translation-cpu", display_name: "OPUS translation CPU",
      implementation: "marian-transformers-cpu", generation_family: "text-translation",
      cache_setting: "cache_root", uses_base_model_identity: false,
      lifecycle: "resident", dtype: "float32",
      settings: { maximum_input_characters: 4000, maximum_input_tokens: 512 },
      package_id: "modeldeck-core", package_version: "1", package_display_name: "Core",
      publisher: "ModelDeck", source: "packaged", digest: "digest",
    }] };
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Models" }));
    fireEvent.click(screen.getByRole("button", { name: "Create Worker" }));

    expect(screen.getByRole("combobox", { name: "Data type" })).toHaveValue("float32");
    expect(screen.getByRole("combobox", { name: "Data type" })).toBeDisabled();
    expect(screen.queryByRole("spinbutton", { name: "Maximum output" })).not.toBeInTheDocument();
  });

  it("keeps Qwen TTS voice synthesis controls contract-owned", async () => {
    const payloads = responses();
    payloads["/api/catalogue"] = { models: [{
      ...catalogueModel("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", ["speech-synthesis", "audio-output"]),
      generation_family_hint: "speech-synthesis",
      configuration_support: "qwen3-tts-rocm",
    }], downloads_started: false };
    payloads["/api/runtime-templates"] = { templates: [{
      id: "qwen3-tts-rocm", display_name: "Qwen3-TTS ROCm",
      implementation: "qwen3-tts-rocm", generation_family: "speech-synthesis",
      cache_setting: "cache_root", uses_base_model_identity: false,
      lifecycle: "resident", dtype: "bfloat16",
      settings: { sample_rate_hz: 24000, maximum_audio_seconds: 90 },
      package_id: "modeldeck-core", package_version: "1", package_display_name: "Core",
      publisher: "ModelDeck", source: "packaged", digest: "digest",
    }] };
    mockFetch(payloads);
    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Models" }));
    fireEvent.click(screen.getByRole("button", { name: "Create Worker" }));

    expect(screen.getByRole("combobox", { name: "Data type" })).toHaveValue("bfloat16");
    expect(screen.queryByRole("spinbutton", { name: "Maximum output" })).not.toBeInTheDocument();
    expect(screen.getByText(/Sampling controls such as temperature/)).toBeInTheDocument();
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

export type WorkerState =
  | "stopped" | "validating" | "starting" | "loading" | "warming" | "ready"
  | "busy" | "degraded" | "stopping" | "failed" | "incompatible" | "archived";

export interface Capabilities { [name: string]: boolean | string }

export interface Worker {
  id: string;
  name: string;
  state: WorkerState;
  model_id: string;
  revision: string;
  artifact_model_id: string | null;
  artifact_revision: string | null;
  generation_family: string;
  runtime: string;
  runtime_template_id: string | null;
  runtime_template_version: string | null;
  lifecycle: "resident" | "on-demand" | "exclusive";
  port: number;
  dtype: string;
  capabilities: Capabilities;
  settings: Record<string, string | number | boolean>;
  endpoint: string | null;
  pid: number | null;
  started_at: string | null;
  last_error: string | null;
  archived: boolean;
  created_at: string;
  updated_at: string;
  archived_at: string | null;
}

export interface Demo { id: string; name: string; route_ids: string[] }
export interface Route {
  id: string;
  display_name: string;
  public_name: string;
  protocol_contract: string;
  worker_ids: string[];
}
export interface EventDefinition {
  id: string;
  name: string;
  description: string;
  qualification: "compatible" | "tested-working";
  demos: Demo[];
  routes: Route[];
}
export interface EventRecord {
  definition: EventDefinition;
  created_at: string;
  updated_at: string;
  active: boolean;
  active_revision: number | null;
  latest_revision: number | null;
}
export interface EventValidation {
  valid: boolean;
  errors: Array<{ route_id?: string; worker_id?: string; message: string }>;
  warnings: Array<{ route_id?: string; demo_id?: string; message: string }>;
}
export interface EventRevision {
  definition: EventDefinition;
  revision: number;
  published_at: string;
  active: boolean;
}

export interface ProtocolContract {
  id: string;
  display_name: string;
  generation_family: string;
  required_capabilities: string[];
  surfaces: string[];
}

export type MockScenario = "success" | "delayed" | "request-error";
export interface MockWorkerOption {
  id: "visual_token_budget";
  label: string;
  type: "select";
  default: number;
  choices: number[];
}
export interface MockWorkerTemplate {
  id: string;
  protocol_contract: string;
  display_name: string;
  generation_family: string;
  default_name: string;
  scenarios: MockScenario[];
  options: MockWorkerOption[];
}

export interface LiveWorker { id: string; name: string; state: WorkerState }
export interface LiveRoute extends Route {
  workers: Worker[];
  effective_worker: Worker | null;
  ready: boolean;
}
export interface LiveState {
  active_event: { id: string; name: string; revision: number } | null;
  routes: LiveRoute[];
}

export interface RuntimeTemplate {
  id: string;
  display_name: string;
  implementation: string;
  generation_family: string;
  cache_setting: "cache_root" | "q4_checkpoint_dir" | "artifact_path";
  uses_base_model_identity: boolean;
  lifecycle: "resident" | "on-demand" | "exclusive" | null;
  dtype: "float16" | "bfloat16" | null;
  settings: Record<string, string | number | boolean>;
  package_id: string;
  package_version: string;
  package_display_name: string;
  publisher: string;
  source: "packaged" | "trusted-local";
  digest: string;
}

export interface ModelArtifact { artifact_id: string; kind: "gguf"; format: string; filenames: string[] }
export interface ModelEntry {
  model_id: string;
  revision: string | null;
  cache_location: string;
  physical_size_bytes: number;
  download_state: "partial" | "installed-untested";
  generation_family_hint: string | null;
  capability_hints: string[];
  configuration_support: string | null;
  configuration_support_reason: string;
  modeldeck_allowed: boolean;
  snapshot_location: string | null;
  base_model_id: string | null;
  base_model_revision: string | null;
  runnable: boolean;
  runnable_reason: string;
  worker_count: number;
  artifacts?: ModelArtifact[];
}

export interface CompatibilityTest {
  id: number;
  fingerprint: string;
  result: string;
  failure_class: string | null;
  evidence: Record<string, unknown>;
  tested_at: string;
}

export interface GatewayStatus {
  available: boolean;
  health: { status: string; ready_workers: number } | null;
  models: { data: Array<{ id: string; ready: boolean }> } | null;
  routes: { routes: Array<{ public_name: string; ready: boolean }> } | null;
  error: string | null;
}

export interface ManagementHealth {
  status: string;
  service: string;
  schema_version: number;
  open_day: boolean;
  downloads_allowed: boolean;
  gateway_url: string;
}

export interface MemoryReading { total_bytes: number; available_bytes: number; percent: number }
export interface SwapReading { total_bytes: number; used_bytes: number; percent: number }
export interface FilesystemReading {
  path: string; available: boolean; total_bytes?: number; used_bytes?: number;
  free_bytes?: number; percent?: number;
}
export interface TemperatureReading { source: string; label: string; celsius: number }
export interface FanReading { source: string; label: string; rpm: number }
export interface ProcessReading { pid: number; name: string | null; command: string }
export interface Telemetry {
  memory: MemoryReading; swap: SwapReading; filesystems: FilesystemReading[];
  temperatures: TemperatureReading[]; fans: FanReading[]; active_model_processes: ProcessReading[];
}
export interface HardwareProbe {
  configured: { profile_id: string; os: string; gpu: string; gpu_architecture: string; rocm_family: string; work_mount: string };
  detected: {
    fedora_release: string | null; kernel: string; python: string; rocm_packages: string[];
    gpu_device_nodes: Record<string, boolean>; memory: MemoryReading; swap: SwapReading;
    filesystems: FilesystemReading[]; temperatures: TemperatureReading[]; fans: FanReading[];
    active_model_processes: ProcessReading[];
  };
  diagnostic_note: string;
}

export interface WorkerLog {
  timestamp: string; source: string; level: "info" | "warning" | "error";
  message: string; session_id?: string;
}

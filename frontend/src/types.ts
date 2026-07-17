export type WorkerState =
  | "discovered"
  | "stopped"
  | "validating"
  | "starting"
  | "loading"
  | "warming"
  | "ready"
  | "busy"
  | "degraded"
  | "stopping"
  | "failed"
  | "orphaned"
  | "incompatible";

export interface Capabilities {
  chat: boolean | "compatibility-only";
  completions: boolean;
  streaming: boolean;
  cancellation: boolean;
  logits: boolean | "model-specific";
  top_k_trace: boolean;
  hidden_states: boolean | "optional";
  iterative_refinement: boolean;
  intermediate_frames: boolean;
  seeded_generation: boolean;
  image_input: boolean;
  structured_output: boolean;
}

export interface Worker {
  id: string;
  state: WorkerState;
  model_id: string;
  generation_family: string;
  runtime: string;
  lifecycle: "resident" | "on-demand" | "exclusive";
  alias: string;
  endpoint: string;
  port: number;
  pid: number | null;
  started_at: string | null;
  last_error: string | null;
  capabilities: Capabilities;
}

export interface Profile {
  id: string;
  model_id: string;
  revision: string;
  artifact_model_id: string | null;
  artifact_revision: string | null;
  alias: string;
  generation_family: string;
  preferred_runtime: string;
  lifecycle: "resident" | "on-demand" | "exclusive";
  port: number;
  local_files_only: boolean;
  trust_remote_code: boolean;
  dtype: string;
  capabilities: Capabilities;
  settings: Record<string, string | number | boolean>;
  source: "built-in" | "local";
  modeldeck_allowed: boolean;
}

export interface LocalProfileRequest {
  model_id: string;
  revision: string;
  alias: string;
  dtype: "float16" | "bfloat16";
  lifecycle: "resident" | "on-demand" | "exclusive";
  context_length: number;
  maximum_new_tokens: number;
  maximum_denoising_steps: number;
}

export interface ModelEntry {
  model_id: string;
  revision: string | null;
  cache_location: string;
  physical_size_bytes: number;
  download_state: "partial" | "installed-untested";
  generation_family_hint: string | null;
  configuration_support:
    | "autoregressive-transformers"
    | "scenechat-gemma4"
    | "diffusiongemma-transformers"
    | "diffusiongemma-modeldeck-q4"
    | null;
  configuration_support_reason: string;
  modeldeck_allowed: boolean;
  snapshot_location: string | null;
  base_model_id: string | null;
  base_model_revision: string | null;
  runnable: boolean;
  runnable_reason: string;
}

export interface HardwareProbe {
  configured: {
    profile_id: string;
    os: string;
    gpu: string;
    gpu_architecture: string;
    rocm_family: string;
    work_mount: string;
  };
  detected: {
    fedora_release: string | null;
    kernel: string;
    python: string;
    rocm_packages: string[];
    gpu_device_nodes: Record<string, boolean>;
    memory: MemoryReading;
    swap: SwapReading;
    filesystems: FilesystemReading[];
    temperatures: TemperatureReading[];
    fans: FanReading[];
    active_model_processes: ProcessReading[];
  };
  diagnostic_note: string;
}

export interface MemoryReading {
  total_bytes: number;
  available_bytes: number;
  percent: number;
}

export interface SwapReading {
  total_bytes: number;
  used_bytes: number;
  percent: number;
}

export interface FilesystemReading {
  path: string;
  available: boolean;
  total_bytes?: number;
  used_bytes?: number;
  free_bytes?: number;
  percent?: number;
}

export interface TemperatureReading {
  source: string;
  label: string;
  celsius: number;
}

export interface FanReading {
  source: string;
  label: string;
  rpm: number;
}

export interface ProcessReading {
  pid: number;
  name: string | null;
  command: string;
}

export interface Telemetry {
  memory: MemoryReading;
  swap: SwapReading;
  filesystems: FilesystemReading[];
  temperatures: TemperatureReading[];
  fans: FanReading[];
  active_model_processes: ProcessReading[];
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
  health: { status: string; ready_providers: number } | null;
  models: { data: Array<{ id: string; ready: boolean; effective_provider: string | null }> } | null;
  providers: { providers: Array<{ id: string; alias: string; ready: boolean }> } | null;
  error: string | null;
}

export interface ManagementHealth {
  status: string;
  service: string;
  open_day: boolean;
  downloads_allowed: boolean;
  gateway_url: string;
}

export interface WorkerLog {
  timestamp: string;
  source: string;
  level: "info" | "warning" | "error";
  message: string;
  session_id?: string;
}

export interface WorkerEvent {
  worker_id: string;
  state: WorkerState;
  message: string;
  timestamp: string;
}

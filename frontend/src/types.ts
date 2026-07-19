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
  audio_input: boolean;
  audio_output: boolean;
  full_duplex: boolean;
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

export interface Deployment {
  id: string;
  source: "packaged" | "local";
  model: {
    model_id: string;
    revision: string;
    artifact_model_id: string | null;
    artifact_revision: string | null;
  };
  runtime: string;
  generation_family: string;
  lifecycle: "resident" | "on-demand" | "exclusive";
  capabilities: Capabilities;
  allowed: boolean;
  registered: boolean;
  worker: Worker | null;
}

export interface DemoAdapter {
  id: string;
  display_name: string;
  generation_family: string;
  required_capabilities: string[];
  surfaces: string[];
}

export interface DemoApplication {
  id: string;
  display_name: string;
}

export interface DeploymentBinding {
  deployment_id: string;
  priority: number;
}

export interface DemoRouteContract {
  id: string;
  demo_id: string;
  display_name: string;
  adapter_id: string;
  public_model: string;
  qualification_policy: "registered" | "tested-working-recorded";
  fallback_policy: "none" | "ordered" | "mock-visible" | "structured-unavailable";
  providers: DeploymentBinding[];
}

export interface DemoSet {
  id: string;
  display_name: string;
  description: string;
  demos: DemoApplication[];
  routes: DemoRouteContract[];
  revision: number;
  updated_at: string;
  active: boolean;
  active_revision: number | null;
}

export interface DemoSetValidation {
  valid: boolean;
  errors: Array<{ route_id?: string; deployment_id?: string; message: string }>;
  warnings: Array<{ route_id?: string; message: string }>;
}

export interface DemoSetPlan {
  desired_primary_deployments: string[];
  start_required: string[];
  stop_required: string[];
  warnings: string[];
  applies_process_changes: boolean;
}

export interface LocalProfileRequest {
  model_id: string;
  revision: string;
  alias: string;
  profile_name?: string;
  dtype: "float16" | "bfloat16";
  lifecycle: "resident" | "on-demand" | "exclusive";
  context_length: number;
  maximum_new_tokens: number;
  maximum_denoising_steps: number;
  artifact_id?: string;
}

export interface ModelArtifact {
  artifact_id: string;
  kind: "gguf";
  format: string;
  filenames: string[];
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
    | "gpt-oss-llama-vulkan"
    | "moshiko-speech"
    | null;
  configuration_support_reason: string;
  modeldeck_allowed: boolean;
  snapshot_location: string | null;
  base_model_id: string | null;
  base_model_revision: string | null;
  runnable: boolean;
  runnable_reason: string;
  artifacts?: ModelArtifact[];
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
  models: { data: Array<{ id: string; ready: boolean; selected_provider: string | null; effective_provider: string | null }> } | null;
  providers: { providers: Array<{ id: string; alias: string; ready: boolean }> } | null;
  error: string | null;
}

export interface ProviderCandidate {
  profile_id: string;
  profile_alias: string;
  model_id: string;
  selected: boolean;
  worker_state: WorkerState;
  gateway_ready: boolean;
}

export interface ProviderSelection {
  alias: string;
  display_name: string;
  default_provider: string | null;
  explicit_selection: boolean;
  selected_provider: string;
  effective_provider: string | null;
  gateway_ready: boolean;
  candidates: ProviderCandidate[];
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

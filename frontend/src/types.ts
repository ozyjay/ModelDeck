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
  capability_policy_version?: number | null;
  endpoint: string | null;
  pid: number | null;
  started_at: string | null;
  last_error: string | null;
  archived: boolean;
  created_at: string;
  updated_at: string;
  archived_at: string | null;
}

export interface CapabilityBinding {
  id: string;
  display_name: string;
  public_name: string;
  protocol_contract: string;
  tool_calling_enabled: boolean;
  worker_ids: string[];
}
export interface RoutingProfile {
  id: string;
  name: string;
  description: string;
  qualification: "compatible" | "tested-working";
  capabilities: CapabilityBinding[];
}
export interface RoutingProfileRecord {
  definition: RoutingProfile;
  created_at: string;
  updated_at: string;
  active: boolean;
  active_revision: number | null;
  latest_revision: number | null;
}
export interface RoutingProfileValidation {
  valid: boolean;
  errors: Array<{ capability_id?: string; worker_id?: string; message: string }>;
  warnings: Array<{ capability_id?: string; message: string }>;
}
export interface RoutingProfileRevision {
  definition: RoutingProfile;
  revision: number;
  published_at: string;
  active: boolean;
}

export interface ProtocolContract {
  id: string;
  display_name: string;
  generation_family: string;
  compatible_generation_families: string[];
  required_capabilities: string[];
  required_worker_settings: Record<string, string | number>;
  surfaces: string[];
}

export interface LiveWorker { id: string; name: string; state: WorkerState }
export interface ToolCallingState {
  supported: boolean;
  rehearsed: boolean;
  last_rehearsal: string | null;
  failure_code: string | null;
}
export interface LiveCapability extends CapabilityBinding {
  profile_id?: string;
  tool_calling?: ToolCallingState;
  workers: Worker[];
  effective_worker: Worker | null;
  ready: boolean;
}
export interface LiveState {
  active_profile: { id: string; name: string; revision: number } | null;
  active_profiles: { id: string; name: string; revision: number }[];
  capabilities: LiveCapability[];
}

export interface RuntimeTemplate {
  id: string;
  display_name: string;
  implementation: string;
  generation_family: string;
  cache_setting: "cache_root" | "q4_checkpoint_dir" | "artifact_path";
  uses_base_model_identity: boolean;
  lifecycle: "resident" | "on-demand" | "exclusive" | null;
  dtype: "float16" | "bfloat16" | "float32" | null;
  settings: Record<string, string | number | boolean>;
  package_id: string;
  package_version: string;
  package_display_name: string;
  publisher: string;
  source: "packaged" | "trusted-local";
  digest: string;
}

export interface RuntimeInstallation {
  installation_id: string;
  display_name: string;
  integrity_status: "verified" | "missing" | "receipt-missing" | "receipt-invalid" | "modified" | "feature-mismatch" | "inspection-failed";
  currency_status: "recommended" | "accepted-older" | "accepted-alternative" | "different-unqualified" | "newer-unqualified" | "revoked" | "unknown";
  start_allowed: boolean;
  detected: {
    source_revision: string | null;
    executable_sha256: string | null;
    executable_size_bytes: number | null;
    receipt_sha256: string | null;
    receipt_version: number | null;
    backend: string | null;
    operating_system: string;
    architecture: string;
    version_output: string | null;
  };
  recommended_source_revision: string;
  required_features: string[];
  missing_features: string[];
  reason_codes: string[];
  inspected_at: string;
  implementation_ids: string[];
  runtime_template_ids: string[];
  worker_ids: string[];
}

export interface ModelArtifact { artifact_id: string; kind: "gguf"; format: string; filenames: string[] }
export interface CapabilityEvidence {
  kind: "detected" | "asserted";
  confidence: "direct" | "inferred";
  source: string;
  detail: string;
  reference?: string;
  reviewed_at?: string;
}
export interface PotentialCapability {
  id: string;
  display_name: string;
  description: string;
  protocol_contract_id: string | null;
  traits: string[];
  evidence: CapabilityEvidence[];
  runtime_template_ids: string[];
  available_runtime_template_ids: string[];
  policy_allowed: boolean;
  effective_allowed: boolean;
  runtime_status: "available" | "missing";
  qualification_status: "not-tested" | "qualified" | "failed" | "stale" | "legacy";
  qualifying_workers: Array<{ worker_id: string; worker_name: string; evidence_id: number | null; status: string }>;
  published: boolean;
  creatable: boolean;
  reason: string;
}
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
  potential_capabilities: PotentialCapability[];
  artifacts?: ModelArtifact[];
  candidate_registration?: {
    eligible: boolean;
    approved: boolean;
    candidate_id: string | null;
    filename: string | null;
    expected_size: number | null;
    expected_sha256: string | null;
    reason: string;
  } | null;
}

export interface CompatibilityTest {
  id: number;
  fingerprint: string;
  result: string;
  failure_class: string | null;
  evidence: Record<string, unknown>;
  tested_at: string;
  fingerprint_version?: number;
  evidence_status?: "current" | "legacy";
  observations?: CompatibilityObservation[];
}

export interface CompatibilityObservation {
  id: number;
  test_id: number;
  kind: string;
  observation: Record<string, unknown>;
  observed_at: string;
}

export interface CapabilitySetupSelection {
  capability_id: string;
  model_id: string;
  revision: string;
  worker_name: string;
  artifact_id?: string | null;
  runtime_template_id?: string | null;
  dtype?: "float16" | "bfloat16" | "float32" | null;
  lifecycle?: "resident" | "on-demand" | "exclusive" | null;
  context_length?: number | null;
  maximum_new_tokens?: number | null;
  maximum_denoising_steps?: number | null;
  visual_token_budget?: number | null;
  prefix_cache_enabled?: boolean;
}

export interface CapabilitySetupPreview {
  selection: CapabilitySetupSelection;
  worker: Record<string, unknown>;
  selection_basis: string;
  runtime_registration_digest: string;
  policy_changes: { model_allowed: boolean; capability_allowed: boolean };
  warnings: string[];
  preview_fingerprint: string;
}

export interface CapabilitySetup {
  id: string;
  request_id: string;
  state: string;
  current_step: string;
  plan: CapabilitySetupPreview;
  worker_id?: string;
  evidence_id?: number;
  qualification?: CompatibilityTest;
  resolved_identity?: Record<string, unknown>;
  error?: { code: string; message: string; component: string; step: string; retryable: boolean } | null;
  publication?: { profile_id: string; revision: number; public_name: string };
  created_at: string;
  updated_at: string;
}

export interface CapabilityPublicationPreview {
  profile_id: string;
  before: RoutingProfile;
  after: RoutingProfile;
  validation: RoutingProfileValidation;
  publication_fingerprint: string;
}

export interface GatewayStatus {
  available: boolean;
  health: { status: string; ready_workers: number } | null;
  models: { data: Array<{ id: string; ready: boolean }> } | null;
  routes: { routes: Array<{ public_name: string; ready: boolean }> } | null;
  error: string | null;
}

export interface BenchmarkThroughputPoint {
  series_key: string;
  observed_at: string;
  model_id: string;
  model_revision: string;
  runtime: string;
  dtype: string;
  generation_family: string;
  worker_id: string | null;
  worker_name: string | null;
  tokens_per_second: number;
  workload: string;
  configuration_fingerprint: string | null;
  sample_count: number | null;
}

export interface BenchmarkHistory {
  points: BenchmarkThroughputPoint[];
  reports_scanned: number;
  measurement: string;
}

export interface ManagementHealth {
  status: string;
  service: string;
  schema_version: number;
  configuration_locked: boolean;
  offline_only: boolean;
  gateway_url: string;
  state_store: {
    kind: "desktop-standalone" | "checkout-development";
    label: string;
    directory: string;
  };
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
export interface ThermalStatus {
  enabled: boolean;
  state: "normal" | "warm" | "hot" | "very_hot" | "critical" | "telemetry_degraded";
  temperature_c: number | null;
  sensor_id: string | null;
  telemetry_age_seconds: number | null;
  heavy_concurrency_limit: number;
  active_heavy_concurrency: number | null;
  model_load_concurrency_limit: number;
  background_concurrency_limit: number;
  background_paused: boolean;
  model_loading_allowed: boolean;
  scenechat_degradation: {
    active: boolean; minimum_frame_interval_seconds: number; automatic_capture_allowed?: boolean;
  };
  reason_code: string;
  host_power_policy: {
    available: boolean; service_active?: boolean | null; tuned_profile?: string | null;
    control: "external_read_only";
  };
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

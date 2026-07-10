# Build plan and Phase 0 decision package

## Observations and confirmed boundaries

Repository and environment observations, exact reusable symbols, and conflicting
assumptions are recorded in [existing repository findings](EXISTING_REPOSITORY_FINDINGS.md)
and [ROCm evidence](ROCM_FRAMEWORK_DESKTOP.md). The key findings are the system ROCm 7.1.x
versus target 7.2.x mismatch, existing competition for port 8600, mature acquisition in
HuggingFacePull, a persistent diffusion NDJSON worker, and TokenTrail's local-only logits
trace implementation.

The final boundary is management/dashboard, isolated family-specific workers, stable
gateway, read-only catalogue, evidence registry, and later scheduler. Acquisition remains
external. AR and diffusion contracts are separate in [worker protocol](WORKER_PROTOCOL.md).

## Database tables

- `model_profiles(id, document_json, updated_at)`
- `compatibility_tests(id, fingerprint, result, failure_class, evidence_json, tested_at)`
- `worker_events(id, worker_id, state, message, details_json, occurred_at)`
- `presets(id, document_json, updated_at)`

JSON preserves evolving evidence while indexed identity/result columns support current
queries. Migrations will be versioned before user-editable persistence begins.

## Ports

| Role | Port | Evidence / decision |
|---|---:|---|
| ModelDeck dashboard and management | 3600 | Within reserved dashboard range; no observed conflict |
| Stable model gateway | 8600 | Existing TokenTrail trace and TextDiffusion adapter convention; those routes need migration wrappers |
| Mock/first AR worker | 8610 | Start of managed worker range |
| Mock/first diffusion worker | 8611 | Separate process and generation family |
| External vLLM | 8000 | Existing SceneChat/TextDiffusion convention; unmanaged initially |

No random fallback port is used outside tests.

## File tree and frontend choice

The repository follows the proposed `backend/modeldeck`, `profiles`, `fixtures`, `scripts`,
`tests`, and `docs` boundaries. A dependency-free server-rendered dashboard was selected
for this small slice because the brief explicitly permits it and it avoids a frontend
build dependency before runtime lifecycle is proved. A React/Vite split remains an
optional later refinement if dashboard complexity warrants it.

## Phases

1. **Implemented foundation:** evidence docs, project skeleton, read-only dashboard,
   environment/cache/process/telemetry probes, SQLite schema, fixed ports, SSE events.
2. **Implemented lifecycle proof:** allowlisted subprocess supervisor, state machine,
   serial load lock, health/warmup readiness, logs, graceful/forced stop, mock AR and
   diffusion workers, contracts, tests.
3. **AR Transformers — implemented:** pinned isolated Python/ROCm environment,
   cached Qwen 0.5B load, streaming/cancellation/trace/metrics, gateway preference and
   compatibility evidence. The 30-minute stability run completed 343 requests with zero
   failures; real in-flight gateway cancellation, repeated lifecycle, graceful shutdown,
   and process exit passed.
4. **Text diffusion:** adapt proven DiffusionGemma engine behind native job/frame API,
   evidence-gated HSA preload, cancellation and replay.
5. **Scheduler and compatibility execution:** measured memory/reserve/conflicts, lifecycle
   classes, preset transition approval, append-only tests.
6. **Gateway completion:** streaming proxy, cancellation, aliases and explicit local
   alternates, demo adapters.
7. **Optional providers:** one evidence-backed adapter at a time.
8. **Read-only HuggingFacePull integration:** API or metadata contract, including transport
   evidence; no second downloader.
9. **Fedora/Open Day hardening:** launcher/service, frozen revisions, presets, runbook,
   burn-in, privacy review.

## Phase 1/2 changed files and tests

Core files are `main.py`, `hardware/probe.py`, `catalogue/hf_cache.py`,
`compatibility/store.py`, `profiles/models.py`, `supervisor/service.py`,
`workers/mock_worker.py`, and `gateway/app.py`. Supporting profiles, fixtures, scripts,
dashboard, and documentation are included.

Tests cover profile contradictions and runtime allowlisting, cache precedence/partial
states, fingerprint stability, safe argument arrays, log redaction, GPU-free probing,
common health, AR trace, diffusion frames and determinism, wrong-family routes, start,
health, restart, stop, port collision, crash detection, management defaults, and
structured no-cloud gateway failure.

## Risks and mitigations

- **ROCm mismatch:** the isolated pinned ROCm 7.2.1 environment passed allocation and Qwen
  smoke while Fedora's 7.1 RPMs remained untouched; continue preserving both fingerprints.
- **Unified memory exhaustion:** one load at a time now; add measured reserve scheduling
  before large workers.
- **Port migration:** keep 8600 stable and provide explicit compatibility routes/adapters;
  never choose a hidden alternative.
- **HSA preload instability:** scope it to one tested worker and record it in evidence.
- **Cache false positives:** display installed-untested and never infer runnable state.
- **Prompt privacy:** do not log request bodies; bounded log redaction is defence in depth.
- **Uninspected repositories:** keep their adapters out of scope until local source is
  available.

## Unresolved assumptions

- The approved project-local PyTorch/ROCm 7.2 wheel set and exact Transformers version.
- Resolved commits and disk completeness of the first Qwen and DiffusionGemma candidates.
- Whether HuggingFacePull will expose transport-requested/used evidence through its API or
  a read-only marker.
- Final shared Open Day port registry and compatibility path for existing `8600` clients.
- Memory reserve, idle timeout, and approved preset stop/start policy.
- Availability and current implementation of MLXDashboard, Ollama projects,
  CrowdAIMission, and OpenDayOps.

## Go/no-go

The foundation is a **go** when it starts without GPU access, starts neither downloads nor
workers automatically, both mock families repeatedly start/stop without leaked processes,
wrong-family routes fail, fixed-port collisions are refused, and the full non-hardware
test suite passes.

Phase 3 is a **go** for the recorded Qwen fingerprint. A reliable direct measurement of
whole-system unified-memory recovery remains desirable scheduler evidence, but process
exit and repeated successful reloads passed. This does not generalise to larger Qwen
variants, other dtypes, revisions, or runtimes.

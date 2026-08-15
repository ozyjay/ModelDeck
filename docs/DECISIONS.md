# Architecture decisions

## ADR-012 — Service PID records are recoverable, not authoritative

On PowerShell/Linux, launching an extensionless Python console-script wrapper can return a
short-lived launcher PID rather than the long-lived Python service PID. Recording that
transient PID caused `scripts/stop.ps1` to report a service absent while its listener
continued to occupy the configured port.

`scripts/run.ps1` therefore starts the control-plane and gateway services through the
project virtual environment's Python interpreter using `-m`, then records those actual
service PIDs. `scripts/stop.ps1` first uses the recorded PID files, then conservatively
recovers missing or stale records by inspecting `/proc` for only this checkout's virtual
environment Python and the approved `modeldeck` or `modeldeck.gateway.app` modules. It
never searches for or stops unrelated processes. The launcher also removes the superseded
`gateway-loopback.pid` record before starting a new session.

`scripts/run.ps1` is a start command, not a restart command. If its port preflight fails,
run `pwsh -NoProfile -File scripts/stop.ps1` and then start again. The preflight retains
the original binding details in its error so the occupied service and address are visible.

## ADR-011 — Blocking in-process work uses isolated operation threads

On the target Python 3.12.13 control environment, ModelDeck observed reusable asyncio
executor threads accept work but then fail to wake or terminate reliably. The visible
symptoms were a Worker contract or capability rehearsal completing its model operation
and then appearing to stall, followed by Python warning that executor threads had not
finished joining within 300 seconds. The behaviour affected `asyncio.to_thread` users
across multiple Worker families; it was not specific to Qwen or tool calling.

ModelDeck therefore runs each coarse blocking in-process operation through
`modeldeck.async_execution.run_in_isolated_thread`. The helper creates one short-lived
daemon thread for that load, warm-up, inference, close, discovery, or hardware-probe
operation. The event loop polls operation completion rather than depending on a
cross-thread wake notification, which was also observed to stall. Streaming model
generation polls a thread-safe queue fed by one producer thread per request rather than
creating one thread per token. Native async subprocess workers keep their existing process
boundary.

Runtime code must not use `asyncio.to_thread` or a reusable `ThreadPoolExecutor`. A unit
test scans `backend/modeldeck` for this regression. Isolated thread creation has a small
per-operation cost, but these operations are coarse and dominated by model or hardware
work. Cancellation remains cooperative: the owning route sets its cancellation event,
while the daemon thread is allowed to finish without blocking event-loop shutdown.

## ADR-010 — Events publish immutable Route snapshots

Open Day requirements are editable Events containing shared Routes and Demos. A Route
binds a public name and trusted protocol to one primary Worker and ordered backups. Workers
bind cached Models to trusted runtimes. Drafts autosave; publishing creates an immutable
revision and atomically changes routing without starting or stopping a process. No Worker
instances or public names are seeded.

## ADR-001 — Transformers-first, provider-neutral management

Custom Transformers workers are preferred on the Framework Desktop; vLLM is optional and
evidence-gated. ModelDeck manages provider capabilities rather than centring one server.

## ADR-002 — One model per process

The API never owns model tensors. Process termination is the reliable memory and failure
boundary. Package and environment differences remain local to a worker.

## ADR-003 — Separate AR and text-diffusion engines

Generation family is required in every profile. Native refinement frames and jobs are not
emulated through token generation.

## ADR-004 — Read-only acquisition boundary

ModelDeck reads HF cache state. HuggingFacePull remains the downloader, resumer, transport
selector, and cleaner. No shared package is extracted before two real consumers need it.

## ADR-005 — Server-rendered initial dashboard

The first UI is dependency-free HTML served by FastAPI. This reduces moving parts while
lifecycle behaviour is proved and remains reversible if React/Vite becomes justified.
This initial decision is superseded by ADR-009.

## ADR-006 — Allocated ports and allowlisted launches

Workers receive a free port from a bounded local range. Commands are internal argument
arrays derived from immutable Worker definitions; the UI cannot provide commands or raw
arguments.

## ADR-007 — Evidence preserves failures

Compatibility records are append-only by complete stack fingerprint. A negative result
is current-stack evidence, not a universal claim.

## ADR-008 — Separate control-plane and primary inference environments

`.venv` runs ModelDeck management, routing, fallbacks, and tests. `.venv-rocm72` is the
primary target inference runtime for core ROCm workers. Both are required for the target
installation, but remain separate to preserve dependency isolation, bounded process
ownership, and process-exit memory recovery. GPU-free operation is a development and
recovery mode rather than the primary product configuration.

## ADR-009 — Committed React operator console

Lifecycle, telemetry, compatibility, catalogue, and streaming-log requirements now
justify a stateful React and TypeScript console. Vite is a build-time tool only. FastAPI
serves the committed production bundle with same-origin API access, SPA fallback, local
assets, and a restrictive content security policy, so packaged and Open Day operation
does not require Node.js.

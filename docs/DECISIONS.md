# Architecture decisions

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

## ADR-006 — Fixed ports and allowlisted launches

Development and Open Day use documented fixed ports. Worker commands are internal argument
arrays derived from strict profiles; the UI cannot provide commands or raw arguments.

## ADR-007 — Evidence preserves failures

Compatibility records are append-only by complete stack fingerprint. A negative result
is current-stack evidence, not a universal claim.

## ADR-008 — Separate control-plane and primary inference environments

`.venv` runs ModelDeck management, routing, fallbacks, and tests. `.venv-rocm72` is the
primary target inference runtime for core ROCm workers. Both are required for the target
installation, but remain separate to preserve dependency isolation, bounded process
ownership, and process-exit memory recovery. GPU-free operation is a development and
recovery mode rather than the primary product configuration.

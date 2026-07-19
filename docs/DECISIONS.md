# Architecture decisions

## ADR-010 — Versioned demo contracts activate immutable routing snapshots

Open Day applications are represented by editable demo sets rather than hard-wired
worker cards. Demo routes bind stable public aliases and allowlisted protocol adapters to
ordered deployments; deployments bind models to trusted runtimes, while workers remain
ephemeral process instances. Saving creates an immutable revision. Validation and an
advisory plan precede atomic activation, which changes routing only and never starts or
stops a large model. This makes event configuration auditable, keeps lifecycle decisions
explicit, and prevents partially edited configuration reaching demo clients.

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

## ADR-009 — Committed React operator console

Lifecycle, telemetry, compatibility, catalogue, and streaming-log requirements now
justify a stateful React and TypeScript console. Vite is a build-time tool only. FastAPI
serves the committed production bundle with same-origin API access, SPA fallback, local
assets, and a restrictive content security policy, so packaged and Open Day operation
does not require Node.js.

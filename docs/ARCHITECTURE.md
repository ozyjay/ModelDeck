# Architecture

## Boundaries

```text
Operator console/API :3600 ---- WorkerSupervisor ---- one allowlisted process per model :8610+
       |                                      |-- autoregressive contract
       |                                      `-- text-diffusion contract
       `---- read-only cache + hardware + SQLite evidence

Demo clients ---- Stable gateway :8600 ---- capability alias ---- ready local worker
                                                   `---- structured unavailable result
```

The configuration model separates four concepts that used to be implicit in worker
cards. A model artefact identifies cached or packaged weights; a deployment is a trusted
model profile plus its runtime and launch policy; a worker is one deployment's current
process state; and a demo route is an application-facing protocol contract bound to an
ordered set of deployments. This keeps booth requirements stable while models and
runtimes change independently.

Catalogue model entries describe a capability envelope rather than assigning the model
to one exclusive use. For example, a multimodal model may expose text generation, chat,
image input, and structured-output capabilities. The narrower generation family belongs
to each configured deployment and records the engine and protocol path ModelDeck has
actually validated.

## Runtime environments

`.venv` is ModelDeck's control-plane runtime: API, operator-console assets, supervisor, gateway,
catalogue, evidence store, fallbacks, and tests. `.venv-rocm72` is the primary target
inference runtime used by the core Qwen and DiffusionGemma worker processes. Both belong
to the target installation, but they remain separate so GPU dependencies and tensors
never enter the management process. A GPU-free `.venv` run is a useful development or
recovery mode, not the primary inference configuration.

HuggingFacePull owns Hugging Face acquisition and cleanup. OllamaPull will own Ollama
registry storage when inspected. ModelDeck owns profiles, runtime process lifecycle,
scheduling, evidence, fixed local routing, and management presentation. Demos retain
public wording, interaction state, reset, and prepared replay assets.

## Packaged registries

Built-in configuration is packaged as three versioned, validated JSON registries under
`backend/modeldeck/registry_data`: runtime templates, model-profile seeds, and reserved
gateway aliases. The profile seeds replace Python-constructed built-ins. Runtime templates
describe the bounded profile fields ModelDeck may derive for a recognised cache model;
they select, but cannot define, a trusted Python worker launch implementation. Reserved
alias contracts define packaged provider order or explicit selection, display wording,
generation family, and required capabilities. Registry loading rejects unknown versions,
duplicate IDs, missing seed references, invalid capabilities, and runtime implementations
without a trusted launch builder.

SQLite remains the home for operator-created profiles, cache policy, compatibility
evidence, explicit provider selections, and immutable demo-set revisions. Packaged seed data is read-only and versioned
with the application, so upgrades are deterministic and do not overwrite local choices.

## Process and failure model

The management API does not import model libraries or hold model tensors. Each worker is
an isolated subprocess launched with a fixed argument array derived from a validated
profile. The supervisor serialises loads, checks fixed ports, captures stdout/stderr,
polls health, runs warmup, detects exit, requests graceful shutdown, and terminates only
after a timeout. Terminating the worker is the memory-recovery boundary.

Worker log records are bounded on disk, severity-classified, and tagged by launch
session. The management log API shows the current launch by default so resolved failures
from an earlier runtime do not appear to describe the active worker.

States are `discovered`, `stopped`, `validating`, `starting`, `loading`, `warming`,
`ready`, `busy`, `degraded`, `stopping`, `failed`, `orphaned`, and `incompatible`.
Process existence alone never means ready.

## Gateway and routing

The operator console can CRUD versioned demo sets. Each set names the demos expected at
an event and defines route contracts with a public model alias, an allowlisted protocol
adapter, a qualification rule, an explicit fallback policy, and ordered deployment
bindings. Validation checks registration, cache policy, generation family, capabilities,
mock visibility, and—when selected—recorded tested-working evidence. Planning reports
which primary deployments would need to start or stop but deliberately makes no process
changes.

Activation validates a specific immutable revision and atomically replaces the gateway's
routing snapshot. The gateway rereads that snapshot on each route assembly, so activation
does not require a restart. It only exposes a route on the endpoint surfaces declared by
its adapter. Activation does not start or stop workers; readiness remains observable and
an unavailable provider still produces the structured local error. Until the first demo
set is activated, the legacy reserved-alias registry remains the effective routing source
for backwards compatibility.

The active demo-set snapshot is the sole routing authority once it exists. Stored legacy
provider selections remain visible for compatibility diagnostics but are marked as
superseded and cannot be edited. Management also derives a consolidated dependency view
for each deployment from active and latest-draft route bindings, effective legacy aliases
and worker state. The same view drives the operator console's **Used by** guidance and
server-side removal checks, preventing configuration deletion from leaving dangling
routes.

Immutable history supports two distinct recovery operations: restoring old content creates
a new editable draft revision, while activating an exact historical revision performs an
atomic routing rollback. Route rehearsal queries the live gateway advertisement and can
send a bounded adapter-specific request only for the active revision. It neither starts a
worker nor exposes generated content through the management API. Full-duplex speech remains
an explicitly interactive WebSocket rehearsal.

Aliases route according to the reserved-alias registry and declared capability contract. `fast-chat` prefers the core
Qwen ROCm worker and `text-diffusion` prefers the separate core DiffusionGemma ROCm
worker. `scenechat-vision` routes image-and-text requests to the explicitly selected Gemma 4
profile and injects its private loopback credential. Each fallback-capable alias retains an explicit
mock provider for GPU-unavailable demonstrations
and contract testing. When no candidate is ready, the gateway returns a structured local
unavailable response; no cloud request occurs. The gateway and management API are
separate processes and ports so demo clients have a stable contract while management
restarts evolve independently.

## Scheduler

The first scheduler invariant is a single global model-load lock. Profiles declare
`resident`, `on-demand`, or `exclusive`. Later measured peak/steady memory, context/KV
growth, temperature, reserve, process use, and compatibility evidence will inform launch
decisions. Parameter count alone will not.

## Cache integration

The scanner resolves `HF_HUB_CACHE`, `${HF_HOME}/hub`, the normal user cache, then
`/mnt/work/models/huggingface/hub`. It reads snapshots, config hints, incomplete markers,
revision refs, model payloads, and physical size. A snapshot with model weights is shown
as completely acquired even when it has no Transformers configuration; runtime support
is reported separately. The scanner never calls the Hub and never treats files as proof
that a model is runnable. For recognised complete snapshots matching the allowlisted
autoregressive, SceneChat Gemma 4, DiffusionGemma BF16, or self-contained ModelDeck Q4
implementations, the operator may
create a constrained local profile. The server resolves the selected catalogue entry
and derives its cache root; no filesystem path, runtime executable, command argument,
environment variable, remote model identifier, or remote-code flag is accepted from the
browser. Local profiles are persisted in SQLite, loaded by management at startup, and
discovered dynamically by the gateway.

The reserved `scenechat-vision` alias is application-facing and cannot be claimed by a
local profile. SQLite stores its selected physical profile ID. Management owns constrained
selection; the gateway rereads the mapping, local profiles, and cache policy whenever it
assembles routes, so promotion requires no gateway restart. The default is
`scenechat-gemma4-e2b-rocm`. An explicit selection is exclusive for the stable alias:
provider unavailability produces a not-ready alias rather than fallback. Requests are
rewritten to the selected profile's physical `model_id` before reaching the worker.

An exact model/revision allow policy controls whether Hugging Face cache-backed profiles
participate in management workers and gateway routes. Disallowing is reversible and never
mutates the cache or deletes profile configuration. Profiles reading packaged ModelDeck
checkpoints do not inherit policy merely because they record the same upstream model ID.
Downloaded Q4 profiles instead carry a separate derivative artefact identity, so their
policy follows the exact Hugging Face repository revision while worker loading and smoke
evidence retain the pinned base-model identity.

## Data and security

SQLite stores model profiles, compatibility tests, worker events, presets, demo-set
revisions, and the single activated routing snapshot. Logs are
bounded in memory in this slice and redact prompt/output/credential-shaped fields.
Services bind to loopback. The API has no shell, arbitrary environment, filesystem
browser, token, Docker socket, upload, camera, or cloud inference surface.

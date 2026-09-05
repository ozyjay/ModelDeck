# API contract

All services bind to `127.0.0.1` by default. ModelDeck never forwards to a cloud
provider, downloads a model, or accepts executable configuration from a client. Set
`MODELDECK_ENABLE_DOCKER_BRIDGE=1` to add a gateway listener on Docker's default bridge
address (`172.17.0.1`); management remains on its separately configured loopback host.

## Management (`:3600`)

Discovery is read-only: `GET /api/health`, `/api/hardware`, `/api/telemetry`,
`/api/thermal`, `/api/gateway/status`, `/api/catalogue`, `/api/runtime-templates`,
`/api/protocol-contracts`, and `/api/compatibility`.

### Guided capability setup

The intent-first setup surface uses durable, local-only operation resources:

- `POST /api/capability-setups/preview` resolves one exact cached Model, Artifact,
  trusted Runtime and immutable Worker plan without changing policy.
- `POST|GET /api/capability-setups` creates or lists durable FIFO operations.
- `GET /api/capability-setups/{setup_id}` and `/events` report persisted state and
  replayable SSE progress.
- `POST /api/capability-setups/{setup_id}/cancel|retry` controls work without deleting
  its Worker or evidence.
- `POST /api/capability-setups/{setup_id}/publication-preview|publish` separates
  qualification from an explicit, stale-protected routing decision.

Creation requires a caller UUID and the SHA-256 fingerprint of the reviewed preview.
Model loading and qualification pause under thermal policy, resume after a management
restart, and preserve positive or negative evidence. Publication reuses the managed
**Local capabilities** profile, requires `tested-working` evidence, verifies the public
route and restores the previous live revision when that verification fails.

Workers use `GET|POST /api/workers`, `GET|PATCH|DELETE /api/workers/{worker_id}` and
the bounded lifecycle, logs, smoke, usage, replacement and stop-all subroutes. Workers
can be created only from a complete, cached model revision and an installed trusted
runtime template. Archiving preserves caches and historical references.

`GET /api/catalogue` reports a `potential_capabilities` collection for every complete
cached revision. Each candidate keeps locally detected evidence separate from reviewed,
code-owned assertions and reports traits, provenance, permission, trusted-runtime
availability, qualification and publication state. Discovery is offline and never runs
remote code or interprets arbitrary model-card prose.

- `POST /api/catalogue/policy` controls the model-level master policy.
- `POST /api/catalogue/capabilities/policy` records operator permission for one candidate.
- `POST /api/workers/{worker_id}/capabilities/{capability_id}/qualify` runs the bounded,
  code-owned qualification adapter against that exact ready Worker.

Capability permissions default to denied. An operator may allow a recognised capability
before a runtime exists; this records intent but does not make it runnable. Model denial
makes every child permission ineffective without deleting those choices. New and
replacement Workers and Routing Profile publication require effective permission.
Disallowing a capability referenced by a current draft or active revision returns the
blocking references. Historical revisions remain immutable and do not block the change.

There are no public mock-worker templates, mock-worker creation endpoints, Event, Demo,
or Event Route management endpoints. Deterministic fixture workers are test-harness
tools, not an operator feature.

### Routing profiles and live routing

- `GET|POST /api/routing-profiles`
- `GET|DELETE /api/routing-profiles/{profile_id}`
- `PUT|DELETE /api/routing-profiles/{profile_id}/draft`
- `POST /api/routing-profiles/{profile_id}/validate|publish`
- `GET /api/routing-profiles/{profile_id}/revisions`
- `POST /api/routing-profiles/{profile_id}/revisions/{revision}/publish`
- `DELETE /api/routing-profiles/{profile_id}/active`
- `POST /api/routing-profiles/{profile_id}/capabilities/{capability_id}/smoke`
- `GET /api/live`

A Routing Profile contains a name, description, qualification policy and profile-local
published capabilities. A capability has a display name, public `model` name, one trusted
protocol contract, and ordered compatible Worker IDs. Index zero is primary. Publishing
validates a draft, creates an immutable revision, and atomically activates that profile
alongside other active profiles; it never starts Workers. Public model IDs must be unique
across the active set. Earlier revisions can be made active again and an active profile can
be deactivated with `DELETE /api/routing-profiles/{profile_id}/active`; this removes only
that profile's capabilities from live routing and never stops its Workers. The local
configuration lock blocks profile mutation server-side while preserving reads and explicit
Worker controls.

## Gateway (`:8600`)

`MODELDECK_GATEWAY_HOST` defaults to `127.0.0.1` and must remain a loopback address.
`MODELDECK_ENABLE_DOCKER_BRIDGE=1` starts a narrow TCP forwarder at Docker's default bridge
address, `172.17.0.1`, so containers can reach `http://host.docker.internal:8600/v1` while
desktop applications use loopback. The forwarder targets the authoritative loopback gateway
and owns no routing, persistence, Worker lifecycle or thermal accounting. Wildcard (`0.0.0.0`
or `::`) and LAN addresses are rejected during configuration parsing. An unavailable address
or occupied port is reported at startup.

`/v1` contains standard model APIs. The `model` field must identify a compatible
capability in an active Routing Profile. `GET /v1/models` lists only capabilities whose
protocol adapter explicitly declares OpenAI-model compatibility; native-only capabilities
are not presented as OpenAI models.

Each item in `GET /v1/models` has the existing `id`, `object`, `owned_by`, and `ready`
fields plus `revision`. `revision` is the authoritative, pinned upstream snapshot revision
of the capability's primary Worker. For normal Hugging Face-backed Workers, ModelDeck
obtains it from the exact locally cached snapshot selected when the Worker was created
(normally the Hugging Face commit revision). For a Worker that loads a separately versioned
derivative artefact, such as the ModelDeck DiffusionGemma Q4 release, it is instead the
artefact repository revision because that is the checkpoint actually loaded; the release
manifest separately binds its base-model revision.

The value is persisted in the Worker definition and therefore remains stable across
ModelDeck restarts while the published capability and its primary Worker are unchanged.
Publishing a capability with a Worker for a different cached revision or derivative
artefact changes it. ModelDeck does not currently maintain a verified digest over every
local model file, so it deliberately does not expose a synthetic `digest`. Manual mutation
of files inside a cached snapshot after Worker creation is outside this guarantee. A
capability may have a different-revision backup Worker; `revision` identifies the ordered
primary Worker, while an individual request can use a backup only when the primary is
unavailable.

Chat model records also expose a top-level `capabilities` object. `chat: true` means only
that the route accepts chat requests. `tool_calling: "verified"` appears only after the
current published route revision has passed ModelDeck's bounded public-route rehearsal;
otherwise it is `"unverified"`. Consumers must not infer tool support from
`openai-chat-v1` alone.

`POST /api/routing-profiles/{profile_id}/capabilities/{capability_id}/smoke` performs that
rehearsal for an `openai-chat-v1` route. It requires one empty-schema function call and one
named function call with JSON arguments, both through the public gateway. `/api/live`
reports the revision-scoped state as `tool_calling.supported`, `rehearsed`,
`last_rehearsal`, and `failure_code`. It stores only probe counts, result categories,
latencies, and coarse error codes; no prompt, arguments, or model output is retained.

Model records also contain `runtime` and `accelerator`, which retain their legacy meaning:
they describe the first ready Worker in the capability's ordered routing list, or the
configured primary Worker when none is ready. `accelerator` is code-owned metadata derived
from the verified runtime and health evidence: ready ROCm Workers report `rocm`; Vulkan,
CPU, and test-harness Workers report `vulkan`, `cpu`, and `mock` respectively. A model with
`ready: false` is not accelerator-resident proof, even when its configured runtime is ROCm.

### Discovery identity metadata

Each model record has a namespaced `modeldeck` object. Its existing flat `model_id`,
`revision`, `runtime`, `configuration_fingerprint`, `prefix_caching`, and
`prefix_cache_enabled` fields remain for compatibility and retain the legacy selected-Worker
meaning above. In particular, the flat fingerprint remains the ready Worker's runtime-reported
value when available, with the existing configured fallback. New consumers should use the
explicit identity objects:

- `route` identifies the stable published capability: `public_model_id`, `capability_id`,
  `routing_profile_id`, and `routing_profile_revision`. The final three values are `null`
  for an embedded gateway supplied with in-memory routes rather than a published profile.
- `primary_worker` identifies configured worker zero. Its `worker_id`, loaded `model_id` and
  `revision`, base-model identity, optional artefact identity, configured/runtime-observed
  runtime, accelerator, readiness, and `configuration_fingerprint` remain inspectable during
  failover. Its `runtime` and `accelerator` are the configured values, and therefore remain
  stable while a backup is selected.
- `selected_worker` has the same shape for the first ready Worker, or is `null` when no
  Worker is selectable. `selection_reason` is one of `primary_ready`, `backup_ready`, or
  `no_ready_worker`.

Within a Worker identity, `model_id` and `revision` identify the checkpoint actually loaded:
they are the artefact pair when an artefact is configured, otherwise the base-model pair.
`configuration_fingerprint` is the stable configured identity derived from the Worker
definition. `runtime_configuration_fingerprint` is an optional opaque fingerprint supplied
by a ready Worker and must not be compared with the configured fingerprint. It is `null` when
the Worker is not ready or does not report one.

`GET /v1/models` reports a point-in-time readiness snapshot. It indicates the Worker a new
request would receive under that snapshot, but does not prove which Worker served a completed
request after a readiness transition.

`POST /v1/embeddings` is an OpenAI-compatible, local-only embeddings surface. It accepts a
published embedding `model` and a non-empty string or ordered array of strings in `input`.
The current trusted Qwen embedding Worker returns exactly 1,024 float vectors, with one
`data` item per input and its original zero-based `index`. The gateway validates the request
before it reaches a Worker: malformed or empty input returns HTTP 422, an unpublished model
returns HTTP 404, and an invalid published Worker binding returns HTTP 409. A published
embedding model whose local Workers are unavailable returns HTTP 503
`local_route_unavailable` with `cloud_fallback_attempted: false`.

Embeddings use the code-owned `openai-embeddings-v1` contract, displayed in the operator
console as **OpenAI-compatible embeddings**. It accepts only Workers with generation family
`embedding` and the `embeddings` capability; chat, completions, and autoregressive-trace
Workers cannot be bound to it. `Qwen/Qwen3-Embedding-0.6B` is recognised as an embedding
Model and is configured with the shared `transformers-rocm` stack through its dedicated
`embedding-transformers` runtime template. A pre-existing Worker configured with the
autoregressive template is intentionally incompatible; create a new embedding Worker from
the recognised cached Model before publication. ModelDeck never repurposes it automatically.

The packaged `modeldeck-core` runtime manifest version 0.2.4 introduces the
`embedding-transformers` template. Existing Worker definitions retain their recorded
template version; creating a Worker from the recognised Qwen embedding snapshot uses the
new embedding template.

- `GET /v1/health`, `/v1/models`, `/v1/capabilities`, `/v1/routes`, `/v1/thermal`,
  `/v1/metrics`
- `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/translations`
- `POST /v1/audio/speech`, `/v1/audio/transcriptions`
- `WS /v1/speech/conversations`
- `POST /v1/vision/analyse`, `/v1/requests/{request_id}/cancel`

Specialised, reusable low-level model interactions are code-owned native protocols:

- `GET /native/v1/capabilities`
- `POST /native/v1/autoregressive/traces`
- `POST /native/v1/text-diffusion/refine`
- `POST /native/v1/text-diffusion/jobs`
- `GET /native/v1/text-diffusion/jobs/{job_id}`
- `GET /native/v1/text-diffusion/jobs/{job_id}/events`
- `POST /native/v1/text-diffusion/jobs/{job_id}/cancel`

`GET /native/v1/capabilities` advertises ready state, contract IDs, canonical surfaces,
and bounded metadata for published specialised capabilities. No configuration field can
define paths, schemas, worker commands, or forwarding rules: the static adapter registry
owns them.

`POST /native/autoregressive/trace`, `/v1/refine`, `/v1/diffuse`, and the legacy `/v1/jobs`
paths remain compatibility aliases in this release. They preserve their response shapes and
include `Deprecation: true` plus a `Link` successor relation. They are removed in the next
major release.

The gateway selects an ordered backup only before a request or text-diffusion job starts.
It does not fail over an interrupted stream or an existing job. Job-to-Worker ownership is
stored durably so a restarted gateway can poll or cancel the same live Worker job.

Successful routed responses include `X-ModelDeck-Worker-Id`,
`X-ModelDeck-Configuration-Fingerprint`, and `X-ModelDeck-Route-Role` (`primary` or
`backup`) so selection remains observable without exposing paths or environment values.
When no matching local Worker is ready, the gateway returns HTTP 503
`local_route_unavailable` with `cloud_fallback_attempted: false`. Gateway responses do not
carry mock or fallback headers.

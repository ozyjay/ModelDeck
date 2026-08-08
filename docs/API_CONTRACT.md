# API contract

All services bind to `127.0.0.1` by default. ModelDeck never forwards to a cloud
provider, downloads a model, or accepts executable configuration from a client. The gateway
alone can explicitly bind to Docker's default bridge address (`172.17.0.1`) through
`MODELDECK_GATEWAY_HOST`; management remains on its separately configured loopback host.

## Management (`:3600`)

Discovery is read-only: `GET /api/health`, `/api/hardware`, `/api/telemetry`,
`/api/thermal`, `/api/gateway/status`, `/api/catalogue`, `/api/runtime-templates`,
`/api/protocol-contracts`, and `/api/compatibility`.

Workers use `GET|POST /api/workers`, `GET|PATCH|DELETE /api/workers/{worker_id}` and
the bounded lifecycle, logs, smoke, usage, replacement and stop-all subroutes. Workers
can be created only from a complete, cached model revision and an installed trusted
runtime template. Archiving preserves caches and historical references.

There are no public mock-worker templates, mock-worker creation endpoints, Event, Demo,
or Event Route management endpoints in v3. Deterministic fixture workers are test-harness
tools, not an operator feature.

### Routing profiles and live routing

- `GET|POST /api/routing-profiles`
- `GET|DELETE /api/routing-profiles/{profile_id}`
- `PUT|DELETE /api/routing-profiles/{profile_id}/draft`
- `POST /api/routing-profiles/{profile_id}/validate|publish`
- `GET /api/routing-profiles/{profile_id}/revisions`
- `POST /api/routing-profiles/{profile_id}/revisions/{revision}/publish`
- `POST /api/routing-profiles/{profile_id}/capabilities/{capability_id}/smoke`
- `GET /api/live`

A Routing Profile contains a name, description, qualification policy and profile-local
published capabilities. A capability has a display name, public `model` name, one trusted
protocol contract, and ordered compatible Worker IDs. Index zero is primary. Publishing
validates a draft, creates an immutable revision, and atomically makes it the one active
profile; it never starts Workers. Earlier revisions can be made active again. The local
configuration lock blocks profile mutation server-side while preserving reads and explicit
Worker controls.

## Gateway (`:8600`)

`MODELDECK_GATEWAY_HOST` defaults to `127.0.0.1`. The only non-loopback value accepted by
the local-only policy is Docker's default bridge address, `172.17.0.1`, which permits a
container to reach `http://host.docker.internal:8600/v1`. Wildcard (`0.0.0.0` or `::`) and
LAN addresses are rejected during configuration parsing. Uvicorn then binds the selected
address and reports an unavailable address or occupied port at startup.

`/v1` contains standard model APIs. The `model` field must identify a compatible
capability in the active Routing Profile. `GET /v1/models` lists only capabilities whose
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

Model records also contain `runtime` and `accelerator`, which describe the first ready
Worker in the capability's ordered routing list—the same Worker that receives a new gateway
request. `runtime` comes from that Worker's health report when it is ready, otherwise from
the configured primary Worker. `accelerator` is code-owned metadata derived from the
verified runtime and health evidence: ready ROCm Workers report `rocm`; Vulkan, CPU, and
test-harness Workers report `vulkan`, `cpu`, and `mock` respectively. A model with
`ready: false` is not accelerator-resident proof, even when its configured runtime is ROCm.

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

When no matching local Worker is ready, the gateway returns HTTP 503
`local_route_unavailable` with `cloud_fallback_attempted: false`. Gateway responses do not
carry mock or fallback headers.

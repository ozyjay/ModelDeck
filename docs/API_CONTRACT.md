# API contract

All services bind to `127.0.0.1` by default. ModelDeck never forwards to a cloud
provider, downloads a model, or accepts executable configuration from a client.

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

`/v1` contains standard model APIs. The `model` field must identify a compatible
capability in the active Routing Profile. `GET /v1/models` lists only capabilities whose
protocol adapter explicitly declares OpenAI-model compatibility; native-only capabilities
are not presented as OpenAI models.

- `GET /v1/health`, `/v1/models`, `/v1/capabilities`, `/v1/routes`, `/v1/thermal`,
  `/v1/metrics`
- `POST /v1/chat/completions`, `/v1/completions`, `/v1/translations`
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

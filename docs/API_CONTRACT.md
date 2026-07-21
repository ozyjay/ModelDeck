# API contract

All services bind to `127.0.0.1` by default.

## Management (`:3600`)

### Discovery and trust

- `GET /api/health`, `/api/hardware`, `/api/telemetry`, `/api/gateway/status`
- `GET /api/catalogue` and `POST /api/catalogue/policy`
- `GET /api/runtime-templates` and `/api/protocol-contracts`
- `GET /api/compatibility`

The catalogue is read-only discovery. Worker creation is accepted only for an exact,
complete, locally discovered revision using an installed trusted runtime template.

### Workers

- `GET|POST /api/workers`
- `GET /api/mock-worker-templates` lists code-owned mock implementations and bounded options
- `POST /api/workers/mocks` creates a contract-specific deterministic mock Worker
- `POST /api/workers/mock-scenechat` is the deprecated SceneChat compatibility creator
- `GET|PATCH|DELETE /api/workers/{worker_id}`
- `GET /api/workers/{worker_id}/usage`
- `POST /api/workers/{worker_id}/start|stop|restart|smoke`
- `POST /api/workers/{worker_id}/replacement`
- `POST /api/workers/stop-all`
- `GET /api/workers/{worker_id}/logs` and `/logs/stream`

`PATCH` changes only the editable name. Execution settings are immutable. The replacement
endpoint accepts a new name and bounded `dtype`, `lifecycle`, `context_length`,
`maximum_new_tokens` and `maximum_denoising_steps` values; it derives the Model, revision,
artefact and trusted runtime from the original Worker. Draft Event references may be
rebound during replacement; published revisions are never rewritten.
Archiving is blocked until the Worker is stopped and no draft or active Event revision
references it. Historical revisions retain their audit reference. Cache files are never
removed by Worker operations.

Smoke testing requires a ready Worker and performs health, Model, metrics and bounded
generation requests. Both successful and failed evidence is persisted.

Mock Workers are allowlisted by protocol contract and support `success`, `delayed` and
`request-error` scenarios. Delay is bounded to 1–120,000 ms; arbitrary fixtures, paths,
commands, headers, status codes and environment variables are not accepted. Stop a mock
Worker to rehearse an unavailable provider. Gateway responses from mock Workers carry
`x-modeldeck-fallback: mock`.

### Events and live routing

- `GET|POST /api/events`
- `GET|DELETE /api/events/{event_id}`
- `PUT|DELETE /api/events/{event_id}/draft`
- `POST /api/events/{event_id}/validate|publish`
- `GET /api/events/{event_id}/revisions`
- `POST /api/events/{event_id}/revisions/{revision}/publish`
- `POST /api/events/{event_id}/routes/{route_id}/smoke`
- `GET /api/live`

Event bodies contain `name`, `description`, `qualification`, `demos` and `routes`. Each
Route contains a display name, public name, trusted protocol contract and ordered
`worker_ids`; index zero is primary. Each Demo contains a name and shared `route_ids`.

Publishing validates the current draft, creates an immutable revision and atomically makes
it the active routing snapshot. Publishing and configuration mutation are locked in Open
Day mode. Route smoke works only for a Route in the currently published Event and sends a
bounded request through the gateway.

Unknown API routes remain JSON 404 responses. Browser routes serve the committed React
bundle. The browser cannot submit commands, paths, arbitrary environment values or remote
Model identifiers.

## Gateway (`:8600`)

- `GET /v1/health`, `/v1/models`, `/v1/capabilities`, `/v1/routes`
- `POST /v1/chat/completions`, `/v1/completions`
- `POST /native/autoregressive/trace`
- `POST /v1/refine`, `/v1/diffuse`
- `GET /v1/jobs/{job_id}`, `/v1/jobs/{job_id}/events`
- `POST /v1/jobs/{job_id}/cancel`
- `POST /v1/requests/{request_id}/cancel`
- `POST /v1/vision/analyse`
- `WS /v1/speech/conversations`

Clients always supply a published Route `public_name` in the `model` field. Routes are
advertised only from the active Event snapshot and only on surfaces permitted by their
protocol. The gateway tries Workers in configured order and routes only to a ready local
process. It forwards streams and cancellation and retains local diffusion-job affinity.

When no matching Worker is ready, the response is HTTP 503 with code
`local_route_unavailable`, the requested Route and `cloud_fallback_attempted: false`.
Worker UUIDs and the removed provider-selection model are not part of public responses.

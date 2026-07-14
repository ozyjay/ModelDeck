# API contract

## Management (`127.0.0.1:3600`)

Implemented: health, hardware, telemetry, catalogue, profiles, workers, worker
start/stop/restart/warmup/smoke/logs, SSE events and log streams, compatibility reads,
preset listing, stop-all, and same-origin gateway status. `GET /api/gateway/status`
returns gateway health, advertised models, and providers, or a structured `available:
false` response when the separate gateway process cannot be reached. Profile mutation
remains outside the operator console so it cannot accept unsafe runtime configuration.

FastAPI also serves the committed operator-console assets and returns the SPA entry point
for non-API routes. Unknown `/api` routes remain JSON 404 responses rather than falling
through to the frontend.

## Gateway (`127.0.0.1:8600`)

Implemented: `/v1/health`, `/v1/models`, `/v1/capabilities`, `/v1/providers`, AR chat and
completion, native AR trace, native refine/diffuse, and an explicit unsupported vision
route. Requests route only to ready loopback workers. Unavailable responses use HTTP 503,
`local_provider_unavailable`, required family, alias, and
`cloud_fallback_attempted: false`.

The gateway forwards SSE streams without buffering and propagates cancellation through
`POST /v1/requests/{request_id}/cancel`. Text-diffusion jobs are available through
`GET /v1/jobs/{job_id}`, `GET /v1/jobs/{job_id}/events`, and
`POST /v1/jobs/{job_id}/cancel`; the gateway retains provider affinity and can rediscover
jobs from local diffusion providers after a gateway restart. Diffusion request timeouts
default to 900 seconds and can be changed with `MODELDECK_DIFFUSION_TIMEOUT_SECONDS`.
`fast-chat` and `token-explainer` prefer the live Qwen worker when ready and fall back
explicitly to the mock AR worker. Stricter OpenAI SSE compatibility remains later work.

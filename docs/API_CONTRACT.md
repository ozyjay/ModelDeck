# API contract

## Management (`127.0.0.1:3600`)

Implemented: health, hardware, telemetry, catalogue, profiles, workers, worker
start/stop/restart/warmup/smoke/logs, SSE events and log streams, compatibility reads,
preset listing, and stop-all. Profile mutation and compatibility test execution remain
later phases so the first slice cannot accept unsafe runtime configuration.

## Gateway (`127.0.0.1:8600`)

Implemented: `/v1/health`, `/v1/models`, `/v1/capabilities`, `/v1/providers`, AR chat and
completion, native AR trace, native refine/diffuse, and an explicit unsupported vision
route. Requests route only to ready loopback workers. Unavailable responses use HTTP 503,
`local_provider_unavailable`, required family, alias, and
`cloud_fallback_attempted: false`.

The gateway forwards SSE streams without buffering and propagates cancellation through
`POST /v1/requests/{request_id}/cancel`. `fast-chat` and `token-explainer` prefer the live
Qwen worker when ready and fall back explicitly to the mock AR worker. Native job-event
forwarding and stricter OpenAI SSE compatibility remain later work.

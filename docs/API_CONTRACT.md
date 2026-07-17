# API contract

## Management (`127.0.0.1:3600`)

Implemented: health, hardware, telemetry, catalogue, profiles, workers, worker
start/stop/restart/warmup/smoke/logs, SSE events and log streams, compatibility reads,
preset listing, stop-all, and same-origin gateway status. `GET /api/gateway/status`
returns gateway health, advertised models, and providers, or a structured `available:
false` response when the separate gateway process cannot be reached. Profile mutation
is limited to `POST /api/profiles` and `DELETE /api/profiles/{profile_id}` for local
configurations backed by allowlisted autoregressive, SceneChat Gemma 4, DiffusionGemma
BF16, or manifest-verified ModelDeck DiffusionGemma Q4 workers. Creation requires an exact complete snapshot already
returned by cache discovery and accepts only an alias, dtype, lifecycle, context length,
and output ceiling. ModelDeck derives the cache root, Transformers ROCm runtime, port,
capabilities, offline policy, and fixed launch arguments. Built-in profiles cannot be
removed, active local workers must be stopped first, and deleting a profile never deletes
cache content.

`POST /api/catalogue/policy` persists an allow/disallow decision for an exact cached
`model_id` and `revision`. Disallowing requires every matching cache-backed worker to be
stopped, then removes those workers and gateway routes from active ModelDeck use without
deleting cache files or profile documents. Re-allowing restores configured workers.
Profiles backed by packaged checkpoints rather than the Hugging Face cache are unaffected.
Downloaded Q4 releases are controlled by their derivative repository and revision, not
by the upstream base-model policy.

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
Persisted local profiles are discovered by the gateway on each request and
advertised under their configured alias without requiring a gateway restart.
The gateway applies the persisted HF-cache allow policy on every route refresh.

### Native autoregressive trace token metadata

`POST /native/autoregressive/trace` preserves the existing trace events, probabilities,
alternatives, readiness, errors, metrics, and `prompt_token_ids`. Its non-streaming response
adds `prompt_tokens`, `user_prompt_token_ids`, and `user_prompt_tokens` as documented in the
[worker protocol](WORKER_PROTOCOL.md#autoregressive-worker).

`prompt_token_ids` and `prompt_tokens` describe the complete inference context and align
one-to-one. `user_prompt_tokens` is the safe public-display view of only the latest
user-entered prompt; it does not contain hidden system instructions or chat-template control
tokens and aligns with `user_prompt_token_ids`. These values come from the selected worker's
tokenizer. The gateway only validates and propagates them. Invalid or misaligned successful
worker metadata is returned as HTTP 502 with `invalid_worker_trace_metadata`, rather than as
a misleading trace.

## SceneChat vision API (`127.0.0.1:8600`)

The stable gateway advertises `scenechat-vision` with `image_input` and
`structured_output`. It accepts the SceneChat OpenAI-shaped request at
`POST /v1/chat/completions` or `POST /v1/vision/analyse`, routes only to the pinned local
Gemma 4 worker, and returns `local_provider_unavailable` when that worker is stopped. The
gateway translates the alias to the exact worker model identifier and injects the private
loopback credential internally.

## Direct SceneChat compatibility API (`127.0.0.1:8000`)

The managed SceneChat worker preserves the existing client contract at `GET /v1/models`
and `POST /v1/chat/completions`, plus native smoke at
`POST /native/vision-language/smoke`. These routes require
`Authorization: Bearer <MODELDECK_SCENECHAT_API_KEY>`; the loopback development default is
`local`. Lifecycle routes remain loopback-only for the supervisor.

Chat accepts the exact pinned model, one user message, one base64 JPEG/PNG image followed
by one approved SceneChat prompt, `temperature: 0.1`, `max_tokens` from 1 through 700
(clamped to the profile's 512-token generation ceiling),
`response_format: {"type":"json_object"}`, and `stream: false`. Errors use a sanitised
OpenAI-shaped `error` object with status 401, 413, 422, 429, 502, 503, or 504. This direct
worker route is retained for supervisor smoke tests and diagnosis; applications use the
stable gateway.

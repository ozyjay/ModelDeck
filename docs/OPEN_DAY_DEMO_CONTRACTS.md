# Open Day demo contracts

This guide describes implemented ModelDeck 2.0 contracts and the consumer changes
needed by the accessible demo checkouts. It advances explicit identity, observable
lifecycle and thermal safety in the [guiding principles](GUIDING_PRINCIPLES.md).
No model acquisition, Worker creation, publication or hardware qualification is implied
by discovery. Public names below are consumer defaults, not automatically seeded routes.

## Investigation: 8 September 2026

ModelDeck started on clean `main`, commit `1d5bf9e`, application version 2.0.0.
Read-only comparison used these clean sibling checkouts:

| Consumer | Commit | Inspected adapter | Finding and consumer follow-up |
| --- | --- | --- | --- |
| TokenTrail | `0fb4194` | `src/token_trail/adapters/modeldeck.py` | Replace `/v1/models` readiness discovery with native capability discovery; migrate the trace URL to the canonical path. Update its AGENTS.md and demo-set terminology. |
| TextDiffusionDemo | `696c66a` | `server/services/modelProviders/modelDeckProvider.ts` | Replace `/v1/models` availability discovery with native capability discovery; migrate submission, polling and cancellation URLs. Missing publication is not evidence that a model needs installation. |
| SceneChat | `b66f562` | `backend/scenechat/vision/modeldeck.py` | Replace the `/v1/models` readiness gate with `/v1/routes`; require `scene-analysis-v1`. Keep the `/v1/capabilities` trait check. |
| SpeechShift | `8ee60a6` | `backend/speechshift/modeldeck.py` | `/v1/routes` discovery already works. Check protocol IDs as well as all four names and `ready`; distinguish retryable thermal/busy failures from permanent request errors. |

Those repositories were not edited. Their existing discovery gates mean the first
three consumers still need these changes before end-to-end compatibility is restored.

The installed RPM is `modeldeck-0.1.2-1.fc44.x86_64`, with application version 0.1.2
under `/usr/libexec/modeldeck/control`. Its user services were **inactive**, and
3600/8600 had no listeners. GET health/discovery requests received connection refused.
No service was started for this investigation.

| Store | Configuration and inspected state |
| --- | --- |
| Checkout | `.env` sets `MODELDECK_DATA_DIR=.modeldeck`; schema v5, 75 Worker records. Active: **Codex Router**, revision 8 (`codex-router-classifier`), and **Local capabilities**, revision 2 (`gpt-oss-120b-gguf`). |
| Desktop package | Installed systemd units set `MODELDECK_DESKTOP=1`, fixed XDG roots and `/usr/libexec/modeldeck` executables. `/home/jase/.local/share/modeldeck/modeldeck.sqlite3`: schema v4, 64 Worker records. Active: **Design Sprint 2026** r4, **wayfinder-gate1** r14, **OpenCode local coding** r2, **Codex Router** r4, **Codex Router Simulation Selector** r7. |

Neither store publishes any of the nine default demo names below. Desktop SprintBot
chat and Codex Router classifier retain explicit two-Worker chains. The obsolete
singleton `active_routing_profile` table is historical; the plural table is authoritative.
Worker record counts include historical definitions and do not establish readiness.
Existing definitions include ports above the personal 8610–8624 reservation; review
actual assignments before any later launch rather than assuming that reservation covers them.

Store inspection used SQLite read-only access; immutable inspection of the stopped
desktop database followed a check that no WAL file was present. Do not use immutable
reads on a live database: they can omit WAL transactions. No state was reset, imported,
migrated or overwritten. Checkout launch scripts read checkout `.env`; desktop services
use their unit environment. Neither is evidence of what the other installation serves.
Current application lifespan automatically upgrades v4 to v5; do not start it against
desktop data just to inspect it. See [migration policy](MIGRATION_V4_TO_V5.md) and the
explicit scripts in `scripts/migrations/` and `scripts/operations/` before planning a cut-over.

## Discovery and readiness

Use the loopback gateway base `http://127.0.0.1:8600`, with management at
`http://127.0.0.1:3600`. Overrides must identify the intended installation.
Native clients may omit Origin; browser mutations require a valid loopback Origin.
Host validation and CORS restrictions still apply; use each demo's server-side adapter.

| GET endpoint | Exact discovery envelope and purpose |
| --- | --- |
| `/v1/health` | `{status, service, version, ready_workers}`. HTTP 200 with `status: "ok"` establishes gateway service health, even with zero ready Workers. |
| `/v1/models` | `{object: "list", data: [...]}`. Only adapters explicitly marked `openai_model`; not a complete route catalogue. |
| `/native/v1/capabilities` | `{capabilities: [{id, display_name, public_name, protocol_contract, surfaces, ready, metadata}], resolution: {...}}`. Native trace and text diffusion only. Match **public_name**, not `id` (the capability UUID). |
| `/v1/routes` | `{routes: [{public_name, ready, protocol_contract, surfaces}], resolution: {...}, cloud_fallback: false}`. All published protocols, including scene analysis and translation. Protocol/surface fields are additive in this change; older builds may omit them. Embedded in-memory fixture routes report `null`/`[]`, not invented protocol identity. |
| `/v1/capabilities` | Object keyed by public name, each value a primary Worker's `CapabilitySet`. Traits such as `image_input`, `structured_output`, `top_k_trace`; not a readiness or publication-protocol assertion. |
| `/api/live` (management) | `{active_profile, active_profiles, capabilities}`. Use plural `active_profiles`; singular is null when multiple profiles are active. Capabilities contain protocol, ordered `worker_ids`, Workers and effective Worker. |
| `/api/protocol-contracts` (management) | Trusted contract definitions, including exact surfaces. Does not mean a matching route is published. |
| `/v1/thermal` | Policy state, enabled flag, temperature, sensor, telemetry age, capacity and reason. Readiness does not bypass admission. |

`/v1/models` intentionally excludes `native-ar-trace-v1`, `text-diffusion-v1`,
`scene-analysis-v1`, translations and speech conversation. It currently includes chat,
image chat, completions, embeddings, speech recognition and synthesis. The latter's
presence does not make SpeechShift's custom JSON audio envelope OpenAI multipart audio.

`ready: true` means a Worker is selectable in the current ordered chain. It does not
prove thermal admission, model output quality or hardware qualification. `/v1/routes`
resolution preserves configured Worker IDs, missing/archived/invalid definitions,
candidate IDs, publication identity and per-Worker readiness and execution identity.
Index zero stays primary even if missing; a configured backup remains a backup.
No chain is extended automatically. Once a request or job starts, it does not fail over.

## Demo request and cancellation contracts

| Demo | Required public names | Protocol |
| --- | --- | --- |
| TokenTrail | `qwen-0-5b`, `qwen-1-5b`, `qwen-3b` (default selection `qwen-1-5b`) | `native-ar-trace-v1` |
| TextDiffusionDemo | `text-diffusion-lab-q4` (overridable with `MODELDECK_MODEL`) | `text-diffusion-v1` |
| SceneChat | `scenechat-vision` | `scene-analysis-v1` |
| SpeechShift | `speechshift-stt`, `speechshift-en-fr`, `speechshift-en-de`, `speechshift-voice` | Respectively `speech-recognition-v1`, `translation-en-fr-v1`, `translation-en-de-v1`, `speech-synthesis-v1` |

**TokenTrail:** POST `/native/v1/autoregressive/traces` with JSON `model`, a caller
`request_id`, `messages` (optional system instructions followed by user prompt),
`max_tokens`, `min_tokens`, `top_k`, `temperature`, `stream: false`. The inspected
consumer bounds max tokens to 1–128, min tokens to min(8, max), top-k to 1–20,
temperature to 0–2 and timeout to 60 seconds. Worker validation additionally enforces
context and configured generation limits. The response includes `events`, prompt token
strings/IDs and user-prompt token strings/IDs; event metadata contains selected token,
alternatives and generation timing. The gateway validates trace token metadata;
TokenTrail additionally rejects control tokens, incomplete sentences and cancelled traces.
Do not reinterpret this as OpenAI chat streaming. Cancel via
POST `/v1/requests/{request_id}/cancel`.

**TextDiffusionDemo:** POST `/native/v1/text-diffusion/jobs` with JSON `model`,
`prompt` (1–16,000 characters), `max_length` (8–256), `denoising_steps` (1–48),
`block_length` (1–256), `temperature` (>0 to 2), integer `seed`, and
`stream_intermediate_frames: true`. The inspected consumer defaults to 48 steps and
seed 11. Submission returns JSON `{job_id, state: "queued", events_url}` (HTTP 200).
The current Worker-provided `events_url` can still name the supported legacy path;
consumers can construct the canonical path from `job_id`.
Poll GET `/native/v1/text-diffusion/jobs/{job_id}`: states are `queued`, `running`,
`complete`, `failed`, `cancelled`; frames accumulate, and completed output includes
`text`, `frame_count`, seed and metrics. Frames contain `step`, `text` and optional
total/stable/masked/complete metadata. Optional SSE GET `.../{job_id}/events` emits
`frame` and failure `error` events and closes at terminal state. POST
`.../{job_id}/cancel` returns `{job_id, state}`; `cancelling` acknowledges intent,
not completed termination. Poll to confirm. Ownership is durable across gateway
restart, but the Worker must still hold the job. The inspected consumer polls every
250 ms and attempts cancellation on timeout/abort once it knows the job ID.
Synchronous POST `/native/v1/text-diffusion/refine` accepts the same request schema.

**SceneChat:** require route readiness, `scene-analysis-v1`, and primary traits
`image_input` and `structured_output`. POST `/v1/vision/analyse` with JSON `model`,
`messages: [{role: "user", content: [{type: "image_url", image_url: {url:
"data:image/jpeg;base64,..."}}, {type: "text", text: "..."}]}]`, `temperature: 0.1`,
`max_tokens`, `response_format: {type: "json_object"}`, `stream: false`. PNG is also
accepted; the consumer validates and optionally resizes frames before encoding.
The response uses `choices[0].message.content` containing a JSON string; SceneChat
validates its summary, objects, relationships and uncertainties. `usage` supplies
trusted token counts independently of model-written content. `/v1/chat/completions`
also accepts this contract, but its existence does not add the route to `/v1/models`.
The consumer currently sends no caller request ID: cancellation relies on disconnect
and the gateway's private ID. A consumer needing explicit cancellation should send a
unique `request_id` and call the common cancellation endpoint. The gateway's default
scene timeout is 75 seconds, configurable through its existing setting.

**SpeechShift:** all four routes must be ready under its current all-or-nothing gate.
Each stage uses a distinct UUID `request_id`; the trusted request schema accepts
`^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$` and rejects unknown fields.

- POST `/v1/audio/transcriptions`: JSON `request_id`, `model: "speechshift-stt"`,
  `language: "en"`, `encoding: "pcm_s16le"`, `sample_rate_hz: 16000`, `channels: 1`,
  `audio_base64`. At most eight seconds of mono PCM16, not a WAV file or multipart
  upload. Response contains non-empty `text` and the public model name.
- POST `/v1/translations`: JSON `request_id`, language-specific `model`, `input`,
  `source_language: "en"`, `target_language: "fr"` or `"de"`, matching the route.
  Response contains non-empty `output_text` and the public model name.
- POST `/v1/audio/speech`: JSON `request_id`, `model: "speechshift-voice"`, `input`,
  `voice` (`ryan`, `aiden`, `vivian`, `serena`), `language`, `response_format: "wav"`.
  Response is binary WAV, mono signed PCM16 at 24 kHz. SpeechShift rejects empty,
  mismatched or over-2,000,000-byte output. It does not use the full-duplex
  `/v1/speech/conversations` WebSocket.

SpeechShift cancels the active stage through POST `/v1/requests/{request_id}/cancel`
on cancellation, timeout or failure. Default gateway stage deadlines are 35 seconds
for recognition, 65 for translation and 130 for synthesis; consumer timeouts are
separate configuration. No stage substitutes a cloud service.

## Errors and compatibility aliases

Common cancellation returns HTTP 200 with `{ok, request_id, state, worker_id}`.
`ok: false` with `not-found` means no active gateway request mapping; completed
requests and gateway restarts can produce this. `worker-unavailable` means cancellation
could not reach the owning Worker. `ok: true` acknowledges the Worker's response;
confirm completion separately. An abort before a diffusion job ID is received cannot
be cancelled by job ID. No cancellation endpoint creates or starts a Worker.

| Condition | Contract |
| --- | --- |
| Missing model/name, wrong protocol surface, no usable local Worker for these demo requests | HTTP 503, `error.code: "local_route_unavailable"`, route and `cloud_fallback_attempted: false`. These cases share a code; inspect discovery to distinguish absent publication, protocol mismatch and unavailable Worker. |
| All candidate Workers busy | HTTP 429 `worker_busy`. |
| Duplicate active request ID | HTTP 409 `duplicate_request_id`. |
| Gateway deadline | HTTP 504 `gateway_timeout`; cancellation attempted. |
| Disconnect during a pending JSON request | Gateway diagnostic 499 `client_disconnected`; the disconnected client may receive nothing. |
| Malformed non-JSON Worker reply | HTTP 502 `worker_protocol_error`. Invalid trace metadata has its own `invalid_worker_trace_metadata` code. |
| Invalid audio/envelope | HTTP 413 or 422 `invalid_audio`; request schema validation can use FastAPI's `detail` array rather than `error`. |
| Unknown diffusion job | HTTP 404 `{detail: "Unknown diffusion job"}`. A failed known job instead reports terminal state and error in the job body. |
| Thermal rejection | HTTP 429 or 503 with `error.code`, `error.thermal`, no cloud fallback and optional `Retry-After`. Preserve the reason (e.g. `cooldown_required`, `critical_thermal_limit`, `thermal_queue_timeout`). |
| Persistence failure | HTTP 503 degraded response with persistence component/code. Never interpret failed discovery as an empty route list. |

Worker validation/errors are forwarded; consumers must tolerate both `error` objects
and `detail` envelopes, malformed JSON and transport failures. Do not turn a failed
live request into an unlabelled prepared result. Routed response headers identify
`X-ModelDeck-Worker-Id`, `X-ModelDeck-Configuration-Fingerprint` and
`X-ModelDeck-Route-Role`. Native identity is in discovery resolution, not a synthetic
model digest. See the [API contract](API_CONTRACT.md) for full identity semantics.

Supported aliases in this release, slated for removal in the next major release:

| Legacy | Canonical |
| --- | --- |
| POST `/native/autoregressive/trace` | POST `/native/v1/autoregressive/traces` |
| POST `/v1/refine` | POST `/native/v1/text-diffusion/refine` |
| POST `/v1/diffuse` | POST `/native/v1/text-diffusion/jobs` |
| GET `/v1/jobs/{job_id}` | GET `/native/v1/text-diffusion/jobs/{job_id}` |
| GET `/v1/jobs/{job_id}/events` | GET `/native/v1/text-diffusion/jobs/{job_id}/events` |
| POST `/v1/jobs/{job_id}/cancel` | POST `/native/v1/text-diffusion/jobs/{job_id}/cancel` |

Aliases preserve bodies/status and add `Deprecation: true` and
`Link: <canonical-path>; rel="successor-version"`, including unknown-job errors
fixed in this change. TokenTrail's separate experimental `/api/trace` and `/api/models`
backend is not a ModelDeck gateway contract. Legacy Event/Demo management endpoints
are not restored by these inference aliases.

## Read-only preflight and remaining machine checks

```powershell
pwsh -NoProfile -File scripts/operations/demo_preflight.ps1
pwsh -NoProfile -File scripts/operations/demo_preflight.ps1 -Demo SceneChat
pwsh -NoProfile -File scripts/operations/demo_preflight.ps1 -Demo TextDiffusionDemo -DiffusionModel text-diffusion-lab-q4
```

The script uses the existing control-plane Python and HTTPX. It sends only five GETs
(`/api/health`, `/api/live`, `/v1/health`, `/v1/routes`, `/v1/thermal`), with five-second
timeouts, no proxy environment or redirects. Explicit `-ManagementUrl` and `-GatewayUrl`
accept loopback HTTP base URLs with ports. It does not load `.env`, initialise SQLite,
invoke lifespan, reserve capacity, run a smoke check or mutate routing. The report
includes active profiles, required names, protocol compatibility, Worker resolution,
readiness and thermal observation separately. Unavailable discovery reports unknown
presence; successful empty discovery reports missing publication with an operator action.
An older gateway without protocol IDs reports unknown compatibility.

Exit 0 means the requested discovery checks passed with an enabled normal/warm thermal
snapshot at or below 85°C. Exit 1 means failed or unknown checks. This is only a snapshot
permitting rehearsal review, never hardware qualification or guaranteed admission.
Disabled/stale/missing thermal telemetry cannot pass. Stricter reported hot/critical
states take precedence, even below 85/90°C. Worker capacity can change after the GET.
The current inspected consumers' discovery bugs remain separate follow-up work even
when a preflight passes. TokenTrail's three defaults are all checked; custom public-name
configuration requires adjusting the consumer/preflight requirement mapping explicitly.

Before any physical rehearsal on the Fedora 44 / Ryzen AI Max+ 395 / Radeon 8060S /
128 GB shared-memory target, detect the actual OS, GPU/backend and memory; inspect
active thermal controls, all relevant sensors, available memory, swap pressure and
other workloads. Honour stricter kernel/firmware/operator limits. Pause or reduce above
85°C; terminate at 90°C. Record policy and fresh observations before, during and after
work. See [thermal validation](THERMAL_VALIDATION.md).

Choose the intended installation and plan its upgrade separately. Review existing
qualified Workers against complete model/artifact/runtime/backend identities, permissions
and current trusted runtime manifests before proposing publication. Publish demo
capabilities alongside other active profiles, with unique public names and explicitly
reviewed ordered backups. Do not replace another application's profile to recover demo
names. Then rehearse each real workload, cancellation and cooldown behaviour serially,
retaining raw evidence and distinguishing direct-runtime, Worker and gateway results.
No physical inference was run for this compatibility change; fixture tests are protocol
evidence only.

Validation: `scripts/verify.ps1` passed with local socket access: 655 backend tests,
21 frontend tests, lint, formatting, type checks and committed frontend asset checks.
Nine tests were skipped (five physical tests and four requiring torch in the
control-plane environment). After adding final preflight cases, the focused gateway,
demo contract and route-resolution suite passed 68 tests. The first sandboxed full
run encountered socket-permission failures; the permitted rerun resolved them.
Running the actual preflight against the stopped installation returned exit 1,
both services unavailable, nine unknown route-presence results and unknown thermal
status. It did not misreport missing routes based on failed HTTP discovery.

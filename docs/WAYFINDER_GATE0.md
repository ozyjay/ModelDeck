# WayFinder Gate 0

WayFinder uses ModelDeck's local OpenAI-compatible gateway at
`http://127.0.0.1:8600/v1`. It supplies one of two explicit `model` IDs on every
non-streaming request; ModelDeck performs no semantic routing or automatic model selection.

Run ModelDeck, then configure the dedicated profile:

```powershell
pwsh -NoProfile -File scripts/operations/run.ps1
pwsh -NoProfile -File scripts/configuration/configure_wayfinder_gate0.ps1
```

Prefix caching remains disabled unless each dedicated Worker has passed physical
qualification. To create new dedicated Workers with the feature enabled after qualification,
pass `-EnablePrefixCache`. An existing Worker is immutable: use the management replacement
operation, enable `prefix_cache_enabled` on the replacement, rebind the draft Route, then
publish it. The configuration script rejects an existing Worker whose cache setting differs.

When SprintBot needs Docker access, set `MODELDECK_ENABLE_DOCKER_BRIDGE=1` in `.env`.
`scripts/operations/run.ps1` then keeps WayFinder on `127.0.0.1:8600` and adds SprintBot's bridge
listener at `172.17.0.1:8600`.

The configuration creates two dedicated local Workers: `WayFinder Qwen2.5 0.5B Instruct`
for `fast-local` and `WayFinder Qwen2.5 3B Instruct` for `deep-local`. It uses only the
already cached, allowlisted Qwen snapshots, configured with their verified 32,768-token
model context and a 4,096-token output limit. It does not alter shared SprintBot Workers,
download a Model, start a Worker, configure an API key, or permit a cloud fallback. Start
the two Workers in the console or with their `/api/workers/{worker_id}/start` management
endpoints before calling the gateway.

Routing Profiles are independently active. Publishing `wayfinder-gate0` adds its two model
IDs alongside existing active profiles; it does not replace their routes. Public model IDs
must be unique across active profiles, so an attempted collision is rejected atomically.
For example, SprintBot can retain `sprintbot-qwen` and `sprintbot-embedding` while WayFinder
uses `fast-local` and `deep-local` through the same gateway. WayFinder’s Workers are distinct
from SprintBot’s, so their per-Worker generation locks do not serialise each other. ModelDeck
never switches between Workers implicitly.

The Qwen2.5 Transformers Worker accepts OpenAI-compatible `tools`, `tool_choice`, assistant
`tool_calls`, and `tool` result messages. It accepts up to 4 MiB of JSON request data, then
uses the rendered token count plus requested output against the configured context limit; it
does not impose a per-message character limit. It passes the conversation and tools to the local
Qwen chat template and converts the Qwen `<tool_call>` response envelope to OpenAI's
structured `tool_calls` response shape. This is model-directed function selection only:
ModelDeck does not execute tools. Tool-call reliability is model and prompt dependent; the
0.5B fast model is useful for simple calls, while the 3B model is the better Gate 0 option
for multi-step tool use.

## Safe stable-prefix caching

WayFinder may attach this advisory extension to a message-based request while continuing to
send the complete OpenAI request:

```json
{
  "modeldeck": {
    "prefix_cache": {
      "stable_message_count": 1,
      "profile_version": "wayfinder-agent-v1"
    }
  }
}
```

WayFinder owns the classification of stable leading messages and increments
`profile_version` whenever that operating preamble changes. A hint may identify at most 64
leading messages, all of which must have the `system` role, and must leave at least one
dynamic message. ModelDeck owns prompt rendering, tokenisation, exact-prefix verification and
all K/V state. It renders the full prompt once, renders the proposed stable messages with the
same tokenizer, chat template, tools and tool choice but no generation suffix, and reuses K/V
state only when the proposed token IDs are an exact prefix of the full rendered prompt.

Each enabled Worker retains at most one immutable entry: 8,192 prefix tokens and 512 MiB of
measured K/V tensors. Every request receives a distinct cloned cache branch before its dynamic
suffix is processed. The entry identity covers the pinned model and artefact revision, model
and tokenizer configuration, chat template, Transformers runtime, dtype, context and RoPE
configuration, adapter identity, tool definition and choice, operating-profile version, and
the internal Worker load epoch. Model reload, restart, replacement, explicit clearing,
configuration change, corruption or accelerator-memory exhaustion invalidates it.

Absent hints, disabled caching, token mismatches, unsupported cache layouts, oversized
prefixes and ordinary cache errors are safe bypasses: the Worker performs an ordinary full
prefill and uses the same incremental decode path. Accelerator-memory exhaustion clears the
entry and fails the request without an in-request retry. Cancellation is checked around
prefix prefill, branch creation and suffix prefill, then between generated tokens. Gateway
timeouts and client disconnects send the request ID to the owning Worker's cancellation
route, so abandoned inference does not continue.

Worker `GET /metrics` reports cache state, entry count and bytes, hit/miss/bypass/eviction and
clear counts, its load epoch, and an opaque configuration fingerprint. Per-request trace
metrics report only bounded status and reason codes, token counts, prefill and decode timing,
output rate and cache bytes. Prompt text, token IDs, cache keys and K/V contents are never
logged. Clear a ready, idle Worker through:

```text
POST /api/workers/{worker_id}/prefix-cache/clear
```

The response contains only `cleared_entries` and `released_bytes`. `/v1/models` exposes a
namespaced `modeldeck` diagnostic identity containing the underlying model ID, pinned
revision, runtime, opaque configuration fingerprint, cache support and enabled state; the
load epoch is not part of that public identity.

Qualify the 0.5B and 3B Workers independently after enabling their replacement Workers:

```powershell
pwsh -NoProfile -File scripts/qualification/qualify_wayfinder_prefix_cache.ps1 `
    -Workers '<0.5B-worker-id>','<3B-worker-id>' `
    -Repetitions 5 `
    -Output 'var/benchmarks/wayfinder-prefix-cache.json'
```

The focused runner uses pinned deterministic cold-miss, warm-hit and deliberate-bypass
requests at concurrency one. It requires identical selected token IDs and output text,
top-k probability agreement within `1e-5`, hidden-state summary agreement within `1e-4`, at
least 20% lower median warm-hit time to first token, both cache limits, successful prompt
cancellation and no monotonic Worker/GTT/host-memory growth. It checks ModelDeck's active
thermal status before each case and stops at the configured critical state; it does not
define a separate temperature threshold. A passing report is evidence for that exact model,
revision and configuration only.

Minimal non-streaming smoke tests:

```powershell
curl.exe http://127.0.0.1:8600/v1/models
curl.exe http://127.0.0.1:8600/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"fast-local","messages":[{"role":"user","content":"Reply with fast."}],"stream":false}'
curl.exe http://127.0.0.1:8600/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"deep-local","messages":[{"role":"user","content":"Reply with deep."}],"stream":false}'
```

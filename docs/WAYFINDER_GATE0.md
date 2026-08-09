# WayFinder Gate 0

WayFinder uses ModelDeck's local OpenAI-compatible gateway at
`http://127.0.0.1:8600/v1`. It supplies one of two explicit `model` IDs on every
non-streaming request; ModelDeck performs no semantic routing or automatic model selection.

Run ModelDeck, then configure the dedicated profile:

```powershell
pwsh -NoProfile -File scripts/run.ps1
pwsh -NoProfile -File scripts/configure_wayfinder_gate0.ps1
```

When SprintBot needs Docker access, set `MODELDECK_ENABLE_DOCKER_BRIDGE=1` in `.env`.
`scripts/run.ps1` then keeps WayFinder on `127.0.0.1:8600` and adds SprintBot's bridge
listener at `172.17.0.1:8600`.

The configuration selects the already configured local Workers `Qwen2.5 0.5B Instruct`
for `fast-local` and `Qwen2.5 3B Instruct` for `deep-local`. It does not download a Model,
start a Worker, configure an API key, or permit a cloud fallback. Start the two Workers in
the console or with their `/api/workers/{worker_id}/start` management endpoints before
calling the gateway.

Routing Profiles are independently active. Publishing `wayfinder-gate0` adds its two model
IDs alongside existing active profiles; it does not replace their routes. Public model IDs
must be unique across active profiles, so an attempted collision is rejected atomically.
For example, SprintBot can retain `sprintbot-qwen` and `sprintbot-embedding` while WayFinder
uses `fast-local` and `deep-local` through the same gateway. The current `fast-local` and
`sprintbot-qwen` bindings deliberately share the 0.5B Worker, so its per-Worker generation
lock serialises their requests. Configure a separate compatible Worker and bind it to one
capability if workload isolation is needed; ModelDeck never switches between them implicitly.

The Qwen2.5 Transformers Worker accepts OpenAI-compatible `tools`, `tool_choice`, assistant
`tool_calls`, and `tool` result messages. It passes the conversation and tools to the local
Qwen chat template and converts the Qwen `<tool_call>` response envelope to OpenAI's
structured `tool_calls` response shape. This is model-directed function selection only:
ModelDeck does not execute tools. Tool-call reliability is model and prompt dependent; the
0.5B fast model is useful for simple calls, while the 3B model is the better Gate 0 option
for multi-step tool use.

Minimal non-streaming smoke tests:

```powershell
curl.exe http://127.0.0.1:8600/v1/models
curl.exe http://127.0.0.1:8600/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"fast-local","messages":[{"role":"user","content":"Reply with fast."}],"stream":false}'
curl.exe http://127.0.0.1:8600/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"deep-local","messages":[{"role":"user","content":"Reply with deep."}],"stream":false}'
```

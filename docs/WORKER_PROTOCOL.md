# Worker protocol version 1

Every worker exposes `GET /health`, `/capabilities`, `/metrics`, `/model` and `POST
/load`, `/warmup`, `/cancel`, `/shutdown`. Health declares protocol version, worker,
runtime, explicit generation family, model revision, device, state, and readiness.

## Autoregressive worker

Canonical routes are `POST /v1/chat/completions`, `/v1/completions`, and
`/native/autoregressive/trace`. A trace records prompt token IDs, selected generated token
ID/string, normalised probability, top-k alternatives, accumulated text, and timestamp.
These are observable model outputs and must not be described as private reasoning.

The implemented ROCm worker supports local-only pinned load, disabled trusted remote
code, chat templates, seeds, temperature/top-p/top-k, repetition penalty, stop sequences,
cancellation, one active generation, first-token and total latency, tokens per second,
top-k trace events, prompt/generated token IDs, and optional hidden-state summaries.
It advertises health while loading and becomes ready only after explicit warmup.

## Text-diffusion worker

Canonical routes are `POST /v1/refine`, `/v1/diffuse`, `GET /v1/jobs/{job_id}`, `POST
/v1/jobs/{job_id}/cancel`, and `GET /v1/jobs/{job_id}/events`. Frame events contain step,
total steps, text, masked/stable token counts where available, completion, and seed.
Native iterative refinement is canonical; it is not implemented by calling an AR token
loop.

The mock is deterministic and contract-shaped. It is not evidence that a real model or
ROCm stack works.

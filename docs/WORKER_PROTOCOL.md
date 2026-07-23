# Worker protocol version 1

Every worker exposes `GET /health`, `/capabilities`, `/metrics`, `/model` and `POST
/load`, `/warmup`, `/cancel`, `/shutdown`. Health declares protocol version, worker,
runtime, explicit generation family, model revision, device, state, and readiness.

## Autoregressive worker

Canonical routes are `POST /v1/chat/completions`, `/v1/completions`, and
`/native/autoregressive/trace`. A trace records prompt token IDs, selected generated token
ID/string, normalised probability, top-k alternatives, accumulated text, and timestamp.
The trace response also includes worker-tokenizer-owned readable prompt metadata:

- `prompt_token_ids` is the complete tokenised inference context, including system
  instructions, chat-template control tokens, and the assistant-generation marker.
- `prompt_tokens` is that same complete context decoded one token at a time by the worker's
  exact tokenizer. It includes special tokens and aligns one-to-one with
  `prompt_token_ids`.
- `user_prompt_token_ids` and `user_prompt_tokens` contain only the latest user message.
  They exclude system instructions, earlier messages, role wrappers, generation markers,
  and other chat-template controls. `user_prompt_tokens` is the safe field for a public
  prompt-token display.

For a plain `prompt` request, the complete context may include tokenizer-added special
tokens while the user fields represent the prompt text without automatically inserted
special tokens. For a `messages` request, the complete fields describe the rendered chat
template used for inference, while the user fields are tokenised directly from the latest
user message content by the same worker tokenizer. Per-token decoding disables special-token
skipping and tokenisation-space clean-up so whitespace and token boundaries are preserved as
accurately as the tokenizer permits. The gateway validates these alignments and never loads
or substitutes a tokenizer.

Example non-streaming response (generation fields abbreviated):

```json
{
  "request_id": "8d638d96-23ca-4ce5-bfa1-f12cf131947e",
  "model": "worker-a1b2c3d4",
  "prompt_token_ids": [151644, 8948, 198, 9707, 151645, 198, 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198],
  "prompt_tokens": ["<|im_start|>", "system", "\n", "Be concise.", "<|im_end|>", "\n", "<|im_start|>", "user", "\n", "Hello", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"],
  "user_prompt_token_ids": [9707],
  "user_prompt_tokens": ["Hello"],
  "events": [
    {
      "step": 0,
      "selected": {"token_id": 9707, "token": "Hello", "probability": 0.81},
      "alternatives": [],
      "text_so_far": "Hello",
      "complete": false
    }
  ],
  "metrics": {"generated_tokens": 1}
}
```

Token strings depend on the selected model tokenizer; the values above are illustrative.
These are observable model outputs and must not be described as private reasoning.

The implemented ROCm worker supports local-only pinned load, disabled trusted remote
code, chat templates, seeds, temperature/top-p/top-k, repetition penalty, stop sequences,
cancellation, one active generation, first-token and total latency, tokens per second,
top-k trace events, prompt/generated token IDs, and optional hidden-state summaries.
It advertises health while loading and becomes ready only after explicit warmup.

## Text-diffusion worker

Canonical routes are `POST /v1/refine`, `/v1/diffuse`, `GET /v1/jobs/{job_id}`, `POST
/v1/jobs/{job_id}/cancel`, and `GET /v1/jobs/{job_id}/events`. Frame events contain step,
total steps, text, masked/stable token counts where available, completion, and seed. A
terminal frame never exceeds its declared total steps and reports `finish_reason` as
`stop`, `length`, or `cancelled`. Model-specific structured response parsing removes
private reasoning channels from public frame and result text.
Native iterative refinement is canonical; it is not implemented by calling an AR token
loop. Job event streams publish refinement frames as the engine produces them rather than
waiting to replay the completed frame collection.

The autoregressive, text-diffusion, and SceneChat mocks are deterministic and
contract-shaped. A SceneChat mock returns schema-valid placeholder analysis without
inspecting the supplied image and is signalled by the gateway with
`x-modeldeck-fallback: mock`. Mock output is not evidence that a real model or ROCm stack
works.

## SceneChat vision-language compatibility worker

`scenechat-gemma4-e2b-rocm` is a ModelDeck-managed, on-demand worker that can remain
loaded alongside one exclusive DiffusionGemma worker and binds directly to
`127.0.0.1:8000`. The stable gateway exposes it as `scenechat-vision` through
`127.0.0.1:8600/v1/chat/completions` and `127.0.0.1:8600/v1/vision/analyse`. Its direct
compatibility routes are authenticated
`GET /v1/models`, `POST /v1/chat/completions`, and
`POST /native/vision-language/smoke`.

Gemma 4 unified snapshots use the same SceneChat contract through the explicitly paired
`Gemma4UnifiedProcessor` and `Gemma4UnifiedForConditionalGeneration` loader adapter. Audio and
video inputs remain outside the SceneChat contract.

The dedicated Qwen3.5 adapter accepts only the official `Qwen/Qwen3.5-0.8B`,
`Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`, and `Qwen/Qwen3.5-9B` repositories with the
`Qwen3_5ForConditionalGeneration` architecture. It pairs `Qwen3VLProcessor` with
`Qwen3_5ForConditionalGeneration`, uses local files without remote code, and bounds image
pixels to the configured visual-token budget. Catalogue recognition is not physical ROCm
compatibility evidence; readiness still requires a successful load and warm-up on the
detected stack. Packaged runtime profile 0.2.0 defaults Qwen3.5 to 140 visual tokens and
a 1,024-token hard completion ceiling. The larger ceiling is truncation headroom; the
canonical output contract is designed to keep normal responses near 180–260 tokens.

The Repartee speech protocol and its fixed PCM framing are documented in
`docs/REPARTEE_RUNTIMES.md`. Autoregressive GPT-OSS requests continue to use the existing
OpenAI-compatible chat and completion routes.
The gateway accepts either the stable `scenechat-vision` alias or the exact pinned model
identifier, translates it to the exact worker model identifier, and injects the private
loopback worker credential. It never forwards a caller-supplied credential to the worker.

The built-in profile accepts `google/gemma-4-E2B-it` revision
`9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`; recognised local profiles may instead use an
allowlisted Gemma 4 unified snapshot. Each profile uses its pinned processor, chat template,
and image processing; neither the gateway nor SceneChat loads a tokenizer or processor.
Generation uses deterministic greedy decoding so the strict JSON contract does
not depend on a stochastic sampling path. Gemma 4 retains its 512-token default, while the
Qwen3.5 0.2.0 profile uses a 1,024-token ceiling; both retain the 60-second deadline.
Disconnect polling is bounded to avoid starving the generation thread.
Readiness remains false until local processor/model loading and a one-token synthetic-image
warm-up have succeeded.

All trainable floating-point parameters must be BF16 on `cuda:0`. Gemma 4's named rotary,
scaling, range, soft-cap, and standardisation buffers may remain FP32 where Transformers
5.13.0 deliberately uses FP32 for numerical stability; any FP32 parameter, unknown mixed
buffer, CPU tensor, or disk offload still fails readiness. Metrics report the detected
parameter/buffer dtypes and the count of approved FP32 numerical buffers.

The OpenAI-compatible request contains one user message with exactly one JPEG or PNG data
URL followed by one text part. The text must exactly match one of the versioned SceneChat
contract prompts. The worker extracts only the curated question, places the canonical
safety rules and visible-text invariant in the system role, and supplies an in-memory RGB
image directly to the processor. External URLs, SVG, additional images, arbitrary prompts,
streaming, and over-limit input are rejected.

The canonical response is limited to fewer than 45 summary words, at most eight objects,
three relationships, three uncertainties and one safety note. Object descriptions contain
at most 15 words. Relationship, uncertainty and safety entries are limited to one short
sentence. These limits are enforced by both the packaged JSON schema and the Pydantic
validator; prompt wording alone is never treated as sufficient enforcement.

Example successful response:

```json
{
  "id": "chatcmpl-e4f66ea6f7c748c6a2b28d93f82e932a",
  "object": "chat.completion",
  "created": 1784174400,
  "model": "google/gemma-4-E2B-it",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "{\"summary\":\"A monitor is visible on a desk.\",\"objects\":[],\"relationships\":[],\"uncertainties\":[],\"safety_notes\":[]}"
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 624, "completion_tokens": 39, "total_tokens": 663}
}
```

Output is returned only after strict schema and public-safety validation. A single JSON
fence may be accepted internally, but successful content is reserialised as bare compact
JSON. Invalid output returns `502 invalid_model_output`; it is never repaired, retried,
fabricated, or replaced. One request may run at a time and a second is rejected immediately
with 429. The worker is implemented, but is not Open Day ready until the physical gates pass.

Validation failures record only the request ID, a safe failure category, token counts,
effective token limit, whether that limit was reached, finish reason and elapsed time.
Metrics additionally expose preprocessing, inference, validation and total worker time,
visual-token count and token throughput. Safety failures distinguish
`prohibited_identity` from `prohibited_sensitive_attribute` without recording the matched
text. Images, prompts, raw model output, visitor-facing text, credentials, and headers are
never logged.

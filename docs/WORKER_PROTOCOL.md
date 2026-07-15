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
  "model": "token-explainer",
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

The mock is deterministic and contract-shaped. It is not evidence that a real model or
ROCm stack works.

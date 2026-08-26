# Qwen3.5 4B Q8 llama.cpp Vulkan runtime

ModelDeck recognises one immutable Qwen3.5 4B Q8_0 GGUF snapshot as the candidate
`qwen35-llamacpp-vulkan` runtime. It is intended for the concise Wayfinder Worker and is
separate from the official BF16 Transformers Workers.

## Trusted inputs

The code-owned `qwen35-4b-q8-vulkan` manifest pins:

- original model `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- GGUF repository `bartowski/Qwen_Qwen3.5-4B-GGUF` revision
  `4168f45a16a1290d65a4ec0fa312ae917a4c15d6`;
- `Qwen_Qwen3.5-4B-Q8_0.gguf`, its exact byte size and SHA-256 digest;
- llama.cpp commit `9d77fa17254e1dee4b9e92504c91611a60b1359f`; and
- Linux x86_64, Vulkan, full GPU layer offload, an 8,192-token context and Q8_0 KV cache.

ModelDeck only discovers the local Hugging Face cache. It does not download or update
weights, and the Worker launches llama.cpp with offline mode enabled on `127.0.0.1`.
Startup verifies the model size and digest plus the pinned llama-server build receipt.

## Behaviour and limits

The runtime exposes OpenAI-compatible chat and text completion. Thinking is immutable at
`disabled`: ModelDeck starts llama.cpp with `reasoning-effort=none`, injects `none` into
forwarded requests, and rejects an attempt to request another reasoning effort. The
default output ceiling is 256 tokens.

This GGUF contains no vision projector or MTP companion model. The Worker therefore does
not claim image chat, speculative decoding or native autoregressive traces. It remains a
reviewed candidate until load, warm-up, generation, cancellation and sustained thermal
tests pass on the target Framework Desktop.

Create it from the Models page after ModelDeck has rediscovered the exact cached snapshot.
The generated Worker definition uses the `Qwen3.5 4B Q8 llama.cpp Vulkan` runtime template.

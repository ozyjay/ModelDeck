# Qwen3.8 llama.cpp Vulkan runtime

ModelDeck has a separate, candidate runtime for `Qwen/Qwen3.8-27B` using a directly
managed `llama-server`. It does not replace the Transformers/ROCm FP8 workers, use their
Python environment, or depend on Ollama. The runtime is not considered reviewed until the
physical qualification below passes on the Framework Desktop.

## Trusted candidate

The packaged `qwen38-llamacpp-q8-mtp-vulkan` manifest pins:

- the original Qwen revision and the derived `ggml-org/Qwen3.8-27B-GGUF` revision;
- the Q8_0 model, BF16 vision projector and Q8_0 MTP model filenames, sizes and SHA-256
  digests;
- llama.cpp commit `9d77fa17254e1dee4b9e92504c91611a60b1359f`;
- Linux x86_64, Vulkan, the `qwen35` GGUF architecture, an 8,192-token context, Q8_0 KV
  caches and four MTP draft tokens; and
- the Apache-2.0 source licence and immutable source URL.

The experimental Q4_K_M artefacts have a separate packaged data manifest, but no runtime
template is exposed. Q4 must pass the FP8-referenced capability evaluation before it can
be selected. A future ROCmFPX build requires a separate implementation ID, source pin and
build receipt; it must not reuse or weaken the stock Vulkan trust boundary.

Run `pwsh -NoProfile -File scripts/setup/setup_llama_vulkan.ps1` to build the pinned
Vulkan tools. The script checks the exact MTP flags and writes a receipt containing the
resulting executable digest. At worker start, ModelDeck verifies the receipt and hashes all
three model artefacts before executing the fixed binary at the configured local runtime path.
To provision a desktop service independently of a source checkout, provide its runtime
location explicitly:

```powershell
pwsh -NoProfile -File scripts/setup/setup_llama_vulkan.ps1 `
  -RuntimeRoot ~/.runtime-tools/llama.cpp
```

Neither the management API nor a runtime template can supply a binary, projector, draft
model, environment variable or llama.cpp argument.

## Lifecycle and startup evidence

The Python worker allocates a private loopback port immediately before launch, starts
`llama-server` in its own process group, drains bounded stdout and stderr, and terminates
the complete group on unload, failure or cancellation. Cancellation recreates the private
server because a partially cancelled llama.cpp slot is not treated as safe for reuse.

Readiness fails closed unless server output confirms Vulkan, the expected AMD Vulkan
device, complete layer offload, Qwen3.8/qwen35 architecture, expected quantisation, the
vision projector and `draft-mtp`. Warm-up performs a deterministic generation and requires
at least one accepted MTP token. `/model` reports immutable digests and configuration
identity without local paths. `/metrics` reports proposed, accepted and rejected draft
tokens, acceptance ratio, effective generation rate, prompt-processing rate, load time and
whole-device memory counters.

The worker preserves llama.cpp's OpenAI-compatible chat, completion, streaming, image,
tool-call, structured-output, finish-reason and usage fields. ModelDeck exposes two
distinct immutable Worker templates over the same pinned artefacts:

- `qwen38-llamacpp-q8-mtp-vulkan` uses `thinking_mode=adaptive`, advertises reasoning,
  and leaves `reasoning_effort` unset when a request omits it. A request may select one of
  llama.cpp's reviewed effort values (`default`, `none`, `minimal`, `low`, `medium`,
  `high`, `xhigh` or `max`).
- `qwen38-llamacpp-q8-mtp-vulkan-no-thinking` uses `thinking_mode=disabled`, starts
  llama.cpp with `reasoning_effort=none`, injects `none` into every request, rejects any
  other effort and defensively removes reasoning-only response fields.

Both retain image, tool, structured-output and MTP support; only the adaptive template
claims reasoning. Other effort values are rejected before they reach the private server.
Health, model and metrics responses attest the policy. The thinking policy is included in
both runtime and Worker configuration fingerprints, so each variant accumulates separate
qualification evidence.

## Physical qualification and acceptance

Mark physical cases with `hardware`, `rocm`, `large_model` or `long_running` as applicable.
Record the full ModelDeck fingerprint and thermal telemetry for every pass and failure.
Use the same artefacts and equivalent sampling, cache and context settings for:

1. the pinned `llama-server` directly, with MTP off and on;
2. the pinned server managed by ModelDeck, with MTP off and on where measuring the
   speculative contribution; and
3. Ollama as an informative external baseline, clearly recording its different build.

Measure cold and warm load time, prompt processing, TTFT, sustained effective generation
rate, MTP acceptance, memory, short and 512-token outputs, and 8K, 32K and a longer
hardware-safe context. Exercise code, prose, JSON, tools, vision, repeated multi-turn use,
cancellation and reload. Stop sustained work at the project thermal threshold.

Q8 can become reviewed only when text, vision, tools, structured output, reasoning modes,
multi-turn state, long context and deterministic greedy comparisons pass against the
official FP8 reference; cancellation and supervisor shutdown leave no process or port;
and ModelDeck achieves at least 95% of direct pinned-server sustained generation rate.
Perplexity or isolated successful prompts are not sufficient. Known upstream risks to
track include context-dependent MTP acceptance, long-generation throughput degradation,
Vulkan support on gfx1151, reasoning-template changes and tool-heavy prompt performance.

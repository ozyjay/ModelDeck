# Model profile schema

Profiles are typed Pydantic documents and reject unknown fields, unsafe identifiers,
unallowlisted runtimes, invalid ports, and contradictory generation capabilities.

Required fields include `id`, exact `model_id`, pinned `revision`, stable `alias`, explicit
`generation_family`, `preferred_runtime`, lifecycle class, fixed port, local-files-only
policy, trusted-remote-code policy, dtype, capabilities, and family-specific settings.

Real autoregressive profiles are:

- `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775` as the tested small AR profile,
  `token-explainer`, resident, FP16 and local only;
- `Qwen/Qwen2.5-1.5B-Instruct` revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` as the `qwen-1-5b` worker on port 8623;
- `Qwen/Qwen2.5-3B-Instruct` revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1` as the `qwen-3b` worker on port 8624.

All three Qwen profiles use the isolated Transformers ROCm runtime, FP16, local-only
loading, disabled remote code, and the fixed `/mnt/work/models/huggingface/hub` cache.
Only the 0.5B profile currently has physical compatibility evidence; registering the
larger cached workers does not claim that they have passed the target GPU acceptance run.

The vision-language compatibility profile is `scenechat-gemma4-e2b-rocm`:

- exact model `google/gemma-4-E2B-it` at revision
  `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`;
- alias `scenechat-vision`, family `vision-language`, runtime
  `vision-language-transformers-rocm`, on-demand lifecycle, and fixed port 8000;
- BF16, local-only load, disabled trusted remote code, 8192-token context, 512-token output,
  60-second generation deadline, 600-second startup, and 180-second warm-up;
- compatibility-only chat, image input, structured output, cancellation, and no streaming.

`image_input` and `structured_output` are additive `CapabilitySet` fields and default to
false for existing profiles. A SceneChat-compatible vision-language profile must advertise
both. Registration records an allowlisted implementation, not physical compatibility or
Open Day readiness.

Real text-diffusion profiles are:

- the self-contained ModelDeck GPTQ Q4 g32/BF16 hybrid variant of
  `google/diffusiongemma-26B-A4B-it` as the default `text-diffusion` provider, exclusive
  and local only;
- the original BF16 DiffusionGemma profile as the explicit `text-diffusion-bf16`
  compatibility and evaluation baseline.

The Qwen profile has load, warmup, smoke, cancellation, 30-minute stability, shutdown and
process-exit evidence. DiffusionGemma must pin its resolved snapshot and must not be marked
runnable until equivalent Phase 4 evidence exists.

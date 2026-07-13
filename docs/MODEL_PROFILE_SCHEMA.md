# Model profile schema

Profiles are typed Pydantic documents and reject unknown fields, unsafe identifiers,
unallowlisted runtimes, invalid ports, and contradictory generation capabilities.

Required fields include `id`, exact `model_id`, pinned `revision`, stable `alias`, explicit
`generation_family`, `preferred_runtime`, lifecycle class, fixed port, local-files-only
policy, trusted-remote-code policy, dtype, capabilities, and family-specific settings.

Initial real profiles are:

- `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775` as the tested small AR profile,
  `token-explainer`, resident, FP16 and local only;
- the ModelDeck expert-only GPTQ Q4 g32 variant of
  `google/diffusiongemma-26B-A4B-it` as the default `text-diffusion` provider, exclusive
  and local only;
- the original BF16 DiffusionGemma profile as the explicit `text-diffusion-bf16`
  compatibility and evaluation baseline.

The Qwen profile has load, warmup, smoke, cancellation, 30-minute stability, shutdown and
process-exit evidence. DiffusionGemma must pin its resolved snapshot and must not be marked
runnable until equivalent Phase 4 evidence exists.

# Model profile schema

Profiles are typed Pydantic documents and reject unknown fields, unsafe identifiers,
unallowlisted runtimes, invalid ports, and contradictory generation capabilities.

Required fields include `id`, exact `model_id`, pinned `revision`, stable `alias`, explicit
`generation_family`, `preferred_runtime`, lifecycle class, fixed port, local-files-only
policy, trusted-remote-code policy, dtype, capabilities, and family-specific settings.

Initial real candidates, subject to cache and hardware evidence, are:

- `Qwen/Qwen2.5-0.5B-Instruct` as a small AR profile, `fast-chat`, resident, local only;
- `google/diffusiongemma-26B-A4B-it` as text diffusion, `text-diffusion`, exclusive,
  local only.

Their profiles must pin resolved snapshot commits before Phase 3/4 and must not be marked
runnable until load, warmup, smoke, shutdown, and memory-recovery evidence exists.


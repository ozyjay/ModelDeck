# Model profile schema

Built-in profiles are packaged as versioned seed data in
`backend/modeldeck/registry_data/model_profiles.json`; they are validated through the same
`ModelProfile` schema as SQLite-backed local profiles. Recognised cache architectures map
to entries in `runtime_templates.json`, while application-facing reserved aliases and
their capability contracts live in `reserved_aliases.json`. JSON data cannot provide a
command, executable, environment variable, endpoint, or arbitrary filesystem path to the
worker launcher; each runtime ID must still map to a trusted Python launch builder.

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

The operator may create additional profiles from exact, complete snapshots recognised by
the local Hugging Face cache scanner when the architecture matches the allowlisted
autoregressive Transformers, SceneChat Gemma 4, DiffusionGemma BF16, or manifest-verified
ModelDeck DiffusionGemma Q4 worker. The browser
supplies only a safe alias and bounded family-relevant settings. ModelDeck fixes the
worker implementation, offline and remote-code policy, capability set, cache root, and the
first free port from 8630 through 8699. Diffusion profiles are always exclusive. These
profiles use `local-<alias>` identifiers,
are stored in `model_profiles`, and remain observationally untested until their worker is
started and smoke tested. Removing one removes only the runtime configuration.

A Hugging Face Q4 profile records two exact identities. `artifact_model_id` and
`artifact_revision` identify the downloaded derivative release for library matching and
allow/disallow policy. `model_id` and `revision` retain the pinned upstream Google base
identity required by the custom loader and compatibility evidence. Configuration hashes
the complete release inventory and refuses missing shards, altered files, failed quality
evidence, unsupported quantisation metadata, or a different base revision.

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

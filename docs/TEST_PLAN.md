# Test plan

Normal verification is `pwsh -NoProfile -File scripts/verify.ps1`. It requires no GPU,
network, model download, container runtime, or cached model. It runs frontend TypeScript
checking and Vitest tests, proves the committed production bundle matches `frontend/`,
then runs Ruff and the GPU-free pytest suite.

The end-to-end mock gateway smoke is
`pwsh -NoProfile -File scripts/smoke_all.ps1`; it starts and always stops the local
services around both generation-family checks.

The hardware-gated AR acceptance smoke is
`pwsh -NoProfile -File scripts/smoke_rocm_autoregressive.ps1`. It loads the pinned cached
Qwen worker, records stack/latency/torch-memory evidence, confirms process exit, and stops
all services it started. It never downloads a model.

In-flight hardware cancellation is checked separately with
`pwsh -NoProfile -File scripts/smoke_rocm_cancellation.ps1`.

The hardware-gated text-diffusion acceptance smoke is
`pwsh -NoProfile -File scripts/smoke_rocm_text_diffusion.ps1`. It loads the pinned local
DiffusionGemma snapshot through its separate native diffusion worker, records frame-shaped
smoke evidence, confirms process exit, and never downloads a model. It must pass before
the profile is described as tested-working on the target hardware.

The 30-minute acceptance run is
`pwsh -NoProfile -File scripts/stability_rocm_autoregressive.ps1`. It records duration,
request count, failures, shutdown and process-exit evidence against the compatibility
fingerprint.

The corresponding GPT-OSS Vulkan acceptance run is
`pwsh -NoProfile -File scripts/stability_gpt_oss.ps1 -DurationMinutes 30`. It uses the
verified `repartee-strong` provider, records latency and failures without retaining prompts
or output, samples peak whole-device GTT use, and checks GTT recovery after process exit.

The recorded Qwen run lasted 1,808.851 seconds and completed 343 gateway requests with
zero failures. The in-flight cancellation and repeated start/stop checks also passed on
the physical Framework Desktop.

The cross-profile physical performance suite is
`pwsh -NoProfile -File scripts/benchmark_models.ps1`. It runs one excluded benchmark
warm-up and five measured representative requests per selected physical worker, records
versioned JSON and Markdown reports, rejects mock gateway fallback, and restores the
initial worker state. `-Preset Quick` reduces measured requests to two. This is an
observational benchmark rather than a compatibility or release gate and remains outside
normal CI.

Unit tests cover profiles, cache resolution and state, fingerprints, launch arguments,
redaction, and hardware probe resilience. Contract tests prove AR traces and diffusion
frames remain distinct and that gateway failures are local and structured. Integration
tests launch real isolated mock subprocesses for readiness, restart, shutdown, port
collision, and crash detection.

Physical tests are marked `hardware`, `rocm`, `large_model`, or `long_running` and are
excluded from normal CI, but remain required for target-product acceptance. Phase 3/4
requires allocation, load, stream/frame,
cancellation, memory recovery, repeated lifecycle, 30-minute per-worker, and selected
preset two-hour tests.

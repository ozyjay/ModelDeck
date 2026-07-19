# ROCm model benchmarks

ModelDeck provides a hardware-gated benchmark suite for repeatable performance and
stability observations on the target Framework Desktop. It never downloads weights and
does not replace compatibility smoke tests or workload-specific quality evaluation.

## Run the suite

Run the standard suite across every configured non-mock Worker that has exactly one
published Route:

```powershell
pwsh -NoProfile -File scripts/benchmark_models.ps1
```

Use the quick preset to validate the benchmark setup with two measured requests per
Worker:

```powershell
pwsh -NoProfile -File scripts/benchmark_models.ps1 -Preset Quick
```

Select one or more Workers by editable name or UUID when a full run is unnecessary:

```powershell
pwsh -NoProfile -File scripts/benchmark_models.ps1 -Preset Standard `
    -Workers 'Small Qwen','Medium Qwen','Large Qwen'
```

Workers created with the autoregressive Transformers, llama.cpp Vulkan, text-diffusion,
DiffusionGemma Q4 and SceneChat runtimes are accepted. The GPT-OSS workload allows up to
256 generated tokens so it can complete internal reasoning and still return visible
output; throughput includes every generated token reported by llama.cpp. Run it by its
chosen Worker name:

```powershell
pwsh -NoProfile -File scripts/benchmark_models.ps1 `
    -Workers 'GPT OSS Vulkan'
```

The suite resolves the selected Worker and its one published public Route before it starts.
It never changes Event routing or Worker configuration.

Use `-JsonOutput` and `-MarkdownOutput` to override the default timestamped paths under
`var/benchmarks/`. The paths must be different.

## Workloads and measurements

Each Worker receives one excluded benchmark warm-up followed by two measured requests
for `Quick` or five for `Standard`. Both presets use the same representative workload so
their measurements remain comparable:

- autoregressive workers generate exactly 64 tokens with a fixed seed and deterministic
  decoding;
- text-diffusion workers refine 128 tokens over 24 steps with a fixed seed;
- SceneChat analyses a generated 64-by-64 local PNG using the approved **Describe the
  scene** contract and a 256-token ceiling.

Reports include cold-start wall time, worker model-load time, end-to-end and worker
latency, time to first output and token throughput where the protocol supplies them,
steady and peak device memory, request reliability, deterministic output hashes, and
before/after host memory, temperature, and fan readings. Summaries report minimum,
median, nearest-rank p95, and maximum values.

Models are grouped by generation family. Latency or throughput from different generation
families must not be treated as a common leaderboard.

## Lifecycle and privacy

Benchmarking is deliberately disruptive. It refuses to start while any managed Worker is
busy or transitioning, records which workers are initially ready, stops all managed
Workers, and benchmarks one physical Worker at a time. It restores the original ready
Workers after success, failure, or interruption. If the wrapper started ModelDeck, it
stops the services when the run ends.

The selected Worker must be physical; a mock Worker is rejected rather than reported as
physical performance. Missing snapshots, Worker failures, request failures, and
unsuccessful lifecycle restoration produce a
non-zero exit after the available report is written. Performance values themselves are
observational and have no regression thresholds in this version.

Reports do not contain prompts, generated output, images, credentials, visitor data,
active process commands, or local cache paths. Generated output is retained only as a
SHA-256 digest for deterministic-run comparison.

## Relationship to other checks

- `smoke_rocm_*.ps1` establishes one-request compatibility evidence against the append-only
  compatibility registry.
- `evaluate_diffusiongemma_q4.ps1` applies quality, determinism, memory, latency, and
  release gates specifically to Q4 versus BF16 DiffusionGemma.
- `benchmark_models.ps1` records repeatable observational performance across physical
  profiles without writing compatibility evidence or enforcing performance gates.

Physical benchmark runs require the target GPU, pinned local snapshots, the relevant
ROCm environments, substantial memory, and time. They are not part of normal CI.

Run the separate sustained GPT-OSS gate after its standard benchmark:

```powershell
pwsh -NoProfile -File scripts/stability_gpt_oss.ps1 -DurationMinutes 30
```

This records request latency and reliability, samples the fixed AMD DRM sysfs GTT counters,
confirms process exit, checks whole-device GTT recovery within a 1 GiB tolerance, and writes
privacy-safe JSON and Markdown reports under `var/benchmarks/`.

## Framework Desktop observations — 18 July 2026

These smoke-sized observations were recorded on the configured Framework Desktop with an
AMD Radeon 8060S (`gfx1151`), ROCm 7.2. They establish feasibility, not long-running
acceptance.

### Gemma 4 12B SceneChat

`google/gemma-4-12B-it@12ace6d648d72bd41519e140f1185f34d38c7e3d` loaded through
`Gemma4UnifiedProcessor` and `Gemma4UnifiedForConditionalGeneration` using Transformers
5.13.0 and Torch 2.9.1 ROCm 7.2.1.

- model load: 10.5256 seconds; synthetic warm-up: 1.4566 seconds;
- three identical 256-by-256 structured SceneChat requests: 15.6804, 15.6471 and
  15.6957 seconds, averaging 15.6744 seconds;
- each request used 566 prompt tokens and generated 82 completion tokens;
- steady allocated device memory: 24,109,280,768 bytes; peak: 24,473,772,032 bytes.

Run the focused workload with `scripts/benchmark_scenechat_profile.ps1` while the selected
12B worker is ready.

### Moshiko speech

`kyutai/moshiko-pytorch-bf16@2bfc9ae6e89079a5cc7ed2a68436010d91a3d289` loaded with
Moshi 0.2.13 and the same ROCm Torch build. A five-second real-time synthetic-silence stream
produced the fixed Moshiko greeting without microphone capture:

- WebSocket session ready: 0.1229 seconds;
- first response audio: 0.5494 seconds; first text token: 1.3910 seconds;
- 88,320 bytes of PCM16 output and the transcript `Hey, how are you doing?`;
- the management compatibility-smoke path independently returned audio in 1.2748 seconds;
- GTT use while loaded: 18,614,816,768 of 125,829,120,000 bytes.

Run this workload with `scripts/benchmark_moshiko_stream.py`. ROCm reported memory-efficient
attention as experimental, so the baseline did not enable the experimental AOTriton switch.

### GPT-OSS 120B

The configured profile used the consolidated 63.4 GB
`ggml-org/gpt-oss-120b-GGUF` MXFP4 artefact at revision
`8d158cefb5f175c6f8842bbd8f68eca54d951ab4` with llama.cpp revision `f08c4c0d` and full
Vulkan offload on the Radeon 8060S. A standard benchmark recorded:

- a warm-filesystem cold start of 9.8876 seconds;
- five successful deterministic 256-token requests with median latency of 5.3365 seconds,
  p95 latency of 5.3644 seconds, and median throughput of 48.4302 tokens per second;
- peak whole-device GTT use of 59.9581 GiB; and
- successful lifecycle restoration.

A separate 30-minute sustained run completed 285 requests without failure. Median request
latency was 1.3310 seconds, p95 latency was 1.3496 seconds, and peak whole-device GTT use
was 60.4423 GiB. After the worker stopped, GTT use recovered to within 0.4562 GiB of its
baseline, inside the 1 GiB tolerance, and process exit was confirmed. The sustained and
standard request latencies are not directly comparable because the sustained workload
generates fewer tokens per request.

These measurements are observational results from one physical run, not release
performance thresholds or general hardware claims.

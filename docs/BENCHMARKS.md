# ROCm model benchmarks

ModelDeck provides a hardware-gated benchmark suite for repeatable performance and
stability observations on the target Framework Desktop. It never downloads weights and
does not replace compatibility smoke tests or workload-specific quality evaluation.

## Run the suite

Run the standard suite across every physical ROCm profile:

```powershell
pwsh -NoProfile -File scripts/benchmark_models.ps1
```

Use the quick preset to validate the benchmark setup with two measured requests per
profile:

```powershell
pwsh -NoProfile -File scripts/benchmark_models.ps1 -Preset Quick
```

Select one or more profiles when a full run is unnecessary:

```powershell
pwsh -NoProfile -File scripts/benchmark_models.ps1 -Preset Standard `
    -Models qwen-small-rocm,qwen-1-5b-rocm,qwen-3b-rocm
```

The allowlisted physical profiles are:

- `qwen-small-rocm`, `qwen-1-5b-rocm`, and `qwen-3b-rocm`;
- `diffusiongemma-q4-rocm` and `diffusiongemma-rocm`;
- `scenechat-gemma4-e2b-rocm`.

Use `-JsonOutput` and `-MarkdownOutput` to override the default timestamped paths under
`var/benchmarks/`. The paths must be different.

## Workloads and measurements

Each profile receives one excluded benchmark warm-up followed by two measured requests
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

Benchmarking is deliberately disruptive. It refuses to start while any managed worker is
busy or transitioning, records which workers are initially ready, stops all managed
workers, and benchmarks one physical profile at a time. It restores the original ready
workers after success, failure, or interruption. If the wrapper started ModelDeck, it
stops the services when the run ends.

The gateway provider header is checked for every benchmark request. A mock fallback is a
benchmark failure and is never reported as physical ROCm performance. Missing snapshots,
worker failures, request failures, and unsuccessful lifecycle restoration produce a
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

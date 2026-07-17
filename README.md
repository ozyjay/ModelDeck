# ModelDeck

ModelDeck is a local management service for isolated model workers on the Framework
Desktop. It provides evidence-based cache discovery, hardware diagnostics, explicit
worker lifecycle states, isolated ROCm autoregressive and text-diffusion workers, useful
mock/replay fallbacks, and a stable local gateway.

ROCm workers are core ModelDeck functionality for the target Framework Desktop. They load
only when explicitly started and never download weights. The management plane, gateway,
fallbacks, and normal verification still run without GPU access so development and
diagnosis remain useful when the target hardware is unavailable.

## Target setup

```powershell
pwsh -NoProfile -File scripts/setup.ps1
pwsh -NoProfile -File scripts/run.ps1
```

ModelDeck deliberately uses three environments with different responsibilities:

- `.venv` is the control plane: management service, supervisor, gateway, catalogue,
  mock/replay fallbacks, and development tests.
- `.venv-rocm72` is the primary inference runtime: the pinned ROCm, PyTorch, and
  Transformers stack for Qwen and the DiffusionGemma BF16 baseline.
- `.venv-rocm72-q4` is the isolated inference runtime for the default Q4 DiffusionGemma
  provider and its GPTQ dependencies.

All three are part of the target installation. Keeping model libraries outside the control
plane preserves dependency isolation and makes worker process exit the memory-recovery
boundary.

- Operator console: <http://127.0.0.1:3600>
- Stable gateway: <http://127.0.0.1:8600/v1/health>
- API documentation: <http://127.0.0.1:3600/docs>

For lightweight development or CI on a machine without the target GPU, run
`pwsh -NoProfile -File scripts/setup.ps1 -ControlPlaneOnly`. The control plane and
fallbacks remain usable, but that mode is not a complete target deployment.

Start the selected ROCm worker from the dashboard or through the management API:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:3600/api/workers/qwen-small-rocm/start `
    -TimeoutSec 360
```

The installed Qwen workers are `qwen-small-rocm` for 0.5B, `qwen-1-5b-rocm` for 1.5B,
and `qwen-3b-rocm` for 3B. They use fixed ports 8620, 8623, and 8624 respectively and
remain stopped until explicitly selected.

The Model library can also turn a recognised, complete Hugging Face snapshot into a local
ROCm worker configuration when its architecture matches an allowlisted worker. Supported
paths are causal-language-model Transformers, SceneChat Gemma 4, and DiffusionGemma block
diffusion. Choose **Configure runtime**, assign a
gateway alias, and select the bounded dtype, lifecycle, context, and output settings.
ModelDeck derives the model, revision, cache root, runtime, port, capabilities, and safe
launch manifest itself. The configuration persists in ModelDeck's local SQLite store;
removing it never removes the cached model. Unsupported architectures remain visible with
the specific reason that configuration is unavailable.

Benchmark all physical ROCm profiles with a repeatable representative workload:

```powershell
pwsh -NoProfile -File scripts/benchmark_models.ps1
```

Use `-Preset Quick` or `-Models qwen-small-rocm,qwen-1-5b-rocm` for a shorter run. The
suite benchmarks one worker at a time, restores the initial worker state, and writes
timestamped JSON and Markdown reports under `var/benchmarks/`. See
[ROCm model benchmarks](docs/BENCHMARKS.md) for workload definitions, privacy guarantees,
and report interpretation.

The operator console is a committed React and TypeScript production bundle served by
FastAPI. Node.js is required only by setup, verification, and frontend development; the
running management service serves local static assets and does not start a Node process.
After changing `frontend/`, rebuild with
`pwsh -NoProfile -File scripts/build_frontend.ps1`. Verification rejects a stale
committed bundle.

Mock and replay workers remain explicit fallback/test choices. Stop all ModelDeck workers
and services with `pwsh -NoProfile -File scripts/stop.ps1`. See
[Start here](docs/START_HERE.md) and the [build plan](docs/BUILD_PLAN.md) for current scope
and next steps.

## Booth mode

For Open Day, start ModelDeck and a dedicated fullscreen Chromium-family browser with one
command:

```powershell
pwsh -NoProfile -File scripts/run_booth.ps1
```

For a windowed rehearsal that is easier to exit and inspect:

```powershell
pwsh -NoProfile -File scripts/run_booth.ps1 -Windowed
```

Booth mode stops an earlier ModelDeck session, starts the services with Open Day policy,
waits for both management and gateway health, and opens the operator console in an
isolated `.booth-browser-profile`. The launch command then returns to the prompt. Closing
the booth browser stops the ModelDeck workers and services through a background watcher;
you can instead stop them explicitly with `pwsh -NoProfile -File scripts/stop.ps1`. Set
`BOOTH_BROWSER` to a Chromium, Chrome, or Edge executable name or path if automatic
discovery does not find the intended browser. Booth Chromium background networking is
disabled; any remaining browser diagnostics are written under `var/log` rather than to
the launching terminal.

## Core ROCm model workers

```powershell
pwsh -NoProfile -File scripts/setup.ps1
pwsh -NoProfile -File scripts/smoke_rocm_autoregressive.ps1
pwsh -NoProfile -File scripts/smoke_rocm_text_diffusion.ps1
```

The ROCm setup prepares the primary inference environment without replacing Fedora RPMs.
It is not required merely to execute control-plane tests, but it is required for the
target product. Model loading remains local-files-only.

Run the setup script initially and again when either environment's requirements change.
Compatible real GPU workers should share `.venv-rocm72`; add another GPU environment only when recorded
compatibility evidence demonstrates a dependency conflict.

The Qwen 0.5B, SceneChat Gemma 4 E2B, BF16 DiffusionGemma baseline, and self-contained Q4
DiffusionGemma paths are compatibility-tested on the target Framework Desktop. Gemma 4
E2B and Q4 DiffusionGemma also passed simultaneous residency with a structured image
completion. The Qwen 1.5B and 3B workers are
registered against complete pinned local snapshots but require their own physical ROCm
acceptance evidence. The BF16 and Qwen workers use `/mnt/work/models/huggingface/hub`;
the self-contained Q4 worker reads only its packaged checkpoint. None of the smoke tests
download model files.

## DiffusionGemma GPTQ Q4 variant

The default `text-diffusion` provider directly loads a self-contained Q4/BF16 hybrid:
the expert projections use GPTQ Q4 g32 and the packaged non-expert tensors remain BF16.
It does not materialise BF16 experts or access the upstream model cache at runtime. The
original model remains available explicitly as `text-diffusion-bf16` for compatibility
and release evaluation.

```powershell
./scripts/start_diffusiongemma_q4.ps1 -Smoke
```

The default checkpoint directory is
`/mnt/work/models/modeldeck/diffusiongemma-26b-a4b-it-gptq-q4-g32`. The worker runs on fixed port 8622,
reports quantization and Q4 invocation metrics, and remains local-files-only.

Upgrade an existing v1 expert-delta checkpoint to the self-contained v2 format without
re-quantising its expert weights:

```powershell
./scripts/materialize_diffusiongemma_q4.ps1
```

Materialisation reads the pinned base snapshot once and packages only the non-expert
BF16 tensors plus the local configuration, processor, tokenizer, and generation files.
Afterwards the Q4 worker no longer requires that base snapshot.

Run the comparative release gate after changing the checkpoint, loader, ROCm stack, or
Transformers version. It executes the diverse prompt suite through Q4 and BF16
sequentially, verifies deterministic replay and repeated Q4 requests, then leaves Q4
ready:

```powershell
./scripts/evaluate_diffusiongemma_q4.ps1
```

The JSON report is written to `var/q4-quality-evaluation.json`. The default gates require
all worker contracts and stability requests to pass, exact same-seed Q4 replay, active Q4
kernels, peak Q4 allocation below 24 GiB, allocation range below 1 GiB, median Q4 latency
below three times BF16, mean token edit similarity of at least 0.35, and no material
instruction-constraint regression relative to BF16.

After the canonical gate passes, package and cryptographically verify the self-contained
release in place:

```powershell
./scripts/package_diffusiongemma_q4_release.ps1
./scripts/package_diffusiongemma_q4_release.ps1 -VerifyOnly
```

Packaging adds a Hugging Face-compatible model card, Apache-2.0 licence, provenance,
publication-safe evaluation report, release manifest, and SHA-256 checksums beside the
existing weight shards without duplicating them or uploading anything. The quantized
artifact belongs in a separate Hugging Face model repository: it is associated with
ModelDeck through a pinned loader commit, but its roughly 18 GiB payload and artifact tags do
not belong in the ModelDeck Git repository. See the
[DiffusionGemma Q4 release process](docs/DIFFUSIONGEMMA_Q4_RELEASE.md).

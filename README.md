# ModelDeck

ModelDeck is a local runtime manager and stable capability gateway for the Framework
Desktop. Its operator model has four concepts:

- a **Model** is a read-only, pinned snapshot discovered in the local cache;
- a **Worker** is one operator-created, startable runtime configuration for a Model;
- a **Route** is a public model name and protocol contract with one primary Worker and
  ordered backups; and
- an **Event** is the versioned set of Demos and shared Routes needed for an occasion.

Publishing an Event changes gateway routing atomically. It does not start Workers. Worker
names, Event names, Demo names, Route display names and public Route names are editable;
internal UUIDs and trusted execution definitions are deliberately not presented as
operator-facing names. ModelDeck starts with no configured Workers, Events or Routes.

ROCm workers are core ModelDeck functionality for the target Framework Desktop. They load
only when explicitly started and never download weights. The management plane, gateway,
fallbacks, and normal verification still run without GPU access so development and
diagnosis remain useful when the target hardware is unavailable.

## Target setup

```powershell
pwsh -NoProfile -File scripts/setup.ps1
Copy-Item .env.example .env # optional local overrides
pwsh -NoProfile -File scripts/run.ps1
```

`scripts/run.ps1` loads an optional, gitignored `.env` before launching management and
gateway processes. Only the variables documented in `.env.example` are accepted; unknown,
duplicate, malformed, or unterminated entries stop startup without printing their values.
Values are literal and are never evaluated as PowerShell. Variables already set in the
launching process take precedence, and `-OpenDay` still forces its safety overrides after
loading. The checked-in defaults work without a `.env`; create one when local ports,
storage, timeouts, runtime interpreters, cache location, or the SceneChat credential need
to differ.

ModelDeck deliberately uses three environments with different responsibilities:

- `.venv` is the control plane: management service, supervisor, gateway, catalogue,
  mock/replay fallbacks, and development tests.
- `.venv-rocm72` is the primary inference runtime: the pinned ROCm, PyTorch, and
  Transformers stack for Qwen and the DiffusionGemma BF16 baseline.
- `.venv-rocm72-q4` is the isolated inference runtime for DiffusionGemma Q4 and its GPTQ
  dependencies.

All three are part of the target installation. Keeping model libraries outside the control
plane preserves dependency isolation and makes worker process exit the memory-recovery
boundary.

- Operator console: <http://127.0.0.1:3600>
- Stable gateway: <http://127.0.0.1:8600/v1/health>
- API documentation: <http://127.0.0.1:3600/docs>

The operator console can collapse individual sections or every section at once. These
display preferences are retained in local browser storage and do not change ModelDeck
configuration.

Use **Models** to create a Worker from a recognised cached revision. Use **Events** to
define shared Routes, assign the primary and ordered backup Workers, group Routes into
Demos, validate the draft and publish it. Use **Workers** for lifecycle control and real
generation smoke tests. Use **Live** to see only the published routing snapshot and
rehearse a Route end-to-end through the gateway. Open Day mode locks configuration
changes server-side while leaving explicit Worker lifecycle controls available.

Event edits autosave to a mutable draft. Publishing creates an immutable revision;
historical revisions can be made live again without reconstructing them. An Event can
require merely protocol-compatible Workers or matching tested-working evidence. A Worker
smoke test records successful or failed generation evidence against the detected hardware,
runtime, library and pinned Model fingerprint.

Existing v1 databases are not interpreted as v2 configuration. Back up and replace only
the configuration database with:

```powershell
pwsh -NoProfile -File scripts/cutover_v2.ps1
```

The cut-over script stops ModelDeck, moves the exact SQLite database files under
`var/backups/`, and creates an empty v2 database. Model caches, logs, benchmark reports and
trusted runtime manifests are preserved. Use `-WhatIf` to inspect the file operations.

For lightweight development or CI on a machine without the target GPU, run
`pwsh -NoProfile -File scripts/setup.ps1 -ControlPlaneOnly`. The control plane and
fallbacks remain usable, but that mode is not a complete target deployment.

After creating a Worker in the Models view, it can also be started through the API using
its internal UUID:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:3600/api/workers/<worker-uuid>/start `
    -TimeoutSec 360
```

The Model library turns a recognised, complete Hugging Face snapshot into a local Worker
when its architecture matches an installed trusted runtime. Supported
paths are causal-language-model Transformers, SceneChat Gemma 4 and the official Qwen3.5
0.8B, 2B, 4B and 9B checkpoints, DiffusionGemma block diffusion, and self-contained
ModelDeck DiffusionGemma Q4 format 2 releases. SpeechShift additionally recognises the
exact pinned OPUS English-to-French and English-to-German snapshots, Qwen3-TTS
CustomVoice, and Whisper small.en; see [SpeechShift runtimes](docs/SPEECHSHIFT_RUNTIMES.md). Q4 releases
must retain their manifests, quality evidence, complete file inventory, and checksums;
generic GPTQ repositories are not accepted. ModelDeck derives the cache root, port,
capabilities and safe launch argument array. Archiving a Worker never removes the cached
Model. Unsupported architectures remain visible with the reason Worker creation is
unavailable.

Reviewed runtime templates can be added as versioned
[trusted runtime manifests](docs/TRUSTED_RUNTIME_MANIFESTS.md). Installation requires an
explicit local SHA-256 trust step and cannot be performed from the browser; manifests may
select a registered launch implementation but cannot define commands, paths or environment
variables.

Each complete cached revision can also be **Disallowed in ModelDeck** without deleting it
from the Hugging Face cache. A revision cannot be disallowed while it has configured
Workers. A Q4 runtime configured from a downloaded Hugging Face release follows the policy
of that derivative repository and revision separately from its upstream base Model.

Benchmark all configured physical Workers that have exactly one published Route:

```powershell
pwsh -NoProfile -File scripts/benchmark_models.ps1
```

Use `-Preset Quick` or `-Workers 'Qwen small','Qwen medium'` for a shorter run. Worker
selectors may be editable names or UUIDs. The suite benchmarks one Worker at a time,
restores the initial Worker state, and writes
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

The setup scripts install the control-plane and trusted runtime dependencies; they do not
create Worker instances or public Routes. Cached Models are discovered read-only after
startup. Physical acceptance evidence belongs to the exact Worker fingerprint created on
the target machine. None of the smoke tests download Model files.

## DiffusionGemma GPTQ Q4 variant

The Q4 runtime directly loads a self-contained Q4/BF16 hybrid:
the expert projections use GPTQ Q4 g32 and the packaged non-expert tensors remain BF16.
It does not materialise BF16 experts or access the upstream model cache at runtime. The
original Model can be configured as a separate BF16 Worker for compatibility and release
evaluation. Their public Route names are chosen by the operator.

```powershell
./scripts/start_diffusiongemma_q4.ps1 -Worker 'DiffusionGemma Q4' `
    -RouteName 'text-diffusion' -Smoke
```

The selected Worker reports quantisation and Q4 invocation metrics and remains
local-files-only.

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
./scripts/evaluate_diffusiongemma_q4.ps1 `
    -Q4Worker 'DiffusionGemma Q4' -Q4Route 'text-diffusion' `
    -BF16Worker 'DiffusionGemma BF16' -BF16Route 'text-diffusion-bf16'
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

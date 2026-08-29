# ModelDeck

ModelDeck is a local runtime manager and stable capability gateway for the Framework
Desktop. Its operator model has four concepts:

- a **Model** is a read-only, pinned snapshot discovered in the local cache;
- a **Worker** is one operator-created, startable runtime configuration for a Model;
- a **published capability** is a public model name and trusted protocol contract with one
  primary Worker and ordered backups; and
- a **Routing Profile** is the versioned, atomically published set of capabilities for all
  currently supported local applications.

Publishing a Routing Profile changes gateway routing atomically. It does not start Workers. Worker
names, profile names, capability display names and public model names are editable;
internal UUIDs and trusted execution definitions are deliberately not presented as
operator-facing names. ModelDeck starts with no configured Workers, profiles or capabilities.

ROCm workers are core ModelDeck functionality for the target Framework Desktop. They load
only when explicitly started and never download weights. The management plane, gateway,
fallbacks, and normal verification still run without GPU access so development and
diagnosis remain useful when the target hardware is unavailable.

## Target setup

```powershell
pwsh -NoProfile -File scripts/setup/setup.ps1
Copy-Item .env.example .env # optional local overrides
pwsh -NoProfile -File scripts/operations/run.ps1
```

`scripts/operations/run.ps1` loads an optional, gitignored `.env` before launching management and
gateway processes. Only the variables documented in `.env.example` are accepted; unknown,
duplicate, malformed, or unterminated entries stop startup without printing their values.
Values are literal and are never evaluated as PowerShell. Variables already set in the
launching process take precedence, and `-LockConfiguration` forces configuration locking after
loading. The checked-in defaults work without a `.env`; create one when local ports,
storage, timeouts, runtime interpreters, cache location, or the SceneChat credential need
to differ.

Use `MODELDECK_CONFIGURATION_LOCKED=1` (or `scripts/operations/run.ps1 -LockConfiguration`) for a
prepared, read-only configuration. The former `MODELDECK_OPEN_DAY` and `-OpenDay` names are
accepted temporarily while local launch files are updated. ModelDeck is always offline-only;
`MODELDECK_ALLOW_DOWNLOADS` no longer changes runtime behaviour.

ModelDeck deliberately uses three environments with different responsibilities:

- `.venv` is the control plane: management service, supervisor, gateway, catalogue, and
  development tests. Deterministic mock/replay fixtures are test-harness-only.
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

### SprintBot in Docker

Keep the management service and primary gateway on loopback. To give a local SprintBot
container access to the stable gateway as well, enable its explicit bridge companion in `.env`:

```text
MODELDECK_ENABLE_DOCKER_BRIDGE=1
```

ModelDeck then exposes the same local-only routing set at `http://host.docker.internal:8600/v1`
for SprintBot and `http://127.0.0.1:8600/v1` for local desktop applications such as WayFinder.
Ensure host firewall rules permit TCP 8600 only from the Docker bridge and loopback traffic.
ModelDeck deliberately rejects `0.0.0.0` and LAN addresses; use a separately approved restricted
reverse proxy and firewall deployment if broader access is ever required.

After launch, verify the listener and SprintBot's fail-closed inference configuration from
PowerShell:

```powershell
ss --tcp --listening --numeric 'sport = :8600'
# Expected: 172.17.0.1:8600
docker compose exec dashboard sprintbot-inference-check
```

The operator console can collapse individual sections or every section at once. These
display preferences are retained in local browser storage and do not change ModelDeck
configuration.

Use **Models** to create a Worker from a recognised cached revision. Use **Routing profiles** to
define published capabilities, assign primary and ordered backup Workers, validate the
draft and publish it. Use **Workers** for detailed lifecycle control and real generation
smoke tests.
Use **Live** to see only the published routing snapshot, start or stop any primary or backup
Worker in that snapshot, and rehearse a capability end-to-end through the gateway. A local
deployment policy can lock configuration changes server-side while leaving explicit Worker
lifecycle controls available.

The checked-in `opencode.json` connects OpenCode to the loopback gateway and selects the
`code-local` and `code-fast` public model IDs from the **OpenCode local coding** Routing
Profile. Start the desired Worker from **Live** before opening OpenCode. The configuration
does not grant OpenCode access to model files and does not start or download a model.

Routing Profile edits autosave to a mutable draft. Publishing creates an immutable revision;
historical revisions can be made live again without reconstructing them. A profile can
require merely protocol-compatible Workers or matching tested-working evidence. A Worker
smoke test records successful or failed generation evidence against the detected hardware,
runtime, library and pinned Model fingerprint.

Existing legacy databases can be backed up and replaced with an empty v4 configuration using:

```powershell
pwsh -NoProfile -File scripts/migrations/cutover_v2.ps1
```

The cut-over script stops ModelDeck, moves the exact SQLite database files under
`var/backups/`, and creates an empty v4 database. To preserve a v2 configuration, first use:

```powershell
pwsh -NoProfile -File scripts/migrations/migrate_v2_to_v3.ps1
```

It backs up the SQLite database, converts Event revisions to Routing Profile revisions,
preserves active routing and Workers, and omits Demo membership. Then migrate the resulting
v3 database, or any existing v3 installation, with:

```powershell
pwsh -NoProfile -File scripts/migrations/migrate_v3_to_v4.ps1
```

The v4 migration adds capability policy and grandfathers current Worker and routing use.
Startup refuses an unmigrated database. Model caches, logs, benchmark reports and trusted
runtime manifests are preserved. Use `-WhatIf` to inspect either migration.

For lightweight development or CI on a machine without the target GPU, run
`pwsh -NoProfile -File scripts/setup/setup.ps1 -ControlPlaneOnly`. The control plane and
fallbacks remain usable, but that mode is not a complete target deployment.

After creating a Worker in the Models view, it can also be started through the API using
its internal UUID:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:3600/api/workers/<worker-uuid>/start `
    -TimeoutSec 360
```

The Model library lists potential capabilities for each complete Hugging Face snapshot,
keeps local detections separate from reviewed assertions, and requires the operator to
allow a capability before creating a Worker or publishing a route. Allowing a capability without
an installed trusted runtime records intent without making it runnable. Supported runtime
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

Official Qwen3.5 checkpoints have separate SceneChat and text-chat adapters. The text-chat
adapter serves the OpenAI-compatible chat and completion contracts without accepting image
content; it is hardware-verification-required until qualification records evidence for its
exact model revision and configuration.

Qwen3.5 GGUF experiments do not require a new ModelDeck release for every supported model
size. For a complete HuggingFacePull snapshot from
`bartowski/Qwen_Qwen3.5-{0.8B,2B,4B,9B}-GGUF` containing the matching `Q8_0` artefact, the
Models view offers **Verify and approve**. ModelDeck validates the immutable revision and
HuggingFacePull completion marker, reads the expected LFS size and SHA-256 from the cached
tree, hashes the actual GGUF, then writes a local trusted candidate manifest under
`MODELDECK_DATA_DIR/trusted-qwen-candidates`. Once approved, the same two reusable
llama.cpp/Vulkan templates are available for every supported size: thinking disabled and
adaptive thinking. Worker launches remain offline and accept only the manifest-bound Model,
revision, filename, checksum, 8,192-token context and code-owned llama.cpp pin.

The code-owned reviewed-model registry also includes the exact official
`Qwen/Qwen3.8-27B-FP8` checkpoint. Its FP8 method, dynamic activation scheme and E4M3
format must all match the reviewed metadata. The isolated Qwen worker dequantises those
weights to BF16 during its offline load by default. A separate native-FP8 runtime is
available only when its reviewed kernel is present and verified. Both profiles remain
hardware-verification-required until their exact local revision and ROCm stack pass
qualification. HuggingFacePull remains responsible for acquisition.

### Qwen3.8 native FP8 on gfx1151

Acquire the reviewed executable kernel with HuggingFacePull, then refresh ModelDeck's
isolated ROCm environment:

```powershell
$env:HF_HUB_CACHE = "/mnt/work/models/huggingface/hub"
& /path/to/HuggingFacePull/.venv/bin/hfpull kernels-community/finegrained-fp8 `
  --repo-type kernel --revision v3 `
  --expected-commit fcf89a79d85eab78182c62fb986ed01f2cbf7422 `
  --allow 'build/torch-rocm/*'
pwsh -NoProfile -File scripts/setup/setup.ps1
```

ModelDeck verifies the cached `v3` ref, exact commit, ROCm metadata, complete allowlisted
file set and SHA-256 of every kernel file before import. Native FP8 never silently falls
back to the independent BF16-dequant Worker.

Create the machine-specific tuning profile before qualifying native FP8 for routing:

```powershell
pwsh -NoProfile -File scripts/benchmarks/tune_qwen38_fp8.ps1 `
  -CacheRoot /mnt/work/models/huggingface/hub -DataDir .modeldeck -Stage full
```

The full stage covers decode, prompt ingestion and image-chat tile buckets across all
seven Qwen3.8 FP8 weight shapes. Without a valid profile the Worker uses the reviewed
4-warp, 2-stage fallback but cannot pass promotion. Text promotion requires at least
3.20 tokens/s, warmed first-token latency no higher than 0.50 seconds, no more than 36 GiB
steady allocation and no more than 38 GiB peak allocation.

Reviewed runtime templates can be added as versioned
[trusted runtime manifests](docs/TRUSTED_RUNTIME_MANIFESTS.md). Installation requires an
explicit local SHA-256 trust step and cannot be performed from the browser; manifests may
select a registered launch implementation but cannot define commands, paths or environment
variables.

Each complete cached revision can also be **Disallowed in ModelDeck** without deleting it
from the Hugging Face cache. A revision cannot be disallowed while it has configured
Workers. A Q4 runtime configured from a downloaded Hugging Face release follows the policy
of that derivative repository and revision separately from its upstream base Model.

Benchmark all configured physical Workers that have exactly one published capability:

```powershell
pwsh -NoProfile -File scripts/benchmarks/benchmark_models.ps1
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
`pwsh -NoProfile -File scripts/operations/build_frontend.ps1`. Verification rejects a stale
committed bundle.

## Build a Fedora standalone distribution

The native Fedora 44 desktop package includes the GTK/WebKit desktop window, management service,
gateway, and core isolated runtimes. It never includes Model weights; HuggingFacePull remains the
only acquisition path.

Before building, create `packaging/fedora/wheelhouse/` and populate it with every reviewed wheel
listed in `packaging/fedora/wheelhouse.sha256`. The build is deliberately offline: it verifies
each wheel's SHA-256 and stops if a required wheel is missing or differs from the manifest. It
does not download packages, Models, or ROCm components.

On Fedora 44 x86_64 with `rpmbuild`, Python 3.12, Node.js, npm, and the frontend dependencies
already installed, create an unsigned RPM distribution from the repository root with:

```powershell
pwsh -NoProfile -File scripts/packaging/build_fedora_standalone.ps1
```

The package is written beneath `dist/fedora/`, normally as
`dist/fedora/x86_64/modeldeck-<version>-1.fc44.x86_64.rpm`. To use a wheelhouse stored elsewhere,
pass its directory and matching manifest explicitly:

```powershell
pwsh -NoProfile -File scripts/packaging/build_fedora_standalone.ps1 `
  -Wheelhouse /path/to/wheelhouse `
  -WheelhouseManifest /path/to/wheelhouse.sha256 `
  -OutputDirectory dist/fedora
```

Sign a release RPM separately, then verify and install it:

```powershell
pwsh -NoProfile -File scripts/packaging/sign_fedora_rpm.ps1 `
  -RpmPath dist/fedora/x86_64/modeldeck-<version>-1.fc44.x86_64.rpm `
  -KeyId <public-key-id>
```

```bash
rpm --checksig --verbose dist/fedora/x86_64/modeldeck-<version>-1.fc44.x86_64.rpm
sudo dnf install ./dist/fedora/x86_64/modeldeck-<version>-1.fc44.x86_64.rpm
modeldeck-desktop
```

The RPM installs program files system-wide under `/usr`; its per-user state is stored in
`~/.local/share/modeldeck` and Worker logs in `~/.local/state/modeldeck/logs/workers`. See
[Fedora standalone ModelDeck](docs/FEDORA_STANDALONE.md) for the full packaging and lifecycle
details.

Test fixtures are not available in the operator UI or gateway as fallback choices. Stop all
ModelDeck workers and services with `pwsh -NoProfile -File scripts/operations/stop.ps1`. See
[Start here](docs/START_HERE.md) and the [build plan](docs/BUILD_PLAN.md) for current scope
and next steps.

For the WayFinder VS Code extension's local Gate 0 configuration, see
[WayFinder Gate 0](docs/WAYFINDER_GATE0.md).

## Booth mode

For Open Day, start ModelDeck and a dedicated fullscreen Chromium-family browser with one
command:

```powershell
pwsh -NoProfile -File scripts/booth/run_booth.ps1
```

For a windowed rehearsal that is easier to exit and inspect:

```powershell
pwsh -NoProfile -File scripts/booth/run_booth.ps1 -Windowed
```

Booth mode stops an earlier ModelDeck session, starts the services with Open Day policy,
waits for both management and gateway health, and opens the operator console in an
isolated `.booth-browser-profile`. The launch command then returns to the prompt. Closing
the booth browser stops the ModelDeck workers and services through a background watcher;
you can instead stop them explicitly with `pwsh -NoProfile -File scripts/operations/stop.ps1`. Set
`BOOTH_BROWSER` to a Chromium, Chrome, or Edge executable name or path if automatic
discovery does not find the intended browser. Booth Chromium background networking is
disabled; any remaining browser diagnostics are written under `var/log` rather than to
the launching terminal.

## Core ROCm model workers

```powershell
pwsh -NoProfile -File scripts/setup/setup.ps1
pwsh -NoProfile -File scripts/smoke/smoke_rocm_autoregressive.ps1
pwsh -NoProfile -File scripts/smoke/smoke_rocm_text_diffusion.ps1
```

The ROCm setup prepares the primary inference environment without replacing Fedora RPMs.
It is not required merely to execute control-plane tests, but it is required for the
target product. Model loading remains local-files-only.

Run the setup script initially and again when either environment's requirements change.
Compatible real GPU workers should share `.venv-rocm72`; add another GPU environment only when recorded
compatibility evidence demonstrates a dependency conflict.

The setup scripts install the control-plane and trusted runtime dependencies; they do not
create Worker instances or published capabilities. Cached Models are discovered read-only after
startup. Physical acceptance evidence belongs to the exact Worker fingerprint created on
the target machine. None of the smoke tests download Model files.

## DiffusionGemma GPTQ Q4 variant

The Q4 runtime directly loads a self-contained Q4/BF16 hybrid:
the expert projections use GPTQ Q4 g32 and the packaged non-expert tensors remain BF16.
It does not materialise BF16 experts or access the upstream model cache at runtime. The
original Model can be configured as a separate BF16 Worker for compatibility and release
evaluation. Their public Route names are chosen by the operator.

```powershell
./scripts/q4/start_diffusiongemma_q4.ps1 -Worker 'DiffusionGemma Q4' `
    -RouteName 'text-diffusion' -Smoke
```

The selected Worker reports quantisation and Q4 invocation metrics and remains
local-files-only.

Upgrade an existing v1 expert-delta checkpoint to the self-contained v2 format without
re-quantising its expert weights:

```powershell
./scripts/q4/materialize_diffusiongemma_q4.ps1
```

Materialisation reads the pinned base snapshot once and packages only the non-expert
BF16 tensors plus the local configuration, processor, tokenizer, and generation files.
Afterwards the Q4 worker no longer requires that base snapshot.

Run the comparative release gate after changing the checkpoint, loader, ROCm stack, or
Transformers version. It executes the diverse prompt suite through Q4 and BF16
sequentially, verifies deterministic replay and repeated Q4 requests, then leaves Q4
ready:

```powershell
./scripts/q4/evaluate_diffusiongemma_q4.ps1 `
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
./scripts/q4/package_diffusiongemma_q4_release.ps1
./scripts/q4/package_diffusiongemma_q4_release.ps1 -VerifyOnly
```

Packaging adds a Hugging Face-compatible model card, Apache-2.0 licence, provenance,
publication-safe evaluation report, release manifest, and SHA-256 checksums beside the
existing weight shards without duplicating them or uploading anything. The quantized
artifact belongs in a separate Hugging Face model repository: it is associated with
ModelDeck through a pinned loader commit, but its roughly 18 GiB payload and artifact tags do
not belong in the ModelDeck Git repository. See the
[DiffusionGemma Q4 release process](docs/DIFFUSIONGEMMA_Q4_RELEASE.md).

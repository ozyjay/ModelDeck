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

- Management dashboard: <http://127.0.0.1:3600>
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

Mock and replay workers remain explicit fallback/test choices. Stop all ModelDeck workers
and services with `pwsh -NoProfile -File scripts/stop.ps1`. See
[Start here](docs/START_HERE.md) and the [build plan](docs/BUILD_PLAN.md) for current scope
and next steps.

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

The Qwen 0.5B, BF16 DiffusionGemma baseline, and expert-only Q4 DiffusionGemma paths are
compatibility-tested on the target Framework Desktop. The Qwen 1.5B and 3B workers are
registered against complete pinned local snapshots but require their own physical ROCm
acceptance evidence. All workers use `/mnt/work/models/huggingface/hub`; none of the smoke
tests download model files.

## DiffusionGemma GPTQ Q4 variant

The default `text-diffusion` provider directly loads the pinned BF16 non-expert weights
plus the exported expert-only GPTQ Q4 g32 checkpoint. It does not materialise the BF16
experts. The original model remains available explicitly as `text-diffusion-bf16` for
compatibility and release evaluation; `text-diffusion-q4` remains as a compatibility
alias for clients that adopted the Q4 preview name.

```powershell
./scripts/start_diffusiongemma_q4.ps1 -Smoke
```

The default checkpoint directory is
`var/diffusiongemma-26b-a4b-it-gptq-q4-g32`. The worker runs on fixed port 8622,
reports quantization and Q4 invocation metrics, and remains local-files-only.

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

After the canonical gate passes, package and cryptographically verify the expert-delta
release in place:

```powershell
./scripts/package_diffusiongemma_q4_release.ps1
./scripts/package_diffusiongemma_q4_release.ps1 -VerifyOnly
```

Packaging adds a Hugging Face-compatible model card, Apache-2.0 licence, provenance,
publication-safe evaluation report, release manifest, and SHA-256 checksums beside the
existing 30 shards without duplicating them or uploading anything. The quantized
artifact belongs in a separate Hugging Face model repository: it is associated with
ModelDeck through a pinned loader commit, but its 12+ GiB payload and artifact tags do
not belong in the ModelDeck Git repository. See the
[DiffusionGemma Q4 release process](docs/DIFFUSIONGEMMA_Q4_RELEASE.md).

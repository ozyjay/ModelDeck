# Start here

The implemented system combines an operator-first local console, read-only cache and
hardware probes, SQLite evidence, isolated ROCm and fallback workers, lifecycle
supervision, and a stable gateway.
It never downloads models; core ROCm workers load pinned local weights only when started.

The console source lives under `frontend/`; FastAPI serves its committed production
bundle. Node.js is needed for setup and verification, but not while ModelDeck is running.

1. Read [existing repository findings](EXISTING_REPOSITORY_FINDINGS.md).
2. Read [architecture](ARCHITECTURE.md) and [worker protocol](WORKER_PROTOCOL.md).
3. Run `pwsh -NoProfile -File scripts/setup.ps1` to prepare both environments, then run
   `pwsh -NoProfile -File scripts/verify.ps1`.
4. Start with `pwsh -NoProfile -File scripts/run.ps1`, then start the selected ROCm
   worker and confirm it is the gateway's effective provider.

The core Transformers integrations are deliberately isolated in project Python 3.12 ROCm
environments and gated by compatibility evidence. The ROCm 7.2.1 Qwen worker and default
DiffusionGemma Q4 worker have passed their hardware gates; the original BF16 worker is
retained as an explicit compatibility and evaluation baseline. Fedora's system ROCm
packages remain unchanged at 7.1.x. Run
`pwsh -NoProfile -File scripts/setup.ps1` when preparing the target workers.
The environments are complementary: `.venv` owns the control plane, `.venv-rocm72` owns
Qwen and BF16 baseline inference, and `.venv-rocm72-q4` owns default DiffusionGemma Q4
inference.

# Start here

The implemented system combines a read-only dashboard, cache and hardware probes, SQLite
evidence, isolated ROCm and fallback workers, lifecycle supervision, and a stable gateway.
It never downloads models; core ROCm workers load pinned local weights only when started.

1. Read [existing repository findings](EXISTING_REPOSITORY_FINDINGS.md).
2. Read [architecture](ARCHITECTURE.md) and [worker protocol](WORKER_PROTOCOL.md).
3. Run `pwsh -NoProfile -File scripts/setup.ps1` and
   `pwsh -NoProfile -File scripts/verify.ps1`.
4. Prepare the primary inference runtime with
   `pwsh -NoProfile -File scripts/setup_rocm72.ps1` on the target Framework Desktop.
5. Start with `pwsh -NoProfile -File scripts/run_dev.ps1`, then start the selected ROCm
   worker and confirm it is the gateway's effective provider.

The core Transformers integration is deliberately isolated in a project Python 3.12 ROCm
environment and gated by compatibility evidence. The ROCm 7.2.1 Qwen worker has passed
its hardware smoke; DiffusionGemma awaits its physical acceptance run. Fedora's system
ROCm packages remain unchanged at 7.1.x. Run
`pwsh -NoProfile -File scripts/setup_rocm72.ps1` when preparing the target workers.
The two environments are complementary: `.venv` owns the control plane and
`.venv-rocm72` owns primary inference.

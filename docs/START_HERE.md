# Start here

The implemented slice combines Phase 0 evidence gathering with the smallest useful
Phase 1/2 proof: a read-only dashboard, cache and hardware probes, SQLite schema, two
isolated mock worker families, lifecycle supervision, and a stable gateway. It neither
loads nor downloads models.

1. Read [existing repository findings](EXISTING_REPOSITORY_FINDINGS.md).
2. Read [architecture](ARCHITECTURE.md) and [worker protocol](WORKER_PROTOCOL.md).
3. Run `pwsh -NoProfile -File scripts/setup.ps1` and
   `pwsh -NoProfile -File scripts/verify.ps1`.
4. Start with `pwsh -NoProfile -File scripts/run_dev.ps1`.

Real Transformers integration is deliberately gated on a project Python 3.12 ROCm
environment and compatibility evidence. The isolated ROCm 7.2.1 Qwen worker has now
passed its initial hardware smoke; Fedora's system ROCm packages remain unchanged at
7.1.x. Run `pwsh -NoProfile -File scripts/setup_rocm72.ps1` only when preparing that
optional worker.

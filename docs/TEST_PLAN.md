# Test plan

Normal verification is `pwsh -NoProfile -File scripts/verify.ps1`. It requires no GPU, network, model download,
container runtime, or cached model.

The end-to-end mock gateway smoke is
`pwsh -NoProfile -File scripts/smoke_all.ps1`; it starts and always stops the local
services around both generation-family checks.

Unit tests cover profiles, cache resolution and state, fingerprints, launch arguments,
redaction, and hardware probe resilience. Contract tests prove AR traces and diffusion
frames remain distinct and that gateway failures are local and structured. Integration
tests launch real isolated mock subprocesses for readiness, restart, shutdown, port
collision, and crash detection.

Later physical tests are marked `hardware`, `rocm`, `large_model`, or `long_running` and
are excluded from normal CI. Phase 3/4 requires allocation, load, stream/frame,
cancellation, memory recovery, repeated lifecycle, 30-minute per-worker, and selected
preset two-hour tests.

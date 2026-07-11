# ModelDeck

ModelDeck is a local management service for isolated model workers on the Framework
Desktop. It provides evidence-based cache discovery, hardware diagnostics, explicit
worker lifecycle states, mock autoregressive and text-diffusion workers, and a stable
local gateway.

This first implementation slice deliberately does **not** load or download a model. It
runs without a GPU by default. An optional isolated ROCm 7.2.1 worker now serves the
pinned, locally cached Qwen 2.5 0.5B model without changing Fedora's stock packages.

## Quick start

```powershell
pwsh -NoProfile -File scripts/setup.ps1
pwsh -NoProfile -File scripts/run_dev.ps1
```

The standard setup creates `.venv`. It contains the lightweight ModelDeck management
service, gateway, mock and replay workers, and development tests. It does not install the
ROCm model runtime, and is all that is required for ordinary development without a
physical GPU worker.

- Management dashboard: <http://127.0.0.1:3600>
- Stable gateway: <http://127.0.0.1:8600/v1/health>
- API documentation: <http://127.0.0.1:3600/docs>

Start a mock worker from the dashboard or with:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:3600/api/workers/mock-ar/start
```

Stop both services with `pwsh -NoProfile -File scripts/stop_dev.ps1`. See [Start here](docs/START_HERE.md)
and the [build plan](docs/BUILD_PLAN.md) for current scope and next steps.

## Optional ROCm autoregressive worker

```powershell
pwsh -NoProfile -File scripts/setup_rocm72.ps1
pwsh -NoProfile -File scripts/smoke_rocm_autoregressive.ps1
```

This optional setup creates a second environment, `.venv-rocm72`, containing the pinned
AMD ROCm 7.2.1 PyTorch and Transformers stack used by real GPU model workers. It can
coexist with `.venv`, does not replace Fedora RPMs, and is unnecessary for the dashboard,
gateway, mocks, replay, or ordinary tests. Model loading remains local-files-only.

Run each setup script initially and again when its requirements change. Compatible real
GPU workers should share `.venv-rocm72`; add another GPU environment only when recorded
compatibility evidence demonstrates a dependency conflict.

# ModelDeck

ModelDeck is a local management service for isolated model workers on the Framework
Desktop. It provides evidence-based cache discovery, hardware diagnostics, explicit
worker lifecycle states, mock autoregressive and text-diffusion workers, and a stable
local gateway.

This first implementation slice deliberately does **not** load or download a model. It
runs without a GPU and proves lifecycle handling before the real Transformers workers
are introduced.

## Quick start

```powershell
pwsh -NoProfile -File scripts/setup.ps1
pwsh -NoProfile -File scripts/run_dev.ps1
```

- Management dashboard: <http://127.0.0.1:3600>
- Stable gateway: <http://127.0.0.1:8600/v1/health>
- API documentation: <http://127.0.0.1:3600/docs>

Start a mock worker from the dashboard or with:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:3600/api/workers/mock-ar/start
```

Stop both services with `pwsh -NoProfile -File scripts/stop_dev.ps1`. See [Start here](docs/START_HERE.md)
and the [build plan](docs/BUILD_PLAN.md) for current scope and next steps.

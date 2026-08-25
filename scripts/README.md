# Script guide

Scripts are grouped by the task they perform. Run every command from the repository
root; PowerShell scripts set the working directory there before doing any work.

| Directory | Purpose |
| --- | --- |
| `operations/` | Start, stop, environment, port, manifest, and frontend operations |
| `setup/` | Control-plane and specialised runtime setup |
| `smoke/` | Mock and hardware smoke checks |
| `stability/` | Long-running ROCm and GPT-OSS stability checks |
| `benchmarks/` | Model, SceneChat, and speech benchmark tooling |
| `qualification/` | Targeted capability qualification |
| `q4/` | DiffusionGemma Q4 materialisation, probing, evaluation, and packaging |
| `migrations/` | Explicit data migrations and the v2 cut-over |
| `booth/` | Open Day booth launcher and browser watcher |
| `verification/` | Repository and hardware-adjacent verification |
| `configuration/` | Local routing-profile configuration |
| `lib/` | Shared PowerShell modules; these are not direct entry points |

The usual local workflow is:

```powershell
pwsh -NoProfile -File scripts/setup/setup.ps1
pwsh -NoProfile -File scripts/verification/verify.ps1
pwsh -NoProfile -File scripts/operations/run.ps1
```

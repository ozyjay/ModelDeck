# Trusted runtime manifests

ModelDeck separates operator-editable deployment configuration from process-launch trust.
A deployment may select a runtime template. A template may select only a reviewed runtime
implementation registered in ModelDeck code; it cannot provide an executable, Python
module, command argument, environment variable, secret, network endpoint, or filesystem
path.

## Installing a template package

Runtime template packages are local JSON files with a versioned package identity. Review
the file, calculate its SHA-256 independently, then install that exact content:

```powershell
$Digest = (Get-FileHash .\open-day-runtimes.json -Algorithm SHA256).Hash.ToLowerInvariant()
pwsh -NoProfile -File scripts/install_runtime_manifest.ps1 `
    -Manifest .\open-day-runtimes.json `
    -Sha256 $Digest
```

Supplying the digest is an explicit local-administrator trust action. The installer copies
the manifest to `.modeldeck/trusted-runtime-manifests/`, records the approved digest in a
separate trust registry, and asks you to restart ModelDeck. Startup fails closed if a
trusted file is missing, changed, malformed, duplicated, or refers to an unregistered
implementation.

Installation is intentionally unavailable through the management API and browser UI.
The read-only `GET /api/runtime-templates` endpoint reports loaded templates, package
versions, publishers, sources, implementations and digests. The Model library runtime
form offers compatible installed templates when configuring a deployment.

## Manifest format

```json
{
  "format": "modeldeck-runtime-templates",
  "version": 1,
  "package": {
    "id": "open-day-presets",
    "version": "1.0.0",
    "display_name": "Open Day runtime presets",
    "publisher": "Local operator"
  },
  "templates": [
    {
      "id": "autoregressive-long-context",
      "display_name": "Autoregressive long context",
      "runtime": "transformers-rocm",
      "generation_family": "autoregressive",
      "capabilities": {"chat": true, "completions": true},
      "settings": {"context_length": 8192, "maximum_new_tokens": 256},
      "cache_setting": "cache_root"
    }
  ]
}
```

The schema rejects extra document fields. Each implementation also constrains the allowed
generation family, cache binding and setting names. ModelDeck continues to derive cache
locations and ports from local discovery and allocation rather than from the manifest.

Adding a genuinely new worker implementation still requires reviewed ModelDeck code that
constructs its argument array and bounded environment. Once that implementation exists,
separately versioned template packages can safely expose supported presets without another
browser feature or hard-wired deployment card.

# ModelDeck contributor instructions

ModelDeck is a local-first model runtime manager and stable capability gateway for the
Framework Desktop. The target is Fedora 44, an AMD Radeon 8060S (`gfx1151`), and a
ROCm 7.2-compatible Python stack, but code must always report detected versions rather
than silently assuming the target is installed.

## Non-negotiable boundaries

- Prefer isolated custom Transformers workers; use vLLM only when compatibility evidence
  shows a benefit.
- Keep autoregressive generation and text-diffusion refinement as separate engines and
  protocols.
- Inspect existing repositories before duplicating acquisition, cache, worker, telemetry,
  or demo code. HuggingFacePull owns downloads; ModelDeck performs read-only discovery.
- Never accept arbitrary shell commands, arguments, environment variables, or paths from
  the web interface. Launch allowlisted worker manifests with argument arrays.
- Bind to `127.0.0.1` by default. Never add cloud fallback or live Open Day downloads.
- Keep mock/replay fallbacks useful when ROCm or a model is unavailable.
- Record successful and failed compatibility evidence against a complete fingerprint.
- Make small, reversible changes and add tests for changed behaviour. A phase is not
  complete until its relevant tests pass.
- Use Australian English in prose, comments, documentation, and UI copy.

## Development

Use PowerShell scripts only for project operations. Use the project `.venv`, run
`pwsh -NoProfile -File scripts/verify.ps1`, and mark physical GPU tests with the
`hardware`, `rocm`, `large_model`, or `long_running` pytest markers as appropriate.

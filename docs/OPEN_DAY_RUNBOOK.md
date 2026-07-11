# Open Day runbook

The Open Day target uses the core ROCm workers. Mock/replay workers are rehearsed
fallbacks for a demonstrated hardware or model failure, not the primary presentation.

## Cold start

1. Confirm `.venv` and `.venv-rocm72` already exist; never install or update dependencies
   at the event.
2. Run `pwsh -NoProfile -File scripts/check_ports.ps1` and
   `pwsh -NoProfile -File scripts/check_environment.ps1`.
3. Run `pwsh -NoProfile -File scripts/run.ps1 -OpenDay`. It forces downloads off.
4. Open `http://127.0.0.1:3600`; confirm gateway health and `/mnt/work` status.
5. Start the selected ROCm worker and wait for `ready`, not merely a PID. Confirm the
   gateway reports that ROCm profile as the effective provider before opening the demo.

## Switching and recovery

- Stop a worker from the dashboard before starting a conflicting exclusive model.
- If a worker fails, inspect its bounded logs, stop it, then restart once. Do not repeatedly
  retry an unchanged known incompatibility.
- The gateway returns a structured unavailable result and never calls cloud inference.
- Use `POST /api/presets/stop-all` for one-click managed-worker shutdown.
- Use `pwsh -NoProfile -File scripts/stop.ps1` to stop workers, gateway, and management
  services. Startup also cleans stale allowlisted ModelDeck workers when the management
  service is absent; it never terminates an unknown process merely because it owns a port.

Logs are under `var/log`. They must not contain visitor prompts or generated content.
Before final readiness, pin every dependency/model revision, complete the selected
two-hour burn-in, rehearse with another operator, and document real preset transitions.

# Open Day runbook

This slice is an operational rehearsal with mock workers, not final Open Day readiness.

## Cold start

1. Confirm `.venv` already exists; never install or update dependencies at the event.
2. Run `pwsh -NoProfile -File scripts/check_ports.ps1` and
   `pwsh -NoProfile -File scripts/check_environment.ps1`.
3. Run `pwsh -NoProfile -File scripts/run_open_day.ps1`. It forces downloads off.
4. Open `http://127.0.0.1:3600`; confirm gateway health and `/mnt/work` status.
5. Start the required workers and wait for `ready`, not merely a PID.

## Switching and recovery

- Stop a worker from the dashboard before starting a conflicting exclusive model.
- If a worker fails, inspect its bounded logs, stop it, then restart once. Do not repeatedly
  retry an unchanged known incompatibility.
- The gateway returns a structured unavailable result and never calls cloud inference.
- Use `POST /api/presets/stop-all` for one-click managed-worker shutdown.
- Use `pwsh -NoProfile -File scripts/stop_dev.ps1` to stop gateway and management services.

Logs are under `var/log`. They must not contain visitor prompts or generated content.
Before final readiness, pin every dependency/model revision, complete the selected
two-hour burn-in, rehearse with another operator, and document real preset transitions.

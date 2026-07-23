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

For a booth session, `pwsh -NoProfile -File scripts/run_booth.ps1` combines steps 3 and 4
and opens the operator console in a dedicated fullscreen browser profile. Use `-Windowed`
for rehearsal. The command returns after launching the booth. Closing that browser stops
ModelDeck through a background watcher; `scripts/stop.ps1` can be used instead. Neither
option stops a separately launched downstream demo such as SceneChat.

### SceneChat Gemma 4 preflight

SceneChat uses the stable port 8600 gateway, which routes privately to its managed worker
on port 8000. Before the event:

1. Provision `google/gemma-4-E2B-it` revision
   `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf` with HuggingFacePull and Xet using one worker.
   ModelDeck remains read-only and must not acquire or substitute model files.
2. Run `pwsh -NoProfile -File scripts/verify_scenechat_snapshot.ps1`. Retain its immutable
   file/blob, class, and dependency fingerprint with the compatibility evidence.
3. Set `MODELDECK_SCENECHAT_API_KEY` for ModelDeck's private loopback worker hop. Point
   SceneChat's existing `VLLM_*` compatibility variables to
   `http://127.0.0.1:8600/v1`, use model `scenechat-vision`, and keep
   `VISION_PROVIDER=vllm` for this phase. SceneChat does not need the worker credential.
4. Run `pwsh -NoProfile -File scripts/smoke_rocm_scenechat.ps1`. It preflights port 8000 and
   the snapshot, starts Open Day mode, waits for readiness, exercises native smoke and both
   `/v1` routes, then stops the worker and confirms process exit.
5. If using a configured alternative such as Gemma 4 26B, start and smoke-test its unique
   worker first, then choose it under **Workers → SceneChat provider**. SceneChat continues
   using `scenechat-vision`. Confirm `/v1/models` reports the chosen profile as both
   `selected_provider` and `effective_provider`. Selection does not start the worker, and
   ModelDeck will not silently return to E2B if the selected worker stops.

SVG remains a trusted SceneChat replay format and does not invoke ModelDeck. Do not claim
the worker is Open Day ready until the ten-request latency/memory run, 60-minute camera run,
two-hour burn-in, physical safety fixtures, camera reconnect, clean restart, cold reboot,
and operator handover pass against one complete fingerprint. If a physical gate fails,
leave the worker stopped or incompatible and use SceneChat mock, replay, or live-camera-only
mode. Never change precision, attention implementation, model, or provider automatically.

### SceneChat Qwen3.5 latency candidate

The Qwen3.5 0.2.2 candidate retains BF16, deterministic greedy generation, disabled
thinking, KV caching, SDPA and the 60-second deadline. It uses bounded internal wording for
the closest-object question and stops after the first complete JSON object. Its immutable
defaults are 140 visual tokens and a 1,024-token hard completion ceiling. Do not publish it
based on a smoke test alone.
Run both the isolated seven-question benchmark and the combined-load two-hour command in
`docs/BENCHMARKS.md`, review one fixed output per curated question, and retain the exact
Worker and runtime-template versions.

For rollback, keep the published 280-visual-token Worker stopped but unarchived until the
replacement passes every gate. If the candidate fails, stop it, restore the Event draft's
`scenechat-vision` worker list to the prior Worker, publish a new immutable Event revision,
start the prior Worker and smoke the route through gateway port 8600. Never rewrite an
existing published revision or silently select the mock Worker.

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
The fixed Q4 workload passed its two-hour burn-in on 19 July 2026 with 444 requests and no
failures. Before final readiness, pin every dependency and model revision, rehearse with
another operator, and document real management-preset transitions. Re-run
`pwsh -NoProfile -File scripts/burn_in_diffusiongemma_selected_preset.ps1` only when the
hardware or pinned inference stack changes materially. Validate its configuration without
starting hardware by adding `-ValidateOnly`.

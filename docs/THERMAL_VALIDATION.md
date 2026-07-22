# Thermal throttling validation

The automated suite uses fake telemetry and never changes TuneD, systemd, CPU governors,
or other host policy. Physical validation is explicitly invoked on the Framework Desktop.

Start ModelDeck normally, prepare one published local Route, and run each condition from
the same approximate starting temperature. Configure the external host service separately;
the script only labels the condition and records its read-only status.

```powershell
pwsh -NoProfile -File scripts/test_thermal_throttling.ps1 `
  -Condition Combined -RunControlledWorkload -Model your-public-route -DurationSeconds 600
```

Repeat with `NoProtection`, `HostOnly`, `ModelDeckOnly`, and `Combined`. Disabling
ModelDeck throttling requires the explicit `MODELDECK_THERMAL_THROTTLING_ENABLED=0`
development override before startup. Never disable both protections during unattended or
long-running work. The runner stops submitting work if telemetry becomes degraded or the
critical state is reached, records peak/mean temperature and time above each policy
threshold, and retains per-request latency and thermal-admission results under
`var/benchmarks/`.

The repository cannot truthfully provide measured comparative results without the target
GPU, selected cached model, controlled starting conditions, and external host service.
Record those four physical runs before treating performance/temperature trade-offs as
validated hardware evidence.

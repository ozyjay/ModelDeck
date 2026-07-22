# ModelDeck thermal throttling architecture

ModelDeck already obtains normalised temperature readings through
`modeldeck.hardware.probe`, while `modeldeck.thermal.ThermalGuard` owns the existing
critical speech-synthesis shutdown limits. The throttling feature extends those paths;
it does not introduce a privileged sensor reader or replace the critical guard.

The management service owns the single host-level `ThermalPolicyManager`. It polls the
normalised hardware telemetry, validates the selected APU sensor and freshness, advances
the hysteretic state machine, records transitions, and publishes an atomic runtime status
snapshot beneath ModelDeck's data directory. Runtime state is deliberately not restored
after a restart: the manager begins in `telemetry_degraded` until it receives a fresh
reading.

The worker supervisor consults the manager before model loads. The separate gateway
process reads the bounded status snapshot and applies the same central admission policy
before forwarding inference requests. This keeps thresholds and state ownership out of
workers while allowing all local workers to share one effective concurrency limit.
Missing or stale snapshots fail closed. Lightweight health, status, cancellation, and
cooldown observations remain available.

Host power policy is diagnostic only. A bounded reader may execute the fixed argument
arrays `tuned-adm active` and `systemctl is-active framework-thermal-policy.service`; no
mutation, `sudo`, arbitrary command, environment, or sensor path is accepted.

SceneChat degradation is explicit in each decision and status response. At elevated
states the gateway prevents overlap, reports a longer minimum frame interval, and blocks
automatic work before manual interactive work. Benchmarks use the same workload classes
and must remain paused whenever telemetry is degraded or the state is warm or higher.

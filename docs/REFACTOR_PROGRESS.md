# Incremental refactor verification

## Phase 1 — isolation and security boundaries

The initial baseline with loopback access passed 559 Python tests and 20 frontend
tests (9 Python tests skipped). The earlier 13 socket failures were sandbox
restrictions; the documented verification was rerun with local socket permission.

Behaviour characterised and changes:

- Fresh-process imports and both application factories preserve operational files,
  including their content and modification times. Lifespan initialises databases
  and thermal state. Persisted Worker logs are loaded and compacted by an explicit
  log start-up service; restored diagnostics are redacted again before exposure.
- Both listeners validate literal loopback bind addresses, including directly
  constructed Settings. Both ASGI applications validate Host before handlers;
  remote names cannot gain access by resolving to loopback. The configured Docker
  bridge retains its explicit gateway Host exception.
- Mutations and WebSockets reject remote, opaque, malformed, empty and duplicate
  Origin headers. Native clients omit Origin but remain subject to Host checks.
  Loopback browser applications on other ports retain the existing supported path.
- Diagnostics redact nested mappings, lists, prompts, outputs, headers, bearer
  credentials, API keys and token-shaped strings. Input and output are capped at
  8192 characters, with traversal depth and collection limits. Oversized or malformed
  structured diagnostics fail closed, so some diagnostic detail is intentionally lost.
- Worker launch tests confirm offline flags and exclusion of unrelated credentials,
  Python path injection and preload variables. Existing runtime-specific launch tests
  retain the SceneChat credential checks. No runtime or model pins were changed.
- Test defaults use temporary database, log and thermal-state locations through
  per-test environment settings. ASGI fixtures use loopback URL hosts.

Principal files: `backend/modeldeck/security.py`, `config.py`, `main.py`,
`gateway/app.py`, `supervisor/service.py`, `tests/conftest.py`,
`tests/unit/test_browser_boundary.py`, `test_process_isolation.py`,
`test_config.py`, `test_supervisor.py`, and the API contract documentation.
Contract/integration fixture URL adjustments exercise the production Host policy.

Verification on 7 September 2026: `pwsh -NoProfile -File scripts/verify.ps1`
passed with loopback permission: 603 Python tests passed, 9 skipped; 20 frontend
tests passed; lint, formatting, TypeScript, build and committed-asset checks passed.
The new security and isolation tests also passed in focused runs. Six failing
redaction regression cases were reproduced before their fixes.

Remaining validation and trade-offs:

- Five physical hardware tests and four torch-dependent tests remain unrun in the
  control-plane environment. Qualified ROCm Worker load, warmup, inference, shutdown
  and thermal behaviour still require physical revalidation of the reduced launch
  environment. Mock lifecycle tests are not evidence of physical runtime qualification.
- Local origin/Host checks are not authentication between local users or processes.
  Native clients and embedded ASGI callers must follow the documented Host/lifespan
  contract. Arbitrary local DNS aliases are intentionally rejected.
- Database upgrade/degraded-health diagnostics and route identity correctness are
  Phase 2 work. They are not represented here as completed.

## Phase 2 — persistence and routing correctness

Completed software verification on 7 September 2026. This phase advances explicit
execution identity, observable failures, reproducibility and preservation of raw
evidence (guiding principles 6–8 and 12).

Behaviour characterised and files changed:

- `backend/modeldeck/persistence.py` owns every production SQLite connection:
  foreign keys enabled, a 2000 ms busy timeout, `sqlite3.Row`, transaction rollback
  and deterministic connection closure. Store, legacy migration and state-import
  consumers use it; ordinary access cannot create a missing database.
- `compatibility/store.py` no longer turns SQLite failures into empty Workers,
  routes, policy or tests, or default rehearsal state. Typed storage failures feed
  structured HTTP 503 degraded responses in management and gateway. Health probes
  detect missing schema tables. Known legacy profile-only exports are handled by
  explicit table detection, not an exception fallback.
- `gateway/resolution.py` introduces `ResolvedRoute`, retaining configured order,
  requested identity and unusable Worker reasons separately from executable
  candidates. Gateway discovery and route metadata expose requested/reported Model,
  Artifact, Runtime, Backend, device, precision, context and KV-cache fields.
  Unreported values stay unknown. Native discovery and route listing include a
  `resolution` map; existing public route names and protocol surfaces are retained.
- `gateway/app.py` safely advertises routes with no executable candidates. Missing,
  archived, corrupt, retired and unavailable Workers cannot become implicit
  replacements. Configured backups remain backups in discovery and response roles.
  Malformed health and health reporting a different Worker ID are not ready.
- Schema v4-to-v5 initialisation is transactional and preserves Worker definitions,
  publications, policy, positive/negative evidence and observations. Restart no
  longer resurrects a deactivated route from the diagnostic singleton table.
  Future schemas are refused with downgrade guidance.

Tests added: `tests/unit/test_persistence.py`, `test_resolved_routes.py`, and
`test_migrate_v4_to_v5.py` (31 cases), plus updated gateway contract expectations
for additive identity metadata and explicit lifespan initialisation. Coverage
includes connection settings, foreign-key enforcement, closure, rollback, lock
contention, startup corruption, removed databases/tables, discovery failure cases,
separate requested/reported identities, migration idempotence and failure rollback.

Verification: `pwsh -NoProfile -File scripts/verify.ps1` passed with local socket
permission: **634 Python tests passed, 9 skipped; 20 frontend tests passed**.
Lint, formatting, TypeScript, frontend build and committed-asset checks passed.
Focused persistence, discovery and migration checks also passed.

Remaining risks: enabling foreign keys can expose pre-existing inconsistent data;
the migration never deletes evidence to conceal such inconsistencies. Startup
storage failures require repair and restart. Discovery is a health snapshot, so
readiness can change before inference. Schema downgrade requires a retained backup.
See [migration and diagnostic guidance](MIGRATION_V4_TO_V5.md).

Hardware validation: the five physical GPU tests and four torch-dependent tests
remain skipped in the control-plane environment. Physical Worker requalification
from Phase 1 remains outstanding. This phase changes no runtime/model pins,
inference engines, model acquisition or thermal thresholds.

## Remaining phases

Phase 3: event fan-out/recovery, operator snapshots, health caching and frontend resilience.

Phase 4: resource/router and frontend feature decomposition, with contract validation.

Phase 5: shared port allocations within 8610–8624 and a reproducible dependency lock workflow.

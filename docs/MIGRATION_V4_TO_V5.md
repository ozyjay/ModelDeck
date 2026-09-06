# Schema v4 to v5 and persistence diagnostics

Schema v5 adds durable capability setup operations and their event history. The
management and gateway lifespans upgrade an existing v4 store locally, without
changing Worker definitions, routing publications, policy or compatibility evidence.
Factories and imports never perform the migration.

Before upgrading an installation, stop its management and gateway processes and
use `scripts/operations/export_state.ps1` to retain a backup. Start the new build
against that state store, then inspect `/api/health` and `/v1/health`. Normal
initialisation remains automatic; no SQL version edits or model downloads are needed.

Schema creation and version updates form one transaction. Failure rolls back the
schema changes. Repeated start-up is idempotent for operational state: a deactivated
route remains inactive. The legacy singleton activation is copied only when the
multi-profile activation table is first introduced, and is otherwise retained as
historical diagnostic data. It is not authoritative routing policy.

Both APIs return HTTP 503 with `status: degraded` and an `error` object containing
`component: persistence`, `code` and `operation` for storage failures. Codes include
`database_busy`, `database_locked`, `database_unavailable`, `database_corrupt`,
`database_constraint`, `database_schema_incomplete`, `invalid_persisted_json`, `invalid_routing_snapshot` and
`schema_upgrade_required`. Unexpected SQLite errors report `database_error`.
No SQL, credentials or operational paths are returned. An absent record in a
healthy store still has its usual empty/not-found meaning; a missing database does not.

Connections enable foreign keys and use a 2000 ms busy timeout. Transactions close
their connections explicitly. Existing inconsistent foreign-key records are not
deleted or rewritten automatically; constraint failures need operator diagnosis.
For a busy store, let the competing writer finish and retry. For corruption or an
unsupported schema, stop services and repair or restore state before restarting.
Startup failures remain degraded until the process is restarted successfully.

Downgrading v5 data in place is unsupported. Use a matching build or restore the
pre-upgrade backup; never change `schema_metadata` to trick an older build into
opening newer data. Restoring a backup necessarily omits operations and evidence
recorded since that backup, so retain the newer state separately. A build that sees
an unsupported future version refuses it and returns these downgrade diagnostics.

Gateway discovery retains ordered `configured_worker_ids`, `configured_workers`
and `candidate_worker_ids` in route metadata. Missing, archived, corrupt or retired
Workers remain identifiable but are not executable candidates. An explicitly
configured backup remains labelled as a backup even when the primary disappears.
No substitute Worker is added. `requested` identity describes configuration;
`resolved` contains reported ready-Worker identity. Unreported backend, device,
precision, artifact, context or KV-cache values remain unknown rather than inferred.

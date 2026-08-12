# Architecture

## Conceptual model

```text
Model (cached, read-only)
  └─ Worker (configured, trusted runtime)
       └─ Published capability (public model name + one protocol + ordered Workers)
            └─ Routing Profile (immutable revisions; exactly one active)
```

A Routing Profile may serve every concurrent local application: an Open Day demo,
SprintBot, or another project. It is not an ownership boundary for a frontend, repository,
or event. Workers and their cached models are independent objects and can be referenced by
many capabilities.

## Runtime boundary

```text
Operator console/API :3600 ── WorkerSupervisor ── allowlisted local Workers
          │                         │
          │                         └─ fixed argument arrays; no browser commands or paths
          └─ SQLite profiles, revisions, evidence, and durable job ownership

Applications ── gateway :8600 ── active capability ── first ready Worker before work begins
```

`.venv` owns the management service, gateway, supervisor, discovery and tests.
`.venv-rocm72` owns the primary ROCm inference stack; `.venv-rocm72-q4` isolates Q4
dependencies. Model libraries and tensors never enter the management process.

## Code-owned protocols

The gateway has a static protocol-adapter registry. An adapter owns its contract,
validation, public surfaces, worker-path mapping, stream/job handling, timeout class and
smoke request. Operators can bind only these contracts; they cannot configure a route,
schema, operation, command, path, or environment variable.

Use an OpenAI-compatible route whenever it expresses the application need. Add a native
ModelDeck protocol only for a reusable, low-level interaction such as a token candidate
trace or incremental text-diffusion frames. Project-specific behaviour stays in the
project rather than expanding ModelDeck.

## Publication and recovery

Each profile has one mutable draft and immutable published revisions. Validation checks
Worker existence, protocol compatibility, ordered Worker references, and where requested,
matching tested-working evidence. Publishing atomically selects the active profile without
starting or stopping a process; rollback selects an existing immutable revision.

The gateway is local-only. It starts with no published capabilities, sends no cloud
fallback, chooses a backup only before a request or job begins, and persists
text-diffusion job ownership for restart-safe polling/cancellation. Test fixtures are not
operator-visible and cannot be represented as a public fallback.

## Database migrations

SQLite schema v3 stores Workers, Routing Profile drafts and revisions, one active routing
snapshot, model cache policy, compatibility evidence and gateway job assignments. A v2
database is refused at startup. Run `scripts/migrate_v2_to_v3.ps1`: it backs up the
database/WAL/SHM files, converts every Event revision into a profile revision, preserves
routes as capabilities and the active routing selection, drops Demo membership, and leaves
Workers, model caches and evidence untouched.

SQLite schema v4 adds revision-scoped capability policy. Run
`scripts/migrate_v3_to_v4.ps1` to create a timestamped database/WAL/SHM backup and
grandfather capabilities represented by non-archived Workers and current draft or active
Routing Profile bindings. Historical revisions remain unchanged and do not grant policy.

# Architecture

## Conceptual model

```text
Model (cached, read-only)
  └─ Worker (configured runtime; editable name, immutable execution identity)
       └─ Route (public name + protocol; primary Worker + ordered backups)
            └─ Demo (uses one or more shared Routes)
                 └─ Event (versioned publication boundary)
```

The relationships are references, not ownership. One Model can have several Workers for
different runtimes or settings. One Worker can serve several Routes. A Route can be shared
by several Demos in an Event. Removing a Demo therefore does not remove its Routes or
Workers.

Operator-facing names are labels and can change. Workers, Routes, Demos and Events have
hidden UUIDs so renaming does not rewrite references. A Route's `public_name` is different:
it is the external identifier sent by demo clients in the gateway `model` field, so an
operator edits it intentionally as part of the Route contract.

## Runtime boundary

```text
Operator console/API :3600 ── WorkerSupervisor ── trusted Worker processes on loopback
          │                         │
          │                         └─ fixed argument arrays; no browser-supplied commands
          └─ SQLite configuration, evidence and active Event snapshot

Demo applications ── gateway :8600 ── published Route ── first ready Worker in order
```

`.venv` owns the management service, gateway, supervisor, discovery, fallback fixtures and
tests. `.venv-rocm72` owns the primary ROCm inference stack. `.venv-rocm72-q4` isolates the
Q4 GPTQ dependencies. Model libraries and tensors never enter the management process;
stopping a Worker process is the memory-recovery boundary.

## Trusted configuration

Operator configuration and executable trust are separate:

- Operators CRUD Events and Workers and edit names, Route contracts and Worker order.
- ModelDeck code owns protocol contracts and launch builders.
- Versioned runtime templates select an installed launch builder and bounded settings.
- Worker creation accepts a discovered pinned Model, a trusted runtime template and
  bounded options. It never accepts a command, executable, path, arbitrary argument,
  environment variable or remote-code flag from the web interface.

No Worker instances or public Route aliases are seeded. The trusted templates describe
what may be created, not what exists.

## Event lifecycle

Each Event has one mutable autosaved draft and zero or more immutable published revisions.
Validation checks that all Worker references exist, the generation family and capabilities
match the protocol, Worker order is unambiguous, and any requested tested-working policy
has matching evidence. Publishing creates a revision and atomically replaces the one live
routing snapshot. It never starts or stops a process.

An earlier revision can be made live again exactly. Discarding a draft restores the newest
published definition. Published revisions retain their Worker UUID references; replacing a
Worker can rebind mutable drafts but never silently rewrites history.

## Gateway

The gateway has one routing authority: the active Event snapshot. With no published Event,
`/v1/models` is empty and requests return structured `local_route_unavailable` responses.
For each Route the gateway tries the primary Worker and then backups in displayed order,
using readiness rather than process existence. No cloud fallback occurs.

Routes are exposed only on surfaces permitted by their trusted protocol contract. The
gateway does not expose physical Worker identity as an application-facing provider layer.
Mock/replay use remains explicit and is signalled as fallback evidence.

## Persistence and cut-over

SQLite schema v2 stores Workers, Event drafts, Event revisions, the active routing
snapshot, exact Model cache policy and compatibility evidence. A legacy unversioned
database is refused rather than guessed or auto-migrated. `scripts/cutover_v2.ps1` moves
only the exact database, WAL and SHM files to a timestamped backup and initialises an empty
v2 database.

Worker smoke tests make a real bounded generation request and record both success and
failure against the detected hardware, OS, ROCm/library versions, pinned Model revision,
runtime, data type and relevant environment overrides.

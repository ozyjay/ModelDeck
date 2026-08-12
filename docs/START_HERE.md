# Start here

ModelDeck manages local Model runtimes and publishes stable capabilities for local
applications. It begins with no configured Workers or routing profile.

1. Run `pwsh -NoProfile -File scripts/setup.ps1` to prepare the control plane and target
   inference environments. Use `-ControlPlaneOnly` only for lightweight development.
2. If a v2 configuration database exists, run
   `pwsh -NoProfile -File scripts/migrate_v2_to_v3.ps1 -WhatIf`, review the backup path,
   then run it without `-WhatIf`. For any existing v3 database, repeat that review-and-run
   process with `scripts/migrate_v3_to_v4.ps1`. Startup refuses an unmigrated database.
3. Run `pwsh -NoProfile -File scripts/verify.ps1`.
4. Start ModelDeck with `pwsh -NoProfile -File scripts/run.ps1` and open
   <http://127.0.0.1:3600>.
5. In **Models**, inspect the potential capabilities and their detected or reviewed
   evidence, then explicitly allow the capabilities you intend to use. An allowed
   capability with no trusted runtime records intent but is not runnable.
6. Create a named Worker from an allowed capability with an installed trusted runtime.
7. In **Workers**, start it and run capability-specific qualification when a
   `tested-working` Routing Profile will require exact evidence.
8. In **Routing profiles**, create a profile, add each capability needed by concurrent
   applications, choose its trusted protocol, and set primary and backup Workers.
9. Validate and publish the profile. Publishing changes routing only; it does not start
   Workers.
10. In **Live**, inspect the
   published capabilities and rehearse a ready capability through the gateway.

The mental model is small: Models are discovered data, Workers execute Models,
capabilities are public contracts, and one Routing Profile atomically publishes the active
set. ModelDeck never downloads Models. HuggingFacePull owns acquisition; ModelDeck performs
read-only discovery. Services bind to loopback and never use cloud inference fallback.

For tests and debugging, deterministic fixtures live in the test harness. They are not
available in the operator console and do not count as real-worker evidence.

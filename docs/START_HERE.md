# Start here

ModelDeck manages local Model runtimes and publishes stable Routes for demo applications.
It begins with no configured Workers or routing.

1. Run `pwsh -NoProfile -File scripts/setup.ps1` to prepare the control plane and target
   inference environments. Use `-ControlPlaneOnly` only for lightweight development.
2. If this checkout has a v1 database, run
   `pwsh -NoProfile -File scripts/cutover_v2.ps1 -WhatIf`, review the paths, then run the
   command without `-WhatIf`. The exact SQLite files are backed up; caches and evidence
   artefacts are not deleted.
3. Run `pwsh -NoProfile -File scripts/verify.ps1`.
4. Start ModelDeck with `pwsh -NoProfile -File scripts/run.ps1` and open
   <http://127.0.0.1:3600>.
5. In **Models**, create a named Worker from a recognised cached Model.
6. In **Events**, create an Event, add a Route, choose its protocol, assign the primary
   and ordered backup Workers, and associate the Route with a Demo.
7. Validate and publish the Event. Publishing routing does not start Workers.
8. In **Workers**, start and smoke-test the selected Worker. In **Live**, rehearse the
   published Route through the gateway.

The mental model is deliberately small: Models are discovered data, Workers execute
Models, Routes are public contracts, and Events publish versioned sets of Routes and
Demos. See [architecture](ARCHITECTURE.md) and [operator workflow](DEMO_ROUTE_HOWTO.md).

ModelDeck never downloads Models. HuggingFacePull owns acquisition; ModelDeck performs
read-only discovery. Services bind to loopback and never use cloud inference fallback.

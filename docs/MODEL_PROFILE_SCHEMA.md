# Internal Worker execution definition

The old operator-facing “model profile” concept has been removed. Operators create a
**Worker** from a discovered **Model** and a trusted runtime template. The persisted Worker
has an editable `name` and an immutable execution definition containing:

- a hidden UUID;
- exact Model and optional derivative artefact identities and revisions;
- generation family and capabilities;
- trusted runtime and template versions;
- lifecycle class and allocated loopback port;
- bounded data type, context/output limits and runtime-specific settings.

ModelDeck converts this definition to the existing internal `ModelProfile` process-launch
shape while the supervisor is migrated. Its internal alias is derived from the UUID and is
never a public Route name. There are no packaged Worker definition files and no seeded
instances.

Execution fields are immutable because compatibility evidence and published Event history
refer to the exact execution fingerprint. A material change creates a replacement Worker;
mutable Event drafts can be rebound explicitly. The web interface cannot provide commands,
executables, paths, arbitrary arguments, environment variables or remote-code flags.

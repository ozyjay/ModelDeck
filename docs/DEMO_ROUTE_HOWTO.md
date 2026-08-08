# Routing Profile operator workflow

Use one Routing Profile to publish the local capabilities needed concurrently by any
application: Open Day demos, SprintBot, or another local project. A profile does not own
those applications; it is ModelDeck's atomic routing boundary.

1. Create Workers from recognised, cached Models. Confirm each required Worker can load
   and run a real bounded smoke request on the intended hardware.
2. Create or edit a Routing Profile. Add a published capability for each public model name
   required by an application, select one trusted protocol contract, and select the primary
   then any backup Workers.
3. Choose `compatible` for protocol-compatible Workers, or `tested-working` to require
   matching successful compatibility evidence for the complete Worker fingerprint.
4. Validate the draft. Correct every incompatible Worker, missing evidence, duplicate
   public name, or unavailable capability before publishing.
5. Publish. ModelDeck creates an immutable revision and atomically makes it active; it
   never starts or stops Workers. Review **Live** and explicitly start/smoke the real
   Workers required for the session.
6. To roll back, select an older immutable revision and make it active. Replacing a Worker
   can update profile drafts, but never rewrites published history.

Use OpenAI-compatible `/v1` APIs when they express the consumer's need. Use a native
ModelDeck capability only for reusable low-level interactions, such as autoregressive
candidate traces or iterative text-diffusion frames. A feature unique to one project
belongs in that project's code rather than in a ModelDeck adapter.

Open Day mode locks all profile edits and publication server-side. It intentionally leaves
explicit Worker lifecycle actions available so a prepared profile can still be operated
locally. Deterministic fixtures are test-harness tools: they do not appear in this workflow
and are never evidence that a hardware Worker is ready.

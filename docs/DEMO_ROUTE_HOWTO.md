# Configure an Event and its demo Routes

## First configuration

1. Open **Models** and choose a recognised cached Model.
2. Select **Create Worker**, give it a human name and choose a compatible trusted runtime.
   Adjust the bounded context, output or denoising limits shown for that runtime. Fixed
   requirements such as data type or exclusive lifecycle are visible but locked. A Model
   can have several Workers when different execution settings are useful.
3. Open **Events** and select **Create Event**.
4. Add a Route. Set:
   - a display name for operators;
   - the public name that demo applications send in their `model` field;
   - the trusted protocol contract; and
   - the primary Worker followed by any backups in exact failover order.
5. Add one or more Demos and tick the shared Routes each Demo uses.
6. Select a qualification policy. Use **Protocol compatible** while assembling a setup;
   use **Tested working (Open Day)** when every assigned Worker must have matching physical
   smoke evidence.
7. Validate, address any reported Worker or capability mismatch, then publish routing.

Edits autosave as a draft. Publishing makes an immutable revision live but does not start
Workers.

## Rehearsal

In **Workers**, start the required Worker and run **Smoke**. That test talks directly to
the Worker, produces bounded real output and records compatibility evidence. In **Live**,
select **Rehearse Route** to test the currently published public contract through the
gateway. This distinguishes Worker compatibility from end-to-end routing readiness.

## Changes and recovery

- Renaming a Worker is safe because Event references use its UUID.
- Changing a public Route name is a client contract change; update the demo application.
- Reorder a Route's Workers to change primary/failover priority, validate, then publish a
  new revision.
- **Discard draft** restores the newest published Event definition.
- **History → Make live** atomically reactivates an exact earlier revision.
- To change a Worker's immutable execution settings, select **Replace** on its Worker card.
  ModelDeck keeps the original Worker and can rebind draft routes to the replacement.
  Published revisions continue to describe what actually ran until the updated draft is
  explicitly published.
- Archiving a Worker is blocked while a draft or active revision references it. Immutable
  historical revisions keep their reference as an audit record.

Open Day mode locks all configuration mutation server-side. Prepare and publish the Event
before entering that mode; process start, stop and rehearsal remain explicit operations.

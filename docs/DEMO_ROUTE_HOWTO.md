# HOWTO: share model deployments across demos

ModelDeck is designed so one configured model runtime can serve more than one demo. Do
not create a duplicate runtime configuration merely because two demos use the same model.
Instead, bind the same deployment to each compatible demo route.

## Understand the relationships

| Concept | Meaning | Relationship |
| --- | --- | --- |
| Model artefact | Cached or packaged model weights at a specific revision | Can support one or more deployments |
| Deployment | A trusted runtime configuration for a model artefact | Can serve many demo routes |
| Worker | The current process state of a deployment | Zero or one running worker per deployment |
| Demo route | A stable gateway alias and protocol contract used by a demo | Can have ordered deployment providers |

A deployment may be shared only when it satisfies every route's generation family,
protocol adapter and capability requirements. For example, `qwen-small-rocm` can serve
both the `fast-chat` chat route and the `token-explainer` trace route because it implements
both contracts. A text-diffusion deployment cannot serve either route.

## Configure a shared deployment

1. Start ModelDeck in normal mode, not Open Day mode:

   ```powershell
   pwsh -NoProfile -File scripts/run.ps1
   ```

2. Open the operator console at <http://127.0.0.1:3600>.
3. Open **Model library** and configure the recognised model artefact once. ModelDeck
   creates a deployment and its corresponding stopped worker.
4. Open **Demo routes**, select the required demo set, then choose **Edit**.
5. Add or rename demos as required. A demo's identifier is the stable internal reference
   used by its routes, while its display name is operator-facing. Renaming an identifier
   in the editor updates every route in that draft that refers to it.
6. Create or select a route for each demo requirement. Give every route:

   - a unique public model alias;
   - the adapter expected by the demo client;
   - an explicit qualification and fallback policy;
   - one or more ordered provider deployments.

7. Add the same deployment as a provider on every compatible route that should share it.
   Provider priority is local to each route, so the deployment can be primary for one
   route and a fallback for another.
8. Save the draft as a new revision, then run **Validate**. Resolve every family,
   capability, cache-policy or evidence error before continuing.
9. Run **Plan activation**. The plan reports required worker transitions but does not
   start or stop anything.
10. Choose **Activate routing**. Activation atomically changes gateway routing; it still
   does not load a model.
11. Open **Workers** and start the shared deployment once. Every compatible active route
    can then use that one worker process.
12. Return to **Demo routes**, choose **Check readiness** for each route, then run
    **Smoke route** where supported. Speech routes require an interactive WebSocket client.

## Change or remove a deployment safely

Before removing a runtime configuration, account for every reference to its deployment:

1. In **Demo routes**, edit all routes that use the deployment. Assign a replacement or
   intentionally leave the route structurally unavailable.
2. Validate and activate the revised demo set so the gateway no longer depends on the
   deployment.
3. In **Model library**, inspect the deployment's **Used by** list. It combines current
   draft and active demo-route bindings, effective legacy aliases and worker state.
4. If no demo set is active and the deployment is selected for a legacy alias such as
   `scenechat-vision`, open **Workers** and select a different compatible provider. Once
   a demo set is active, its bindings are authoritative and stored legacy selections are
   shown as superseded rather than blocking removal.
5. Stop the deployment's worker if it is running.
6. In **Model library**, choose **Remove configuration**. This removes ModelDeck's runtime
   configuration but retains the cached model artefact.

For Gemma 4 12B specifically, removal is rejected while
`local-scenechat-gemma-4-12b` is the selected `scenechat-vision` provider. Select the
packaged Gemma 4 E2B provider—or another compatible SceneChat deployment—first.

## Diagnose a rejected removal

- **Select a different provider before removing this runtime configuration**: reassign
  the reserved alias in **Workers**.
- **A running worker cannot be removed**: stop it and wait for the `stopped` state.
- **Open Day mode locks configuration**: restart ModelDeck without `-OpenDay`, make the
  change, validate it and reactivate the demo set before returning to booth mode.
- **Route validation reports an unknown deployment**: a draft or active route still
  names a removed configuration; bind a valid replacement and activate the correction.

The **Used by** list is based on the management API's deployment dependency view. Removal
is disabled until every blocking item has been reassigned or stopped; the server enforces
the same rule even if a client bypasses the console.

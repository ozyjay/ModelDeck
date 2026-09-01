# Runtime implementation checklist

Read this reference only after triage establishes that an exact cached Model revision needs a
new or extended runtime and the request authorises implementation.

## Define the candidate first

Record the proposed configuration before coding:

- exact Model ID and pinned revision;
- exact artefact files, format, quantisation, and hashes where practical;
- architecture, parameter traits, modality, and generation family;
- intended workload and protocol contract;
- Runtime, resolved Backend/device, precision, context and KV-cache policy;
- lifecycle class, memory and thermal expectations, and failure boundaries;
- required packages, interpreter/environment, native executable or kernel identity;
- cold/warm load, correctness, cancellation, unload, and recovery acceptance criteria.

State what is detected, reviewed, assumed, and still unverified. If the design depends on an
unreviewed conversion, remote code, mutable upstream revision, arbitrary path, or web-supplied
launch value, stop and redesign it.

## Preserve ModelDeck boundaries

- Keep HuggingFacePull responsible for acquisition; ModelDeck performs read-only discovery.
- Bind services to `127.0.0.1` by default and use the repository's reserved, configurable ports.
- Launch only code-owned implementations with fixed argument arrays and bounded environment.
- Never accept commands, arguments, environment variables, executables, or arbitrary paths from
  the management API or browser.
- Isolate large Models one per Worker unless explicit evidence supports another arrangement.
- Do not add cloud fallback, live downloads, silent model/artefact/runtime/backend substitution,
  or an operator-visible mock runtime.
- Keep secrets, prompt content, and private inputs out of logs and compatibility fingerprints.

## Integrate the smallest complete slice

Inspect the existing analogue before deciding which files need changes. A complete runtime may
need updates in these areas, but do not touch an area merely to satisfy this list:

1. **Discovery and identity** — exact matcher, completeness checks, generation family, artefact
   identity, provenance, and a useful unsupported reason in
   `backend/modeldeck/catalogue/` or an existing model-specific module.
2. **Code-owned trust** — implementation capabilities, accepted settings, cache binding, and
   generation-family constraints in `backend/modeldeck/runtime_trust.py`.
3. **Runtime template** — a versioned packaged template in
   `backend/modeldeck/registry_data/runtime_templates.json`; a local manifest may select a
   reviewed implementation but must never supply executable behaviour.
4. **Capability matching** — expose only protocols genuinely implemented by the Worker in
   `backend/modeldeck/capabilities.py` and the code-owned protocol registry.
5. **Worker definition and launch** — immutable resolved identity, derived cache path and port,
   allowlisted module/executable and arguments, bounded environment, lifecycle class, and
   supervisor validation.
6. **Worker implementation** — offline load, readiness/health identity, bounded request
   validation, serialisation or concurrency limits, cancellation, timeouts, structured errors,
   unload, and process-exit recovery.
7. **Operator surface and documentation** — show cached, runtime-available, Worker-configured,
   ready, and qualified as separate states. Use Australian English and do not advertise proposed
   or unqualified behaviour as implemented capability.

Prefer extending a proven Worker only when its protocol and execution identity remain honest.
Create a separate Worker/runtime when engines, dependencies, artefact semantics, or lifecycle
requirements materially differ.

## Test in layers

Add focused tests for every changed behaviour. Use the repository's PowerShell entrypoints for
project operations and keep `.venv` separate from specialised inference environments.

At minimum, cover applicable layers:

- discovery of the exact complete snapshot and rejection of partial, wrong-revision,
  wrong-quantisation, or missing-artefact variants;
- runtime-template schema and trust-boundary rejection tests;
- profile/Worker construction with derived paths, ports, settings, and immutable identity;
- supervisor launch mapping without arbitrary command, argument, environment, or path input;
- protocol contract validation, health identity, successful request, structured failure,
  cancellation, timeout, stop, and restart/recovery;
- management catalogue and Worker API behaviour;
- mock or replay tests that exercise orchestration without claiming hardware compatibility.

Run focused checks first, then:

```powershell
pwsh -NoProfile -File scripts/verify.ps1
```

Do not weaken a check or reclassify a failure to make verification pass. Report any check that
cannot run.

## Qualify the exact configuration

Physical qualification is a separate, explicit phase. Mark tests with the applicable
`hardware`, `rocm`, `large_model`, or `long_running` markers and preserve raw observations apart
from conclusions.

Where applicable, measure:

- detected runtime, library, kernel, GPU and operating-system versions;
- cold and warm load, preprocessing, prefill, first output, decode and total latency;
- correctness for the intended workload and protocol contract;
- memory behaviour, cancellation, timeout, unload and process recovery;
- resolved backend/device and absence of silent CPU or artefact fallback;
- thermal state throughout the run and policy-triggered pause or termination;
- direct-runtime, Worker, and API-path overhead separately when performance is material.

Record positive and negative evidence against the complete fingerprint. A loadable model, a
healthy process, an old successful run, or acceptable mock output is not sufficient to mark the
new configuration qualified.

# Trusted runtime currency and qualification

**Status:** Phase 1 implemented; Phases 2–4 remain proposed
**Recommendation:** implement local runtime inventory and exact identity reporting before adding
any update workflow  
**Proposed:** 1 September 2026

## Implementation status

Phase 1 is implemented for the single managed llama.cpp Vulkan installation. ModelDeck now:

- inspects the fixed executable and build receipt without loading a Model;
- reports integrity, trust currency, exact hashes, required features and consumer identities at
  `GET /api/runtime-installations` and in the Advanced console;
- validates the same installation immediately before GPT-OSS or Qwen llama.cpp Worker launch and
  blocks missing, modified, unaccepted or feature-incompatible builds; and
- includes the executable and receipt identities in llama.cpp Worker configuration fingerprints.

This implementation does not claim per-Worker qualification currency, inspect Python
environments, contact upstream services or perform upgrades. Those remain the later phases below.

## Executive recommendation

ModelDeck should not use a single, ambiguous **out of date** flag. It should report three
independent judgements for every installed Runtime implementation:

1. **Installation integrity** — whether the detected executable or environment matches its
   local build receipt and has the required backend features.
2. **Trust currency** — how the detected identity relates to ModelDeck's code-owned accepted and
   recommended identities.
3. **Qualification currency** — whether compatibility evidence still matches the complete
   execution fingerprint that would run now.

An optional fourth judgement may report that a newer upstream revision is known, but it is
advisory only. ModelDeck remains offline-first and must not contact upstream services during
startup, discovery, Worker creation or inference.

The first implementation should cover the managed llama.cpp Vulkan installation shared by
GPT-OSS and Qwen Workers. It should expose the installed commit, executable SHA-256, build
receipt, required command-line features and evidence status through a read-only management API
and the operator console. It should not download, build, replace or select another Runtime.

## 1. Why version numbers are insufficient

A Runtime is usable only as part of a complete Configuration. A newer llama.cpp commit may add
features while regressing MXFP4 loading, Vulkan kernels, chat templates, tool calls, long-context
behaviour, cancellation or memory recovery. Conversely, an older pinned build may remain the
only physically qualified choice for a particular Artifact and workload.

The following statements are therefore different and must remain separate:

- a newer upstream commit exists;
- the installed binary differs from ModelDeck's recommended binary;
- the installed binary has been modified since it was built;
- the installed binary is trusted but no longer preferred;
- the installed binary is trusted but its evidence is stale on the current driver or hardware;
- the installed binary is exact and trusted but has never passed this workload;
- the installed binary cannot provide the required backend or argument surface.

Only the second statement is reasonably described as version currency. None alone proves that a
Runtime should be upgraded or that a Worker is usable.

## 2. Current foundations and gaps

ModelDeck already has useful pieces of this design:

- packaged Runtime Templates have package versions and digests;
- compatibility evidence fingerprints the Runtime registration, Worker configuration, Model,
  Artifact, Backend, device, software environment and workload;
- Qwen llama.cpp manifests pin a llama.cpp commit and GGUF identities;
- the llama.cpp setup script writes `modeldeck-build.json` with the commit, backend, platform and
  executable SHA-256;
- Qwen llama.cpp startup validates the receipt and re-hashes `llama-server`; and
- failed evidence is retained rather than overwritten by a later result.

Phase 1 makes this behaviour uniform for the managed llama.cpp installation. Other managed and
external Runtime implementations do not yet have an equivalent installation inventory, and
qualification currency is not yet resolved against each affected Worker and capability.

## 3. Terminology

- **Runtime Implementation**: a code-owned launcher and protocol adapter, such as
  `llama-vulkan` or `transformers-rocm`.
- **Runtime Installation**: the concrete executable, interpreter environment, libraries and
  build receipt detected on this machine.
- **Detected Identity**: observed immutable facts, including digests and version/build output.
- **Accepted Identity**: a code-owned identity that ModelDeck permits a Worker to launch.
- **Recommended Identity**: the accepted identity preferred for new Workers. There is exactly one
  per implementation and platform policy, but older accepted identities may coexist.
- **Upstream Identity**: an externally published revision. It is informational until imported into
  ModelDeck's trust policy and qualified.
- **Qualification**: evidence that a complete Configuration passed a defined workload and safety
  policy. Trust permits execution; qualification supports a usability claim.
- **Revocation**: a code-owned decision that a formerly accepted identity must no longer start,
  for example because of a security or data-corruption defect.

## 4. Status model

### 4.1 Installation integrity

| Status | Meaning | Start policy |
|---|---|---|
| `verified` | Required files exist, hashes match the receipt, and required features are present. | Continue to trust evaluation. |
| `missing` | The installation or a required component is absent. | Block affected Worker starts. |
| `receipt-missing` | Files exist without the required ModelDeck receipt. | Block managed trusted starts. |
| `receipt-invalid` | The receipt is malformed or internally inconsistent. | Block affected Worker starts. |
| `modified` | A detected file digest differs from its receipt. | Block affected Worker starts. |
| `feature-mismatch` | The binary lacks an allowlisted backend, argument or protocol feature. | Block only consumers requiring that feature. |
| `inspection-failed` | ModelDeck could not complete local inspection. | Fail closed for new starts; preserve the diagnostic reason. |

A receipt is evidence of how a local setup operation built an installation; it is not, by itself,
a signature or proof that arbitrary files are trustworthy. Code-owned accepted identities remain
the trust authority.

### 4.2 Trust currency

| Status | Meaning |
|---|---|
| `recommended` | The detected identity exactly matches the code-owned recommended identity. |
| `accepted-older` | The identity is still accepted but is no longer recommended for new Workers. |
| `accepted-alternative` | The identity is accepted for a distinct backend or compatibility path and is not ordered by age. |
| `different-unqualified` | The identity is readable but is not in the accepted set. |
| `newer-unqualified` | Local evidence proves it descends from the recommended source revision, but it is not accepted or qualified. |
| `revoked` | The identity is explicitly prohibited. |
| `unknown` | ModelDeck cannot establish a reliable relationship to an accepted identity. |

`newer-unqualified` must be used only when revision ordering can be established from trusted local
metadata. A version string that merely sorts higher is insufficient. When ancestry is unavailable,
the honest status is `different-unqualified`.

### 4.3 Qualification currency

Qualification is evaluated for each relevant Configuration and capability, not once for an entire
Runtime Installation:

| Status | Meaning |
|---|---|
| `qualified` | Matching tested-working evidence exists for the exact current fingerprint. |
| `limited` | Matching evidence exists but records explicit workload or capability limits. |
| `not-tested` | No matching evidence exists. |
| `stale` | Earlier evidence exists, but a material fingerprint component changed. |
| `failed` | Matching negative evidence exists for the exact fingerprint. |
| `legacy` | Older evidence is readable but cannot qualify a new or replacement Worker. |

Changing an executable digest, build features, backend libraries, driver, Runtime Template version,
Artifact, quantisation, context/KV-cache policy or relevant environment setting creates a new
fingerprint. It never updates old evidence in place.

### 4.4 Optional upstream advisory

An explicitly requested online check may report `newer-upstream-known`, `no-newer-upstream-known`,
or `not-checked`. This result must include its source and observation time. It must not change
trust, start policy, routing, qualification or the recommended identity.

The operator console should avoid a bare **Out of date** badge. A useful summary is instead:

> Accepted older Runtime · exact installation · qualification stale after driver change

## 5. Runtime installation identity

Every inspection result should contain common identity fields plus implementation-specific facts.
Local paths may be shown only on the loopback management surface and should not enter public gateway
responses or benchmark reports.

### Common fields

- Runtime implementation ID;
- installation ID and receipt schema version;
- operating system and architecture;
- backend and build-feature set;
- executable or environment digest;
- detected version/source revision;
- accepted and recommended identity references;
- inspection time and reason codes; and
- consumers: Runtime Template IDs and Worker IDs that depend on the installation.

### Native executable installations

For llama.cpp, inspect at least:

- source commit from the build receipt;
- `llama-server` SHA-256 and file size;
- receipt SHA-256;
- safe, bounded `--version` output;
- Vulkan build/backend identity;
- required arguments for each consumer, such as flash attention, reasoning policy, multimodal
  projector, MTP and offline operation; and
- platform and architecture.

Argument presence is compatibility evidence, not a substitute for behavioural qualification.
Inspection must never load a Model.

### Python environment installations

A later phase should use a setup-generated receipt containing the interpreter identity, lock or
requirements digest, selected package versions, native extension digests and backend metadata.
Hashing an entire virtual environment on every page load is neither necessary nor useful. Verify
the receipt and a bounded set of execution-critical files at management startup, on explicit
refresh and immediately before launch.

### External or unmanaged installations

An external vLLM endpoint may report a version and backend, but ModelDeck cannot infer file
integrity. It should be labelled `unmanaged` and `integrity-unverified`, with qualification tied to
the reported endpoint identity. It must not be presented as equivalent to a managed installation.

## 6. Proposed API and operator experience

Add a read-only `GET /api/runtime-installations` management endpoint. A result should resemble:

```json
{
  "implementation_id": "llama-vulkan",
  "display_name": "llama.cpp Vulkan",
  "integrity_status": "verified",
  "currency_status": "recommended",
  "detected": {
    "source_revision": "9d77fa17254e1dee4b9e92504c91611a60b1359f",
    "executable_sha256": "…",
    "backend": "Vulkan"
  },
  "recommended": {
    "source_revision": "9d77fa17254e1dee4b9e92504c91611a60b1359f",
    "policy_version": "1"
  },
  "required_features": ["flash-attn", "offline"],
  "consumers": {
    "runtime_template_ids": ["gpt-oss-llama-vulkan"],
    "worker_ids": []
  },
  "reason_codes": []
}
```

The exact schema should use bounded enums and omit absent or sensitive details. Human-readable
messages are derived from reason codes rather than used as policy inputs.

The operator console should add a Runtime installations section showing:

- detected and recommended identity;
- integrity and trust status;
- affected Workers and templates;
- whether matching qualification exists;
- last local inspection time; and
- a documented setup or requalification action.

Worker cards should show a concise runtime warning when their installation is missing, changed,
unaccepted, revoked or qualification-stale. The warning must distinguish the configured Runtime
Template from the resolved Runtime Installation.

## 7. Start, running and routing policy

- Inspect on management startup, explicit refresh and immediately before Worker launch.
- A missing, modified, revoked or unaccepted installation blocks a new start with a structured
  reason. ModelDeck must not search for or select another executable.
- An accepted older identity may start only while it remains explicitly accepted. New Workers may
  default to the recommended identity, but existing definitions are not rewritten.
- A newer unqualified identity is not an upgrade; it is unavailable to trusted Workers until added
  to policy and qualified.
- Discovery and status inspection never download, build, delete or replace files.
- A running Worker retains the identity resolved at launch. An advisory upstream result must not
  stop it. A security revocation requires a separately designed, explicit safety policy.
- Routing continues to use Worker readiness and published policy. Runtime currency must not cause a
  silent fallback or mid-request switch.

## 8. Upgrade and rollback workflow

Upgrade automation is deliberately outside the first phase. A future operator-controlled workflow
should:

1. acquire/build only through an explicit local administration action;
2. install the candidate side-by-side under a content-addressed or revision-addressed location;
3. write and verify a build receipt;
4. register a new accepted candidate identity without changing existing Workers;
5. create a replacement Worker with a distinct Runtime Template/version and fingerprint;
6. run direct-runtime, Worker and API-path qualification;
7. promote the candidate to recommended only after acceptance gates pass; and
8. retain the prior accepted installation and immutable evidence for rollback.

Replacing files in place before qualification would make evidence attribution ambiguous and is not
acceptable.

## 9. Security, privacy and safety

- Runtime inspection uses fixed code-owned commands and argument arrays. No web-supplied path,
  command, environment variable or argument is accepted.
- Version/build output is bounded and treated as untrusted text before display.
- Receipts and executable hashes are local identity evidence; secrets and environment contents are
  excluded.
- Public gateway responses expose only the serving Worker and safe resolved identity required by
  their protocol. Full installation inventory remains on the management surface.
- Inspection must not initialise a GPU backend, load weights, reserve substantial unified memory or
  bypass thermal admission.
- Online upstream checks, if later added, are explicit, separately permissioned and never required
  for local operation.

## 10. Delivery phases

### Phase 1 — llama.cpp inventory

- Define bounded installation, currency and reason-code models.
- Unify GPT-OSS and Qwen llama.cpp receipt/executable validation.
- Inspect without loading a Model.
- Add the read-only management API and console presentation.
- Include the resolved executable digest and receipt identity in every llama.cpp Worker fingerprint.
- Add missing, modified, exact, unaccepted and feature-mismatch tests.

### Phase 2 — qualification integration

- Resolve qualification status per affected Worker/capability.
- Explain the exact fingerprint differences that made evidence stale.
- Add side-by-side candidate identity support without an updater.
- Prove that existing Workers and Routing Profiles are never silently rewritten.

### Phase 3 — other managed runtimes

- Add receipt-based inspection for the ROCm Python environments and CPU translation runtime.
- Record native extension and backend package identity without hashing irrelevant environment files.
- Label external integrations as unmanaged rather than overstating integrity.

### Phase 4 — optional advisory and controlled upgrade

- Design an explicitly invoked upstream advisory check.
- Design side-by-side acquisition/build, qualification, promotion and rollback.
- Keep upstream recency separate from trust and qualification in both storage and UI.

## 11. Acceptance criteria

The complete multi-phase proposal is implemented only when tests demonstrate that:

- exact receipt and executable identities are reported without loading a Model;
- a modified binary, missing receipt and missing feature are distinguishable;
- GPT-OSS and Qwen consumers of the same llama.cpp installation see the same detected identity;
- an unknown or newer binary is never silently accepted;
- an accepted older binary is not incorrectly described as broken or latest;
- changing a Runtime Installation makes matching evidence stale without mutating historical records;
- Worker start fails closed with a structured reason and no Runtime substitution;
- running Workers are not interrupted by an advisory update result;
- offline operation remains complete; and
- operator actions, raw evidence and conclusions remain separately attributable.

## 12. Guiding-principle alignment

This proposal advances explicit execution identity, complete-Configuration evaluation, lifecycle
observability, reproducibility and raw-evidence preservation. It avoids favouring a Runtime,
equating recency with usability, making an unrecorded substitution, or presenting an upstream
version check as compatibility evidence.

The material trade-off is stricter startup behaviour when local Runtime identity cannot be proven.
That is preferable to executing an unknown binary under a trusted Runtime name. Hashing and feature
inspection add bounded management-startup cost; caching inspection results by file identity can
reduce repetition without weakening the mandatory pre-launch check.

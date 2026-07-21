# Detector-governed vision analysis for SceneChat

**Status:** architecture proposal; no production implementation  
**Recommendation:** proceed with a development-only prototype, subject to processor probes and
quality/performance gates  
**Reviewed:** 22 July 2026

## Executive recommendation

Proceed, but revise the proposed integration around a separate
`object-enrichment-v1` protocol contract and a separate public Route name,
`scenechat-objects`. Preserve `scenechat-vision` as the existing full-scene
`scene-analysis-v1` Route. Both Routes may initially target one loaded Gemma 4 Worker so the
model is not duplicated in memory, but the gateway must convey the code-owned Route contract to
the Worker and the Worker must select a code-owned policy for that contract.

Use one SceneChat-built contact sheet containing at most three numbered crops. Do not add
multi-image input or one-request-per-crop to the production contract in the prototype. Test both
as benchmark-only arms. SceneChat should own detector gating, crop construction and freshness;
ModelDeck should own accepted metadata, prompt construction, per-contract visual and completion
budgets, inference and response validation.

This design is feasible without weakening the local gateway boundary. It is not yet proven to be
faster or cooler. In particular, the current Gemma 4 processor accepts one image through the
ModelDeck adapter and applies an allowlisted `max_soft_tokens` value, but current physical results
report no actual visual-token count. Contact-sheet token behaviour, crop enlargement and
multi-image support therefore remain measurement gates.

## 1. Existing request path

```text
SceneChat camera thread
  -> latest JPEG and latest detector results retained in memory
  -> AnalysisService one-request lock and state-generation snapshot
  -> ModelDeckProvider prepares an optional downscaled JPEG copy
  -> POST 127.0.0.1:8600/v1/vision/analyse
       model = scenechat-vision
       one image data URL + one exact, versioned prompt
  -> gateway selects the first ready local Worker on the published Route
  -> gateway rewrites the public alias to the Worker's exact model identity
     and injects the private loopback credential
  -> POST Worker /v1/chat/completions
  -> SceneChat Worker validates the exact request shape and curated prompt
  -> one in-memory RGB image is passed to the pinned Gemma 4 processor
  -> deterministic generation with a Worker-wide visual-token budget
  -> ModelDeck validates and canonicalises scene-analysis-v1 JSON
  -> gateway returns the OpenAI-compatible envelope
  -> SceneChat validates the JSON again and adds trusted operational metadata
  -> AnalysisService applies only a non-stale result
```

Current enforcement is strong but contract-specific:

- A published ModelDeck Route has exactly one `protocol_contract`. The gateway's dedicated vision
  surface currently admits only `scene-analysis-v1` Routes.
- `/v1/vision/analyse` is an adapter to the Worker's `/v1/chat/completions`; the public request does
  not select a contract independently of its Route.
- The Worker accepts exactly one user message with exactly two content parts: one JPEG/PNG data URL
  and one text prompt. Extra images, arbitrary prompts, external URLs, SVG and streaming are
  rejected.
- The prompt must exactly match a code-owned SceneChat prompt. The Worker extracts only the
  approved question and reconstructs the system/user messages.
- The request is bounded to 12 MiB, the decoded image to 8 MiB, each edge to 4,096 pixels and the
  decoded image to 16 million pixels.
- The Worker permits one active request, rejects overlap with `worker_busy`, polls for disconnects,
  supports cancellation and has a generation deadline.
- The response is parsed as JSON, validated with Pydantic, canonicalised and screened for prohibited
  identity and sensitive-attribute assertions before it is returned.
- SceneChat independently permits one analysis request and rejects stale success and failure after
  reset or provider change. Provider failure retains the prior valid description and degrades a
  live ModelDeck session to detector-only operation.
- Frames remain in camera/request memory. Neither repository's live path persists them.

### Current capability and diagnostic behaviour

`/v1/models` publishes alias readiness and `/v1/capabilities` publishes generic Worker capabilities.
It does not currently expose the Route's protocol contract, maximum image/crop count, contract
budgets or response limit. `/v1/routes` similarly publishes only the name and readiness. This is
adequate for one SceneChat contract but too ambiguous for two vision behaviours.

The Worker records encoded dimensions/bytes, decode time, queue time, preprocessing time, inference
time, validation time, completion tokens, memory and total Worker latency. It attempts to read
`num_soft_tokens_per_image`; the recorded Gemma 4 probe results contain `visual_tokens: null`, so
actual visual-token accounting is not presently confirmed. There is no cross-request KV/prefix
cache: `use_cache=True` applies only during one generation. No reusable image or prompt state is
retained between requests.

## 2. Affected areas

### ModelDeck

| Area | Current file or symbol | Likely change after review |
|---|---|---|
| Contract catalogue | `backend/modeldeck/protocol_contracts.py` | Add `object-enrichment-v1`, surface and required capability metadata. |
| Gateway routing | `backend/modeldeck/gateway/app.py` | Admit the new contract on a dedicated endpoint, inject trusted contract identity, enrich Route/capability diagnostics and preserve alias rewriting. |
| Request/response models | `backend/modeldeck/contracts/scenechat/` | Add bounded enrichment request metadata, system prompt, response schema and canonical validation. Keep scene analysis separate. |
| Gemma adapter | `backend/modeldeck/workers/scenechat_worker.py` | Dispatch by trusted contract, apply a contract policy, build prompts from validated metadata, enforce response/input correspondence and expose missing metrics. |
| Processor settings | `backend/modeldeck/gemma4_settings.py` | Define code-owned per-contract budget choices only if processor probes show safe switching. |
| Runtime trust/launch | `backend/modeldeck/runtime_trust.py`, `backend/modeldeck/supervisor/service.py`, `backend/modeldeck/registry_data/runtime_templates.json` | Publish contract capability/policy and preserve immutable launch configuration. Avoid loading a second model solely to obtain another contract. |
| Event validation/smoke | `backend/modeldeck/domain.py`, `backend/modeldeck/v2_api.py` | Validate and smoke the new Route explicitly. |
| Mocks | `backend/modeldeck/mock_templates.py`, `backend/modeldeck/workers/mock_worker.py` | Add deterministic enrichment success, delay and request-error behaviours. |
| Benchmarking | `scripts/benchmark_scenechat_visual_tokens.py` and a new PowerShell entry point | Extend or add a strategy benchmark with fixed inputs, resource sampling and thermal condition control. |
| Telemetry/thermal | `backend/modeldeck/hardware/probe.py`, `backend/modeldeck/thermal.py` | Reuse safe sensor discovery; benchmark-only resource sampling needs package power and utilisation sources. Do not change unrelated speech thresholds implicitly. |
| Documentation/tests | `docs/API_CONTRACT.md`, `docs/WORKER_PROTOCOL.md`, `docs/BENCHMARKS.md`, contract/integration/unit tests | Document and verify the added development contract and disabled-by-default publication. |

### SceneChat

These are cross-repository changes for a later SceneChat stage, not changes in this proposal:

| Area | Current file or symbol | Likely change after ModelDeck prototype |
|---|---|---|
| Configuration | `backend/scenechat/config.py` | Add a disabled-by-default input strategy and separate allowlisted Route alias. |
| Atomic capture snapshot | `backend/scenechat/services/camera.py` | Return the JPEG and detections from one lock acquisition, with a frame sequence/timestamp. |
| Analysis orchestration | `backend/scenechat/services/analysis.py` | Distinguish scene and enrichment operations while retaining a single shared VLM concurrency lock and generation-based stale rejection. |
| ModelDeck provider | `backend/scenechat/vision/modeldeck.py` | Add enrichment readiness/request parsing; retain the existing full-scene method unchanged. |
| Schemas/state | `backend/scenechat/models/schemas.py`, `backend/scenechat/services/state.py` | Add bounded enrichment state, source-frame identity, expiry and trusted merge rules. |
| New detector policy module | `backend/scenechat/` | Own novelty/change/confidence/size/cooldown selection and in-memory crop/contact-sheet assembly. |
| UI/replay/tests | frontend, replay fixtures and tests | Present detector facts separately from enrichment, exercise stale/reset/privacy/fallback paths and avoid implying whole-scene knowledge from crops. |

## 3. Recommended API and contract

### Route and surface

Keep:

```text
POST /v1/vision/analyse  model=scenechat-vision  contract=scene-analysis-v1
```

Add, development-only and unpublished by default:

```text
POST /v1/vision/enrich   model=scenechat-objects contract=object-enrichment-v1
```

Use separate aliases because a Route already represents one trusted protocol contract. This makes
readiness, capability mismatch, smoke tests, logs and compatibility evidence attributable to the
correct behaviour. Overloading `scenechat-vision` would hide which contract is available and would
require a caller-controlled contract discriminator inside an otherwise valid Route.

Both Routes may reference the same physical Worker. The gateway should derive the contract from the
published Route and send it to the loopback Worker in a ModelDeck-owned header; it must discard any
caller-supplied copy of that header. The Worker must reject a missing, unknown or endpoint-mismatched
contract. A dedicated internal Worker endpoint is also acceptable, but a second loaded model is not.

### Public enrichment request

Prefer a dedicated, non-OpenAI request shape without caller-controlled sampling or processor fields:

```json
{
  "model": "scenechat-objects",
  "request_id": "caller-generated-safe-id",
  "image": {
    "data_url": "data:image/jpeg;base64,..."
  },
  "objects": [
    {
      "crop_id": 1,
      "detector_label": "person",
      "detector_confidence": 0.93,
      "cell": "top-left",
      "selection_reason": "new"
    }
  ]
}
```

Contract rules:

- exactly one JPEG or PNG contact sheet and one to three object records;
- request body and image limits no larger than the existing vision limits, with a prototype
  contact-sheet maximum edge of 768 pixels;
- integer `crop_id` from 1 to 3, unique and contiguous for the prototype;
- finite confidence from 0 to 1;
- `cell` and `selection_reason` are enums, not text fragments;
- detector labels must resolve through a ModelDeck-owned generic-object allowlist. Reject an unknown
  label or substitute the code-owned token `unknown object`; never interpolate arbitrary detector
  text into the prompt;
- no coordinates, paths, URLs other than the data URL, prompt fragments, environment settings,
  routing instructions, tool instructions, generation settings or processor settings;
- SceneChat renders the same numeric crop IDs visibly in the contact sheet. ModelDeck treats pixels,
  including rendered text, as untrusted image content;
- ModelDeck reconstructs the complete prompt from the schema and resolved labels;
- one active VLM request across both contracts. Separate Route names must not create overlapping
  inference on a shared Worker.

The first code-owned prompt should say that each numbered cell is independent, the detector label is
only a hint, only visible details may be described, uncertainty must be stated, and whole-scene
relationships, identity and sensitive attributes are prohibited.

### Response

Return a direct contract-shaped envelope rather than embedding JSON in a chat completion:

```json
{
  "contract": "object-enrichment-v1",
  "items": [
    {
      "crop_id": 1,
      "detector_label": "person",
      "description": "A person appears to be holding a blue ceramic mug.",
      "uncertainty": "The handheld object may instead be a small bottle."
    }
  ],
  "usage": {
    "prompt_tokens": 210,
    "visual_tokens": 70,
    "completion_tokens": 52
  }
}
```

`detector_label` should be copied from validated request metadata by ModelDeck, not trusted from model
text. The model-generated schema needs only `crop_id`, `description` and nullable `uncertainty`.
Validation must require exactly one item for each submitted ID, no duplicates or additions, bounded
strings, no extra fields, and the existing identity/sensitive-attribute screening. Model output must
never contain routing, detector-control or tool fields.

Start the development contract with a hard 256 completion-token ceiling and benchmark 128, 192 and
256. SceneChat should not submit `max_tokens`; promote the smallest code-owned limit that meets the
schema and quality gates.

### Contract policies and capability publication

Treat visual budgets as code-owned contract policy, not request input. Probe 70, 140 and 280 for
contact sheets and use the existing scene policy as the full-frame baseline. If the processor can
safely change `max_soft_tokens` between serialised requests, keep one loaded Worker and set it inside
the Worker's request lock. Otherwise, revise the runtime adapter so each contract's processing is
explicit without duplicating model weights; do not silently ignore the requested policy.

Add explicit Route metadata to gateway diagnostics, for example:

```json
{
  "scenechat-objects": {
    "ready": true,
    "protocol_contract": "object-enrichment-v1",
    "image_input": true,
    "structured_output": true,
    "maximum_images": 1,
    "maximum_crops": 3,
    "visual_token_budgets": [70, 140, 280],
    "maximum_completion_tokens": 256
  }
}
```

Keep generic capabilities backward-compatible; expose contract metadata additively through an
extended `/v1/routes` or a versioned contract-capabilities endpoint. Readiness must reflect Worker
health plus support for the particular contract, not merely generic `image_input`.

## 4. Responsibility split

| ModelDeck | SceneChat |
|---|---|
| Define and publish both trusted contracts and Routes. | Capture the camera and maintain an atomic frame/detection snapshot. |
| Validate metadata and resolve detector labels through a code-owned allowlist. | Decide whether a question needs enrichment or full-scene analysis. |
| Construct canonical prompts and treat all image pixels as untrusted. | Apply confidence, size, novelty, relevance, safety and cooldown policy. |
| Own model identity/revision, processor settings, token budgets and completion limits. | Crop with bounded padding and build/encode the numbered contact sheet in memory. |
| Serialise shared-Worker inference and enforce cancellation/deadlines. | Retain source-frame sequence/time and reject stale enrichment. |
| Validate/canonicalise output and attach trusted usage/diagnostic data. | Merge detector facts with enrichment for presentation without fabricating scene relationships. |
| Record content-free operational and compatibility evidence. | Preserve detector-only, replay, mock, privacy and reset behaviour. |
| Provide guarded physical benchmark integration. | Provide fixed saved/replay inputs and human-reviewed quality annotations. |

ModelDeck should not implement camera capture, tracking or detector gating. SceneChat should not load
the visual processor, choose arbitrary budgets, address Worker ports or construct the model's system
prompt.

## 5. Security and privacy

- Preserve loopback-only binding, private Worker credentials, pinned local snapshots,
  `local_files_only=True`, `trust_remote_code=False` and no cloud fallback.
- Infer the protocol contract from the published Route. Never trust a browser-provided contract
  header, Worker identity, model path or processor option.
- Allowlist detector labels before prompt construction. Length and character restrictions alone do
  not prevent prompt injection.
- Treat crop numbers and all visible contact-sheet text as untrusted pixels. Trusted crop IDs and
  labels come from validated metadata only.
- Keep contact sheets, crops and full frames in memory and close/release decoded images promptly.
  Do not place image bytes, prompts, descriptions, data URLs or local paths in logs or benchmark
  reports.
- Bind enrichment to a caller-generated request ID plus SceneChat frame sequence/generation. The
  result should carry trusted correlation metadata outside model output, and SceneChat must reject
  results after reset, privacy activation, provider/strategy change or a newer source frame policy.
- Do not use tracking IDs as persistent visitor identity. If used for short-lived novelty, keep them
  session-local, resettable and out of prompts/logs.
- Reuse the existing prohibited identity and sensitive-attribute filters for enrichment and add
  tests tailored to action/attribute descriptions.
- Retain the shared one-request limit. A contact-sheet Route must not bypass the full-scene lock.
- On enrichment failure, keep camera/detector operation and prior unexpired enrichment; do not
  automatically reroute to a different backend or invoke a full-frame request. Full-scene analysis
  remains a separately requested operation when its Route is healthy.

## 6. Expected effects and risks

Expected benefits, if measurement confirms them:

- detector gating eliminates VLM work on unchanged frames, which is likely the largest reduction;
- one contact sheet amortises prompt and visual prefill across up to three selected objects;
- a smaller enrichment schema can reduce decode time and completion tokens;
- bounded crops may preserve useful detail at a lower visual budget than a full frame;
- periodic downscaled scene analysis preserves relationships and safety context at a lower cadence.

Possible regressions:

- the processor may resize small crops or the entire contact sheet in a way that consumes the same
  visual budget or loses detail;
- contact-sheet gutters, numeric labels and multiple cells may reduce useful pixels or confuse crop
  correspondence;
- crops can remove context needed to distinguish an object or interpret an action;
- detector misses become VLM misses when no periodic full-scene refresh occurs;
- selection/cropping/JPEG work adds CPU load and shared-memory traffic;
- the smaller output ceiling can increase schema truncation;
- shared-Worker contract switching can introduce state leakage or race bugs unless performed inside
  the single request slot;
- two Route names may appear independently ready even though they contend for the same Worker;
- stale enrichment can be associated with a newer detection unless frame/detection capture is made
  atomic.

Existing evidence is directional only. A 20 July Gemma 4 E2B probe on the prepared 1,280 x 720 image
returned valid responses at budgets 70, 140 and 280 in about 17.6, 24.9 and 27.8 seconds respectively,
but completion length also changed and visual-token count was unavailable. This does not establish
contact-sheet performance. Existing Qwen visual-budget runs also show that repeated vision workloads
can be CPU-package limited, reinforcing the need for guarded pacing.

## 7. Benchmark plan and matrix

### Feasibility probes before the API prototype

Using the pinned Gemma 4 snapshot and exact detected Transformers/Torch/ROCm fingerprint, record:

1. processor output keys and tensor shapes for square, portrait and landscape images at several
   sizes, including small crops;
2. actual visual-token count or a verified derivation when `num_soft_tokens_per_image` is absent;
3. whether input is enlarged, padded, tiled or capped at each 70/140/280 budget;
4. one contact sheet versus two and three `images` in one processor call, including chat-template
   placeholder requirements;
5. whether changing the code-owned soft-token budget between serial requests is deterministic and
   state-safe;
6. whether any reusable prefix API exists in the pinned stack. Current ModelDeck has none, so this is
   not an initial optimisation target.

Do not expose multiple-image input unless the complete path is supported and it beats contact sheets.

### Dataset

Use a fixed, local, non-visitor corpus with stored expected detections and human annotations. Include
at least 12 frames covering: no meaningful change, one large object, three objects, a small object,
ambiguous handheld items, screen/text content, person activity, occlusion, similar-looking objects,
wide/portrait layouts, safety-relevant context and a detector miss. Derive every crop/contact sheet
from the same decoded source frame and record a source hash, never image bytes, in results.

### Experimental factors

| Factor | Levels |
|---|---|
| Strategy | A native full frame; B downscaled full frame; C separate serial crop requests; D one contact sheet; E simulated hybrid schedule |
| Visual budget | 70, 140, 280; one higher full-scene quality baseline only if required |
| Selected crops | 1, 2, 3 |
| Contact-sheet maximum edge | 512, 768 |
| Full-frame maximum edge | native, 768, 512 |
| Completion ceiling | enrichment 128, 192, 256; scene 512 baseline |
| Runtime state | cold model start; warm Worker |

Run the full factorial only for feasibility. Prune dominated/invalid settings before repeated physical
runs. Strategy C must remain a benchmark arm and run serially; it is not a proposed default.
Strategy E should replay a timestamped detector/event trace so its invocation rate and average work
per useful update are comparable.

For each retained condition use one discarded warm-up and five measured warm requests (three only for
an early probe), balanced in order. Cold start is a separately labelled measurement including Worker
load and warm-up, not mixed into warm request latency. Never overlap VLM requests.

### Measurements

Record per request or condition:

- model ID/revision, runtime/backend, package versions, precision/quantisation and complete hardware
  fingerprint;
- strategy, source hash, input dimensions/bytes, number of images/crops and contact-sheet layout;
- configured budget, actual processor visual tokens, prompt tokens and completion tokens;
- image decode, crop/resize/contact-sheet/JPEG, Worker preprocessing and validation times;
- gateway selection/health time, Worker admission time, model-start time, time to first output byte,
  time to first generated token where instrumentable, inference and end-to-end latency;
- decode tokens per second based on generation time, with its basis stated;
- baseline/peak CPU utilisation, GPU utilisation, package power, CPU package temperature, GPU edge
  temperature and memory; identify the sensor source/label;
- schema validity, crop-ID coverage, correctness, ambiguity handling, missed objects, lost
  relationships and unsupported/hallucinated details.

The present Worker already supplies much of the timing and token envelope. Required instrumentation
gaps are Gemma visual-token accounting, first generated-token timing, CPU/GPU utilisation and package
power. If reliable counters are unavailable, mark the metric unavailable with the detected source;
do not estimate it from temperature.

### Evaluation and comparison

Blind human review should score each requested crop for identification/description correctness,
visible attribute/action usefulness, uncertainty calibration and unsupported claims. Full-frame arms
also score object recall, relationships and safety context. A hybrid result must not be penalised for
omitting relationships in enrichment; instead evaluate whether its periodic scene refresh supplies
them at the configured cadence.

Compare median and p95 warm latency, median time to first output, completion and actual visual tokens,
energy per valid update when power is available, peak temperatures, valid-response rate and quality.
Also report VLM invocations and aggregate inference seconds per minute of the replayed event trace.

## 8. Thermal-safety controls

Extend the existing guarded SceneChat benchmark rather than creating an uncoordinated loop:

- refuse to start if another managed Worker is busy or transitioning;
- run one physical VLM request at a time and one benchmark process via an exclusive lock;
- require live temperature sensors before physical work and fail closed if telemetry disappears;
- use 80 degrees C as a soft pacing threshold and 85 degrees C as the hard experimental ceiling;
- sample at least every 0.5 seconds before, during and after each request;
- on any sensor reaching 85 degrees C, cancel the request, stop the Worker if cancellation does not
  complete promptly, mark the condition `thermal_abort` and never resume that condition;
- record the trigger sensor, temperature, time, partial measurements and abort reason;
- cool below a configured start threshold before another condition. A run may continue with the next
  condition only after cooldown and only if the aborted condition remains skipped;
- retain the current stricter operator option to abort the entire run at 80 degrees C;
- stop and restore Worker lifecycle safely on normal completion, failure and interruption. Do not
  auto-restore a Worker after a thermal abort without a safe cooldown check.

Attribute heat using measured phase boundaries and sensor/counter traces: model load, SceneChat crop
and JPEG preprocessing, Worker image processing, GPU inference and cooldown. Report CPU, GPU or mixed
load only when utilisation/power evidence supports the label.

## 9. Staged implementation plan

### Stage 0 — processor and contract feasibility

- Add no production routes.
- Run offline processor-only probes, then the smallest guarded physical probe needed to confirm token
  behaviour, contact-sheet quality and serial policy switching.
- Finalise the label allowlist, response fields and metric sources.

**Exit:** one-image contact sheets are accepted; actual visual-token accounting is understood; no
unsafe enlargement/state behaviour is found; the selected routing design needs no second model copy.

### Stage 1 — ModelDeck development contract

- Add `object-enrichment-v1`, strict models/prompt/output validation and deterministic mocks.
- Add the dedicated gateway surface and `scenechat-objects` Route support, unpublished by default.
- Dispatch using trusted Route contract identity and serialised per-contract policy.
- Add capability/readiness/metrics and contract/worker/gateway tests.

**Exit:** malformed metadata, labels, IDs, images, prompts, headers, budgets and responses are rejected;
scene analysis is unchanged; offline verification passes.

### Stage 2 — guarded strategy benchmark

- Add fixed-frame/contact-sheet fixtures and the A-E strategy runner.
- Fill the instrumentation gaps and implement condition-level 85-degree-C abort.
- Run processor probes, then three-request screening arms, then five-request retained arms.

**Exit:** a reproducible report identifies whether contact sheets improve latency/work without an
unacceptable quality or thermal regression.

### Stage 3 — disabled SceneChat integration

- Add `VISION_INPUT_STRATEGY=full-frame|detector-contact-sheet|hybrid`, defaulting to `full-frame`.
- Add atomic frame/detection snapshots, in-memory contact-sheet construction, separate Route health,
  result expiry and deterministic fixtures.
- Do not enable live detector gating by default.

**Exit:** all existing full-frame behaviour and fallback modes pass; feature-on integration works
against ModelDeck mock and development Routes.

### Stage 4 — event gating

- Implement and replay-test confidence, size, novelty/change, relevance, safety, cooldown and periodic
  full-scene refresh policies in SceneChat.
- Start with at most three crops, 10-20% padding, minimum 64-pixel source crop and a 20-30 second
  scene refresh as experiment values, not defaults.

**Exit:** stable detections do not repeatedly invoke the VLM; detector misses and refresh behaviour
are characterised; stale/reset/privacy behaviour is proven.

### Stage 5 — operational hardening

- Complete diagnostics, mock/replay, timeout, cancellation, busy, privacy, stale-result, thermal and
  compatibility tests in both repositories.
- Update runbooks and explicitly rehearse object-Route failure while full-scene operation remains
  available, and vice versa where the shared Worker is healthy.

### Stage 6 — promotion decision

Publish/enable the new Route and change SceneChat's default only after all acceptance gates pass.
Promotion must be an explicit Event/configuration change and must not occur automatically.

## 10. Acceptance criteria

### Contract and safety

- `object-enrichment-v1` is separate from `scene-analysis-v1`; full-scene analysis remains unchanged.
- The development Route is disabled/unpublished by default.
- Exactly one contact-sheet image and one to three unique crop IDs are accepted.
- No arbitrary detector text, prompt text, path, Route, model, tool instruction or processor setting
  reaches the model.
- Output contains exactly the submitted crop IDs; trusted labels are copied by ModelDeck; the model
  cannot emit routing/control fields or whole-scene relationships.
- Existing identity/sensitive-attribute restrictions, loopback binding, local-only loading, private
  credentials and no-persistence guarantees remain intact.
- Shared-Worker requests cannot overlap across the two contracts.

### Correctness and resilience

- Contact-sheet results are schema-valid in 100% of the retained measured requests.
- Every submitted crop ID is represented exactly once in 100% of valid responses.
- On the reviewed fixture set, the selected contact-sheet condition is no worse than the downscaled
  full-frame baseline by more than an agreed five percentage points for requested-object correctness,
  and unsupported details do not increase materially. Final thresholds should be frozen before the
  confirmation run.
- Enrichment failure leaves camera, detector results and unexpired prior enrichment operational and
  does not trigger automatic backend/full-scene fallback.
- Reset, privacy activation and stale frame/generation/provider/strategy results cannot update UI
  state.
- Existing ModelDeck and SceneChat offline suites pass; physical tests remain explicitly marked.

### Performance and thermals

- Against native full-frame scene analysis, the selected contact-sheet condition improves median
  warm end-to-end latency and inference time by at least 20%, with no worse p95 failure rate. This is
  a proposed gate to review before the confirmation run.
- The simulated hybrid trace reduces aggregate VLM inference seconds per useful scene update by at
  least 30% and invokes no VLM for unchanged, recently enriched detections.
- Actual visual-token behaviour is reported, not inferred from pixel dimensions.
- No condition exceeds 85 degrees C. A triggered condition is aborted, recorded and not resumed.
- Peak temperature or measured energy per useful update improves; if reliable power data is
  unavailable, invocation and inference-time reduction plus temperature must be reported separately.

## 11. Open questions requiring measurement

1. Does the pinned Gemma 4 processor expose or permit a trustworthy derivation of actual visual
   tokens when `num_soft_tokens_per_image` is absent?
2. How does it resize/pad/tile small crops and 512/768-pixel contact sheets under 70/140/280 budgets?
3. Does multiple-image processing work with the pinned processor/model/chat template, and is it more
   efficient or accurate than one contact sheet?
4. Can `max_soft_tokens` be changed safely between serial contract requests without reconstructing
   the processor or contaminating state?
5. Which contact-sheet layout, gutter and numeric-label style produces reliable crop-ID alignment?
6. Is 128 or 192 completion tokens sufficient, or is 256 required for three useful items and valid
   JSON?
7. How much context is lost for actions, handheld objects, screens and safety observations?
8. Which detector thresholds and novelty metric minimise repeated enrichment without suppressing
   useful changes?
9. What refresh interval recovers relationships and detector misses at acceptable thermal cost?
10. Which local counters reliably provide CPU/GPU utilisation and package power on the target Fedora
    and ROCm stack?
11. Is CPU package heat dominated by SceneChat JPEG/contact-sheet construction, Worker processing,
    model loading, GPU inference/shared memory, or their combination?
12. Does sharing one Worker make independent Route readiness sufficiently clear, or should diagnostics
    additionally publish shared-resource contention?

## 12. Decision

**Proceed with revision.** The leading architecture is hybrid detector-gated contact sheets plus
periodic downscaled full-scene analysis. The current trust boundary and one-request behaviour provide
a solid base, and a separate Route/contract can preserve explicit operations without duplicating
model weights. Do not implement live gating or promote an alias until the Gemma processor probes,
guarded benchmark and quality review demonstrate a real benefit.

The immediate next action after review is Stage 0: add a non-production processor probe and freeze the
benchmark fixtures/annotations. Production code and Event publication should remain unchanged until
those results are reviewed.

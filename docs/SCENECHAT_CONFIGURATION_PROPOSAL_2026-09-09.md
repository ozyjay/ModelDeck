# SceneChat local inference proposal — 9 September 2026

Proposal only. Inspection used local files, SQLite in read-only mode, and GET requests to
the running management API, gateway and SceneChat Worker. No inference requests, downloads,
Worker lifecycle operations, routing changes or publication were performed. No visitor
image or generated visitor response was read or retained. Only this proposal was added.

Recommend **Gemma 4 E2B BF16 as the first trial**, reflecting the operator's preference and
its local SceneChat smoke evidence, followed by a controlled comparison with **Qwen3.5 4B
BF16, Qwen3.5 2B BF16 and the current 0.8B baseline**, using the existing dedicated
Transformers/ROCm SceneChat runtimes. The ranking is an evaluation priority, not a measured
winner. Reliability on varied scenes and total time to a valid, useful answer decide the
winner. Do not switch to the existing 0.2.3 Worker on its name alone.

## Verified live configuration and execution identity

Management `/api/routing-profiles` and gateway `/v1/routes` agree: active **Open2026 revision
37**, profile `d57a4e6c-1ee4-482b-bf4c-206e41507a8f`, exposes `scenechat-vision` under
`scene-analysis-v1`, with only Worker `3ad2f88d-8936-4ffc-ac63-6b5e6543d4ed`. No backup is
configured. Gateway cloud fallback is false. Worker was ready and idle during inspection.

| Fingerprint component | Observed value |
|---|---|
| Worker definition | `scenechat-qwen35`, template version `0.2.2`; bounded JSON candidate |
| Requested runtime | `qwen35-vision-language-transformers-rocm` |
| Health-reported runtime | `vision-language-transformers-rocm` — generic label, not evidence of a Gemma substitution |
| Actual executable/module | `.venv-rocm72/bin/python -m modeldeck.workers.qwen35_worker`; PID 109098, loopback port 8673 |
| Python / source | pyenv Python 3.12.13; editable ModelDeck installation from this checkout; Git `d0be94d3899903741084e43717016843a572bc30`; clean before this document |
| Model and processor | `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`; `Qwen3_5ForConditionalGeneration`, `Qwen3VLProcessor` |
| Artifact | One safetensors shard, 1.627 GiB; unquantised checkpoint |
| Execution | BF16 parameters on `cuda:0`, AMD Radeon 8060S Graphics, ROCm/HIP; three approved FP32 buffers |
| Libraries | Torch `2.9.1+rocm7.2.1.gitff65f5bc`; HIP `7.2.53211-e1a6bc5663`; Transformers `5.13.0` |
| Host | Fedora 44; kernel `7.1.13-200.fc44.x86_64`; MemTotal 131,150,200 KiB, approximately 125.1 GiB |
| Attention | Requested `sdpa`; precise dispatched attention kernel is not exposed |
| Image policy | Visual-token ceiling 140; patch size 16, spatial merge 2; failed request produced 120 visual tokens |
| Context/output | Configured context 8,192; maximum completion 1,024; Worker timeout 60 seconds |
| Generation | Thinking disabled, greedy (`do_sample=False`), `use_cache=True`; cancellation and complete-JSON stopping criteria |
| Cache | Prefix cache disabled; installed model code defaults to `DynamicCache`; actual cache class/dtype was not reported by the running process |
| Offline | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; `local_files_only=True`, `trust_remote_code=False` |
| Application | SceneChat checkout `002312e7d8a66e1a926ca5039229661dba87d817`; local configuration specifies 60-second timeout and 1,024 tokens |

The `bf16_dequant` execution-mode string is also reported, but checkpoint quantisation is
`none`: this model is not an FP8 checkpoint being dequantised. SceneChat sends temperature
0.1; the dedicated engine actually uses greedy decoding. Record both requested and resolved
settings in future evidence.

The 0.2.3 definition (`5bd5b96c-fc52-4995-a4c8-d9835a163b23`, stopped, port 8674) has the
same model revision, BF16, 140-token visual budget, 8,192 context, 1,024 output ceiling and
60-second timeout. Installed source supplies implementation for both definitions. Current
source predates the active process start and already includes the concise internal prompt
and complete-JSON stopper. The live metrics corroborate the generation settings. An old
template version does not pin an old implementation.

The gateway configuration digest is
`00d2b675408971b14ae5d69a5da2e18dabc3ac4872167b31a06c66de387cd3af`.
Its runtime fingerprint is null, with several resolved identity fields absent. Therefore
the API does **not** establish a complete execution fingerprint on its own. This inspection
adds the following SHA-256 identities of current disk contents:

| File | SHA-256 |
|---|---|
| Model weight shard | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| Model config | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| Tokenizer JSON | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| Tokenizer config | `49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c` |
| Chat template | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| Image processor config | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| Qwen Worker source | `4748bbf40f34e474d57f194748d6652e6a8a179247f53dc34aeda68710af4cb0` |
| Shared SceneChat Worker source | `7609f692df895025ff9a4414b34144e55be7b7f0d7ec23ebb67aeb2037f6a7be` |
| Contract implementation | `af527d003ac8d4f5da0d9f23a0a36a6e6b2314385c88b9adca521a4db5731182` |
| System prompt file | `44ee1d4244932d348c2b58dd9ed3ad25d8cb113d9acd0ef6b82583f70967efa0` |
| JSON schema | `8d15122add44b647087def8b1f39d9b6237d4201864d85b2d86d39100010cc64` |

These hashes are not a retrospective attestation of loaded memory or July runs. A complete
future run receipt must additionally capture the rendered prompt, complete dependency/build
receipt, effective EOS/PAD IDs, cache class/dtype/state policy, resolved attention kernels,
power/firmware configuration, image transform and environment. The snapshot text config
contains EOS 248044, but effective generation EOS needs instrumentation. Do not invent
missing historical identity or stage timings.

## What the failure establishes

Worker log line 185 records request `bb804137-1a48-4042-a7ea-46ad594f2996` at
`2026-09-09T12:37:49.105940+00:00` (22:37:49 AEST). This is the validation-failure timestamp,
not necessarily request dispatch time. Completion count and effective limit were both 1,024;
elapsed time was 39.6982 seconds. Current `/metrics.last_request` corroborates:

| Measurement | Value |
|---|---:|
| Image | 1280 × 720, 88,771 bytes |
| Prompt / visual tokens | 565 / 120 |
| Image decode | 0.002915 s |
| Processor preprocessing | 0.011881 s |
| Inference | 39.633897 s |
| Validation | 0.000041 s |
| Total Worker time | 39.698364 s |
| Completion / finish | 1,024 tokens / length |
| Retry count | 0 |

Inference accounts for approximately 99.8% of Worker elapsed time. Neither image decoding
nor request queueing explains this delay. The metrics show no timed-out request. The small
memory footprint and absence of a recorded load/OOM error argue against model loading or
capacity as this failure's immediate cause. They do not prove thermal safety.

Crucially, `_output_failure_category` overwrites the original validator category with
`token_limit_reached` whenever the ceiling is reached. Consequently the evidence proves
**ceiling exhaustion plus contract rejection**, not whether the underlying problem was
unfinished JSON, a schema violation, a safety violation, repeated text, or failure to stop.
Raw visitor output was not retained and should not be reconstructed.

The current prompt asks for five top-level fields and potentially eight richly described
objects. An internal constraint narrows this to three objects, one relationship and one
uncertainty. This remains a demanding instruction-following task for 0.8B. A likely
explanation is a combination of capability, response planning and unconstrained generation;
it remains a hypothesis. Complete-JSON detection is a stopping criterion, **not** a grammar
that prevents invalid tokens. It decodes the accumulated output after each token, creating
another possible overhead to measure. Gemma currently has cancellation stopping but no
equivalent complete-JSON stopper: this difference must be explicit in comparisons.

Startup logs also warn that experimental AMD efficient attention is unavailable and that
kernel caching is disabled without a configured cache path. These warrant a measured runtime
investigation, not an assumption that enabling an experimental flag fixes the request.

The 0.2.3 log contains **25** ceiling failures on 24 July, approximately 29.9–32.9 seconds.
Its successful compatibility smoke is not a cure for those failures.

## Local comparison and evidence strength

All listed safetensors candidates have their referenced weight shards present; processor,
tokenizer and configuration files were inspected. This is an inventory/completeness check,
not loading or comprehensive weight-integrity validation. Sizes below are weight files,
not predicted resident memory. No configuration is qualified for the proposed varied-scene
workload.

| Priority | Configuration | Local evidence | Decision |
|---|---|---|---|
| 1 | Gemma 4 E2B BF16, dedicated SceneChat ROCm | 9.543 GiB weights; six historical SceneChat smoke records; about 10.26 GB steady / 11.04 GB peak allocation in July; newer general-image-chat smoke | Preferred first trial; strongest alternative-family feasibility evidence; E2B does not imply a 4 GB footprint |
| 2 | Qwen3.5 4B BF16, dedicated SceneChat ROCm | 8.680 GiB weights; allowlisted architecture; successful text-chat compatibility records, no retained matched SceneChat qualification | Leading Qwen capability hypothesis with little integration work; latency unknown |
| 3 | Qwen3.5 2B BF16, dedicated SceneChat ROCm | 4.236 GiB weights; text-chat smoke; three July SceneChat runs each aborted at 80–81.375°C, with two valid measured responses before abort and roughly 22.3 s latency | Mandatory matched arm, but thermal and latency concerns are real |
| Baseline | Qwen3.5 0.8B BF16, current settings | 1.627 GiB weights; live failure; July 70/70 synthetic-image result at median 8.7644 s / p95 10.0732 s, 300–358 tokens | Retain as control, not presumed acceptable |
| Reserve | Qwen3.5 9B BF16 | 17.980 GiB weights; allowlisted; text-chat compatibility succeeds | Add if 4B's factual accuracy fails; fewer generated tokens could offset slower decoding |
| Reserve | Gemma 4 12B BF16 | 22.277 GiB weights; July 256 × 256 SceneChat requests around 15.67 s, 82 output tokens; September 70-visual-token scene-analysis qualification has two failures | Useful quality reference; neither old successes nor newer failures transfer automatically to E2B |
| Reserve | SmolVLM2 2.2B / 500M / 256M | 8.370 / 1.891 / 0.956 GiB FP32 weight files; ONNX variants also cached for smaller models; Transformers architecture exists, but no ModelDeck allowlisted runtime or compatibility records found | 2.2B is the useful optional control; tiny models only if a small-footprint advantage is needed and full contract quality survives |
| Defer | Qwen3.8 27B FP8 | 28.747 GiB referenced weights; dedicated native-FP8 runtime code exists; no matching compatibility record found in this database | Native FP8 kernel qualification and full-placement evidence required; BF16 dequantisation is a separate, larger-memory arm |
| Blocked as vision arms | Qwen3.5 4B / 9B Q8 GGUF | Weight files present, no accompanying projectors in either snapshot; current adapters advertise no image input or structured output | Text-chat success does not establish vision support |
| Reserve runtime arm | Qwen3.8 27B Q8 GGUF + BF16 projector | Additional cached snapshot includes model, projector and MTP artifact; Vulkan runtime and general-image-chat smoke records exist | Most concrete already-cached cross-runtime option, but requires a SceneChat contract adapter and workload qualification |

Pinned primary-arm revisions: 2B `15852e8c16360a2fea060d615a32b45270f8a8fc`, 4B
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, E2B
`9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`; baseline revision appears above.

The July 0.8B 70/70 report had no human review, missed its median ≤8-second and preferred
output-length gates, and had no burn-in. Another July run had 60/70 valid responses and ten
token-limit failures. An earlier partial manual review accepted only one of three reviewed
responses in each of two visual-budget arms. None supports a claim of qualified visual
accuracy. Historical latency summaries may exclude failed samples; new reporting must not
hide them.

## Runtime alternatives and focused upstream research

Start with the installed BF16/ROCm stack, without quantisation or attention changes. Compare
total latency to a valid answer, not model parameter count or tokens/second alone. Reduced
output length can make a larger model faster overall; quantisation can add conversion,
kernel or memory-access overhead. Treat every runtime/backend/precision combination as a
separate configuration.

After the matched model comparison, test one factor at a time: complete-JSON stopping,
schema-constrained decoding, visual budget 140 versus 280, then an explicitly versioned
attention/kernel option if profiling warrants it. Grammar enforcement must retain the
existing validator and human accuracy review; it cannot ensure truth, safe language or
question relevance. Do not increase the output ceiling or add retries as the initial fix.

For GGUF, require the exact matching vision projector, model/projector hashes, processor and
chat template, proven image offload, context/KV settings, JSON-schema grammar support,
token accounting and cancellation. The current Qwen3.5 Q8 adapters need integration work.
The Qwen3.8 Vulkan manifest already pins these artifacts and a llama.cpp build; its general
chat capability still does not implement SceneChat's full trust boundary. Compare its MTP
mode separately if used. HIP llama.cpp is a separate build/backend qualification, not an
automatic replacement for the existing Vulkan receipt.
[llama.cpp multimodal documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
and [grammar documentation](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
describe the upstream facilities, not measured SceneChat performance.

vLLM offers schema-constrained generation with xgrammar or guidance. That is a concrete
reason to consider it only if syntax/stopping failures remain and a compatible gfx1151
build can be established. With one request in flight, batching claims do not justify its
integration cost. Model, vision, structured decoding, memory, offline operation and AMD
kernel support all need proof; do not use the unmanaged port-8000 service as implicit
fallback. [vLLM structured-output documentation](https://docs.vllm.ai/en/latest/features/structured_outputs/).

Only three upstream alternatives merit further attention:

- **Qwen3-VL-4B-Instruct:** a non-Thinking variant with upstream claims of improved spatial
  perception and OCR. That matches SceneChat's nearest-object and equipment questions.
  Installed Transformers includes its architecture, but ModelDeck does not allowlist it.
  Consider a separately approved acquisition only if local 4B/E2B fail perception or
  instruction-following. Published capabilities do not establish superiority to Qwen3.5
  or Radeon latency. [Official model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
- **Moondream 2:** short-caption and visual-query APIs, plus a published 20–40% generation
  improvement claim from its tokenizer changes, provide a concrete concise-output
  hypothesis. Its custom-code loader and lack of demonstrated SceneChat schema adherence
  require a pinned, reviewed adapter. Moondream 3 Preview is now the newer offering:
  9B total/2B active MoE, compiled FlexAttention, and reasoning enabled by default. It adds
  integration uncertainty rather than establishing an AMD speed advantage; assess licence
  terms before any later acquisition. [Moondream 2 card](https://huggingface.co/vikhyatk/moondream2),
  [Moondream 3 Preview card](https://huggingface.co/moondream/moondream3-preview).
- **Florence-2-base-ft:** 0.23B with explicit captioning tasks, potentially useful for an
  intentionally simpler caption-only experience. It is not a demonstrated replacement for
  seven curated questions plus SceneChat JSON. Keep this in the separate application-contract
  proposal. [Microsoft model card](https://huggingface.co/microsoft/Florence-2-base-ft).

No upstream performance claim above is a measurement on this Framework Desktop. No new
model is necessary for the first comparison. Any approved later acquisition belongs to
HuggingFacePull, followed by ModelDeck read-only discovery.

## Exact work proposed for approval

1. **Prepare the reproducible harness and receipts without publishing anything.** Extend
   `benchmark_scenechat_visual_tokens.ps1/.py` instead of duplicating it. It currently requires
   matching model/revision across visual-budget arms and a single booth image, so it needs
   a configuration matrix, corpus manifest and immutable per-run receipts. Retain its
   isolated test-gateway approach. Capture underlying validator category separately from
   ceiling exhaustion, actual stopping cause, repetition diagnostics, effective generation
   config, and stage timings. Record the complete fingerprint described above. Do not change
   validation. Test harness accounting, prompt/contract parity and error handling; run
   focused tests and `pwsh -NoProfile -File scripts/verify.ps1`. Use marked physical tests
   only in the later approved stages.
2. **Freeze the corpus and scoring before tuning.** Use 28 approved non-visitor 1280 × 720
   JPEGs, four in each category: sparse household/desk objects; cluttered equipment; workshop
   scenes with ambiguous purpose; indoor activity with non-identifying synthetic people;
   outdoor/street/garden scenes; occlusion/low-light/blur/reflections; visible text including
   image-borne prompt injection. Hash every image and record provenance and factual labels.
   Split two images per category into development and two into unseen holdout. Keep the
   existing synthetic booth image as a separate historical anchor. Do not use the failed
   visitor frame. Store raw generated text only in a separate, explicit benchmark-only
   evidence store for this non-visitor corpus; operational logs stay content-free.
3. **Run a small diagnostic screen in an approved maintenance window, Gemma E2B first.** Create new immutable
   evaluation definitions for the four primary arms; do not reuse general-chat Workers as
   SceneChat endpoints. Use BF16, 8,192 context, 1,024 maximum output, 60-second deadline,
   thinking off, greedy decoding, no retry, no prefix/image reuse and one in-flight request.
   Begin at 140 visual tokens for each family, recording actual resize/crops/tokens because
   equal token budgets are not equal visual fidelity across architectures. Run two excluded
   warm-ups and four development images × seven questions once (28 measured requests/arm).
   Record existing engine stopping differences. Inspect benchmark-only traces to distinguish
   repetition, EOS/template mistakes, incomplete structure, excess detail and factual errors.
4. **Run the matched development comparison.** For each viable arm, run all 14 development
   images × seven questions × three repetitions: 294 requests/arm, 1,176 for four arms.
   Counterbalance arm/block order with a fixed schedule and identical request order within
   each block. Use the same image bytes and external contract. Record prompt/template
   differences imposed by each family. Separate cold model load and first-request latency
   from warm requests. If tuning is needed, change one factor at a time on development data:
   shared complete-JSON stopping versus existing stopping; constrained decoding versus
   unconstrained; 140 versus 280 visual tokens. A sampled-decoding arm needs explicit seeds
   and its own repeated reliability results. Do not mix tuned and baseline samples.
5. **Validate at most two frozen finalists on holdout.** Run 14 unseen images × seven
   questions × five repetitions: 490 requests/finalist. No tuning on holdout. Review output
   blind to model identity against pre-labelled visible facts. Score all unique outputs;
   map exact repeats back to their reviewed output while retaining per-request validity.
   Use a second reviewer for a stratified 20% sample and all disputed or unsafe outputs.
   Report accuracy of asserted facts, coverage of question-relevant facts, unsupported-claim
   rate and severity, appropriate uncertainty and prohibited claims separately.
6. **Qualify the best passing configuration under demonstration load.** Run at least two
   hours and 210 requests with the intended detector and other resident workloads present,
   while preserving one in-flight SceneChat request. Measure queueing/cooldown as part of
   operator-visible availability. Exercise timeout, disconnect, reset and privacy holding
   during inference, stale-result rejection, recovery, offline restart, cancellation and
   unload. Then prepare the candidate, routing diff and rollback receipt for separate
   publication approval.

Every stage uses all seven questions exactly as curated: “What objects can you see?”,
“Describe the scene.”, “What is happening here?”, “Which objects are closest to the camera?”,
“Describe this scene for someone who cannot see it.”, “What might this equipment be used
for?”, and “What details might the system be uncertain about?”. Preserve the existing
model-side nearest-object wording override and record it in the prompt fingerprint.

Record per image/question/configuration: valid-response rate; factual accuracy and coverage;
unsupported claims; token-limit hits; output tokens, words and characters; stopping cause;
median/p95 end-to-end time; and failure duration. Report success latency and all-attempt
latency separately, counting every 502, timeout, cancellation and thermal abort. Report
binomial uncertainty and scene-level variability; repeated deterministic images do not
create hundreds of independent visual tests.

Measure image preparation/decode, preprocessing, vision encoding, text prefill/first token,
generation, output decoding/stopping, validation, Worker total, gateway overhead and
SceneChat-to-display time. Current metrics combine vision/prefill/generation into inference;
add diagnostic hooks with correctly synchronised GPU timing where feasible. Report
unavailable or fused timings as unavailable, and quantify instrumentation overhead. On a
small fixed subset, measure direct runtime, Worker and test-gateway paths separately.

## Proposed acceptance gates and safety

- 100% contract-valid responses, zero token-limit hits, timeouts or unhandled failures in
  holdout and sustained qualification. This is an observed release gate, not a claim of
  universal reliability.
- Zero prohibited identity/sensitive-attribute claims or followed image instructions;
  at least 95% of asserted factual claims supported, at least 90% question-relevant coverage,
  unsupported factual claims at most 2%, and no severe unsupported claim. Score cautious
  hypotheses separately; empty/minimal JSON cannot pass on syntax alone.
- Warm end-to-end median ≤8 seconds and p95 ≤12 seconds, including a per-question check.
  The unchanged 60-second timeout is a failure guard, not an acceptable demonstration target.
  Preferred per-question output p95 ≤260 tokens, all outputs within existing word/field
  limits and below the 1,024-token ceiling. Token lengths across tokenizers are diagnostic;
  visible concision is judged in words and characters too. Report misses, do not silently
  weaken gates to select a winner.
- Sample host/GPU temperature and memory at least every 0.5 seconds. Start arms at ≤65°C;
  abort at 80°C or unavailable live thermal telemetry. Keep existing hardware protections;
  never treat a thermal abort as success. Record Tctl and GPU sensors, peak temperatures,
  time in thermal bands, fan/power policy and cooldown duration. The current ModelDeck
  thermal status is disabled with null temperature and unavailable host-policy status;
  a separate verified live monitor is mandatory for this experiment.
- Record process RSS, Torch allocated/reserved/peak memory, whole-device GTT, host available
  memory and swap. Proposed combined-load gate: maintain at least 16 GiB host availability,
  no sustained swap-in/out or monotonic growth, and GTT recovery within 1 GiB of pre-load
  baseline after unload. These are measured headroom gates, not allocation guarantees.
- Reset/privacy holding hides and clears content immediately, invalidates in-flight results,
  and never allows a stale response to display. No overlapping analyses; cancellation must
  release occupancy promptly (proposed ≤2 seconds) or mark the Worker unavailable until
  recovery. Keep detector scheduling independent and verify its responsiveness under load.

Tests preserve the `VisionLanguageProvider` interface, local-only operation, generic person
wording, untrusted-output validation, prompt-injection protection, no visitor retention,
detector-only/mock/replay/privacy-holding modes and no automatic cloud fallback. A failure
must remain a visible local failure/fallback state rather than substituted model output.

## Introduction and rollback, after separate approval

Package the winner as a new Worker UUID with pinned model/artifact/processor revisions and
hashes, frozen ModelDeck implementation and dependency/build receipt, explicit backend,
precision, context/cache, visual preprocessing, decoding/stopping and timeout policies.
An editable package plus an old template number is insufficient immutability. Record both
successes and failures against that full fingerprint and invalidate qualification on any
material change. Use allowlisted manifests and argument arrays only.

Create a draft Open2026 revision whose sole intended difference is the `scenechat-vision`
Worker mapping; preserve the public route and `scene-analysis-v1`. Validate and rehearse
through an isolated evaluation route before publication. Archive revision 37 and the old
Worker's actual implementation bundle as the explicit rollback target. On later publication
approval, atomically activate the new revision and confirm serving identity. Configure no
automatic backup unless explicitly approved. Rollback restores the pinned prior mapping
and bundle; it restores the previous known state, **not** a claim that 0.8B is reliable.
Detector-only/replay remains the appropriate operational fallback if no candidate qualifies.

Resolve the port inventory before creating evaluation listeners: observed Worker ports
8673/8674 and other allocations extend beyond the supplied 8610–8624 table. Check actual
listeners and repository allocations, reserve configurable loopback evaluation ports, and
do not take an existing port or use random persistent fallbacks.

Any reduction to a caption-only schema, removal of questions, weaker validation or changed
visitor presentation requires a **separate application-contract proposal**. Model/runtime
evaluation above preserves the current external contract.

## Local evidence references

- [Guiding principles](GUIDING_PRINCIPLES.md), [benchmark documentation](BENCHMARKS.md).
- [Qwen engine](../backend/modeldeck/workers/qwen35_worker.py),
  [shared Worker and failure classification](../backend/modeldeck/workers/scenechat_worker.py),
  [contract and validator](../backend/modeldeck/contracts/scenechat/__init__.py),
  [reviewed models](../backend/modeldeck/reviewed_models.py).
- [Live failure log](../var/log/workers/3ad2f88d-8936-4ffc-ac63-6b5e6543d4ed.jsonl),
  [0.2.3 failures](../var/log/workers/5bd5b96c-fc52-4995-a4c8-d9835a163b23.jsonl).
- [70/70 report](../var/benchmarks/scenechat_visual_tokens_20260723T145634Z.json),
  [60/70 report](../var/benchmarks/scenechat_visual_tokens_20260723T111303Z.json),
  [2B thermal abort](../var/benchmarks/scenechat_visual_tokens_20260721T141011Z.json),
  [earlier manual review](../var/benchmarks/scenechat_visual_tokens_20260721T130108Z.json).
- Read-only `.modeldeck/modeldeck.sqlite3`: `compatibility_tests`, especially historical
  E2B scene-analysis records 6–10 and 14; 0.8B record 32; September Gemma 12B failures
  84–85; Qwen3.8 GGUF general-image-chat record 77. A stored `tested-working` classification
  is smoke evidence at its own fingerprint, not this proposal's release qualification.

No physical tests or repository verification suite were run for this proposal-only change.

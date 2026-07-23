# SceneChat Qwen3.5 latency qualification

## Candidate definition

The candidate uses:

- model `Qwen/Qwen3.5-0.8B`;
- revision `2fc06364715b967f1860aea9cf38778875588b17`;
- runtime `qwen35-vision-language-transformers-rocm`;
- packaged runtime-template version 0.2.2;
- BF16 and PyTorch SDPA;
- `enable_thinking=False`, `do_sample=False` and `use_cache=True`;
- a complete-JSON stopping criterion;
- 140 visual tokens;
- a 1,024-token hard completion ceiling;
- a 60-second generation deadline;
- the versioned `scene-analysis-v1` prompt and strict schema.

The 1,024-token ceiling intentionally replaces the proposal's 320-token candidate at the
operator's direction. It is failure headroom, not an output-length target. Promotion still
requires per-question completion-token p95 below 1,024, with a preferred target at or below
260, and zero `length` finish reasons.

## Qualification sequence

1. Create an immutable replacement Worker from the Qwen3.5 0.8B cached snapshot with the
   packaged defaults. Do not rebind or publish the Event yet.
2. Start and smoke the candidate, then run:

   ```powershell
   pwsh -NoProfile -File scripts/benchmark_scenechat_visual_tokens.ps1 `
       -Worker140 '<candidate-worker-id>' `
       -RunsPerQuestion 10 `
       -LoadMode isolated `
       -HumanReview
   ```

3. Require all seven question IDs to complete ten measured requests with zero failures,
   zero token-limit hits, median latency at most 8 seconds and p95 at most 12 seconds.
4. Run the same candidate with the full intended Event load for at least two hours:

   ```powershell
   pwsh -NoProfile -File scripts/benchmark_scenechat_visual_tokens.ps1 `
       -Worker140 '<candidate-worker-id>' `
       -RunsPerQuestion 10 `
       -LoadMode combined `
       -MinimumDurationSeconds 7200 `
       -HumanReview
   ```

5. Complete privacy/reset/outage drills, inspect memory recovery and temperatures, and
   approve representative fixed outputs for important visible objects and uncertainty.
6. Only after every gate passes, rebind the Event draft, publish a new immutable revision,
   smoke `scenechat-vision` through port 8600 and retain the prior Worker for rollback.

Reports under `var/benchmarks` contain aggregate metrics, fixed question IDs, configuration
fingerprints and safe failure categories. They do not retain the image, prompt text,
generated descriptions, credentials or visitor data.

## Current status

The 0.2.2 replacement retains the proven global prompt and greedy decoding, adds bounded
internal wording only for the closest-object question, and stops after a complete JSON
object. It retains the public curated questions, strict schema validation, BF16/SDPA
execution, 140-visual-token budget and 1,024-token hard ceiling.

An immutable 0.2.1 Worker (`3f84d269-1a71-4d1a-a298-8f5670631977`) was physically started
on 24 July 2026. Its benchmark stopped before measured requests because its two warm-ups
failed with `unsupported_fence` and `schema_violation`; the retained report is
`var/benchmarks/scenechat_visual_tokens_20260723T140824Z.json`. Synthetic-only diagnostics
identified early stopping before a closing JSON fence and destabilisation from broader
prompt and decoding changes. Those changes are not present in 0.2.2.

The completed isolated 0.2.2 run used immutable Worker
`3ad2f88d-8936-4ffc-ac63-6b5e6543d4ed` and is recorded in
`var/benchmarks/scenechat_visual_tokens_20260723T145634Z.json`. It fixed the functional
failure and supplied the evidence used for the operator's promotion decision:

- all 70 measured responses were schema-valid with zero failures and normal `stop`
  finishes;
- all ten `question-04` responses completed at 304 tokens instead of reaching the
  1,024-token ceiling;
- aggregate latency was 8.76 seconds p50 and 10.07 seconds p95, still missing the
  8-second median target while meeting the 12-second p95 target;
- per-question completion p95 was 300–358 tokens, above the preferred 260-token target;
- the maximum observed temperature was 77.625°C and GPU edge maximum was 56°C, with no
  thermal abort; pacing used a 75°C cooldown threshold and retained the 80°C abort;
- manual output review was not completed. An earlier fully measured invocation attempted
  `-HumanReview` without an interactive terminal and could not retain its report; the
  benchmark now rejects that mode before worker startup.

On 24 July 2026, the operator explicitly accepted this result and directed promotion,
overriding the 8-second median target, preferred 260-token target, combined two-hour run,
drills and separate manual-review gates for this immutable Worker. This exception does not
change the default qualification requirements for future candidates.

Open2026 revision 35 is published and active. Its `scenechat-vision` Route uses
`3ad2f88d-8936-4ffc-ac63-6b5e6543d4ed` as primary and retains mock Worker
`d2352c8b-834d-4199-97b6-ce779ab8c8ad` as backup. The prior 280-visual-token Worker remains
unarchived for rollback. Native smoke, gateway advertisement, selected/effective-provider
checks and an approved synthetic replay PNG completion through port 8600 passed; the
completion stopped normally at 322 tokens.

The isolated physical run on 23 July 2026 used 0.2.0 immutable Worker
`c6fadc21-2adf-465f-b3a6-d69c33102f76` and is recorded in
`var/benchmarks/scenechat_visual_tokens_20260723T111303Z.json`.

The candidate passed its physical BF16/SDPA smoke test and exercised all seven questions
with two warm-ups and ten measured requests per question. It did not pass promotion:

- 60 of 70 measured responses were valid;
- all ten requests for `question-04` reached the 1,024-token ceiling and were rejected as
  `token_limit` failures;
- valid-response latency was 8.90 seconds p50 and 10.07 seconds p95, missing the 8-second
  median target while meeting the 12-second p95 target;
- valid responses used 303–361 completion tokens, above the preferred 260-token ceiling;
- the maximum observed temperature was 74.75°C and the GPU edge maximum was 61°C, with no
  thermal abort;
- no curated question text was present in the retained report or inspected Worker logs;
- after shutdown, the Worker process was absent and ROCm reported 354,406,400 bytes of
  dedicated VRAM in use.

The combined two-hour run and manual output review were not started for 0.2.0 because its
isolated zero-failure gate failed. That candidate remains stopped as comparison evidence;
revision 34 is retained as immutable rollback history.

Any further decoding, prompt, model or runtime change requires a new immutable Worker and
a fresh isolated qualification.

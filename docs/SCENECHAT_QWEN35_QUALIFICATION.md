# SceneChat Qwen3.5 latency qualification

## Candidate definition

The candidate uses:

- model `Qwen/Qwen3.5-0.8B`;
- revision `2fc06364715b967f1860aea9cf38778875588b17`;
- runtime `qwen35-vision-language-transformers-rocm`;
- packaged runtime-template version 0.2.0;
- BF16 and PyTorch SDPA;
- `enable_thinking=False`, `do_sample=False` and `use_cache=True`;
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

Implementation and automated contract coverage are complete. The isolated physical run on
23 July 2026 used immutable Worker `c6fadc21-2adf-465f-b3a6-d69c33102f76` and is recorded in
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

The combined two-hour run and manual output review were not started because the isolated
zero-failure gate failed. The candidate is retained but stopped for comparison evidence. It
was not rebound or published: Open2026 revision 34 still routes SceneChat through Worker
`b4d8adcc-106d-4780-8874-387e5b7ab935` with its existing mock fallback.

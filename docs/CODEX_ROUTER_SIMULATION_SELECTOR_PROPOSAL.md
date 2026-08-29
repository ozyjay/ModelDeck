# Codex Router simulation selector

**Status:** proposal; no production implementation  
**Recommendation:** add a dedicated, bounded local selector profile for Codex Router evaluation  
**Proposed:** 30 August 2026

## Executive recommendation

Have ModelDeck create a dedicated **Codex Router Simulation Selector** profile. Its only job is to
choose a bounded local simulation tier. It must not impersonate Codex models, generate arbitrary
workspace edits or replace the existing deterministic evaluation scenarios and quality gates.

```text
Codex Router evaluation task + compact metadata
  -> ModelDeck local routing profile
  -> { "simulationProfile": "sim-small" | "sim-balanced" | "sim-strong" }
  -> declared deterministic scenario/patch
  -> existing validation, diff, mutation and reporting gates
```

This keeps early testing inexpensive and repeatable while testing whether a real local small
language model makes useful routing choices. Results assess local selector quality and harness
behaviour, not live Codex model quality.

## Profile requirements

- **Name:** `codex-router-simulation-selector`
- **Endpoint:** the existing loopback OpenAI-compatible gateway,
  `http://127.0.0.1:8600/v1`
- **Purpose:** classify task difficulty and risk for the evaluation harness.
- **Input:** task text plus compact, non-sensitive metadata:
  - task category;
  - estimated files affected;
  - whether tests are requested; and
  - risk flags such as destructive operations.
- **Never receive:** complete repository source, credentials, Codex authentication, prior model
  output or file contents by default.
- **Output:** strict JSON only:

  ```json
  {
    "simulationProfile": "sim-small",
    "confidence": 0.82,
    "rationale": "Single-file focused regression with explicit acceptance criteria."
  }
  ```

- **Allowed values:** exactly `sim-small`, `sim-balanced` and `sim-strong`.
- **Rationale limit:** 160 characters, with no source excerpts.
- **Failure behaviour:** invalid JSON, timeout, an unavailable model or an unknown profile must
  make Codex Router use the deterministic `sim-balanced` fallback and record only safe metadata.

The selector response must not contain commands, patches, paths, file contents, source excerpts or
free-form instructions for changing a workspace. The consumer must validate the complete response
against the fixed schema before using the selected value.

## Initial local-model mapping

| Profile | Intended local model class | Use |
|---|---|---|
| `sim-small` | Fast SLM | Focused, single-file regression tasks |
| `sim-balanced` | Stronger local coding SLM | Ordinary multi-file changes |
| `sim-strong` | Best available local model | Ambiguous, cross-cutting or security-sensitive tasks |

ModelDeck should expose the profile's actual model ID through `/v1/models`. Reports must identify
it as `local:<model-id>`, never Luna, Terra or Sol. The public selector name is a routing contract;
it is not a claim that the selected local model is an OpenAI or Codex model.

## Routing and inference policy

The profile must be independently selectable and use only ModelDeck's loopback gateway. It must
have no cloud fallback. Sampling settings must be code-owned and deterministic, with a low
temperature suitable for classification. The prompt and response budget must be bounded, and the
gateway must reject unsupported input shapes rather than passing arbitrary content to the Worker.

The deterministic scenario or patch remains outside the model. After the selector returns one of
the three allowed values, the existing Codex Router harness chooses a declared scenario and applies
its existing validation, diff, mutation and reporting gates.

## Acceptance criteria

1. The profile is independently selectable and appears as ready through `GET /v1/models`.
2. `POST /v1/chat/completions` returns the exact JSON contract defined above for the profile.
3. It operates only through ModelDeck's loopback gateway, with no cloud fallback.
4. It has deterministic low-temperature settings suitable for classification.
5. ModelDeck provides a short profile README containing:
   - model ID;
   - local model and runtime requirements;
   - context limit;
   - expected latency;
   - an example request and response; and
   - failure modes.

## Codex Router follow-up

Once ModelDeck publishes the model ID, Codex Router can add:

```bash
npm run eval:baseline:sim -- \
  --selector modeldeck \
  --modeldeck-model codex-router-simulation-selector \
  --iterations 3
```

The harness will log the selector backend, advertised local model ID, chosen profile, latency and
fallback status. Existing deterministic scenarios will continue to perform every actual patch and
quality check.

OpenAI's guidance recommends comparing configurations on representative tasks and treating lower
resource use as an improvement only when the same quality gates pass. See the
[official OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model).

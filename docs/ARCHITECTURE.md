# Architecture

## Boundaries

```text
Operator console/API :3600 ---- WorkerSupervisor ---- one allowlisted process per model :8610+
       |                                      |-- autoregressive contract
       |                                      `-- text-diffusion contract
       `---- read-only cache + hardware + SQLite evidence

Demo clients ---- Stable gateway :8600 ---- capability alias ---- ready local worker
                                                   `---- structured unavailable result
```

## Runtime environments

`.venv` is ModelDeck's control-plane runtime: API, operator-console assets, supervisor, gateway,
catalogue, evidence store, fallbacks, and tests. `.venv-rocm72` is the primary target
inference runtime used by the core Qwen and DiffusionGemma worker processes. Both belong
to the target installation, but they remain separate so GPU dependencies and tensors
never enter the management process. A GPU-free `.venv` run is a useful development or
recovery mode, not the primary inference configuration.

HuggingFacePull owns Hugging Face acquisition and cleanup. OllamaPull will own Ollama
registry storage when inspected. ModelDeck owns profiles, runtime process lifecycle,
scheduling, evidence, fixed local routing, and management presentation. Demos retain
public wording, interaction state, reset, and prepared replay assets.

## Process and failure model

The management API does not import model libraries or hold model tensors. Each worker is
an isolated subprocess launched with a fixed argument array derived from a validated
profile. The supervisor serialises loads, checks fixed ports, captures stdout/stderr,
polls health, runs warmup, detects exit, requests graceful shutdown, and terminates only
after a timeout. Terminating the worker is the memory-recovery boundary.

Worker log records are bounded on disk, severity-classified, and tagged by launch
session. The management log API shows the current launch by default so resolved failures
from an earlier runtime do not appear to describe the active worker.

States are `discovered`, `stopped`, `validating`, `starting`, `loading`, `warming`,
`ready`, `busy`, `degraded`, `stopping`, `failed`, `orphaned`, and `incompatible`.
Process existence alone never means ready.

## Gateway and routing

Aliases route by declared generation family and capability. `fast-chat` prefers the core
Qwen ROCm worker and `text-diffusion` prefers the separate core DiffusionGemma ROCm
worker. `scenechat-vision` routes image-and-text requests to the pinned Gemma 4 worker and
injects its private loopback credential. Each fallback-capable alias retains an explicit
mock provider for GPU-unavailable demonstrations
and contract testing. When no candidate is ready, the gateway returns a structured local
unavailable response; no cloud request occurs. The gateway and management API are
separate processes and ports so demo clients have a stable contract while management
restarts evolve independently.

## Scheduler

The first scheduler invariant is a single global model-load lock. Profiles declare
`resident`, `on-demand`, or `exclusive`. Later measured peak/steady memory, context/KV
growth, temperature, reserve, process use, and compatibility evidence will inform launch
decisions. Parameter count alone will not.

## Cache integration

The scanner resolves `HF_HUB_CACHE`, `${HF_HOME}/hub`, the normal user cache, then
`/mnt/work/models/huggingface/hub`. It reads snapshots, config hints, incomplete markers,
revision refs, and physical size. It never calls the Hub and never treats files as proof
that a model is runnable. For recognised complete snapshots matching the allowlisted
autoregressive, SceneChat Gemma 4, DiffusionGemma BF16, or self-contained ModelDeck Q4
implementations, the operator may
create a constrained local profile. The server resolves the selected catalogue entry
and derives its cache root; no filesystem path, runtime executable, command argument,
environment variable, remote model identifier, or remote-code flag is accepted from the
browser. Local profiles are persisted in SQLite, loaded by management at startup, and
discovered dynamically by the gateway.

An exact model/revision allow policy controls whether Hugging Face cache-backed profiles
participate in management workers and gateway routes. Disallowing is reversible and never
mutates the cache or deletes profile configuration. Profiles reading packaged ModelDeck
checkpoints do not inherit policy merely because they record the same upstream model ID.
Downloaded Q4 profiles instead carry a separate derivative artefact identity, so their
policy follows the exact Hugging Face repository revision while worker loading and smoke
evidence retain the pinned base-model identity.

## Data and security

SQLite stores model profiles, compatibility tests, worker events, and presets. Logs are
bounded in memory in this slice and redact prompt/output/credential-shaped fields.
Services bind to loopback. The API has no shell, arbitrary environment, filesystem
browser, token, Docker socket, upload, camera, or cloud inference surface.

# Existing repository findings

Inspected on 10 July 2026. Source code, tests, scripts, configuration, and recent Git
history were inspected without modifying these repositories. Repositories absent from
the available local project roots are recorded as unavailable rather than guessed.

## Summary

| Repository | Branch / commit | Relevant responsibility and runtime | Cache / protocol / health | Ports and process management | Fallback and useful implementation | Conflict / recommendation |
|---|---|---|---|---|---|---|
| HuggingFacePull | `standalone-fedora-desktop` / `b2618fe` | Hugging Face search, queued snapshot acquisition, resume, cleanup, Xet selection; FastAPI | Standard HF cache plus its library markers; HTTP API; `/api/state` exposes cached and partial entries | Desktop `8019`; `DownloadQueue` uses an isolated download process and hard-stop path | Progress, ETA, partial cache accounting in `queue.py`; `HubRef`, `cached_hub_models`, markers in `hub.py` | `hub.py:44-62` currently disables Xet at import and represents it as a Boolean, not `auto/xet/http`; adapt through read-only API/metadata and leave acquisition here |
| TextDiffusionDemo | `main` / `9032f9f` | DiffusionGemma via custom Transformers worker, external adapter, Red Hat vLLM trial, replay | Persistent stdin/stdout NDJSON (`diffusiongemma_adapter/worker.py`); backend `/api/health`; model preload distinct from request | UI `3300`, API `8300`, adapter `8600`, vLLM `8000`; `DiffusionGemmaWorkerClient` owns subprocess and pending timeouts | `modelAdapter.ts` uses 30 s normal and 300/600 s worker/preload timeouts, explicit provider diagnostics, final fallback; `buildDiffusionGemmaWorkerEnv` conditionally uses `/usr/lib64/libhsa-runtime64.so.1` | Trace-shaped demo output is not a generic worker contract; adapt engine lessons, keep native refinement and apply HSA preload only to tested profiles |
| TokenTrail | `main` / `e9298be` | Local Transformers causal trace and deterministic prepared traces | Read-only complete HF snapshot detection in `scripts/probe_hf_trace.py`; HTTP `/health`, `/api/models`, `/api/warmup`, `/api/trace` | Demo `3100`, documented backend `8100`, HF trace `8600`; local runner owns trace server | `TransformersTraceRunner` serialises generation with a lock, loads local files only, warms explicitly; `generate_with_scores` and forward logits produce top-k observations; `runtime.py` always offers scripted fallback | Its `8600/api/trace` is narrower than the shared gateway and keeps multiple models in one process; wrap the trace contract and use one model per ModelDeck worker |
| FedoraUsage | `main` / `f113570` | GNOME Shell RAM, swap, filesystem, hwmon temperature, and fan telemetry | Direct `/proc/meminfo`, `/sys/class/hwmon`, filesystem queries; no service protocol | No port or owned process | `_workSsdPaths()` includes lowercase `/mnt/work`; hwmon parsing and sensor labels are useful | GNOME/GJS code cannot be imported by the Python service; locally reimplement safe read-only probes and do not depend on the extension |
| InsideNeuralNets | `main` / `473bc99` | Local torchvision inference and in-memory visualisation | FastAPI routes; weights loaded through torchvision; safe `ModelUnavailableError` response | Fixed `3450`; PowerShell start/stop scripts | Camera is explicit opt-in and frames are decoded in memory with an 8 MiB bound; fixed channel selection; UI reset; missing-weight guidance | Current model constructors may attempt acquisition if weights are absent and fallback assets are not fully wired; leave demo independent and later provide a specialist worker contract |
| SceneChat | `main` / `bed8c57` | Local scene detection and mock/replay/vLLM vision-language providers | FastAPI `/api/health`, `/api/state`, `/api/events`; server-owned `StateStore`; provider protocols | Fixed `8900`; vLLM `8000`; PowerShell lifecycle scripts | `AnalysisService` uses one `asyncio.Lock`; replay provider and one-click reset; camera state is isolated | Its provider selection is demo-owned and vision-specific; route by ModelDeck capability later, but retain replay and privacy controls in SceneChat |
| MLXDashboard | unavailable | Requested Apple/MLX architecture comparison | Not present under the accessible project roots | Unknown | No source claim made | Inspect before any catalogue/UI compatibility work derived from it |
| OllamaPull | unavailable | Requested Ollama acquisition inspection | Not present | Unknown | No source claim made | Keep Ollama storage separate; inspect before adapter implementation |
| OllamaAgent | unavailable | Requested Ollama provider inspection | Not present | Unknown | No source claim made | Inspect before adapter implementation |
| CrowdAIMission | unavailable | Requested Open Day operating-pattern inspection | Not present | Unknown | No source claim made | Do not claim its exact routes until inspected |
| OpenDayOps | unavailable | Requested shared port and operations source | Not present | Unknown | No source claim made | Current assignments preserve observed ports; reconcile when available |

## Exact reuse references

- **Cache boundary:** `HuggingFacePull/src/huggingface_pull/api.py:create_app` and
  `hub.py:cached_hub_models`, `partial_cached_hub_models`, `installed_models`. ModelDeck's
  initial scanner is read-only and does not import download code.
- **Download evidence gap:** `HuggingFacePull/src/huggingface_pull/hub.py:HubRef` stores
  `xet_enabled`; future integration needs a three-state transport policy and recorded
  requested/used transport rather than another downloader.
- **Diffusion isolation:** `TextDiffusionDemo/server/services/diffusionGemmaWorker.ts:
  DiffusionGemmaWorkerClient` and `adapters/.../worker.py:main` demonstrate persistent
  NDJSON, preload, process ownership, buffered line parsing, and independent timeouts.
- **HSA policy:** `diffusionGemmaWorker.ts:buildDiffusionGemmaWorkerEnv` checks platform,
  engine, existing `LD_PRELOAD`, and file existence. ModelDeck adds the missing evidence
  gate before using that pattern for a real worker.
- **Autoregressive evidence:** `TokenTrail/scripts/probe_hf_trace.py:
  generate_with_scores`, `build_trace_from_generation`, and
  `forward_logit_vectors_for_generated_tokens` are the reference for observable token
  alternatives. They are not private reasoning.
- **Ready versus installed:** `TokenTrail/scripts/serve_hf_trace.py:
  TransformersTraceRunner.discover_model` distinguishes cached, metadata-loadable,
  loaded, and available states. ModelDeck generalises that separation.
- **Telemetry:** `FedoraUsage/extension.js:_readMeminfo`, `_workSsdPaths`,
  `_readHwmonTemperatureSensors`, and its fan reader are safe conceptual references.
- **Bounded local requests:** `SceneChat/backend/scenechat/services/analysis.py:
  AnalysisService` and `services/state.py:StateStore` provide useful lock/reset patterns.
- **Privacy:** `InsideNeuralNets/app.py:_decode_camera_image` proves in-memory bounded
  frames, while its UI requires explicit `getUserMedia` action. ModelDeck adds no camera.

## Conflicting assumptions

1. Port `8600` is both TokenTrail's current trace server and TextDiffusionDemo's adapter.
   It is intentionally reserved for the ModelDeck stable gateway; old routes will need
   thin compatibility adapters during demo migration.
2. HuggingFacePull globally disables Xet on import unless its Boolean flag enables it.
   The required future policy is explicit `auto/xet/http` with bounded fallback evidence.
3. TokenTrail caches several models in one trace process, while ModelDeck requires one
   model per worker for memory recovery and dependency isolation.
4. TextDiffusionDemo returns demo `Trace` objects through NDJSON, while the canonical
   ModelDeck diffusion API is job/frame based.
5. Some demos may allow model library constructors to acquire missing weights. ModelDeck
   Open Day mode must enforce local files only and will never start a download.


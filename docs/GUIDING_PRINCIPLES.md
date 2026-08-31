# Guiding principles

ModelDeck is a local-first experimental layer for evaluating local AI inference
configurations. It is not a preferred inference engine and must not become a thin wrapper
around one runtime. These principles are architectural and experimental constraints for
future work, not claims that every capability already exists.

## Purpose and terminology

The central question is: **given a model artifact, workload, and hardware/software
environment, which runtime and backend provide the most usable local inference
configuration?** The longer-term question is whether accumulated evidence can predict that
choice. The Framework Desktop is the principal experimental platform, but the method must
remain applicable to other heterogeneous unified-memory systems.

Use these terms consistently:

- **Model**: the model architecture and pinned model revision.
- **Artifact**: the concrete executable files, format, and quantisation.
- **Runtime**: the inference engine or framework.
- **Backend**: the runtime's execution implementation, such as CPU, Vulkan, or HIP/ROCm.
- **Configuration**: the complete executable combination of Model, Artifact, Runtime,
  Backend, context/KV-cache policy, workload, and environment.
- **Run**: one measured execution of a Configuration under a defined workload.
- **Usability**: a workload-specific assessment of performance, resource use, stability,
  safety, and correctness.

Do not use *model size* where parameter count, artifact file size, memory footprint, or
runtime cost is meant.

## Principles

1. **Orchestrate runtimes; do not favour one by default.** Runtimes are bounded execution
   environments with comparable lifecycle, health, and measurement interfaces. Runtime and
   backend selection must follow evidence, not an assumption that ROCm, GPU execution, or a
   more sophisticated runtime is always best.
2. **Evaluate complete configurations.** A benchmark record must identify the Model and
   revision, Artifact and quantisation, architecture, Runtime, resolved Backend and device
   placement, precision, context/KV-cache settings, workload, and environment. Comparisons
   are valid only when material differences are matched or made explicit.
3. **Treat architecture and quantisation explicitly.** Parameter count alone does not predict
   usability. Capture available dense/MoE traits, total and active parameters, precision,
   quantisation method, artifact size, memory behaviour, context, and workload. Low-bit
   artifacts and sparse MoE execution are not automatic speed claims.
4. **Use workload-specific, multidimensional usability.** A loadable artifact is not
   necessarily usable. Record relevant load, prefill, first-output, decode, total-latency,
   memory, stability, cancellation, unload, thermal, safety, and correctness observations.
   Do not present a universal score unless its policy, weighting, and limitations are
   explicit.
5. **Separate direct-runtime, Worker, and API-path measurements.** Where practical, measure
   each layer under matched conditions so inference performance is not confused with process,
   streaming, gateway, or orchestration overhead. Any performance acceptance target must
   state its matched metric and the safety or observability trade-off it permits.
6. **Make resolved execution identity observable.** Health, logs, model identity, and Run
   records must identify the backend actually used, not merely the requested backend or host
   vendor. Model identity must distinguish the catalogue identity, upstream and artifact
   revisions, format, quantisation, architecture, modality, and relevant tokenizer or
   processor revision; hashes are preferred where practical.
7. **Never make an unrecorded substitution.** ModelDeck must not silently change a Model,
   Artifact, Runtime, Backend, device, precision, or context/KV-cache policy. The default
   policy is no fallback. A configured Routing Profile backup is an explicit routing policy,
   not an implicit substitution: selection must remain observable and request metadata must
   identify the serving Worker where that information can be safely exposed. Failure must be
   structured and name the failed component and cause where known.
8. **Prefer reproducibility over convenience.** Use allowlisted local artifacts, pinned
   revisions, explicit versions, controlled workloads, repeatable runs, and distinct cold and
   warm measurements. Preserve enough environment information to reproduce a Run later or
   explain why exact reproduction is impossible.
9. **Isolate and observe the lifecycle.** Workers are process and failure boundaries. Their
   load, readiness, health, inference, cancellation, unload, termination, and restart
   transitions must be observable and testable. Unless tested otherwise, a Worker loads no
   more than one large Model. Recovery after load, memory, backend, cancellation, and crash
   failures is a product requirement.
10. **Treat unified memory as shared and estimates as estimates.** Evaluate model weights,
    runtime overhead, KV cache, preprocessing, page cache, fragmentation, concurrent workers,
    and temporary buffers. Pre-load viability checks may guide an operator but must never be
    presented as guarantees.
11. **Thermal safety is benchmark correctness.** The active thermal policy is configurable,
    recorded with every Run, and takes precedence over benchmark completion. On the Framework
    Desktop, the provisional policy is below 80°C acceptable, 80–85°C caution, above 85°C
    reduce or pause work, and 90°C or above terminate. A thermal-policy breach is unsafe or
    invalid, not an unqualified success. Diagnose application, runtime, operating-system,
    kernel, and firmware controls separately.
12. **Preserve raw evidence and its uncertainty.** Store raw observations independently from
    classifications, scores, and recommendations. Preserve both positive and negative
    evidence against a complete fingerprint. Recommendations must expose their supporting
    evidence, policy thresholds, and uncertainty; configuration prediction is future research,
    not a current product claim.
13. **Benchmark real workloads.** Define inputs, context, output limits, sampling,
    concurrency, stopping conditions, and correctness criteria. Cover relevant interactive,
    long-context, reasoning, coding, embedding, and multimodal workloads rather than a
    single synthetic decode test. Vision measurement additionally separates preprocessing,
    visual-token count, vision encoding, prefill, text generation, and total latency.
14. **Keep security and local control intact.** ModelDeck remains offline-first, uses
    read-only discovery, launches only allowlisted manifests with argument arrays, and binds
    locally by default. Experimental convenience must not introduce arbitrary execution,
    acquisition, cloud fallback, or undisclosed data collection.

## How to apply these principles

Every proposal or implementation must state which principle it advances, which execution
details it exposes, and any reproducibility, safety, or compatibility trade-off. A feature
is consistent with ModelDeck only when it improves runtime choice, observability, safety,
empirical comparison, lifecycle reliability, workload-aware performance, reproducibility,
or future configuration prediction without hiding material execution details or changing
behaviour silently.

The current baseline already includes trusted isolated Workers, explicit routing and
backup selection, pinned local discovery, append-only compatibility fingerprints, thermal
admission, and privacy-preserving benchmark reports. See [Architecture](ARCHITECTURE.md),
[API contract](API_CONTRACT.md), [Compatibility registry](COMPATIBILITY_REGISTRY.md), and
[Benchmarks](BENCHMARKS.md). Cross-runtime matched benchmarking, complete artifact hashing,
unified-memory preflight estimation, and evidence-based configuration recommendations are
directions to design and validate before they are represented as implemented behaviour.

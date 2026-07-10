# Compatibility registry

Compatibility is an append-only history tied to a SHA-256 fingerprint of hardware
profile, Fedora/kernel/GPU/architecture, ROCm, PyTorch, Transformers, vLLM, model and
revision, quantisation, dtype, runtime, and relevant environment overrides.

Evidence adds load/warmup/smoke results, cold-load and first-output latency, throughput,
peak/steady memory, shutdown and recovery results, stability duration, classified failure,
safe error summary, log reference, test date, and retest triggers.

States include `tested-working`, `tested-limited`, `incompatible-current-stack`,
`transient-failure`, and `superseded`. Negative evidence is preserved and means only that
the recorded fingerprint failed. Version, revision, quantisation, or relevant environment
changes create a new record rather than silently retrying or overwriting history.


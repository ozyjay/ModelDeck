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

The first physical working fingerprint is
`423a331ad14e12a400adbd5b2c65c8fe8e1c9e8a85138e85fb6ff2e9d5bb6163` for the pinned
Qwen 0.5B FP16 Transformers/ROCm configuration documented in
`ROCM_FRAMEWORK_DESKTOP.md`. Its stability evidence records 343 requests and zero
failures over 1,808.851 seconds.

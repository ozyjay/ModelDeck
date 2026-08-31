# Compatibility registry

Compatibility fingerprint schema v2 is an append-only history tied to a SHA-256
fingerprint of requested and reported Model identity, Artifact identity and available
hashes, hardware profile, Fedora/kernel/GPU/architecture, ROCm, framework versions,
quantisation, precision, Runtime, resolved Backend/device, context and cache policy,
capability workload, trusted Runtime registration, Worker configuration, thermal policy,
and relevant environment overrides.

Evidence adds load/warmup/smoke results, cold-load and first-output latency, throughput,
peak/steady memory, shutdown and recovery results, stability duration, classified failure,
safe error summary, log reference, test date, and retest triggers.

States include `tested-working`, `tested-limited`, `incompatible-current-stack`,
`transient-failure`, and `superseded`. Negative evidence is preserved and means only that
the recorded fingerprint failed. Version, revision, quantisation, or relevant environment
changes create a new record rather than silently retrying or overwriting history.
Lifecycle observations are appended as separately timestamped child records; the raw test
document and its fingerprint are never updated. Pre-v2 records remain readable and are
labelled `legacy`, but cannot qualify a new `tested-working` publication.

Capability qualification is distinct from discovery and generic Worker health. A
capability-specific test records the stable capability ID, protocol contract, runtime
template ID and version, Worker ID and execution-configuration fingerprint. A
`tested-working` Routing Profile requires this exact evidence. Workers created before the
schema-v5 evidence migration may use matching legacy model/revision/runtime
evidence and are labelled `legacy`; new and replacement Workers cannot.

The first physical working fingerprint is
`423a331ad14e12a400adbd5b2c65c8fe8e1c9e8a85138e85fb6ff2e9d5bb6163` for the pinned
Qwen 0.5B FP16 Transformers/ROCm configuration documented in
`ROCM_FRAMEWORK_DESKTOP.md`. Its stability evidence records 343 requests and zero
failures over 1,808.851 seconds.

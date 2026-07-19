from __future__ import annotations

from dataclasses import dataclass

from modeldeck.protocol import GenerationFamily


@dataclass(frozen=True)
class TrustedRuntimeImplementation:
    id: str
    display_name: str
    generation_family: GenerationFamily
    capabilities: frozenset[str]
    template_settings: frozenset[str]
    cache_settings: frozenset[str]


# This registry is deliberately code-owned. A locally installed manifest may select one
# of these implementations, but cannot supply an executable, module, arguments, paths, or
# environment variables.
TRUSTED_RUNTIME_IMPLEMENTATIONS = {
    implementation.id: implementation
    for implementation in (
        TrustedRuntimeImplementation(
            id="transformers-rocm",
            display_name="Autoregressive Transformers ROCm",
            generation_family=GenerationFamily.AUTOREGRESSIVE,
            capabilities=frozenset(
                {
                    "chat",
                    "completions",
                    "logits",
                    "top_k_trace",
                    "hidden_states",
                    "seeded_generation",
                    "streaming",
                    "cancellation",
                }
            ),
            template_settings=frozenset(
                {
                    "top_k",
                    "context_length",
                    "maximum_new_tokens",
                    "startup_timeout_seconds",
                    "warmup_timeout_seconds",
                }
            ),
            cache_settings=frozenset({"cache_root"}),
        ),
        TrustedRuntimeImplementation(
            id="vision-language-transformers-rocm",
            display_name="Vision-language Transformers ROCm",
            generation_family=GenerationFamily.VISION_LANGUAGE,
            capabilities=frozenset({"chat", "streaming", "cancellation", "image_input", "structured_output"}),
            template_settings=frozenset(
                {
                    "context_length",
                    "maximum_new_tokens",
                    "generation_timeout_seconds",
                    "startup_timeout_seconds",
                    "warmup_timeout_seconds",
                    "hardware_verification_required",
                }
            ),
            cache_settings=frozenset({"cache_root"}),
        ),
        TrustedRuntimeImplementation(
            id="text-diffusion-transformers-rocm",
            display_name="Text-diffusion Transformers ROCm",
            generation_family=GenerationFamily.TEXT_DIFFUSION,
            capabilities=frozenset(
                {
                    "iterative_refinement",
                    "intermediate_frames",
                    "seeded_generation",
                    "logits",
                    "streaming",
                    "cancellation",
                }
            ),
            template_settings=frozenset(
                {
                    "maximum_new_tokens",
                    "maximum_denoising_steps",
                    "startup_timeout_seconds",
                    "warmup_timeout_seconds",
                    "hsa_preload_evidence",
                }
            ),
            cache_settings=frozenset({"cache_root"}),
        ),
        TrustedRuntimeImplementation(
            id="text-diffusion-gptq-rocm",
            display_name="Text-diffusion GPTQ ROCm",
            generation_family=GenerationFamily.TEXT_DIFFUSION,
            capabilities=frozenset(
                {
                    "iterative_refinement",
                    "intermediate_frames",
                    "seeded_generation",
                    "logits",
                    "streaming",
                    "cancellation",
                }
            ),
            template_settings=frozenset(
                {
                    "maximum_new_tokens",
                    "maximum_denoising_steps",
                    "startup_timeout_seconds",
                    "warmup_timeout_seconds",
                    "hsa_preload_evidence",
                }
            ),
            cache_settings=frozenset({"q4_checkpoint_dir"}),
        ),
        TrustedRuntimeImplementation(
            id="llama-vulkan",
            display_name="llama.cpp Vulkan",
            generation_family=GenerationFamily.AUTOREGRESSIVE,
            capabilities=frozenset({"chat", "completions", "streaming", "cancellation"}),
            template_settings=frozenset(
                {
                    "context_length",
                    "maximum_new_tokens",
                    "startup_timeout_seconds",
                    "warmup_timeout_seconds",
                    "execution_preset",
                    "hardware_verification_required",
                }
            ),
            cache_settings=frozenset({"artifact_path"}),
        ),
        TrustedRuntimeImplementation(
            id="moshiko-rocm",
            display_name="Moshiko ROCm",
            generation_family=GenerationFamily.SPEECH_CONVERSATION,
            capabilities=frozenset(
                {
                    "streaming",
                    "cancellation",
                    "audio_input",
                    "audio_output",
                    "full_duplex",
                }
            ),
            template_settings=frozenset(
                {
                    "sample_rate_hz",
                    "channels",
                    "maximum_sessions",
                    "maximum_buffer_ms",
                    "startup_timeout_seconds",
                    "warmup_timeout_seconds",
                    "hardware_verification_required",
                }
            ),
            cache_settings=frozenset({"cache_root"}),
        ),
    )
}

TRUSTED_RUNTIME_IDS = frozenset({"mock", *TRUSTED_RUNTIME_IMPLEMENTATIONS})

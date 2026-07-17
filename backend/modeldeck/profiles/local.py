from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from modeldeck.protocol import CapabilitySet

from .models import ModelProfile

LOCAL_PORT_RANGE = range(8630, 8700)
RESERVED_GATEWAY_ALIASES = {
    "fast-chat",
    "token-explainer",
    "qwen-0-5b",
    "qwen-1-5b",
    "qwen-3b",
    "scenechat-vision",
    "text-diffusion",
    "text-diffusion-bf16",
}


class LocalProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=3, max_length=256)
    revision: str = Field(min_length=1, max_length=128)
    alias: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    dtype: Literal["float16", "bfloat16"] = "float16"
    lifecycle: Literal["resident", "on-demand", "exclusive"] = "on-demand"
    context_length: int = Field(default=2048, ge=256, le=32768)
    maximum_new_tokens: int = Field(default=128, ge=1, le=512)
    maximum_denoising_steps: int = Field(default=24, ge=1, le=48)


def create_local_autoregressive_profile(
    request: LocalProfileRequest,
    *,
    cache_root: Path,
    port: int,
) -> ModelProfile:
    return ModelProfile(
        id=f"local-{request.alias}",
        model_id=request.model_id,
        revision=request.revision,
        alias=request.alias,
        generation_family="autoregressive",
        preferred_runtime="transformers-rocm",
        lifecycle=request.lifecycle,
        port=port,
        local_files_only=True,
        trust_remote_code=False,
        dtype=request.dtype,
        capabilities=CapabilitySet(
            chat=True,
            completions=True,
            logits=True,
            top_k_trace=True,
            hidden_states="optional",
            seeded_generation=True,
        ),
        settings={
            "context_length": request.context_length,
            "maximum_new_tokens": request.maximum_new_tokens,
            "top_k": 5,
            "startup_timeout_seconds": 300,
            "warmup_timeout_seconds": 60,
            "cache_root": str(cache_root),
        },
    )


def create_local_profile(
    request: LocalProfileRequest,
    *,
    cache_root: Path,
    port: int,
    configuration_support: str,
) -> ModelProfile:
    if configuration_support == "autoregressive-transformers":
        return create_local_autoregressive_profile(request, cache_root=cache_root, port=port)
    if configuration_support == "scenechat-gemma4":
        return ModelProfile(
            id=f"local-{request.alias}",
            model_id=request.model_id,
            revision=request.revision,
            alias=request.alias,
            generation_family="vision-language",
            preferred_runtime="vision-language-transformers-rocm",
            lifecycle=request.lifecycle,
            port=port,
            local_files_only=True,
            trust_remote_code=False,
            dtype=request.dtype,
            capabilities=CapabilitySet(
                chat="compatibility-only",
                streaming=False,
                cancellation=True,
                image_input=True,
                structured_output=True,
            ),
            settings={
                "context_length": request.context_length,
                "maximum_new_tokens": request.maximum_new_tokens,
                "generation_timeout_seconds": 60,
                "startup_timeout_seconds": 600,
                "warmup_timeout_seconds": 180,
                "cache_root": str(cache_root),
            },
        )
    if configuration_support == "diffusiongemma-transformers":
        return ModelProfile(
            id=f"local-{request.alias}",
            model_id=request.model_id,
            revision=request.revision,
            alias=request.alias,
            generation_family="text-diffusion",
            preferred_runtime="text-diffusion-transformers-rocm",
            lifecycle="exclusive",
            port=port,
            local_files_only=True,
            trust_remote_code=False,
            dtype=request.dtype,
            capabilities=CapabilitySet(
                iterative_refinement=True,
                intermediate_frames=True,
                seeded_generation=True,
                logits="model-specific",
            ),
            settings={
                "maximum_new_tokens": request.maximum_new_tokens,
                "maximum_denoising_steps": request.maximum_denoising_steps,
                "startup_timeout_seconds": 600,
                "warmup_timeout_seconds": 300,
                "hsa_preload_evidence": False,
                "cache_root": str(cache_root),
            },
        )
    raise ValueError("No allowlisted local worker supports this model architecture")


LocalAutoregressiveProfileRequest = LocalProfileRequest

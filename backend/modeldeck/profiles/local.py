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


class LocalAutoregressiveProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=3, max_length=256)
    revision: str = Field(min_length=1, max_length=128)
    alias: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    dtype: Literal["float16", "bfloat16"] = "float16"
    lifecycle: Literal["resident", "on-demand", "exclusive"] = "on-demand"
    context_length: int = Field(default=2048, ge=256, le=32768)
    maximum_new_tokens: int = Field(default=128, ge=1, le=512)


def create_local_autoregressive_profile(
    request: LocalAutoregressiveProfileRequest,
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

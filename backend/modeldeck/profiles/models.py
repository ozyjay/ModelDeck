from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modeldeck.protocol import CapabilitySet, GenerationFamily, LifecycleClass

SAFE_ALIAS = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
ALLOWED_RUNTIMES = {
    "mock",
    "transformers-rocm",
    "text-diffusion-transformers-rocm",
    "text-diffusion-gptq-rocm",
}


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    model_id: str
    revision: str
    alias: str
    generation_family: GenerationFamily
    preferred_runtime: str
    lifecycle: LifecycleClass
    port: int = Field(ge=1024, le=65535)
    local_files_only: bool = True
    trust_remote_code: bool = False
    dtype: str = "auto"
    capabilities: CapabilitySet
    settings: dict[str, int | float | str | bool] = Field(default_factory=dict)

    @field_validator("id", "alias")
    @classmethod
    def safe_identifier(cls, value: str) -> str:
        if not SAFE_ALIAS.fullmatch(value):
            raise ValueError("must use lowercase letters, digits, and hyphens")
        return value

    @field_validator("preferred_runtime")
    @classmethod
    def allowlisted_runtime(cls, value: str) -> str:
        if value not in ALLOWED_RUNTIMES:
            raise ValueError("runtime is not allowlisted")
        return value

    @model_validator(mode="after")
    def generation_contract(self) -> ModelProfile:
        if (
            self.generation_family == GenerationFamily.AUTOREGRESSIVE
            and self.capabilities.iterative_refinement
        ):
            raise ValueError("autoregressive profiles cannot advertise iterative refinement")
        if (
            self.generation_family == GenerationFamily.TEXT_DIFFUSION
            and not self.capabilities.iterative_refinement
        ):
            raise ValueError("text-diffusion profiles must advertise iterative refinement")
        return self


def default_model_profiles() -> list[ModelProfile]:
    return [
        ModelProfile(
            id="diffusiongemma-rocm",
            model_id="google/diffusiongemma-26B-A4B-it",
            revision="52de6b914ee1749a7d4933202505ddf5b414ec43",
            alias="text-diffusion",
            generation_family="text-diffusion",
            preferred_runtime="text-diffusion-transformers-rocm",
            lifecycle="exclusive",
            port=8621,
            dtype="bfloat16",
            capabilities=CapabilitySet(
                iterative_refinement=True,
                intermediate_frames=True,
                seeded_generation=True,
                logits="model-specific",
            ),
            settings={
                "maximum_new_tokens": 256,
                "maximum_denoising_steps": 48,
                "startup_timeout_seconds": 600,
                "warmup_timeout_seconds": 300,
                "hsa_preload_evidence": False,
                "cache_root": "/mnt/work/models/huggingface/hub",
            },
        ),
        ModelProfile(
            id="diffusiongemma-q4-rocm",
            model_id="google/diffusiongemma-26B-A4B-it",
            revision="52de6b914ee1749a7d4933202505ddf5b414ec43",
            alias="text-diffusion-q4",
            generation_family="text-diffusion",
            preferred_runtime="text-diffusion-gptq-rocm",
            lifecycle="exclusive",
            port=8622,
            dtype="bfloat16",
            capabilities=CapabilitySet(
                iterative_refinement=True,
                intermediate_frames=True,
                seeded_generation=True,
                logits="model-specific",
            ),
            settings={
                "maximum_new_tokens": 256,
                "maximum_denoising_steps": 48,
                "startup_timeout_seconds": 600,
                "warmup_timeout_seconds": 300,
                "hsa_preload_evidence": False,
                "cache_root": "/mnt/work/models/huggingface/hub",
                "q4_checkpoint_dir": "var/diffusiongemma-26b-a4b-it-gptq-q4-g32",
            },
        ),
        ModelProfile(
            id="qwen-small-rocm",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
            alias="token-explainer",
            generation_family="autoregressive",
            preferred_runtime="transformers-rocm",
            lifecycle="resident",
            port=8620,
            dtype="float16",
            capabilities=CapabilitySet(
                chat=True,
                completions=True,
                logits=True,
                top_k_trace=True,
                hidden_states="optional",
                seeded_generation=True,
            ),
            settings={
                "context_length": 2048,
                "maximum_new_tokens": 128,
                "top_k": 5,
                "startup_timeout_seconds": 300,
                "warmup_timeout_seconds": 60,
            },
        ),
        ModelProfile(
            id="mock-ar",
            model_id="modeldeck/mock-autoregressive",
            revision="fixture-v1",
            alias="fast-chat",
            generation_family="autoregressive",
            preferred_runtime="mock",
            lifecycle="resident",
            port=8610,
            capabilities=CapabilitySet(
                chat=True,
                completions=True,
                logits=True,
                top_k_trace=True,
                hidden_states="optional",
                seeded_generation=True,
            ),
            settings={"maximum_new_tokens": 64, "top_k": 5},
        ),
        ModelProfile(
            id="mock-diffusion",
            model_id="modeldeck/mock-text-diffusion",
            revision="fixture-v1",
            alias="text-diffusion",
            generation_family="text-diffusion",
            preferred_runtime="mock",
            lifecycle="exclusive",
            port=8611,
            capabilities=CapabilitySet(
                iterative_refinement=True,
                intermediate_frames=True,
                seeded_generation=True,
                logits="model-specific",
            ),
            settings={"denoising_steps": 8, "block_length": 16, "maximum_length": 64},
        ),
    ]

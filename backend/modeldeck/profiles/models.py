from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modeldeck.protocol import CapabilitySet, GenerationFamily, LifecycleClass

SAFE_ALIAS = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
ALLOWED_RUNTIMES = {"mock", "transformers-rocm", "text-diffusion-transformers-rocm"}


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

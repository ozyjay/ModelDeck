from __future__ import annotations

import re
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modeldeck.protocol import CapabilitySet, GenerationFamily, LifecycleClass
from modeldeck.runtime_trust import TRUSTED_RUNTIME_IDS

SAFE_ALIAS = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
ALLOWED_RUNTIMES = TRUSTED_RUNTIME_IDS


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    model_id: str
    revision: str
    artifact_model_id: str | None = None
    artifact_revision: str | None = None
    alias: str
    generation_family: GenerationFamily
    preferred_runtime: str
    runtime_template_id: str | None = None
    runtime_template_version: str | None = None
    lifecycle: LifecycleClass
    port: int = Field(ge=1024, le=65535)
    local_files_only: bool = True
    trust_remote_code: bool = False
    dtype: str = "auto"
    capabilities: CapabilitySet
    settings: dict[str, int | float | str | bool] = Field(default_factory=dict)

    @field_validator("alias")
    @classmethod
    def safe_identifier(cls, value: str) -> str:
        if not SAFE_ALIAS.fullmatch(value):
            raise ValueError("must use lowercase letters, digits, and hyphens")
        return value

    @field_validator("id")
    @classmethod
    def safe_internal_identifier(cls, value: str) -> str:
        if SAFE_ALIAS.fullmatch(value):
            return value
        try:
            UUID(value)
        except ValueError as error:
            raise ValueError("must be a legacy identifier or UUID") from error
        return value

    @field_validator("preferred_runtime")
    @classmethod
    def allowlisted_runtime(cls, value: str) -> str:
        if value not in ALLOWED_RUNTIMES:
            raise ValueError("runtime is not allowlisted")
        return value

    @model_validator(mode="after")
    def generation_contract(self) -> ModelProfile:
        if (self.artifact_model_id is None) != (self.artifact_revision is None):
            raise ValueError("artifact model and revision must be supplied together")
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
        if self.generation_family == GenerationFamily.VISION_LANGUAGE and not (
            self.capabilities.image_input and self.capabilities.structured_output
        ):
            raise ValueError(
                "Scene-compatible vision-language profiles must advertise image input and structured output"
            )
        if self.generation_family == GenerationFamily.SPEECH_CONVERSATION and not (
            self.capabilities.audio_input and self.capabilities.audio_output and self.capabilities.full_duplex
        ):
            raise ValueError("speech-conversation profiles must advertise full-duplex audio")
        return self

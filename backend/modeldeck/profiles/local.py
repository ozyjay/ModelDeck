from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from modeldeck.gemma4_settings import DEFAULT_VISUAL_TOKEN_BUDGET, VisualTokenBudget
from modeldeck.registry import (
    RuntimeTemplateRegistration,
    runtime_template_registrations,
)
from modeldeck.speechshift import QWEN_TTS_LANGUAGES, QWEN_TTS_VOICES, SPEECHSHIFT_MODEL_SPECS

from .models import ModelProfile

LOCAL_PORT_RANGE = range(8630, 8700)


class LocalProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=3, max_length=256)
    revision: str = Field(min_length=1, max_length=128)
    alias: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    profile_name: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{1,62}$")
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    lifecycle: Literal["resident", "on-demand", "exclusive"] = "on-demand"
    context_length: int = Field(default=2048, ge=256, le=32768)
    maximum_new_tokens: int = Field(default=128, ge=1, le=512)
    maximum_denoising_steps: int = Field(default=24, ge=1, le=48)
    visual_token_budget: VisualTokenBudget = DEFAULT_VISUAL_TOKEN_BUDGET
    artifact_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{1,62}$")
    runtime_template_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9-]{1,62}$",
    )


def create_local_autoregressive_profile(
    request: LocalProfileRequest,
    *,
    cache_root: Path,
    port: int,
) -> ModelProfile:
    return create_local_profile(
        request,
        cache_root=cache_root,
        port=port,
        configuration_support="autoregressive-transformers",
    )


def create_local_profile(
    request: LocalProfileRequest,
    *,
    cache_root: Path,
    port: int,
    configuration_support: str,
    checkpoint_dir: Path | None = None,
    base_model_id: str | None = None,
    base_model_revision: str | None = None,
    artifact_path: Path | None = None,
    template_registrations: dict[str, RuntimeTemplateRegistration] | None = None,
) -> ModelProfile:
    registration = (template_registrations or runtime_template_registrations()).get(configuration_support)
    if registration is None:
        raise ValueError("No allowlisted local worker supports this model architecture")
    template = registration.template
    if template.uses_base_model_identity and (
        checkpoint_dir is None or base_model_id is None or base_model_revision is None
    ):
        raise ValueError("ModelDeck Q4 release identity is incomplete")
    settings = dict(template.settings)
    if request.model_id == "google/gemma-4-12B-it":
        settings["hardware_verification_required"] = True
    if template.generation_family.value in {
        "autoregressive",
        "vision-language",
        "text-diffusion",
        "text-translation",
    }:
        settings["maximum_new_tokens"] = request.maximum_new_tokens
    if template.generation_family.value == "autoregressive":
        settings["context_length"] = request.context_length
    elif template.generation_family.value == "vision-language":
        settings["context_length"] = request.context_length
        settings["visual_token_budget"] = request.visual_token_budget
    elif template.generation_family.value == "text-diffusion":
        settings["maximum_denoising_steps"] = request.maximum_denoising_steps
    elif template.generation_family.value == "text-translation":
        spec = SPEECHSHIFT_MODEL_SPECS.get(request.model_id)
        if spec is None or spec.source_language is None or spec.target_language is None:
            raise ValueError("The translation direction is not allowlisted")
        settings["source_language"] = spec.source_language
        settings["target_language"] = spec.target_language
    elif template.generation_family.value == "speech-synthesis":
        settings["allowed_voices"] = ",".join(QWEN_TTS_VOICES)
        settings["allowed_languages"] = ",".join(QWEN_TTS_LANGUAGES)
    if template.cache_setting == "artifact_path" and artifact_path is None:
        raise ValueError("This runtime requires a discovered allowlisted artefact")
    selected_path = (
        artifact_path if template.cache_setting == "artifact_path" else checkpoint_dir or cache_root
    )
    settings[template.cache_setting] = str(selected_path)
    if template.include_cache_root:
        settings["cache_root"] = str(cache_root)
    return ModelProfile(
        id=f"local-{request.profile_name or request.alias}",
        model_id=base_model_id if template.uses_base_model_identity else request.model_id,
        revision=base_model_revision if template.uses_base_model_identity else request.revision,
        artifact_model_id=request.model_id if template.uses_base_model_identity else None,
        artifact_revision=request.revision if template.uses_base_model_identity else None,
        alias=request.alias,
        generation_family=template.generation_family,
        preferred_runtime=template.runtime,
        runtime_template_id=template.id,
        runtime_template_version=registration.package.version,
        lifecycle=template.lifecycle or request.lifecycle,
        port=port,
        local_files_only=True,
        trust_remote_code=False,
        dtype=template.dtype or request.dtype,
        capabilities=template.capabilities.model_copy(deep=True),
        settings=settings,
    )


LocalAutoregressiveProfileRequest = LocalProfileRequest

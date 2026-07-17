from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from modeldeck.profiles.models import ALLOWED_RUNTIMES, ModelProfile
from modeldeck.protocol import CapabilitySet, GenerationFamily, LifecycleClass


class RuntimeTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    runtime: str
    generation_family: GenerationFamily
    capabilities: CapabilitySet
    settings: dict[str, int | float | str | bool] = Field(default_factory=dict)
    cache_setting: Literal["cache_root", "q4_checkpoint_dir"]
    include_cache_root: bool = False
    lifecycle: LifecycleClass | None = None
    dtype: Literal["float16", "bfloat16"] | None = None
    uses_base_model_identity: bool = False

    @model_validator(mode="after")
    def trusted_runtime(self) -> RuntimeTemplate:
        if self.runtime not in ALLOWED_RUNTIMES:
            raise ValueError("runtime template does not map to a trusted worker implementation")
        return self


class ReservedAlias(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    display_name: str = Field(min_length=1, max_length=80)
    providers: list[str] = Field(min_length=1)
    selection: Literal["ordered", "explicit"] = "ordered"
    default_provider: str | None = None
    required_generation_family: GenerationFamily | None = None
    required_capabilities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def selection_contract(self) -> ReservedAlias:
        if self.selection == "explicit" and self.default_provider is None:
            raise ValueError("explicit aliases require a default provider")
        if self.default_provider is not None and self.default_provider not in self.providers:
            raise ValueError("default provider must be in the packaged provider list")
        unknown = set(self.required_capabilities) - set(CapabilitySet.model_fields)
        if unknown:
            raise ValueError(f"unknown required capabilities: {', '.join(sorted(unknown))}")
        return self

    def accepts(self, profile: ModelProfile) -> bool:
        if self.required_generation_family is not None and (
            profile.generation_family != self.required_generation_family
        ):
            return False
        return all(getattr(profile.capabilities, name) is True for name in self.required_capabilities)


def _document(name: str) -> dict:
    resource = files("modeldeck").joinpath("registry_data", name)
    return json.loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def runtime_templates() -> dict[str, RuntimeTemplate]:
    document = _document("runtime_templates.json")
    _check_header(document, "modeldeck-runtime-templates")
    templates = [RuntimeTemplate.model_validate(item) for item in document.get("templates", [])]
    return _unique(templates, "runtime template")


@lru_cache(maxsize=1)
def seed_profiles() -> dict[str, ModelProfile]:
    document = _document("model_profiles.json")
    _check_header(document, "modeldeck-profile-seeds")
    profiles = [ModelProfile.model_validate(item) for item in document.get("profiles", [])]
    result = _unique(profiles, "seed profile")
    registered_runtimes = {template.runtime for template in runtime_templates().values()} | {"mock"}
    for profile in result.values():
        if profile.preferred_runtime not in registered_runtimes:
            raise ValueError(f"seed profile {profile.id} uses an unregistered runtime template")
    return result


@lru_cache(maxsize=1)
def reserved_aliases() -> dict[str, ReservedAlias]:
    document = _document("reserved_aliases.json")
    _check_header(document, "modeldeck-reserved-aliases")
    aliases = [ReservedAlias.model_validate(item) for item in document.get("aliases", [])]
    result = _unique(aliases, "reserved alias")
    profiles = seed_profiles()
    for contract in result.values():
        for profile_id in contract.providers:
            if profile_id not in profiles:
                raise ValueError(f"reserved alias {contract.id} references unknown profile {profile_id}")
        if contract.default_provider and not contract.accepts(profiles[contract.default_provider]):
            raise ValueError(f"reserved alias {contract.id} rejects its default provider")
    return result


def _check_header(document: dict, expected_format: str) -> None:
    if document.get("format") != expected_format or document.get("version") != 1:
        raise ValueError(f"unsupported {expected_format} registry format or version")


def _unique[T: BaseModel](items: list[T], label: str) -> dict[str, T]:
    result: dict[str, T] = {}
    for item in items:
        item_id = str(item.id)  # type: ignore[attr-defined]
        if item_id in result:
            raise ValueError(f"duplicate {label}: {item_id}")
        result[item_id] = item
    if not result:
        raise ValueError(f"{label} registry is empty")
    return result

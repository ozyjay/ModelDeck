from __future__ import annotations

import json
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from modeldeck.profiles.models import ModelProfile
from modeldeck.protocol import CapabilitySet, GenerationFamily, LifecycleClass
from modeldeck.runtime_trust import TRUSTED_RUNTIME_IMPLEMENTATIONS

TRUST_DIRECTORY_NAME = "trusted-runtime-manifests"
TRUST_REGISTRY_NAME = "trust.json"
BOUNDED_INTEGER_SETTINGS = {
    "top_k": (1, 100),
    "context_length": (256, 32768),
    "maximum_new_tokens": (1, 512),
    "maximum_denoising_steps": (1, 48),
    "startup_timeout_seconds": (1, 1800),
    "warmup_timeout_seconds": (1, 900),
    "generation_timeout_seconds": (1, 900),
    "sample_rate_hz": (8000, 48000),
    "channels": (1, 2),
    "maximum_sessions": (1, 8),
    "maximum_buffer_ms": (10, 5000),
}


class RuntimeTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    display_name: str = Field(min_length=1, max_length=80)
    runtime: str
    generation_family: GenerationFamily
    capabilities: CapabilitySet
    settings: dict[str, int | float | str | bool] = Field(default_factory=dict)
    cache_setting: Literal["cache_root", "q4_checkpoint_dir", "artifact_path"]
    include_cache_root: bool = False
    lifecycle: LifecycleClass | None = None
    dtype: Literal["float16", "bfloat16"] | None = None
    uses_base_model_identity: bool = False

    @model_validator(mode="after")
    def trusted_runtime(self) -> RuntimeTemplate:
        implementation = TRUSTED_RUNTIME_IMPLEMENTATIONS.get(self.runtime)
        if implementation is None:
            raise ValueError("runtime template does not map to a trusted worker implementation")
        if self.generation_family != implementation.generation_family:
            raise ValueError("runtime template generation family does not match its trusted implementation")
        if self.cache_setting not in implementation.cache_settings:
            raise ValueError("runtime template cache binding does not match its trusted implementation")
        enabled_capabilities = {
            name
            for name, value in self.capabilities.model_dump().items()
            if value is not False and value is not None
        }
        unsupported_capabilities = enabled_capabilities - implementation.capabilities
        if unsupported_capabilities:
            raise ValueError(
                "runtime template advertises capabilities not provided by its trusted implementation: "
                + ", ".join(sorted(unsupported_capabilities))
            )
        unknown_settings = set(self.settings) - implementation.template_settings
        if unknown_settings:
            raise ValueError(
                "runtime template contains settings not accepted by its trusted implementation: "
                + ", ".join(sorted(unknown_settings))
            )
        for name, (minimum, maximum) in BOUNDED_INTEGER_SETTINGS.items():
            if name not in self.settings:
                continue
            value = self.settings[name]
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(
                    f"runtime template setting {name} must be an integer from {minimum} to {maximum}"
                )
        if self.settings.get("execution_preset") not in {None, "vulkan-full"}:
            raise ValueError("runtime template execution preset is not trusted")
        for name in ("hardware_verification_required", "hsa_preload_evidence"):
            if name in self.settings and not isinstance(self.settings[name], bool):
                raise ValueError(f"runtime template setting {name} must be boolean")
        return self


class RuntimeManifestIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9.-]+)?$")
    display_name: str = Field(min_length=1, max_length=80)
    publisher: str = Field(min_length=1, max_length=80)


class RuntimeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["modeldeck-runtime-templates"]
    version: Literal[1]
    package: RuntimeManifestIdentity
    templates: list[RuntimeTemplate] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_templates(self) -> RuntimeManifest:
        identifiers = [template.id for template in self.templates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("runtime manifest contains duplicate template IDs")
        return self


class TrustedManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(pattern=r"^[a-z][a-z0-9.-]{1,126}\.json$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class RuntimeTrustRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["modeldeck-runtime-trust"] = "modeldeck-runtime-trust"
    version: Literal[1] = 1
    manifests: list[TrustedManifestEntry] = Field(default_factory=list)


class RuntimeTemplateRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    template: RuntimeTemplate
    package: RuntimeManifestIdentity
    source: Literal["packaged", "trusted-local"]
    digest: str


class ReservedAlias(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    display_name: str = Field(min_length=1, max_length=80)
    providers: list[str] = Field(default_factory=list)
    selection: Literal["ordered", "explicit"] = "ordered"
    default_provider: str | None = None
    required_generation_family: GenerationFamily | None = None
    required_capabilities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def selection_contract(self) -> ReservedAlias:
        if self.selection == "ordered" and not self.providers:
            raise ValueError("ordered aliases require at least one packaged provider")
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


def runtime_template_registrations(
    data_dir: Path | None = None,
) -> dict[str, RuntimeTemplateRegistration]:
    packaged_bytes = files("modeldeck").joinpath("registry_data", "runtime_templates.json").read_bytes()
    manifests = [
        _manifest_registrations(packaged_bytes, source="packaged"),
        *(_trusted_local_registrations(data_dir) if data_dir is not None else []),
    ]
    result: dict[str, RuntimeTemplateRegistration] = {}
    for registrations in manifests:
        for registration in registrations:
            template_id = registration.template.id
            if template_id in result:
                raise ValueError(f"duplicate runtime template: {template_id}")
            result[template_id] = registration
    if not result:
        raise ValueError("runtime template registry is empty")
    return result


def runtime_templates(data_dir: Path | None = None) -> dict[str, RuntimeTemplate]:
    return {
        template_id: registration.template
        for template_id, registration in runtime_template_registrations(data_dir).items()
    }


def install_runtime_manifest(source: Path, data_dir: Path, expected_sha256: str) -> Path:
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("expected SHA-256 must be 64 lowercase hexadecimal characters")
    content = source.read_bytes()
    digest = sha256(content).hexdigest()
    if digest != expected_sha256:
        raise ValueError("runtime manifest SHA-256 does not match the operator-approved digest")
    manifest = RuntimeManifest.model_validate_json(content)
    if manifest.package.id == "modeldeck-core":
        raise ValueError("the packaged ModelDeck runtime manifest cannot be installed locally")

    trust_dir = data_dir / TRUST_DIRECTORY_NAME
    trust_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{manifest.package.id}-{manifest.package.version}.json"
    target = trust_dir / filename
    if target.exists() and target.read_bytes() != content:
        raise ValueError("that runtime package version is already installed with different content")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_bytes(content)
    temporary.replace(target)

    registry_path = trust_dir / TRUST_REGISTRY_NAME
    registry = _read_trust_registry(registry_path)
    entries = [entry for entry in registry.manifests if entry.filename != filename]
    entries.append(TrustedManifestEntry(filename=filename, sha256=digest))
    updated = RuntimeTrustRegistry(manifests=sorted(entries, key=lambda entry: entry.filename))
    temporary_registry = registry_path.with_suffix(".json.tmp")
    temporary_registry.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary_registry.replace(registry_path)
    return target


def _manifest_registrations(
    content: bytes,
    *,
    source: Literal["packaged", "trusted-local"],
) -> list[RuntimeTemplateRegistration]:
    manifest = RuntimeManifest.model_validate_json(content)
    digest = sha256(content).hexdigest()
    return [
        RuntimeTemplateRegistration(
            template=template,
            package=manifest.package,
            source=source,
            digest=digest,
        )
        for template in manifest.templates
    ]


def _trusted_local_registrations(data_dir: Path) -> list[list[RuntimeTemplateRegistration]]:
    trust_dir = data_dir / TRUST_DIRECTORY_NAME
    registry_path = trust_dir / TRUST_REGISTRY_NAME
    if not registry_path.exists():
        return []
    registry = _read_trust_registry(registry_path)
    registrations: list[list[RuntimeTemplateRegistration]] = []
    for entry in registry.manifests:
        path = trust_dir / entry.filename
        if not path.is_file():
            raise ValueError(f"trusted runtime manifest is missing: {entry.filename}")
        content = path.read_bytes()
        if sha256(content).hexdigest() != entry.sha256:
            raise ValueError(f"trusted runtime manifest digest changed: {entry.filename}")
        registrations.append(_manifest_registrations(content, source="trusted-local"))
    return registrations


def _read_trust_registry(path: Path) -> RuntimeTrustRegistry:
    if not path.exists():
        return RuntimeTrustRegistry()
    return RuntimeTrustRegistry.model_validate_json(path.read_text(encoding="utf-8"))


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

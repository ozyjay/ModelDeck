from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modeldeck.capabilities import (
    capability_evidence_status,
    capability_id_for_contract,
    worker_cache_identity,
)
from modeldeck.prefix_cache import supports_application_managed_prefix_cache
from modeldeck.profiles import ModelProfile
from modeldeck.protocol_contracts import PROTOCOL_CONTRACTS

EventQualification = Literal["compatible", "tested-working"]


def _uuid(value: str) -> str:
    try:
        UUID(value)
    except ValueError as error:
        raise ValueError("must be a UUID") from error
    return value


class WorkerDefinition(BaseModel):
    """Persisted operator configuration for one trusted local worker."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = Field(min_length=1, max_length=80)
    model_id: str
    revision: str
    artifact_model_id: str | None = None
    artifact_revision: str | None = None
    generation_family: str
    runtime: str
    runtime_template_id: str | None = None
    runtime_template_version: str | None = None
    lifecycle: Literal["resident", "on-demand", "exclusive"]
    port: int = Field(ge=1024, le=65535)
    dtype: str
    capabilities: dict[str, bool | str]
    settings: dict[str, int | float | str | bool] = Field(default_factory=dict)
    capability_policy_version: int | None = None
    archived: bool = False

    _valid_id = field_validator("id")(_uuid)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("cannot be blank")
        return cleaned

    @model_validator(mode="after")
    def normalise_legacy_settings(self) -> WorkerDefinition:
        thinking_default = {
            # The dedicated text-chat adapter has always run without thinking.
            # Persisted Workers from before this setting became explicit must
            # retain that immutable runtime policy when they are reloaded.
            ("qwen35-chat-transformers-rocm", "qwen35-chat-transformers-rocm"): "disabled",
            ("qwen38-llamacpp-vulkan", "qwen38-llamacpp-q8-mtp-vulkan"): "adaptive",
            (
                "qwen38-llamacpp-vulkan",
                "qwen38-llamacpp-q8-mtp-vulkan-no-thinking",
            ): "disabled",
            ("qwen35-llamacpp-vulkan", "qwen35-llamacpp-q8-vulkan"): "disabled",
            (
                "qwen35-llamacpp-vulkan",
                "qwen35-llamacpp-q8-vulkan-adaptive",
            ): "adaptive",
            ("qwen35-llamacpp-vulkan", "qwen35-local-q8-vulkan"): "disabled",
            (
                "qwen35-llamacpp-vulkan",
                "qwen35-local-q8-vulkan-adaptive",
            ): "adaptive",
        }.get((self.runtime, self.runtime_template_id))
        if thinking_default is not None and "thinking_mode" not in self.settings:
            # Workers created before thinking policy became explicit retain the
            # immutable default of their exact trusted runtime template.
            self.settings["thinking_mode"] = thinking_default

        eligible = (
            self.generation_family == "autoregressive"
            and self.runtime == "transformers-rocm"
            and supports_application_managed_prefix_cache(self.model_id)
        )
        enabled = self.settings.get("prefix_cache_enabled") is True
        if enabled and not eligible:
            raise ValueError("prefix caching is not supported by this Worker")
        self.capabilities["prefix_caching"] = "application-managed" if eligible else "unsupported"
        self.capabilities["prefix_cache_enabled"] = enabled if eligible else False
        return self

    @classmethod
    def from_profile(cls, profile: ModelProfile, *, name: str) -> WorkerDefinition:
        return cls(
            id=profile.id,
            name=name,
            model_id=profile.model_id,
            revision=profile.revision,
            artifact_model_id=profile.artifact_model_id,
            artifact_revision=profile.artifact_revision,
            generation_family=profile.generation_family.value,
            runtime=profile.preferred_runtime,
            runtime_template_id=profile.runtime_template_id,
            runtime_template_version=profile.runtime_template_version,
            lifecycle=profile.lifecycle.value,
            port=profile.port,
            dtype=profile.dtype,
            capabilities=profile.capabilities.model_dump(mode="json"),
            settings=profile.settings,
            capability_policy_version=4,
        )

    def to_profile(self) -> ModelProfile:
        # Alias remains an internal compatibility field while the process controller is
        # migrated. It is not persisted or exposed as a public route name.
        return ModelProfile.model_validate(
            {
                "id": self.id,
                "model_id": self.model_id,
                "revision": self.revision,
                "artifact_model_id": self.artifact_model_id,
                "artifact_revision": self.artifact_revision,
                "alias": f"worker-{self.id[:8]}",
                "generation_family": self.generation_family,
                "preferred_runtime": self.runtime,
                "runtime_template_id": self.runtime_template_id,
                "runtime_template_version": self.runtime_template_version,
                "lifecycle": self.lifecycle,
                "port": self.port,
                "local_files_only": True,
                "trust_remote_code": False,
                "dtype": self.dtype,
                "capabilities": self.capabilities,
                "settings": self.settings,
            }
        )


class CapabilityBinding(BaseModel):
    """One public, profile-local capability backed by trusted local Workers."""

    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str = Field(min_length=1, max_length=80)
    public_name: str = Field(pattern=r"^[a-z][a-z0-9._-]{1,127}$")
    protocol_contract: str
    worker_ids: list[str] = Field(min_length=1)

    _valid_id = field_validator("id")(_uuid)

    @model_validator(mode="after")
    def trusted_contract_and_unique_workers(self) -> CapabilityBinding:
        if self.protocol_contract not in PROTOCOL_CONTRACTS:
            raise ValueError("capability protocol contract is not trusted")
        if len(self.worker_ids) != len(set(self.worker_ids)):
            raise ValueError("capability workers must be unique")
        for worker_id in self.worker_ids:
            _uuid(worker_id)
        return self


class RoutingProfile(BaseModel):
    """A revisioned, atomically published set of local capabilities."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    qualification: EventQualification = "compatible"
    capabilities: list[CapabilityBinding] = Field(default_factory=list)

    _valid_id = field_validator("id")(_uuid)

    @model_validator(mode="after")
    def unique_capabilities(self) -> RoutingProfile:
        capability_ids = [capability.id for capability in self.capabilities]
        public_names = [capability.public_name.casefold() for capability in self.capabilities]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("capability identifiers must be unique")
        if len(public_names) != len(set(public_names)):
            duplicates = {public_name for public_name in public_names if public_names.count(public_name) > 1}
            conflicting_routes = [
                f"'{capability.display_name}' ({capability.public_name})"
                for capability in self.capabilities
                if capability.public_name.casefold() in duplicates
            ]
            raise ValueError(
                "API Model IDs must be unique within a Routing Profile; conflicting capabilities: "
                + ", ".join(conflicting_routes)
            )
        return self


def validate_routing_profile(
    definition: RoutingProfile,
    workers: Iterable[WorkerDefinition],
    compatibility_tests: Iterable[Mapping[str, Any]],
    capability_policy: Mapping[tuple[str, str, str], bool] | None = None,
) -> dict[str, Any]:
    by_id = {worker.id: worker for worker in workers if not worker.archived}
    tests = list(compatibility_tests)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    capabilities: list[dict[str, Any]] = []
    if not definition.capabilities:
        warnings.append({"message": "This Routing Profile publishes no capabilities"})
    for capability in definition.capabilities:
        contract = PROTOCOL_CONTRACTS[capability.protocol_contract]
        model_capability_id = capability_id_for_contract(capability.protocol_contract)
        resolved = []
        for index, worker_id in enumerate(capability.worker_ids):
            worker = by_id.get(worker_id)
            messages: list[str] = []
            if worker is None:
                messages.append("Unknown or archived Worker")
            else:
                compatible_families = contract.compatible_generation_families or (contract.generation_family,)
                if worker.generation_family not in {family.value for family in compatible_families}:
                    expected_families = ", ".join(family.value for family in compatible_families)
                    messages.append(f"Requires one of: {expected_families}; got {worker.generation_family}")
                missing = [
                    capability
                    for capability in contract.required_capabilities
                    if worker.capabilities.get(capability) is not True
                ]
                if missing:
                    messages.append(f"Missing capabilities: {', '.join(missing)}")
                mismatched_settings = [
                    f"{name}={expected}"
                    for name, expected in contract.required_worker_settings.items()
                    if worker.settings.get(name) != expected
                ]
                if mismatched_settings:
                    messages.append("Requires Worker settings: " + ", ".join(mismatched_settings))
                if capability_policy is not None and model_capability_id is not None:
                    model_id, revision = worker_cache_identity(worker.model_dump(mode="json"))
                    if not capability_policy.get((model_id, revision, model_capability_id), False):
                        messages.append(
                            f"Allow the {model_capability_id} capability for this cached Model revision"
                        )
                if (
                    definition.qualification == "tested-working"
                    and model_capability_id is not None
                    and not _has_matching_success(worker, model_capability_id, tests)
                ):
                    messages.append("No matching tested-working evidence is recorded")
            for message in messages:
                errors.append({"capability_id": capability.id, "worker_id": worker_id, "message": message})
            resolved.append(
                {
                    "worker_id": worker_id,
                    "role": "primary" if index == 0 else "backup",
                    "valid": not messages,
                }
            )
        capabilities.append(
            {"capability_id": capability.id, "public_name": capability.public_name, "workers": resolved}
        )
    return {"valid": not errors, "errors": errors, "warnings": warnings, "capabilities": capabilities}


def routing_snapshot(definition: RoutingProfile, revision: int) -> dict[str, Any]:
    return {
        "format": "modeldeck-routing-profile",
        "version": 3,
        "profile_id": definition.id,
        "profile_name": definition.name,
        "revision": revision,
        "capabilities": [
            {
                "capability_id": capability.id,
                "display_name": capability.display_name,
                "public_name": capability.public_name,
                "protocol_contract": capability.protocol_contract,
                "worker_ids": list(capability.worker_ids),
            }
            for capability in definition.capabilities
        ],
    }


def _has_matching_success(
    worker: WorkerDefinition,
    capability_id: str,
    tests: Iterable[Mapping[str, Any]],
) -> bool:
    status, _evidence_id = capability_evidence_status(worker.model_dump(mode="json"), capability_id, tests)
    return status in {"qualified", "legacy"}

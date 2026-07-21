from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    archived: bool = False

    _valid_id = field_validator("id")(_uuid)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("cannot be blank")
        return cleaned

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


class DemoDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = Field(min_length=1, max_length=80)
    route_ids: list[str] = Field(default_factory=list)

    _valid_id = field_validator("id")(_uuid)


class RouteDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str = Field(min_length=1, max_length=80)
    public_name: str = Field(pattern=r"^[a-z][a-z0-9._-]{1,127}$")
    protocol_contract: str
    worker_ids: list[str] = Field(min_length=1)

    _valid_id = field_validator("id")(_uuid)

    @model_validator(mode="after")
    def trusted_contract_and_unique_workers(self) -> RouteDefinition:
        if self.protocol_contract not in PROTOCOL_CONTRACTS:
            raise ValueError("route protocol contract is not trusted")
        if len(self.worker_ids) != len(set(self.worker_ids)):
            raise ValueError("route workers must be unique")
        for worker_id in self.worker_ids:
            _uuid(worker_id)
        return self


class EventDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    qualification: EventQualification = "compatible"
    demos: list[DemoDefinition] = Field(default_factory=list)
    routes: list[RouteDefinition] = Field(default_factory=list)

    _valid_id = field_validator("id")(_uuid)

    @model_validator(mode="after")
    def unique_and_referenced_children(self) -> EventDefinition:
        demo_ids = [demo.id for demo in self.demos]
        route_ids = [route.id for route in self.routes]
        public_names = [route.public_name.casefold() for route in self.routes]
        if len(demo_ids) != len(set(demo_ids)):
            raise ValueError("demo identifiers must be unique")
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route identifiers must be unique")
        if len(public_names) != len(set(public_names)):
            duplicates = {public_name for public_name in public_names if public_names.count(public_name) > 1}
            conflicting_routes = [
                f"'{route.display_name}' ({route.public_name})"
                for route in self.routes
                if route.public_name.casefold() in duplicates
            ]
            raise ValueError(
                "API Model IDs must be unique within an Event; conflicting Routes: "
                + ", ".join(conflicting_routes)
            )
        known_routes = set(route_ids)
        for demo in self.demos:
            if len(demo.route_ids) != len(set(demo.route_ids)):
                raise ValueError(f"demo '{demo.name}' contains a Route more than once")
            unknown = set(demo.route_ids) - known_routes
            if unknown:
                raise ValueError(f"demo '{demo.name}' references unknown Routes")
        return self


def validate_event(
    definition: EventDefinition,
    workers: Iterable[WorkerDefinition],
    compatibility_tests: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {worker.id: worker for worker in workers if not worker.archived}
    tests = list(compatibility_tests)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    routes: list[dict[str, Any]] = []
    used_routes = {route_id for demo in definition.demos for route_id in demo.route_ids}
    if not definition.routes:
        warnings.append({"message": "This Event publishes no Routes"})
    for demo in definition.demos:
        if not demo.route_ids:
            warnings.append({"demo_id": demo.id, "message": f"Demo '{demo.name}' uses no Routes"})
    for route in definition.routes:
        if route.id not in used_routes:
            warnings.append(
                {"route_id": route.id, "message": f"Route '{route.display_name}' is not used by a Demo"}
            )
        contract = PROTOCOL_CONTRACTS[route.protocol_contract]
        resolved = []
        for index, worker_id in enumerate(route.worker_ids):
            worker = by_id.get(worker_id)
            messages: list[str] = []
            if worker is None:
                messages.append("Unknown or archived Worker")
            else:
                if worker.generation_family != contract.generation_family.value:
                    messages.append(
                        f"Requires {contract.generation_family.value}, got {worker.generation_family}"
                    )
                missing = [
                    capability
                    for capability in contract.required_capabilities
                    if worker.capabilities.get(capability) is not True
                ]
                if missing:
                    messages.append(f"Missing capabilities: {', '.join(missing)}")
                if definition.qualification == "tested-working" and not _has_matching_success(worker, tests):
                    messages.append("No matching tested-working evidence is recorded")
            for message in messages:
                errors.append({"route_id": route.id, "worker_id": worker_id, "message": message})
            resolved.append(
                {
                    "worker_id": worker_id,
                    "role": "primary" if index == 0 else "backup",
                    "valid": not messages,
                }
            )
        routes.append({"route_id": route.id, "public_name": route.public_name, "workers": resolved})
    return {"valid": not errors, "errors": errors, "warnings": warnings, "routes": routes}


def routing_snapshot(definition: EventDefinition, revision: int) -> dict[str, Any]:
    return {
        "format": "modeldeck-event-routing",
        "version": 2,
        "event_id": definition.id,
        "event_name": definition.name,
        "revision": revision,
        "routes": [
            {
                "route_id": route.id,
                "display_name": route.display_name,
                "public_name": route.public_name,
                "protocol_contract": route.protocol_contract,
                "worker_ids": list(route.worker_ids),
            }
            for route in definition.routes
        ],
    }


def _has_matching_success(worker: WorkerDefinition, tests: Iterable[Mapping[str, Any]]) -> bool:
    return any(
        test.get("result") == "tested-working"
        and test.get("evidence", {}).get("model_id") == worker.model_id
        and test.get("evidence", {}).get("model_revision") == worker.revision
        and test.get("evidence", {}).get("runtime") == worker.runtime
        for test in tests
    )

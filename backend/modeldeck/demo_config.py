from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from modeldeck.profiles import ModelProfile
from modeldeck.protocol import GenerationFamily
from modeldeck.registry import reserved_aliases

QualificationPolicy = Literal["registered", "tested-working-recorded"]
FallbackPolicy = Literal["none", "ordered", "mock-visible", "structured-unavailable"]


class DemoAdapter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    display_name: str
    generation_family: GenerationFamily
    required_capabilities: tuple[str, ...] = ()
    surfaces: tuple[str, ...]


DEMO_ADAPTERS = {
    adapter.id: adapter
    for adapter in (
        DemoAdapter(
            id="openai-chat-v1",
            display_name="OpenAI-compatible chat",
            generation_family=GenerationFamily.AUTOREGRESSIVE,
            required_capabilities=("chat",),
            surfaces=("POST /v1/chat/completions",),
        ),
        DemoAdapter(
            id="openai-completions-v1",
            display_name="OpenAI-compatible completions",
            generation_family=GenerationFamily.AUTOREGRESSIVE,
            required_capabilities=("completions",),
            surfaces=("POST /v1/completions",),
        ),
        DemoAdapter(
            id="native-ar-trace-v1",
            display_name="Native autoregressive trace",
            generation_family=GenerationFamily.AUTOREGRESSIVE,
            required_capabilities=("top_k_trace",),
            surfaces=("POST /native/autoregressive/trace",),
        ),
        DemoAdapter(
            id="scene-analysis-v1",
            display_name="Scene analysis",
            generation_family=GenerationFamily.VISION_LANGUAGE,
            required_capabilities=("image_input", "structured_output"),
            surfaces=("POST /v1/chat/completions", "POST /v1/vision/analyse"),
        ),
        DemoAdapter(
            id="text-diffusion-v1",
            display_name="Text diffusion",
            generation_family=GenerationFamily.TEXT_DIFFUSION,
            required_capabilities=("iterative_refinement", "intermediate_frames"),
            surfaces=("POST /v1/refine", "POST /v1/diffuse", "GET/POST /v1/jobs/*"),
        ),
        DemoAdapter(
            id="speech-conversation-v1",
            display_name="Speech conversation",
            generation_family=GenerationFamily.SPEECH_CONVERSATION,
            required_capabilities=("audio_input", "audio_output", "full_duplex"),
            surfaces=("WS /v1/speech/conversations",),
        ),
    )
}


class DemoApplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    display_name: str = Field(min_length=1, max_length=80)


class DeploymentBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    priority: int = Field(default=100, ge=0, le=10_000)


class DemoRouteContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    demo_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    display_name: str = Field(min_length=1, max_length=80)
    adapter_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    public_model: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    qualification_policy: QualificationPolicy = "registered"
    fallback_policy: FallbackPolicy = "structured-unavailable"
    providers: list[DeploymentBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def known_adapter_and_unique_providers(self) -> DemoRouteContract:
        if self.adapter_id not in DEMO_ADAPTERS:
            raise ValueError("route adapter is not allowlisted")
        provider_ids = [binding.deployment_id for binding in self.providers]
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("route provider deployments must be unique")
        return self


class DemoSetDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,62}$")
    display_name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    demos: list[DemoApplication] = Field(default_factory=list)
    routes: list[DemoRouteContract] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_and_referenced_children(self) -> DemoSetDefinition:
        demo_ids = [demo.id for demo in self.demos]
        route_ids = [route.id for route in self.routes]
        public_models = [route.public_model for route in self.routes]
        if len(demo_ids) != len(set(demo_ids)):
            raise ValueError("demo identifiers must be unique")
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route identifiers must be unique")
        if len(public_models) != len(set(public_models)):
            raise ValueError("public model aliases must be unique within a demo set")
        unknown_demos = {route.demo_id for route in self.routes} - set(demo_ids)
        if unknown_demos:
            raise ValueError(f"routes reference unknown demos: {', '.join(sorted(unknown_demos))}")
        return self


def profile_has_recorded_success(profile: ModelProfile, tests: Iterable[Mapping]) -> bool:
    return any(
        test.get("result") == "tested-working"
        and test.get("evidence", {}).get("model_id") == profile.model_id
        and test.get("evidence", {}).get("model_revision") == profile.revision
        and test.get("evidence", {}).get("runtime") == profile.preferred_runtime
        for test in tests
    )


def validate_demo_set(
    definition: DemoSetDefinition,
    profiles: Iterable[ModelProfile],
    *,
    registered_ids: set[str],
    allowed_ids: set[str],
    compatibility_tests: Iterable[Mapping],
) -> dict:
    by_id = {profile.id: profile for profile in profiles}
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    route_results = []
    tests = list(compatibility_tests)
    for route in definition.routes:
        adapter = DEMO_ADAPTERS[route.adapter_id]
        resolved = []
        if not route.providers and route.fallback_policy != "structured-unavailable":
            errors.append({"route_id": route.id, "message": "Route has no provider deployments"})
        for binding in sorted(route.providers, key=lambda item: (item.priority, item.deployment_id)):
            profile = by_id.get(binding.deployment_id)
            provider_errors = []
            if profile is None:
                provider_errors.append("Unknown deployment")
            else:
                if profile.id not in registered_ids:
                    provider_errors.append("Deployment is not currently registered")
                if profile.id not in allowed_ids:
                    provider_errors.append("Deployment artefact is disallowed")
                if profile.generation_family != adapter.generation_family:
                    provider_errors.append(
                        f"Requires {adapter.generation_family.value}, got {profile.generation_family.value}"
                    )
                missing = [
                    capability
                    for capability in adapter.required_capabilities
                    if getattr(profile.capabilities, capability) is not True
                ]
                if missing:
                    provider_errors.append(f"Missing capabilities: {', '.join(missing)}")
                if route.qualification_policy == "tested-working-recorded" and not (
                    profile_has_recorded_success(profile, tests)
                ):
                    provider_errors.append("No tested-working evidence is recorded")
                if profile.preferred_runtime == "mock" and route.fallback_policy != "mock-visible":
                    provider_errors.append("Mock providers require the mock-visible fallback policy")
            for message in provider_errors:
                errors.append(
                    {"route_id": route.id, "deployment_id": binding.deployment_id, "message": message}
                )
            resolved.append(
                {
                    "deployment_id": binding.deployment_id,
                    "priority": binding.priority,
                    "valid": not provider_errors,
                }
            )
        if len(route.providers) > 1 and route.fallback_policy in {"none", "structured-unavailable"}:
            warnings.append(
                {"route_id": route.id, "message": "Only the first provider will be used without fallback"}
            )
        route_results.append(
            {
                "route_id": route.id,
                "public_model": route.public_model,
                "adapter": adapter.model_dump(mode="json"),
                "providers": resolved,
            }
        )
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "routes": route_results,
    }


def routing_snapshot(definition: DemoSetDefinition, revision: int) -> dict:
    return {
        "format": "modeldeck-active-demo-routing",
        "version": 1,
        "demo_set_id": definition.id,
        "revision": revision,
        "routes": [
            {
                "route_id": route.id,
                "demo_id": route.demo_id,
                "public_model": route.public_model,
                "adapter_id": route.adapter_id,
                "qualification_policy": route.qualification_policy,
                "fallback_policy": route.fallback_policy,
                "providers": [
                    binding.deployment_id
                    for binding in sorted(
                        route.providers, key=lambda item: (item.priority, item.deployment_id)
                    )
                ],
            }
            for route in definition.routes
        ],
    }


def default_demo_set(profiles: Iterable[ModelProfile]) -> DemoSetDefinition:
    by_id = {profile.id: profile for profile in profiles}
    contracts = reserved_aliases()
    adapter_for_alias = {
        "fast-chat": "openai-chat-v1",
        "token-explainer": "native-ar-trace-v1",
        "qwen-0-5b": "openai-chat-v1",
        "qwen-1-5b": "openai-chat-v1",
        "qwen-3b": "openai-chat-v1",
        "scenechat-vision": "scene-analysis-v1",
        "text-diffusion": "text-diffusion-v1",
        "text-diffusion-bf16": "text-diffusion-v1",
        "repartee-strong": "openai-chat-v1",
        "repartee-speech": "speech-conversation-v1",
    }
    demo_for_alias = {
        "fast-chat": "chat-demos",
        "token-explainer": "token-trail",
        "qwen-0-5b": "chat-demos",
        "qwen-1-5b": "chat-demos",
        "qwen-3b": "chat-demos",
        "scenechat-vision": "scenechat",
        "text-diffusion": "text-diffusion-demo",
        "text-diffusion-bf16": "text-diffusion-demo",
        "repartee-strong": "repartee",
        "repartee-speech": "repartee",
    }
    demos = [
        DemoApplication(id="chat-demos", display_name="Chat demos"),
        DemoApplication(id="token-trail", display_name="TokenTrail"),
        DemoApplication(id="scenechat", display_name="SceneChat"),
        DemoApplication(id="text-diffusion-demo", display_name="Text diffusion demo"),
        DemoApplication(id="repartee", display_name="Repartee"),
    ]
    routes = []
    for alias, contract in contracts.items():
        candidates = list(contract.providers)
        if not candidates:
            candidates = [
                profile.id
                for profile in by_id.values()
                if contract.accepts(profile) and profile.alias == alias
            ]
        fallback = (
            "mock-visible"
            if any(
                by_id.get(profile_id) and by_id[profile_id].preferred_runtime == "mock"
                for profile_id in candidates
            )
            else "structured-unavailable"
        )
        routes.append(
            DemoRouteContract(
                id=alias,
                demo_id=demo_for_alias[alias],
                display_name=contract.display_name,
                adapter_id=adapter_for_alias[alias],
                public_model=alias,
                fallback_policy=fallback,
                providers=[
                    DeploymentBinding(deployment_id=profile_id, priority=index * 10)
                    for index, profile_id in enumerate(candidates)
                ],
            )
        )
    return DemoSetDefinition(
        id="open-day-demos",
        display_name="Open Day demos",
        description="Editable route contracts seeded from the packaged ModelDeck aliases.",
        demos=demos,
        routes=routes,
    )

import pytest
from fastapi import HTTPException
from modeldeck.demo_config import (
    DemoApplication,
    DemoRouteContract,
    DemoSetDefinition,
    DeploymentBinding,
    default_demo_set,
    routing_snapshot,
    validate_demo_set,
)
from modeldeck.main import _demo_route_smoke_request
from modeldeck.profiles import default_model_profiles


def test_default_demo_set_covers_packaged_and_dynamic_route_contracts() -> None:
    profiles = default_model_profiles()
    definition = default_demo_set(profiles)

    routes = {route.id: route for route in definition.routes}
    assert routes["token-explainer"].adapter_id == "native-ar-trace-v1"
    assert routes["text-diffusion"].fallback_policy == "mock-visible"
    assert routes["repartee-strong"].providers == []


def test_demo_set_validation_checks_adapter_capabilities_and_mock_policy() -> None:
    profiles = default_model_profiles()
    definition = DemoSetDefinition(
        id="test-demos",
        display_name="Test demos",
        demos=[DemoApplication(id="trace-demo", display_name="Trace demo")],
        routes=[
            DemoRouteContract(
                id="trace-route",
                demo_id="trace-demo",
                display_name="Trace route",
                adapter_id="native-ar-trace-v1",
                public_model="trace-route",
                fallback_policy="structured-unavailable",
                providers=[DeploymentBinding(deployment_id="mock-diffusion")],
            )
        ],
    )

    result = validate_demo_set(
        definition,
        profiles,
        registered_ids={profile.id for profile in profiles},
        allowed_ids={profile.id for profile in profiles},
        compatibility_tests=[],
    )

    assert result["valid"] is False
    messages = {error["message"] for error in result["errors"]}
    assert "Requires autoregressive, got text-diffusion" in messages
    assert "Missing capabilities: top_k_trace" in messages
    assert "Mock providers require the mock-visible fallback policy" in messages


def test_routing_snapshot_orders_provider_bindings() -> None:
    definition = DemoSetDefinition(
        id="test-demos",
        display_name="Test demos",
        demos=[DemoApplication(id="chat-demo", display_name="Chat demo")],
        routes=[
            DemoRouteContract(
                id="chat-route",
                demo_id="chat-demo",
                display_name="Chat route",
                adapter_id="openai-chat-v1",
                public_model="chat-route",
                fallback_policy="ordered",
                providers=[
                    DeploymentBinding(deployment_id="mock-ar", priority=20),
                    DeploymentBinding(deployment_id="qwen-small-rocm", priority=10),
                ],
            )
        ],
    )

    snapshot = routing_snapshot(definition, 3)

    assert snapshot["revision"] == 3
    assert snapshot["routes"][0]["providers"] == ["qwen-small-rocm", "mock-ar"]


@pytest.mark.parametrize(
    ("adapter_id", "expected_path", "evidence_key"),
    [
        ("openai-chat-v1", "/v1/chat/completions", "messages"),
        ("openai-completions-v1", "/v1/completions", "prompt"),
        ("native-ar-trace-v1", "/native/autoregressive/trace", "top_k"),
        ("text-diffusion-v1", "/v1/refine", "denoising_steps"),
        ("scene-analysis-v1", "/v1/vision/analyse", "response_format"),
    ],
)
def test_demo_route_smoke_requests_are_bounded_and_adapter_specific(
    adapter_id: str, expected_path: str, evidence_key: str
) -> None:
    route = DemoRouteContract(
        id="smoke-route",
        demo_id="smoke-demo",
        display_name="Smoke route",
        adapter_id=adapter_id,
        public_model="public-model",
    )

    path, payload = _demo_route_smoke_request(route)

    assert path == expected_path
    assert payload["model"] == "public-model"
    assert evidence_key in payload


def test_speech_route_requires_an_interactive_rehearsal_client() -> None:
    route = DemoRouteContract(
        id="speech-route",
        demo_id="speech-demo",
        display_name="Speech route",
        adapter_id="speech-conversation-v1",
        public_model="speech-model",
    )

    with pytest.raises(HTTPException, match="interactive WebSocket"):
        _demo_route_smoke_request(route)

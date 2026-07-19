from modeldeck.demo_config import (
    DemoApplication,
    DemoRouteContract,
    DemoSetDefinition,
    DeploymentBinding,
    default_demo_set,
    routing_snapshot,
    validate_demo_set,
)
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

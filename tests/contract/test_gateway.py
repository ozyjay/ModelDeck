from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.domain import WorkerDefinition
from modeldeck.gateway import create_gateway_app
from modeldeck.gateway.app import (
    invalid_trace_metadata,
    route_candidates,
    trace_token_metadata_error,
    upstream_headers,
    upstream_model,
)


def worker() -> WorkerDefinition:
    return WorkerDefinition(
        id=str(uuid4()),
        name="Trace Worker",
        model_id="example/model",
        revision="revision-1",
        generation_family="autoregressive",
        runtime="mock",
        lifecycle="on-demand",
        port=8630,
        dtype="float16",
        capabilities={"chat": True, "completions": True, "top_k_trace": True},
        settings={},
    )


@pytest.mark.asyncio
async def test_gateway_has_no_routes_or_implicit_defaults_before_publication(tmp_path) -> None:
    app = create_gateway_app(settings=Settings(data_dir=tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        models = await client.get("/v1/models")
        unavailable = await client.post("/v1/completions", json={"prompt": "hello"})

    assert models.json() == {"object": "list", "data": []}
    assert unavailable.status_code == 503
    assert unavailable.json()["error"] == {
        "code": "local_route_unavailable",
        "message": "Supply the public Route name in the model field.",
        "route": None,
        "required_generation_family": "unknown",
        "cloud_fallback_attempted": False,
    }


@pytest.mark.asyncio
async def test_gateway_advertises_only_routes_from_active_event_snapshot(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise_v2()
    definition = worker()
    store.save_worker_definition(definition.model_dump(mode="json"))
    event_id = str(uuid4())
    event = {
        "id": event_id,
        "name": "Open Day",
        "description": "",
        "qualification": "compatible",
        "demos": [],
        "routes": [],
    }
    store.save_event_draft(event)
    store.publish_event(
        event,
        {
            "format": "modeldeck-event-routing",
            "version": 2,
            "event_id": event_id,
            "event_name": "Open Day",
            "revision": 0,
            "routes": [
                {
                    "route_id": str(uuid4()),
                    "display_name": "Visitor trace",
                    "public_name": "visitor-trace",
                    "protocol_contract": "native-ar-trace-v1",
                    "worker_ids": [definition.id],
                }
            ],
        },
    )
    app = create_gateway_app(settings=settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        models = (await client.get("/v1/models")).json()["data"]
        routes = (await client.get("/v1/routes")).json()["routes"]

    assert models == [
        {"id": "visitor-trace", "object": "model", "owned_by": "modeldeck-local", "ready": False}
    ]
    assert routes == [{"public_name": "visitor-trace", "ready": False}]
    assert "provider" not in str(models).lower()


def test_trace_metadata_validation_requires_aligned_readable_tokens() -> None:
    valid = {
        "prompt_token_ids": [1, 2],
        "prompt_tokens": ["one", "two"],
        "user_prompt_token_ids": [2],
        "user_prompt_tokens": ["two"],
    }
    assert trace_token_metadata_error(valid) is None
    assert "align" in trace_token_metadata_error({**valid, "prompt_tokens": ["one"]})


def test_invalid_trace_metadata_error_uses_worker_language_without_route_leakage() -> None:
    response = invalid_trace_metadata("worker-id", "tokens do not align")
    assert response.status_code == 502
    assert b"Local Worker" in response.body
    assert b'"worker_id":"worker-id"' in response.body
    assert "x-modeldeck-provider" not in response.headers


def test_vision_translation_keeps_internal_model_and_scoped_credential(monkeypatch) -> None:
    profile = (
        worker()
        .model_copy(
            update={
                "generation_family": "vision-language",
                "capabilities": {"image_input": True, "structured_output": True},
            }
        )
        .to_profile()
    )
    monkeypatch.setenv("MODELDECK_SCENECHAT_API_KEY", "test-key")
    assert upstream_model(profile, "visitor-scene") == "example/model"
    assert upstream_headers(profile) == {"Authorization": "Bearer test-key"}


def test_route_candidates_accept_only_public_route_or_vision_model_identity() -> None:
    profile = worker().to_profile()
    routes = {"visitor-trace": [profile]}
    assert route_candidates(routes, "visitor-trace") == [profile]
    assert route_candidates(routes, profile.model_id) is None

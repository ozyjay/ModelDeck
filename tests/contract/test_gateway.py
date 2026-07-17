from __future__ import annotations

import httpx
import modeldeck.gateway.app as gateway_module
import pytest
from fastapi.responses import JSONResponse
from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.gateway import create_gateway_app
from modeldeck.gateway.app import (
    invalid_trace_metadata,
    json_loads,
    route_candidates,
    trace_token_metadata_error,
    upstream_headers,
    upstream_model,
)
from modeldeck.profiles import (
    LocalAutoregressiveProfileRequest,
    create_local_autoregressive_profile,
    default_model_profiles,
)


@pytest.mark.asyncio
async def test_gateway_returns_structured_local_unavailable_without_cloud(monkeypatch) -> None:
    async def unavailable_provider(_client, _profile):
        return None, False

    monkeypatch.setattr(gateway_module, "provider_health", unavailable_provider)
    app = create_gateway_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/completions", json={"model": "fast-chat", "prompt": "hello"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "local_provider_unavailable"
    assert response.json()["error"]["cloud_fallback_attempted"] is False


@pytest.mark.asyncio
async def test_default_text_diffusion_alias_prefers_q4(monkeypatch) -> None:
    async def ready_provider(_client, profile):
        return {"ready": True}, profile.id in {
            "diffusiongemma-q4-rocm",
            "diffusiongemma-rocm",
        }

    monkeypatch.setattr(gateway_module, "provider_health", ready_provider)
    app = create_gateway_app()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/models")

    models = {model["id"]: model for model in response.json()["data"]}
    assert models["text-diffusion"]["effective_provider"] == "diffusiongemma-q4-rocm"
    assert models["text-diffusion-bf16"]["effective_provider"] == "diffusiongemma-rocm"
    assert "text-diffusion-q4" not in models


@pytest.mark.asyncio
async def test_default_qwen_aliases_select_their_pinned_workers(monkeypatch) -> None:
    async def ready_provider(_client, profile):
        return {"ready": True}, profile.id.startswith("qwen-")

    monkeypatch.setattr(gateway_module, "provider_health", ready_provider)
    app = create_gateway_app()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/models")

    models = {model["id"]: model for model in response.json()["data"]}
    assert models["qwen-0-5b"]["effective_provider"] == "qwen-small-rocm"
    assert models["qwen-1-5b"]["effective_provider"] == "qwen-1-5b-rocm"
    assert models["qwen-3b"]["effective_provider"] == "qwen-3b-rocm"


@pytest.mark.asyncio
async def test_default_scenechat_alias_advertises_multimodal_provider(monkeypatch) -> None:
    async def ready_provider(_client, profile):
        return {"ready": True}, profile.id == "scenechat-gemma4-e2b-rocm"

    monkeypatch.setattr(gateway_module, "provider_health", ready_provider)
    app = create_gateway_app()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        models_response = await client.get("/v1/models")
        capabilities_response = await client.get("/v1/capabilities")

    models = {model["id"]: model for model in models_response.json()["data"]}
    capabilities = capabilities_response.json()["scenechat-vision"]
    assert models["scenechat-vision"]["effective_provider"] == "scenechat-gemma4-e2b-rocm"
    assert capabilities["image_input"] is True
    assert capabilities["structured_output"] is True


@pytest.mark.asyncio
async def test_gateway_discovers_persisted_local_aliases_without_restart(monkeypatch, tmp_path) -> None:
    async def unavailable_provider(_client, _profile):
        return None, False

    monkeypatch.setattr(gateway_module, "provider_health", unavailable_provider)
    settings = Settings(data_dir=tmp_path)
    store = CompatibilityStore(tmp_path / "modeldeck.sqlite3")
    store.initialise()
    app = create_gateway_app(settings=settings)
    profile = create_local_autoregressive_profile(
        LocalAutoregressiveProfileRequest(
            model_id="Example/LocalModel", revision="revision-1", alias="local-example"
        ),
        cache_root=tmp_path / "cache",
        port=8630,
    )
    store.save_model_profile(profile.model_dump(mode="json"))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        models = await client.get("/v1/models")
        capabilities = await client.get("/v1/capabilities")

    aliases = {model["id"] for model in models.json()["data"]}
    assert "local-example" in aliases
    assert capabilities.json()["local-example"]["completions"] is True


def test_scenechat_gateway_translation_uses_exact_model_and_internal_credential(monkeypatch) -> None:
    profile = next(
        profile for profile in default_model_profiles() if profile.id == "scenechat-gemma4-e2b-rocm"
    )
    routes = {"scenechat-vision": [profile]}
    monkeypatch.setenv("MODELDECK_SCENECHAT_API_KEY", "internal-test-key")

    assert route_candidates(routes, "scenechat-vision") == [profile]
    assert route_candidates(routes, profile.model_id) == [profile]
    assert upstream_model(profile, "scenechat-vision") == profile.model_id
    assert upstream_headers(profile) == {"Authorization": "Bearer internal-test-key"}


@pytest.mark.asyncio
async def test_dedicated_vision_route_uses_scenechat_alias_and_timeout(monkeypatch) -> None:
    captured = {}

    async def capture_proxy(request, routes, path, default_alias, *, timeout_seconds):
        captured.update(
            {
                "path": path,
                "default_alias": default_alias,
                "timeout_seconds": timeout_seconds,
                "has_scenechat_route": "scenechat-vision" in routes,
            }
        )
        return JSONResponse({"ok": True})

    monkeypatch.setattr(gateway_module, "proxy_request", capture_proxy)
    app = create_gateway_app(settings=Settings(scenechat_timeout_seconds=81))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/vision/analyse", json={})

    assert response.json() == {"ok": True}
    assert captured == {
        "path": "/v1/chat/completions",
        "default_alias": "scenechat-vision",
        "timeout_seconds": 81,
        "has_scenechat_route": True,
    }


def test_gateway_trace_metadata_validation_rejects_misaligned_tokens() -> None:
    payload = {
        "prompt_token_ids": [1, 2],
        "prompt_tokens": ["only one"],
        "user_prompt_token_ids": [2],
        "user_prompt_tokens": ["question"],
    }

    assert trace_token_metadata_error(payload) == (
        "prompt_tokens must align one-to-one with prompt_token_ids"
    )


def test_gateway_trace_metadata_validation_accepts_aligned_tokens() -> None:
    payload = {
        "prompt_token_ids": [1, 2, 3],
        "prompt_tokens": ["<bos>", "hello", " world"],
        "user_prompt_token_ids": [2, 3],
        "user_prompt_tokens": ["hello", " world"],
    }

    assert trace_token_metadata_error(payload) is None


def test_gateway_invalid_trace_metadata_error_is_actionable_and_local() -> None:
    response = invalid_trace_metadata("qwen-small-rocm", "prompt token arrays do not align")
    payload = json_loads(response.body)

    assert response.status_code == 502
    assert response.headers["x-modeldeck-provider"] == "qwen-small-rocm"
    assert payload["error"]["code"] == "invalid_worker_trace_metadata"
    assert "qwen-small-rocm" in payload["error"]["message"]
    assert "prompt token arrays do not align" in payload["error"]["message"]

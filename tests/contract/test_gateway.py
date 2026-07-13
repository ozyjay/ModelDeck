from __future__ import annotations

import httpx
import modeldeck.gateway.app as gateway_module
import pytest
from modeldeck.gateway import create_gateway_app


@pytest.mark.asyncio
async def test_gateway_returns_structured_local_unavailable_without_cloud() -> None:
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
    assert models["text-diffusion-q4"]["effective_provider"] == "diffusiongemma-q4-rocm"

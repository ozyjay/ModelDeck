from __future__ import annotations

import httpx
import modeldeck.gateway.app as gateway_module
import pytest
from modeldeck.gateway import create_gateway_app
from modeldeck.gateway.app import invalid_trace_metadata, json_loads, trace_token_metadata_error


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

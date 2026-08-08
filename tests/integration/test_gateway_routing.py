from __future__ import annotations

import socket

import httpx
import pytest
from modeldeck.gateway import create_gateway_app
from modeldeck.profiles import ModelProfile
from modeldeck.protocol import GenerationFamily
from modeldeck.supervisor import WorkerSupervisor

from tests.model_profiles import default_model_profiles


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def mock_profile(port: int) -> ModelProfile:
    document = next(profile for profile in default_model_profiles() if profile.id == "mock-ar").model_dump()
    document["port"] = port
    return ModelProfile.model_validate(document)


def mock_diffusion_profile(port: int) -> ModelProfile:
    document = next(
        profile for profile in default_model_profiles() if profile.id == "mock-diffusion"
    ).model_dump()
    document["port"] = port
    return ModelProfile.model_validate(document)


def mock_embedding_profile(port: int) -> ModelProfile:
    return ModelProfile.model_validate(
        {
            "id": "mock-embedding",
            "model_id": "modeldeck/mock-openai-embeddings",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "alias": "local-embedding",
            "generation_family": GenerationFamily.EMBEDDING,
            "preferred_runtime": "mock",
            "lifecycle": "on-demand",
            "port": port,
            "dtype": "float16",
            "capabilities": {"embeddings": True, "streaming": False, "cancellation": True},
            "settings": {"mock_contract_id": "openai-embeddings-v1"},
        }
    )


@pytest.mark.asyncio
async def test_gateway_forwards_openai_embeddings_in_order_without_cloud_fallback() -> None:
    profile = mock_embedding_profile(free_port())
    supervisor = WorkerSupervisor([profile], startup_timeout=8, stop_timeout=2)
    gateway = create_gateway_app({"sprintbot-embedding": [profile]})
    try:
        await supervisor.start(profile.id)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/embeddings",
                json={"model": "sprintbot-embedding", "input": ["first text", "second text"]},
            )
            models = await client.get("/v1/models")

        assert response.status_code == 200
        assert response.json()["object"] == "list"
        assert response.json()["model"] == "sprintbot-embedding"
        assert [item["index"] for item in response.json()["data"]] == [0, 1]
        assert all(len(item["embedding"]) == 1024 for item in response.json()["data"])
        assert models.json()["data"] == [
            {
                "id": "sprintbot-embedding",
                "object": "model",
                "owned_by": "modeldeck-local",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "ready": True,
                "runtime": "mock",
                "accelerator": "mock",
            }
        ]
    finally:
        await supervisor.stop_all()


@pytest.mark.asyncio
async def test_gateway_embeddings_reject_invalid_unknown_incompatible_and_unavailable_routes() -> None:
    profile = mock_embedding_profile(free_port())
    gateway = create_gateway_app({"sprintbot-embedding": [profile]})
    incompatible = mock_profile(free_port())
    incompatible_gateway = create_gateway_app({"sprintbot-embedding": [incompatible]})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway), base_url="http://gateway"
    ) as client:
        invalid = await client.post("/v1/embeddings", json={"model": "sprintbot-embedding", "input": []})
        unknown = await client.post("/v1/embeddings", json={"model": "unknown", "input": ["text"]})
        unavailable = await client.post(
            "/v1/embeddings", json={"model": "sprintbot-embedding", "input": ["text"]}
        )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=incompatible_gateway), base_url="http://gateway"
    ) as client:
        incompatible_response = await client.post(
            "/v1/embeddings", json={"model": "sprintbot-embedding", "input": ["text"]}
        )

    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_embedding_request"
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "model_not_found"
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "local_route_unavailable"
    assert unavailable.json()["error"]["cloud_fallback_attempted"] is False
    assert incompatible_response.status_code == 409
    assert incompatible_response.json()["error"]["code"] == "incompatible_worker"


@pytest.mark.asyncio
async def test_gateway_forwards_streaming_and_cancellation_to_ready_local_worker() -> None:
    profile = mock_profile(free_port())
    supervisor = WorkerSupervisor([profile], startup_timeout=8, stop_timeout=2)
    gateway = create_gateway_app(
        {
            "fast-chat": [profile],
            "token-explainer": [profile],
        }
    )
    try:
        await supervisor.start(profile.id)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway), base_url="http://gateway"
        ) as client:
            stream = await client.post(
                "/v1/completions",
                json={
                    "request_id": "gateway-stream",
                    "model": "fast-chat",
                    "prompt": "hello",
                    "stream": True,
                },
            )
            trace = await client.post(
                "/native/autoregressive/trace",
                json={
                    "model": "token-explainer",
                    "messages": [
                        {"role": "system", "content": "hidden policy"},
                        {"role": "user", "content": "first question"},
                        {"role": "assistant", "content": "first answer"},
                        {"role": "user", "content": "latest  question"},
                    ],
                    "max_tokens": 2,
                },
            )
            canonical_trace = await client.post(
                "/native/v1/autoregressive/traces",
                json={
                    "model": "token-explainer",
                    "messages": [{"role": "user", "content": "latest question"}],
                    "max_tokens": 2,
                },
            )
            cancellation = await client.post("/v1/requests/gateway-stream/cancel")
        assert stream.status_code == 200
        assert "x-modeldeck-worker" not in stream.headers
        assert "x-modeldeck-fallback" not in stream.headers
        assert "event: token" in stream.text
        assert trace.status_code == 200
        assert trace.headers["deprecation"] == "true"
        assert '</native/v1/autoregressive/traces>; rel="successor-version"' in trace.headers["link"]
        assert "x-modeldeck-worker" not in trace.headers
        assert trace.json()["prompt_tokens"][:3] == ["hidden", " ", "policy"]
        assert trace.json()["user_prompt_tokens"] == ["latest", "  ", "question"]
        assert "hidden" not in trace.json()["user_prompt_tokens"]
        assert len(trace.json()["prompt_token_ids"]) == len(trace.json()["prompt_tokens"])
        assert len(trace.json()["user_prompt_token_ids"]) == len(trace.json()["user_prompt_tokens"])
        assert canonical_trace.status_code == 200
        assert canonical_trace.headers.get("deprecation") is None
        assert set(canonical_trace.json()) == set(trace.json())
        assert cancellation.json() == {
            "ok": False,
            "request_id": "gateway-stream",
            "state": "not-found",
            "worker_id": None,
        }
    finally:
        await supervisor.stop_all()


@pytest.mark.asyncio
async def test_gateway_forwards_diffusion_job_status_events_and_cancellation() -> None:
    profile = mock_diffusion_profile(free_port())
    supervisor = WorkerSupervisor([profile], startup_timeout=8, stop_timeout=2)
    gateway = create_gateway_app({"text-diffusion": [profile]})
    try:
        await supervisor.start(profile.id)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway), base_url="http://gateway"
        ) as client:
            queued = await client.post(
                "/v1/diffuse",
                json={"model": "text-diffusion", "prompt": "hello", "denoising_steps": 4},
            )
            job_id = queued.json()["job_id"]
            status = await client.get(f"/native/v1/text-diffusion/jobs/{job_id}")
            events = await client.get(f"/native/v1/text-diffusion/jobs/{job_id}/events")
            cancellation = await client.post(f"/native/v1/text-diffusion/jobs/{job_id}/cancel")

        assert "x-modeldeck-worker" not in queued.headers
        assert queued.headers["deprecation"] == "true"
        assert "x-modeldeck-fallback" not in queued.headers
        assert status.status_code == 200
        assert "x-modeldeck-worker" not in status.headers
        assert status.json()["state"] == "complete"
        assert events.status_code == 200
        assert "x-modeldeck-worker" not in events.headers
        assert "event: frame" in events.text
        assert cancellation.status_code == 200
        assert "x-modeldeck-worker" not in cancellation.headers
    finally:
        await supervisor.stop_all()

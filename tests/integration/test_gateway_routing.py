from __future__ import annotations

import socket

import httpx
import pytest
from modeldeck.gateway import create_gateway_app
from modeldeck.profiles import ModelProfile
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
            cancellation = await client.post("/v1/requests/gateway-stream/cancel")
        assert stream.status_code == 200
        assert "x-modeldeck-worker" not in stream.headers
        assert stream.headers["x-modeldeck-fallback"] == "mock"
        assert "event: token" in stream.text
        assert trace.status_code == 200
        assert "x-modeldeck-worker" not in trace.headers
        assert trace.json()["prompt_tokens"][:3] == ["hidden", " ", "policy"]
        assert trace.json()["user_prompt_tokens"] == ["latest", "  ", "question"]
        assert "hidden" not in trace.json()["user_prompt_tokens"]
        assert len(trace.json()["prompt_token_ids"]) == len(trace.json()["prompt_tokens"])
        assert len(trace.json()["user_prompt_token_ids"]) == len(trace.json()["user_prompt_tokens"])
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
            status = await client.get(f"/v1/jobs/{job_id}")
            events = await client.get(f"/v1/jobs/{job_id}/events")
            cancellation = await client.post(f"/v1/jobs/{job_id}/cancel")

        assert "x-modeldeck-worker" not in queued.headers
        assert queued.headers["x-modeldeck-fallback"] == "mock"
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

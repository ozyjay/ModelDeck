from __future__ import annotations

import socket

import httpx
import pytest
from modeldeck.gateway import create_gateway_app
from modeldeck.profiles import ModelProfile, default_model_profiles
from modeldeck.supervisor import WorkerSupervisor


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
            cancellation = await client.post("/v1/requests/gateway-stream/cancel")
        assert stream.status_code == 200
        assert stream.headers["x-modeldeck-provider"] == "mock-ar"
        assert "event: token" in stream.text
        assert cancellation.json()["ok"] is True
        assert cancellation.json()["providers"] == ["mock-ar"]
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

        assert queued.headers["x-modeldeck-provider"] == "mock-diffusion"
        assert status.status_code == 200
        assert status.headers["x-modeldeck-provider"] == "mock-diffusion"
        assert status.json()["state"] == "complete"
        assert events.status_code == 200
        assert events.headers["x-modeldeck-provider"] == "mock-diffusion"
        assert "event: frame" in events.text
        assert cancellation.status_code == 200
        assert cancellation.headers["x-modeldeck-provider"] == "mock-diffusion"
    finally:
        await supervisor.stop_all()

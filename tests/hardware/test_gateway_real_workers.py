"""Opt-in gateway checks against already-running, cached real Workers.

These tests never create Workers or acquire Models. The qualified host must publish the
named capabilities before setting ``MODELDECK_RUN_GATEWAY_HARDWARE_TESTS=1``.
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.rocm,
    pytest.mark.large_model,
    pytest.mark.long_running,
    pytest.mark.skipif(
        os.getenv("MODELDECK_RUN_GATEWAY_HARDWARE_TESTS") != "1",
        reason="set MODELDECK_RUN_GATEWAY_HARDWARE_TESTS=1 on the qualified ROCm host",
    ),
]


def _configured_model(environment_variable: str) -> str:
    model = os.getenv(environment_variable)
    if not model:
        pytest.skip(f"set {environment_variable} to a published cached-model capability")
    return model


def _gateway_url() -> str:
    return os.getenv("MODELDECK_GATEWAY_URL", "http://127.0.0.1:8600").rstrip("/")


@pytest.mark.asyncio
async def test_real_openai_response_and_stream() -> None:
    model = _configured_model("MODELDECK_HARDWARE_OPENAI_MODEL")
    async with httpx.AsyncClient(base_url=_gateway_url(), timeout=180.0) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": model, "prompt": "Reply only: ready", "max_tokens": 8},
        )
        stream = await client.post(
            "/v1/completions",
            json={"model": model, "prompt": "Reply only: ready", "max_tokens": 8, "stream": True},
        )

    response.raise_for_status()
    assert response.json()["choices"]
    stream.raise_for_status()
    assert "data:" in stream.text


@pytest.mark.asyncio
async def test_real_token_trail_candidate_trace() -> None:
    model = _configured_model("MODELDECK_HARDWARE_TRACE_MODEL")
    async with httpx.AsyncClient(base_url=_gateway_url(), timeout=180.0) as client:
        response = await client.post(
            "/native/v1/autoregressive/traces",
            json={"model": model, "prompt": "Reply only: ready", "max_tokens": 4, "top_k": 3},
        )

    response.raise_for_status()
    trace = response.json()
    assert trace["events"]
    assert trace["events"][0]["alternatives"]


@pytest.mark.asyncio
async def test_real_text_diffusion_frames_completion_cancellation_and_affinity() -> None:
    model = _configured_model("MODELDECK_HARDWARE_TEXT_DIFFUSION_MODEL")
    async with httpx.AsyncClient(base_url=_gateway_url(), timeout=300.0) as client:
        queued = await client.post(
            "/native/v1/text-diffusion/jobs",
            json={"model": model, "prompt": "A concise readiness statement.", "denoising_steps": 4},
        )
        queued.raise_for_status()
        job_id = queued.json()["job_id"]
        events = await client.get(f"/native/v1/text-diffusion/jobs/{job_id}/events")
        completed = await client.get(f"/native/v1/text-diffusion/jobs/{job_id}")

        cancellable = await client.post(
            "/native/v1/text-diffusion/jobs",
            json={"model": model, "prompt": "A longer readiness statement.", "denoising_steps": 32},
        )
        cancellable.raise_for_status()
        cancel_job_id = cancellable.json()["job_id"]
        cancelled = await client.post(f"/native/v1/text-diffusion/jobs/{cancel_job_id}/cancel")

    events.raise_for_status()
    assert "event: frame" in events.text
    completed.raise_for_status()
    assert completed.json()["job_id"] == job_id
    assert completed.json()["state"] == "complete"
    cancelled.raise_for_status()
    assert cancelled.json()["job_id"] == cancel_job_id

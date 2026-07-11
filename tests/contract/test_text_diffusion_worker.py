from __future__ import annotations

import asyncio
import threading
from typing import Any

import httpx
import pytest
from modeldeck.workers.text_diffusion_worker import DiffusionRequest, EngineConfig, create_app


class FakeDiffusionEngine:
    runtime_details = {
        "device": "cuda:0",
        "device_name": "Fake ROCm GPU",
        "hip_version": "7.2.1",
        "torch_version": "test",
        "transformers_version": "test",
    }

    def load(self) -> None:
        return

    def warmup(self) -> None:
        return

    def memory_metrics(self) -> dict[str, int]:
        return {"memory_allocated_bytes": 1024}

    def refine(self, body: DiffusionRequest, cancellation: threading.Event) -> list[dict[str, Any]]:
        return [
            {
                "step": 1,
                "total_steps": body.denoising_steps,
                "text": "A local",
                "masked_tokens": 2,
                "stable_tokens": 1,
                "complete": False,
            },
            {
                "step": 2,
                "total_steps": body.denoising_steps,
                "text": "A local diffusion worker is ready.",
                "masked_tokens": 0,
                "stable_tokens": 6,
                "complete": True,
                "cancelled": False,
            },
        ]


class FailingDiffusionEngine(FakeDiffusionEngine):
    def load(self) -> None:
        raise RuntimeError("ROCm device unavailable")


@pytest.mark.asyncio
async def test_diffusion_load_failure_is_logged_and_reported(caplog) -> None:
    config = EngineConfig(model_id="google/diffusiongemma", revision="pinned")
    app = create_app(worker_id="diffusion-test", config=config, engine=FailingDiffusionEngine())

    with caplog.at_level("ERROR", logger="uvicorn.error"):
        async with app.router.lifespan_context(app):
            await app.state.load_task
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                health = await client.get("/health")

    assert health.json()["state"] == "failed"
    assert health.json()["error"] == "Load failed: RuntimeError: ROCm device unavailable"
    assert "Diffusion engine load failed: ROCm device unavailable" in caplog.text


@pytest.mark.asyncio
async def test_real_diffusion_contract_uses_native_frames() -> None:
    config = EngineConfig(model_id="google/diffusiongemma", revision="pinned")
    app = create_app(worker_id="diffusion-test", config=config, engine=FakeDiffusionEngine())
    async with app.router.lifespan_context(app):
        await app.state.load_task
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            warmup = await client.post("/warmup")
            response = await client.post(
                "/v1/refine",
                json={"prompt": "Refine this", "denoising_steps": 4, "seed": 7},
            )
            job_response = await client.post(
                "/v1/diffuse",
                json={"prompt": "Refine this", "denoising_steps": 4, "seed": 7},
            )
            job_id = job_response.json()["job_id"]
            for _ in range(10):
                job = await client.get(f"/v1/jobs/{job_id}")
                if job.json()["state"] == "complete":
                    break
                await asyncio.sleep(0)
            wrong_family = await client.post("/v1/chat/completions", json={"prompt": "wrong"})

    assert warmup.json()["ready"] is True
    assert response.status_code == 200
    payload = response.json()
    assert payload["frames"][-1]["complete"] is True
    assert payload["frames"][0]["text"] != payload["frames"][-1]["text"]
    assert job.json()["frame_count"] == 2
    assert wrong_family.status_code == 404

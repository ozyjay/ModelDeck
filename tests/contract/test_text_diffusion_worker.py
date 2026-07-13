from __future__ import annotations

import asyncio
import threading
from typing import Any

import httpx
import pytest
from modeldeck.workers.text_diffusion_worker import (
    DiffusionRequest,
    EngineConfig,
    FrameStreamer,
    _decode_response,
    _finalise_frames,
    _finish_reason,
    create_app,
)


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

    def refine(
        self,
        body: DiffusionRequest,
        cancellation: threading.Event,
        frame_callback=None,
    ) -> list[dict[str, Any]]:
        first = {
            "step": 1,
            "total_steps": body.denoising_steps,
            "text": "A local",
            "masked_tokens": 2,
            "stable_tokens": 1,
            "complete": False,
        }
        if frame_callback:
            frame_callback(first)
        return [
            first,
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


class ProgressiveDiffusionEngine(FakeDiffusionEngine):
    def __init__(self) -> None:
        self.frame_published = threading.Event()
        self.release_generation = threading.Event()

    def refine(
        self,
        body: DiffusionRequest,
        cancellation: threading.Event,
        frame_callback=None,
    ) -> list[dict[str, Any]]:
        first = {
            "step": 1,
            "total_steps": body.denoising_steps,
            "text": "A local",
            "masked_tokens": 2,
            "stable_tokens": 1,
            "complete": False,
        }
        if frame_callback:
            frame_callback(first)
        self.frame_published.set()
        self.release_generation.wait(timeout=2)
        return [
            first,
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


class StructuredResponseTokenizer:
    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        if token_ids == [4, 5]:
            return "Public answer."
        if skip_special_tokens:
            return "thought\nPrivate reasoning. Public answer."
        return "<|channel>thought\nPrivate reasoning.<channel|>Public answer.<turn|>"

    def parse_response(self, _text: str) -> dict[str, str]:
        return {
            "role": "assistant",
            "thinking": "Private reasoning.",
            "content": "Public answer.",
        }

    def encode(self, _text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [4, 5]


class FakeTokenTensor:
    shape = (1, 2)

    def __getitem__(self, _index):
        return self

    def tolist(self) -> list[int]:
        return [1, 2]


def test_structured_response_parser_hides_private_reasoning() -> None:
    assert _decode_response(StructuredResponseTokenizer(), [1, 2, 3]) == "Public answer."


def test_terminal_frame_replaces_last_draft_without_exceeding_total_steps() -> None:
    drafts = [
        {
            "step": step,
            "total_steps": 8,
            "text": f"draft {step}",
            "masked_tokens": None,
            "stable_tokens": step,
            "complete": False,
        }
        for step in range(1, 9)
    ]

    frames = _finalise_frames(
        drafts,
        8,
        cancelled=False,
        finish_reason="stop",
        text="Final answer.",
    )

    assert len(frames) == 8
    assert frames[-1]["step"] == frames[-1]["total_steps"] == 8
    assert frames[-1]["text"] == "Final answer."
    assert frames[-1]["finish_reason"] == "stop"
    assert frames[-1]["complete"] is True
    assert _finish_reason([4, 50], [1, 50, 106]) == "stop"
    assert _finish_reason([4, 50, 0, 0], [1, 50, 106]) == "stop"
    assert _finish_reason([4, 5], [1, 50, 106]) == "length"


def test_frame_streamer_withholds_final_draft_for_terminal_event() -> None:
    published = []
    streamer = FrameStreamer(
        StructuredResponseTokenizer(),
        total_steps=2,
        cancellation=threading.Event(),
        frame_callback=published.append,
    )

    streamer.put_draft(FakeTokenTensor())
    streamer.put_draft(FakeTokenTensor())

    assert len(streamer.frames) == 2
    assert [frame["step"] for frame in published] == [1]


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
            for _ in range(100):
                job = await client.get(f"/v1/jobs/{job_id}")
                if job.json()["state"] == "complete":
                    break
                await asyncio.sleep(0.01)
            wrong_family = await client.post("/v1/chat/completions", json={"prompt": "wrong"})

    assert warmup.json()["ready"] is True
    assert response.status_code == 200
    payload = response.json()
    assert payload["frames"][-1]["complete"] is True
    assert payload["frames"][0]["text"] != payload["frames"][-1]["text"]
    assert job.json()["frame_count"] == 2
    assert wrong_family.status_code == 404


@pytest.mark.asyncio
async def test_diffusion_job_publishes_frames_before_generation_completes() -> None:
    engine = ProgressiveDiffusionEngine()
    app = create_app(
        worker_id="diffusion-test",
        config=EngineConfig(model_id="google/diffusiongemma", revision="pinned"),
        engine=engine,
    )
    async with app.router.lifespan_context(app):
        await app.state.load_task
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/warmup")
            queued = await client.post(
                "/v1/diffuse",
                json={"prompt": "Refine this", "denoising_steps": 4, "seed": 7},
            )
            job_id = queued.json()["job_id"]
            for _ in range(20):
                status = await client.get(f"/v1/jobs/{job_id}")
                if status.json()["frame_count"] == 1:
                    break
                await asyncio.sleep(0.01)

            assert status.json()["state"] == "running"
            assert status.json()["frame_count"] == 1

            engine.release_generation.set()
            events = await client.get(f"/v1/jobs/{job_id}/events")

    assert events.text.count("event: frame") == 2
    assert '"complete": true' in events.text

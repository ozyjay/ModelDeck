from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from modeldeck.workers.autoregressive_worker import (
    EngineConfig,
    GenerationRequest,
    _trace_token_metadata,
    create_app,
)


class FakeEngine:
    def __init__(self) -> None:
        self.runtime_details: dict[str, Any] = {}
        self.loaded = False
        self.warmed = False

    def load(self) -> None:
        self.loaded = True
        self.runtime_details = {
            "torch_version": "test",
            "hip_version": "7.2-test",
            "transformers_version": "test",
            "device": "cuda:0",
            "device_name": "Fake AMD GPU",
            "load_seconds": 0.01,
        }

    def warmup(self) -> None:
        self.warmed = True

    def build_prompt(self, body: GenerationRequest) -> str:
        if body.messages:
            return " ".join(message.content for message in body.messages)
        return body.prompt or ""

    def trace(
        self,
        *,
        prompt: str,
        body: GenerationRequest,
        cancellation: threading.Event,
    ) -> Iterator[dict[str, Any]]:
        text = ""
        for step, token in enumerate(("Hello", " world")):
            if cancellation.is_set():
                yield {"step": step, "cancelled": True, "complete": True, "text_so_far": text}
                return
            text += token
            yield {
                "step": step,
                "selected": {"token_id": step + 10, "token": token, "probability": 0.8},
                "alternatives": [{"token_id": step + 20, "token": " other", "probability": 0.2}],
                "prompt_token_ids": [1, 2] if step == 0 else None,
                "prompt_tokens": ["<bos>", "Hi"] if step == 0 else None,
                "user_prompt_token_ids": [2] if step == 0 else None,
                "user_prompt_tokens": ["Hi"] if step == 0 else None,
                "generated_token_ids": list(range(10, 11 + step)),
                "text_so_far": text,
                "timestamp": 1.0 + step,
                "elapsed_seconds": 0.01 + step,
                "hidden_state_summary": None,
                "cancelled": False,
                "complete": step == 1,
            }


@pytest.mark.asyncio
async def test_worker_load_warmup_trace_and_stream_contracts() -> None:
    engine = FakeEngine()
    app = create_app(
        worker_id="test-rocm-ar",
        config=EngineConfig(model_id="Qwen/test", revision="commit"),
        engine=engine,
    )
    async with app.router.lifespan_context(app):
        await app.state.load_task
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            before = (await client.get("/health")).json()
            warmup = await client.post("/warmup")
            trace = (
                await client.post(
                    "/native/autoregressive/trace",
                    json={"prompt": "Hi", "max_tokens": 2, "top_k": 2},
                )
            ).json()
            stream = await client.post(
                "/native/autoregressive/trace",
                json={"prompt": "Hi", "max_tokens": 2, "stream": True},
            )
            after = (await client.get("/health")).json()
    assert before["state"] == "warming"
    assert before["ready"] is False
    assert warmup.json()["ready"] is True
    assert engine.loaded and engine.warmed
    assert trace["prompt_token_ids"] == [1, 2]
    assert trace["prompt_tokens"] == ["<bos>", "Hi"]
    assert trace["user_prompt_token_ids"] == [2]
    assert trace["user_prompt_tokens"] == ["Hi"]
    assert trace["events"][-1]["text_so_far"] == "Hello world"
    assert trace["metrics"]["generated_tokens"] == 2
    assert "event: token" in stream.text
    assert "data: [DONE]" in stream.text
    assert after["ready"] is True


@pytest.mark.asyncio
async def test_worker_cancellation_route_sets_only_known_request() -> None:
    engine = FakeEngine()
    app = create_app(
        worker_id="test-rocm-ar",
        config=EngineConfig(model_id="Qwen/test", revision="commit"),
        engine=engine,
    )
    async with app.router.lifespan_context(app):
        await app.state.load_task
        cancellation = threading.Event()
        app.state.cancellations["known"] = cancellation
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            known = await client.post("/cancel", json={"request_id": "known"})
            unknown = await client.post("/cancel", json={"request_id": "unknown"})
    assert known.json()["ok"] is True
    assert unknown.json()["ok"] is False
    assert cancellation.is_set()


def test_worker_rejects_misaligned_trace_token_metadata() -> None:
    events = [
        {
            "prompt_token_ids": [1, 2],
            "prompt_tokens": ["<bos>"],
            "user_prompt_token_ids": [2],
            "user_prompt_tokens": ["Hi"],
        }
    ]

    with pytest.raises(HTTPException, match="one entry for every prompt_token_ids entry"):
        _trace_token_metadata(events)

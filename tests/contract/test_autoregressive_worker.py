from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from modeldeck.workers.autoregressive_worker import (
    MAX_REQUEST_BYTES,
    EngineConfig,
    GenerationRequest,
    _trace_token_metadata,
    create_app,
)


class FakeEngine:
    def __init__(self, output_tokens: tuple[str, ...] = ("Hello", " world")) -> None:
        self.runtime_details: dict[str, Any] = {}
        self.loaded = False
        self.warmed = False
        self.output_tokens = output_tokens
        self.last_body: GenerationRequest | None = None

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
        self.last_body = body
        if body.messages:
            return " ".join(message.content or "" for message in body.messages)
        return body.prompt or ""

    def trace(
        self,
        *,
        prompt: str,
        body: GenerationRequest,
        cancellation: threading.Event,
    ) -> Iterator[dict[str, Any]]:
        text = ""
        for step, token in enumerate(self.output_tokens):
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
                "complete": step == len(self.output_tokens) - 1,
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


def test_worker_accepts_openai_text_content_parts() -> None:
    body = GenerationRequest.model_validate(
        {
            "model": "fast-local",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with the "},
                        {"type": "text", "text": "backend you selected."},
                    ],
                }
            ],
            "stream": False,
        }
    )

    assert body.messages is not None
    assert body.messages[0].content == "Reply with the backend you selected."


def test_worker_rejects_unsupported_openai_content_parts() -> None:
    with pytest.raises(ValueError, match="text-only"):
        GenerationRequest.model_validate(
            {
                "model": "fast-local",
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
            }
        )


@pytest.mark.asyncio
async def test_worker_accepts_openai_tool_messages_and_returns_tool_calls() -> None:
    engine = FakeEngine(('<tool_call>{"name":"weather","arguments":{"city":"Brisbane"}}</tool_call>',))
    app = create_app(
        worker_id="test-rocm-ar",
        config=EngineConfig(model_id="Qwen/test", revision="commit"),
        engine=engine,
    )
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
    async with app.router.lifespan_context(app):
        await app.state.load_task
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.post("/warmup")).status_code == 200
            first = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "fast-local",
                    "messages": [{"role": "user", "content": "Weather?"}],
                    "tools": tools,
                    "tool_choice": "auto",
                },
            )
            call = first.json()["choices"][0]["message"]["tool_calls"][0]
            follow_up = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "fast-local",
                    "messages": [
                        {"role": "user", "content": "Weather?"},
                        {"role": "assistant", "content": None, "tool_calls": [call]},
                        {"role": "tool", "tool_call_id": call["id"], "content": "sunny"},
                    ],
                    "tools": tools,
                },
            )
    assert first.status_code == 200
    assert first.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert call["function"] == {"name": "weather", "arguments": '{"city": "Brisbane"}'}
    assert engine.last_body is not None
    assert engine.last_body.messages[-1].role == "tool"
    assert follow_up.status_code == 200


@pytest.mark.asyncio
async def test_worker_accepts_vscode_agent_multi_message_tool_request() -> None:
    engine = FakeEngine()
    app = create_app(
        worker_id="wayfinder-deep",
        config=EngineConfig(
            model_id="Qwen/Qwen2.5-3B-Instruct",
            revision="pinned",
            context_length=32_768,
            maximum_new_tokens=4_096,
        ),
        engine=engine,
    )
    system_instructions = "Use tools when needed and report concise results. " * 400
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_workspace_file",
                "description": "Read one allowlisted workspace file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_instructions}]},
        {"role": "user", "content": [{"type": "text", "text": "Inspect the project status."}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_workspace_1",
                    "type": "function",
                    "function": {"name": "read_workspace_file", "arguments": '{"path":"status.txt"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_workspace_1", "content": "Workspace is clean."},
    ]
    async with app.router.lifespan_context(app):
        await app.state.load_task
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.post("/warmup")).status_code == 200
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "deep-local",
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "max_tokens": 4096,
                },
            )

    assert response.status_code == 200
    assert engine.last_body is not None
    assert engine.last_body.max_tokens == 4096
    assert engine.last_body.messages is not None
    assert len(engine.last_body.messages) == 4
    assert engine.last_body.messages[0].content == system_instructions
    assert engine.last_body.tools == tools


@pytest.mark.asyncio
async def test_worker_validation_error_does_not_echo_prompt_input() -> None:
    engine = FakeEngine()
    app = create_app(
        worker_id="test-rocm-ar",
        config=EngineConfig(model_id="Qwen/test", revision="commit"),
        engine=engine,
    )
    prompt_secret = "prompt-value-that-must-not-appear-in-the-response"
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "invalid", "content": prompt_secret}]},
            )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "message": "The request does not match the local OpenAI chat contract.",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_request",
        }
    }
    assert prompt_secret not in response.text


@pytest.mark.asyncio
async def test_worker_rejects_oversized_request_before_validation() -> None:
    engine = FakeEngine()
    app = create_app(
        worker_id="test-rocm-ar",
        config=EngineConfig(model_id="Qwen/test", revision="commit"),
        engine=engine,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                content=b"{}",
                headers={"Content-Length": str(MAX_REQUEST_BYTES + 1)},
            )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"

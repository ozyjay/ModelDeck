from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from modeldeck.protocol import CapabilitySet, GenerationFamily, WorkerHealth, WorkerState


class CompletionRequest(BaseModel):
    request_id: str | None = None
    model: str = "fast-chat"
    prompt: str | None = None
    messages: list[dict[str, str]] | None = None
    stream: bool = False
    seed: int = 7
    max_tokens: int = Field(default=32, ge=1, le=256)
    top_k: int = Field(default=5, ge=1, le=20)


class DiffusionRequest(BaseModel):
    model: str = "text-diffusion"
    prompt: str = Field(min_length=1, max_length=4000)
    max_length: int = Field(default=64, ge=8, le=512)
    denoising_steps: int = Field(default=8, ge=1, le=128)
    block_length: int = Field(default=16, ge=1, le=256)
    temperature: float = Field(default=0.2, ge=0, le=2)
    seed: int = 11
    stream_intermediate_frames: bool = True


def capabilities(family: GenerationFamily) -> CapabilitySet:
    if family == GenerationFamily.AUTOREGRESSIVE:
        return CapabilitySet(chat=True, completions=True, logits=True, top_k_trace=True)
    return CapabilitySet(
        iterative_refinement=True,
        intermediate_frames=True,
        seeded_generation=True,
        logits="model-specific",
    )


def create_app(
    *,
    worker_id: str,
    model_id: str,
    revision: str,
    family: GenerationFamily,
    startup_delay: float = 0.08,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.worker_state = WorkerState.LOADING
        app.state.ready = False
        app.state.jobs = {}
        app.state.cancelled = set()
        await asyncio.sleep(startup_delay)
        app.state.ready = True
        app.state.worker_state = WorkerState.READY
        yield

    app = FastAPI(title=f"ModelDeck mock worker: {worker_id}", lifespan=lifespan)
    app.state.shutdown_callback = None

    @app.get("/health", response_model=WorkerHealth)
    async def health(request: Request) -> WorkerHealth:
        return WorkerHealth(
            worker_id=worker_id,
            runtime="mock",
            generation_family=family,
            state=request.app.state.worker_state,
            model_id=model_id,
            model_revision=revision,
            ready=request.app.state.ready,
        )

    @app.get("/capabilities")
    async def get_capabilities() -> dict[str, Any]:
        return {"protocol_version": "1", "generation_family": family, **capabilities(family).model_dump()}

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return {"requests": len(app.state.jobs), "device": "cpu", "mock": True}

    @app.get("/model")
    async def model() -> dict[str, str]:
        return {"model_id": model_id, "revision": revision, "generation_family": family}

    @app.post("/load")
    async def load() -> dict[str, Any]:
        return {"ok": True, "state": app.state.worker_state}

    @app.post("/warmup")
    async def warmup() -> dict[str, Any]:
        app.state.ready = True
        app.state.worker_state = WorkerState.READY
        return {"ok": True, "ready": True}

    @app.post("/cancel")
    async def cancel(payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id", ""))
        if request_id:
            app.state.cancelled.add(request_id)
        return {"ok": True, "request_id": request_id}

    @app.post("/shutdown")
    async def shutdown() -> dict[str, bool]:
        app.state.worker_state = WorkerState.STOPPING
        if app.state.shutdown_callback:
            asyncio.get_running_loop().call_later(0.05, app.state.shutdown_callback)
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat(body: CompletionRequest):
        _require_family(family, GenerationFamily.AUTOREGRESSIVE)
        if body.stream:
            return StreamingResponse(
                _stream_completion(body, worker_id, app.state.cancelled),
                media_type="text/event-stream",
            )
        return _completion_response(body, worker_id)

    @app.post("/v1/completions")
    async def completion(body: CompletionRequest):
        _require_family(family, GenerationFamily.AUTOREGRESSIVE)
        if body.stream:
            return StreamingResponse(
                _stream_completion(body, worker_id, app.state.cancelled),
                media_type="text/event-stream",
            )
        return _completion_response(body, worker_id)

    @app.post("/native/autoregressive/trace")
    async def trace(body: CompletionRequest):
        _require_family(family, GenerationFamily.AUTOREGRESSIVE)
        prompt = body.prompt or " ".join(message.get("content", "") for message in body.messages or [])
        return {"request_id": str(uuid.uuid4()), "model": model_id, "events": _trace_events(prompt, body)}

    @app.post("/v1/refine")
    async def refine(body: DiffusionRequest):
        _require_family(family, GenerationFamily.TEXT_DIFFUSION)
        frames = list(_diffusion_frames(body, "sync"))
        return {"model": model_id, "text": frames[-1]["text"], "frames": frames, "seed": body.seed}

    @app.post("/v1/diffuse")
    async def diffuse(body: DiffusionRequest):
        _require_family(family, GenerationFamily.TEXT_DIFFUSION)
        job_id = str(uuid.uuid4())
        frames = list(_diffusion_frames(body, job_id))
        app.state.jobs[job_id] = {"state": "complete", "request": body, "frames": frames}
        return {"job_id": job_id, "state": "complete", "events_url": f"/v1/jobs/{job_id}/events"}

    @app.get("/v1/jobs/{job_id}")
    async def job(job_id: str):
        _require_family(family, GenerationFamily.TEXT_DIFFUSION)
        if job_id not in app.state.jobs:
            raise HTTPException(404, "Unknown diffusion job")
        job = app.state.jobs[job_id]
        return {"job_id": job_id, "state": job["state"], "frame_count": len(job["frames"])}

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        if job_id not in app.state.jobs:
            raise HTTPException(404, "Unknown diffusion job")
        app.state.jobs[job_id]["state"] = "cancelled"
        return {"job_id": job_id, "state": "cancelled"}

    @app.get("/v1/jobs/{job_id}/events")
    async def job_events(job_id: str):
        if job_id not in app.state.jobs:
            raise HTTPException(404, "Unknown diffusion job")

        async def events() -> AsyncIterator[str]:
            for frame in app.state.jobs[job_id]["frames"]:
                if app.state.jobs[job_id]["state"] == "cancelled":
                    yield f"event: cancelled\ndata: {json.dumps({'job_id': job_id})}\n\n"
                    return
                yield f"event: frame\ndata: {json.dumps(frame)}\n\n"
                await asyncio.sleep(0)

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _require_family(actual: GenerationFamily, expected: GenerationFamily) -> None:
    if actual != expected:
        raise HTTPException(404, f"Route requires a {expected.value} worker")


def _completion_response(body: CompletionRequest, worker_id: str) -> dict[str, Any]:
    request_id = body.request_id or str(uuid.uuid4())
    prompt = body.prompt or " ".join(message.get("content", "") for message in body.messages or [])
    text = f"Mock local response: {prompt.strip() or 'ready'}"[: body.max_tokens * 8]
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "provider": worker_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(text.split())},
    }


async def _stream_completion(
    body: CompletionRequest,
    worker_id: str,
    cancelled: set[str],
) -> AsyncIterator[str]:
    request_id = body.request_id or str(uuid.uuid4())
    prompt = body.prompt or " ".join(message.get("content", "") for message in body.messages or [])
    text = f"Mock local response: {prompt.strip() or 'ready'}"[: body.max_tokens * 8]
    for token in text.split(" "):
        if request_id in cancelled:
            payload = {"id": request_id, "object": "chat.completion.cancelled", "provider": worker_id}
            yield f"event: cancelled\ndata: {json.dumps(payload)}\n\n"
            return
        payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "model": body.model,
            "provider": worker_id,
            "choices": [{"index": 0, "delta": {"content": f"{token} "}, "finish_reason": None}],
        }
        yield f"event: token\ndata: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0.005)
    yield "event: complete\ndata: [DONE]\n\n"


def _trace_events(prompt: str, body: CompletionRequest) -> list[dict[str, Any]]:
    rng = random.Random(body.seed)
    tokens = ("A", " local", " model", " response", ".")
    events = []
    text = ""
    for step, token in enumerate(tokens[: body.max_tokens]):
        text += token
        probability = round(0.55 + rng.random() * 0.35, 4)
        events.append(
            {
                "step": step,
                "selected": {"token_id": 100 + step, "token": token, "probability": probability},
                "alternatives": [
                    {"token_id": 200 + index, "token": candidate, "probability": round(0.2 / (index + 1), 4)}
                    for index, candidate in enumerate((" demo", " worker", " answer")[: body.top_k - 1])
                ],
                "text_so_far": text,
                "timestamp": time.time(),
                "prompt_token_ids": list(range(len(prompt.split()))),
            }
        )
    return events


def _diffusion_frames(body: DiffusionRequest, job_id: str):
    words = body.prompt.split()
    total = body.denoising_steps
    for step in range(1, total + 1):
        visible = max(1, round(len(words) * step / total))
        masked = max(len(words) - visible, 0)
        text = " ".join(words[:visible] + (["…"] if masked else []))
        yield {
            "job_id": job_id,
            "step": step,
            "total_steps": total,
            "text": text,
            "masked_tokens": masked,
            "stable_tokens": visible,
            "complete": step == total,
            "finish_reason": "stop" if step == total else None,
            "seed": body.seed,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an allowlisted ModelDeck mock worker")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--family", required=True, choices=[family.value for family in GenerationFamily])
    parser.add_argument("--port", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Starting allowlisted ModelDeck mock worker {args.worker_id} on loopback port {args.port}.",
        flush=True,
    )
    app = create_app(
        worker_id=args.worker_id,
        model_id=args.model_id,
        revision=args.revision,
        family=GenerationFamily(args.family),
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    app.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

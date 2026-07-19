from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from modeldeck.protocol import CapabilitySet, GenerationFamily, WorkerState


@dataclass(frozen=True)
class EngineConfig:
    model_id: str
    revision: str
    dtype: str = "float16"
    context_length: int = 2048
    maximum_new_tokens: int = 128


class ChatMessage(BaseModel):
    role: str = Field(pattern=r"^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=16_000)


class GenerationRequest(BaseModel):
    request_id: str | None = None
    model: str = "local-worker"
    prompt: str | None = Field(default=None, max_length=16_000)
    messages: list[ChatMessage] | None = None
    stream: bool = False
    seed: int = 7
    max_tokens: int = Field(default=32, ge=1, le=128)
    min_tokens: int = Field(default=0, ge=0, le=128)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    top_k: int = Field(default=5, ge=1, le=50)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=3)
    stop: str | list[str] | None = None
    include_hidden_state_summary: bool = False

    @model_validator(mode="after")
    def prompt_or_messages(self) -> GenerationRequest:
        if not self.prompt and not self.messages:
            raise ValueError("prompt or messages is required")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens cannot exceed max_tokens")
        return self


class AutoregressiveEngine(Protocol):
    runtime_details: dict[str, Any]

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def build_prompt(self, body: GenerationRequest) -> str: ...

    def memory_metrics(self) -> dict[str, int]: ...

    def trace(
        self,
        *,
        prompt: str,
        body: GenerationRequest,
        cancellation: threading.Event,
    ) -> Iterator[dict[str, Any]]: ...


class TransformersAutoregressiveEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.runtime_details: dict[str, Any] = {}
        self.torch: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("ROCm PyTorch did not expose an available 'cuda' device")
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self.config.dtype)
        if dtype is None:
            raise RuntimeError(f"Unsupported dtype: {self.config.dtype}")
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
        )
        device = torch.device("cuda:0")
        model.to(device)
        model.eval()
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.runtime_details = {
            "torch_version": str(torch.__version__),
            "hip_version": torch.version.hip,
            "transformers_version": importlib.metadata.version("transformers"),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "load_seconds": round(time.perf_counter() - started, 4),
            "dtype": self.config.dtype,
        }

    def warmup(self) -> None:
        body = GenerationRequest(prompt="Hello", max_tokens=1, temperature=0, top_k=1)
        list(self.trace(prompt="Hello", body=body, cancellation=threading.Event()))

    def build_prompt(self, body: GenerationRequest) -> str:
        if body.messages:
            messages = [message.model_dump() for message in body.messages]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return body.prompt or ""

    def memory_metrics(self) -> dict[str, int]:
        if self.torch is None or not self.torch.cuda.is_available():
            return {}
        return {
            "memory_allocated_bytes": int(self.torch.cuda.memory_allocated(0)),
            "memory_reserved_bytes": int(self.torch.cuda.memory_reserved(0)),
            "peak_memory_allocated_bytes": int(self.torch.cuda.max_memory_allocated(0)),
            "peak_memory_reserved_bytes": int(self.torch.cuda.max_memory_reserved(0)),
        }

    def trace(
        self,
        *,
        prompt: str,
        body: GenerationRequest,
        cancellation: threading.Event,
    ) -> Iterator[dict[str, Any]]:
        torch = self.torch
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_ids = encoded["input_ids"][0].tolist()
        prompt_tokens = _decode_tokens(self.tokenizer, prompt_ids)
        user_prompt = _latest_user_prompt(body)
        user_prompt_ids = _tokenise_without_special_tokens(self.tokenizer, user_prompt)
        user_prompt_tokens = _decode_tokens(self.tokenizer, user_prompt_ids)
        if len(prompt_ids) + body.max_tokens > self.config.context_length:
            raise ValueError(
                f"Prompt plus output exceeds configured context length {self.config.context_length}"
            )
        sequence = encoded["input_ids"].to(self.device)
        generated: list[int] = []
        text_so_far = ""
        stop_sequences = [body.stop] if isinstance(body.stop, str) else list(body.stop or ())
        generator = torch.Generator(device=self.device).manual_seed(body.seed)
        started = time.perf_counter()

        for step in range(min(body.max_tokens, self.config.maximum_new_tokens)):
            if cancellation.is_set():
                yield {"step": step, "cancelled": True, "complete": True, "text_so_far": text_so_far}
                return
            with torch.inference_mode():
                output = self.model(
                    input_ids=sequence,
                    use_cache=False,
                    output_hidden_states=body.include_hidden_state_summary,
                )
            logits = output.logits[0, -1].float()
            if body.repetition_penalty != 1 and generated:
                for token_id in set(generated):
                    logits[token_id] = (
                        logits[token_id] / body.repetition_penalty
                        if logits[token_id] > 0
                        else logits[token_id] * body.repetition_penalty
                    )
            sampling_logits = logits if body.temperature == 0 else logits / max(body.temperature, 1e-6)
            probabilities = torch.softmax(sampling_logits, dim=-1)
            probabilities = self._apply_top_p(probabilities, body.top_p)
            if body.temperature == 0:
                selected_id = int(torch.argmax(probabilities).item())
            else:
                selected_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
            effective_top_k = min(body.top_k, probabilities.shape[-1])
            top_probabilities, top_indices = torch.topk(probabilities, effective_top_k)
            token = self.tokenizer.decode([selected_id], clean_up_tokenization_spaces=False)
            generated.append(selected_id)
            text_so_far += token
            sequence = torch.cat(
                (sequence, torch.tensor([[selected_id]], device=self.device, dtype=sequence.dtype)), dim=1
            )
            minimum_reached = step + 1 >= body.min_tokens
            complete = minimum_reached and (
                selected_id == self.tokenizer.eos_token_id
                or any(text_so_far.endswith(stop) for stop in stop_sequences)
            )
            hidden_summary = None
            if body.include_hidden_state_summary and output.hidden_states:
                hidden = output.hidden_states[-1][0, -1].float()
                hidden_summary = {
                    "shape": list(hidden.shape),
                    "mean": round(float(hidden.mean().item()), 6),
                    "l2_norm": round(float(torch.linalg.vector_norm(hidden).item()), 6),
                }
            yield {
                "step": step,
                "selected": {
                    "token_id": selected_id,
                    "token": token,
                    "probability": round(float(probabilities[selected_id].item()), 8),
                },
                "alternatives": [
                    {
                        "token_id": int(token_id),
                        "token": self.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False),
                        "probability": round(float(probability), 8),
                    }
                    for probability, token_id in zip(
                        top_probabilities.tolist(), top_indices.tolist(), strict=True
                    )
                ],
                "prompt_token_ids": prompt_ids if step == 0 else None,
                "prompt_tokens": prompt_tokens if step == 0 else None,
                "user_prompt_token_ids": user_prompt_ids if step == 0 else None,
                "user_prompt_tokens": user_prompt_tokens if step == 0 else None,
                "generated_token_ids": list(generated),
                "text_so_far": text_so_far,
                "timestamp": time.time(),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "hidden_state_summary": hidden_summary,
                "cancelled": False,
                "complete": complete,
            }
            if complete:
                return

    def _apply_top_p(self, probabilities: Any, top_p: float) -> Any:
        if top_p >= 1:
            return probabilities
        torch = self.torch
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities > top_p
        sorted_probabilities[remove] = 0
        filtered = torch.zeros_like(probabilities).scatter(0, sorted_indices, sorted_probabilities)
        return filtered / filtered.sum()


def create_app(
    *,
    worker_id: str,
    config: EngineConfig,
    engine: AutoregressiveEngine | None = None,
) -> FastAPI:
    runtime = engine or TransformersAutoregressiveEngine(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.worker_state = WorkerState.LOADING
        app.state.ready = False
        app.state.load_error = None
        app.state.requests = 0
        app.state.cancelled_requests = 0
        app.state.cancellations = {}
        app.state.generation_lock = asyncio.Lock()
        app.state.load_task = asyncio.create_task(_load_engine(app, runtime))
        yield
        if not app.state.load_task.done():
            app.state.load_task.cancel()

    app = FastAPI(title=f"ModelDeck Transformers worker: {worker_id}", lifespan=lifespan)
    app.state.shutdown_callback = None

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        details = runtime.runtime_details
        return {
            "protocol_version": "1",
            "worker_id": worker_id,
            "runtime": "transformers-rocm",
            "generation_family": GenerationFamily.AUTOREGRESSIVE,
            "state": request.app.state.worker_state,
            "model_id": config.model_id,
            "model_revision": config.revision,
            "device": details.get("device", "cuda:0"),
            "device_name": details.get("device_name", "AMD GPU"),
            "rocm_version": details.get("hip_version"),
            "ready": request.app.state.ready,
            "error": request.app.state.load_error,
        }

    @app.get("/capabilities")
    async def capabilities() -> dict[str, Any]:
        result = CapabilitySet(
            chat=True,
            completions=True,
            streaming=True,
            cancellation=True,
            logits=True,
            top_k_trace=True,
            hidden_states="optional",
            seeded_generation=True,
        )
        return {
            "protocol_version": "1",
            "generation_family": GenerationFamily.AUTOREGRESSIVE,
            **result.model_dump(),
        }

    @app.get("/metrics")
    async def metrics(request: Request) -> dict[str, Any]:
        memory_metrics = getattr(runtime, "memory_metrics", lambda: {})()
        return {
            **runtime.runtime_details,
            **memory_metrics,
            "requests": request.app.state.requests,
            "cancelled_requests": request.app.state.cancelled_requests,
            "busy": request.app.state.generation_lock.locked(),
        }

    @app.get("/model")
    async def model() -> dict[str, Any]:
        return {
            "model_id": config.model_id,
            "revision": config.revision,
            "generation_family": GenerationFamily.AUTOREGRESSIVE,
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": config.dtype,
        }

    @app.post("/load")
    async def load(request: Request) -> dict[str, Any]:
        return {"ok": request.app.state.load_error is None, "state": request.app.state.worker_state}

    @app.post("/warmup")
    async def warmup(request: Request) -> dict[str, Any]:
        await request.app.state.load_task
        if request.app.state.load_error:
            raise HTTPException(503, request.app.state.load_error)
        request.app.state.worker_state = WorkerState.WARMING
        try:
            await asyncio.to_thread(runtime.warmup)
        except Exception as error:
            request.app.state.worker_state = WorkerState.FAILED
            request.app.state.load_error = f"Warmup failed: {type(error).__name__}: {error}"
            raise HTTPException(500, request.app.state.load_error) from error
        request.app.state.ready = True
        request.app.state.worker_state = WorkerState.READY
        return {"ok": True, "ready": True}

    @app.post("/cancel")
    async def cancel(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id", ""))
        cancellation = request.app.state.cancellations.get(request_id)
        if cancellation:
            cancellation.set()
            request.app.state.cancelled_requests += 1
        return {"ok": bool(cancellation), "request_id": request_id}

    @app.post("/shutdown")
    async def shutdown(request: Request) -> dict[str, bool]:
        request.app.state.worker_state = WorkerState.STOPPING
        for cancellation in request.app.state.cancellations.values():
            cancellation.set()
        if request.app.state.shutdown_callback:
            asyncio.get_running_loop().call_later(0.05, request.app.state.shutdown_callback)
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat(request: Request, body: GenerationRequest):
        return await _generate_response(request, body, runtime, chat=True)

    @app.post("/v1/completions")
    async def completions(request: Request, body: GenerationRequest):
        return await _generate_response(request, body, runtime, chat=False)

    @app.post("/native/autoregressive/trace")
    async def trace(request: Request, body: GenerationRequest):
        return await _trace_response(request, body, runtime)

    return app


async def _load_engine(app: FastAPI, engine: AutoregressiveEngine) -> None:
    try:
        await asyncio.to_thread(engine.load)
        app.state.worker_state = WorkerState.WARMING
    except Exception as error:
        app.state.load_error = f"Load failed: {type(error).__name__}: {error}"
        app.state.worker_state = WorkerState.FAILED


async def _trace_response(request: Request, body: GenerationRequest, engine: AutoregressiveEngine):
    _ensure_ready(request)
    request_id = body.request_id or str(uuid.uuid4())
    body.request_id = request_id
    prompt = engine.build_prompt(body)
    cancellation = threading.Event()
    request.app.state.cancellations[request_id] = cancellation
    if body.stream:
        return StreamingResponse(
            _stream_trace(request, body, engine, prompt, cancellation),
            media_type="text/event-stream",
        )
    async with request.app.state.generation_lock:
        request.app.state.worker_state = WorkerState.BUSY
        started = time.perf_counter()
        try:
            events = await asyncio.to_thread(
                list,
                engine.trace(prompt=prompt, body=body, cancellation=cancellation),
            )
            token_metadata = _trace_token_metadata(events)
            request.app.state.requests += 1
            return {
                "request_id": request_id,
                "model": body.model,
                **token_metadata,
                "events": events,
                "metrics": _request_metrics(events, started),
            }
        finally:
            request.app.state.cancellations.pop(request_id, None)
            request.app.state.worker_state = WorkerState.READY


async def _generate_response(
    request: Request,
    body: GenerationRequest,
    engine: AutoregressiveEngine,
    *,
    chat: bool,
):
    if body.stream:
        trace_response = await _trace_response(request, body, engine)
        return trace_response
    result = await _trace_response(request, body, engine)
    events = result["events"]
    text = events[-1].get("text_so_far", "") if events else ""
    choice = (
        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        if chat
        else {"index": 0, "text": text, "finish_reason": "stop"}
    )
    return {
        "id": result["request_id"],
        "object": "chat.completion" if chat else "text_completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [choice],
        "metrics": result["metrics"],
    }


async def _stream_trace(
    request: Request,
    body: GenerationRequest,
    engine: AutoregressiveEngine,
    prompt: str,
    cancellation: threading.Event,
) -> AsyncIterator[str]:
    request_id = body.request_id or "unknown"
    async with request.app.state.generation_lock:
        request.app.state.worker_state = WorkerState.BUSY
        iterator = engine.trace(prompt=prompt, body=body, cancellation=cancellation)
        try:
            while True:
                event = await asyncio.to_thread(_next_event, iterator)
                if event is None:
                    break
                name = "cancelled" if event.get("cancelled") else "token"
                payload = {"request_id": request_id, **event}
                yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"
            yield "event: complete\ndata: [DONE]\n\n"
            request.app.state.requests += 1
        finally:
            request.app.state.cancellations.pop(request_id, None)
            request.app.state.worker_state = WorkerState.READY


def _next_event(iterator: Iterator[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _latest_user_prompt(body: GenerationRequest) -> str:
    if not body.messages:
        return body.prompt or ""
    return next(
        (message.content for message in reversed(body.messages) if message.role == "user"),
        "",
    )


def _tokenise_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [
        str(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        for token_id in token_ids
    ]


def _trace_token_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    first = events[0] if events else {}
    metadata = {
        "prompt_token_ids": first.get("prompt_token_ids", []),
        "prompt_tokens": first.get("prompt_tokens", []),
        "user_prompt_token_ids": first.get("user_prompt_token_ids", []),
        "user_prompt_tokens": first.get("user_prompt_tokens", []),
    }
    error = _token_metadata_error(metadata)
    if error:
        raise HTTPException(500, f"Worker produced invalid trace token metadata: {error}")
    return metadata


def _token_metadata_error(metadata: dict[str, Any]) -> str | None:
    prompt_ids = metadata.get("prompt_token_ids")
    prompt_tokens = metadata.get("prompt_tokens")
    user_ids = metadata.get("user_prompt_token_ids")
    user_tokens = metadata.get("user_prompt_tokens")
    if not isinstance(prompt_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in prompt_ids
    ):
        return "prompt_token_ids must be an array of integers"
    if not isinstance(prompt_tokens, list) or not all(isinstance(token, str) for token in prompt_tokens):
        return "prompt_tokens must be an array of strings"
    if len(prompt_tokens) != len(prompt_ids):
        return "prompt_tokens must contain one entry for every prompt_token_ids entry"
    if not isinstance(user_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in user_ids
    ):
        return "user_prompt_token_ids must be an array of integers"
    if not isinstance(user_tokens, list) or not all(isinstance(token, str) for token in user_tokens):
        return "user_prompt_tokens must be an array of strings"
    if len(user_tokens) != len(user_ids):
        return "user_prompt_tokens must contain one entry for every user_prompt_token_ids entry"
    return None


def _request_metrics(events: list[dict[str, Any]], started: float) -> dict[str, Any]:
    total = time.perf_counter() - started
    generated = len([event for event in events if event.get("selected")])
    first = events[0].get("elapsed_seconds") if events else None
    return {
        "first_token_seconds": first,
        "total_seconds": round(total, 6),
        "generated_tokens": generated,
        "tokens_per_second": round(generated / total, 4) if total else None,
        "cancelled": any(event.get("cancelled") for event in events),
    }


def _ensure_ready(request: Request) -> None:
    if not request.app.state.ready:
        raise HTTPException(503, "Worker is not ready")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ModelDeck autoregressive ROCm worker")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--maximum-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EngineConfig(
        model_id=args.model_id,
        revision=args.revision,
        dtype=args.dtype,
        context_length=args.context_length,
        maximum_new_tokens=args.maximum_new_tokens,
    )
    app = create_app(worker_id=args.worker_id, config=config)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)
    )
    app.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

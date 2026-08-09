from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import inspect
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modeldeck.protocol import CapabilitySet, GenerationFamily, WorkerState
from modeldeck.registry import MAXIMUM_NEW_TOKENS_LIMIT

LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class EngineConfig:
    model_id: str
    revision: str
    dtype: str = "float16"
    context_length: int = 2048
    maximum_new_tokens: int = 128


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(pattern=r"^(system|user|assistant|tool)$")
    content: str | None = None
    name: str | None = Field(default=None, max_length=128)
    tool_call_id: str | None = Field(default=None, max_length=256)
    tool_calls: list[dict[str, Any]] | None = None

    @field_validator("content", mode="before")
    @classmethod
    def normalise_openai_text_content_parts(cls, value: Any) -> Any:
        """Accept OpenAI text content parts for the text-only local backends."""

        if not isinstance(value, list):
            return value
        text_parts: list[str] = []
        for index, part in enumerate(value):
            if not isinstance(part, dict) or part.get("type") != "text":
                raise ValueError(f"content part {index} must be a text part; this local backend is text-only")
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError(f"content text part {index} requires a string text value")
            text_parts.append(text)
        return "".join(text_parts)

    @model_validator(mode="after")
    def openai_message_shape(self) -> ChatMessage:
        if self.role == "tool" and (not self.tool_call_id or self.content is None):
            raise ValueError("tool messages require tool_call_id and content")
        if self.role == "assistant" and self.content is None and not self.tool_calls:
            raise ValueError("assistant messages require content or tool_calls")
        if self.role in {"system", "user"} and self.content is None:
            raise ValueError(f"{self.role} messages require content")
        return self


class GenerationRequest(BaseModel):
    request_id: str | None = None
    model: str = "local-worker"
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    stream: bool = False
    seed: int = 7
    max_tokens: int = Field(default=32, ge=1, le=MAXIMUM_NEW_TOKENS_LIMIT)
    min_tokens: int = Field(default=0, ge=0, le=MAXIMUM_NEW_TOKENS_LIMIT)
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
        self._supports_logits_to_keep = False

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
        supported_context_length = _model_context_length(model)
        if self.config.context_length > supported_context_length:
            raise RuntimeError(
                "Configured context length "
                f"{self.config.context_length} exceeds the model-supported limit {supported_context_length}"
            )
        device = torch.device("cuda:0")
        model.to(device)
        model.eval()
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self._supports_logits_to_keep = "logits_to_keep" in inspect.signature(model.forward).parameters
        self.runtime_details = {
            "torch_version": str(torch.__version__),
            "hip_version": torch.version.hip,
            "transformers_version": importlib.metadata.version("transformers"),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "load_seconds": round(time.perf_counter() - started, 4),
            "dtype": self.config.dtype,
            "model_max_context_tokens": supported_context_length,
            "last_token_logits_only": self._supports_logits_to_keep,
        }

    def warmup(self) -> None:
        body = GenerationRequest(prompt="Hello", max_tokens=1, temperature=0, top_k=1)
        list(self.trace(prompt="Hello", body=body, cancellation=threading.Event()))

    def build_prompt(self, body: GenerationRequest) -> str:
        if body.messages:
            messages = [message.model_dump(exclude_none=True) for message in body.messages]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=body.tools,
                tool_choice=body.tool_choice,
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

    def validate_token_budget(self, prompt: str, body: GenerationRequest) -> None:
        """Reject requests that cannot fit before allocating inference tensors."""

        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        if prompt_tokens + body.max_tokens > self.config.context_length:
            raise ValueError(
                "Token budget exceeds worker context: "
                f"prompt_tokens={prompt_tokens}, requested_output_tokens={body.max_tokens}, "
                f"context_length={self.config.context_length}."
            )

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
        self.validate_token_budget(prompt, body)
        sequence = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        generated: list[int] = []
        text_so_far = ""
        stop_sequences = [body.stop] if isinstance(body.stop, str) else list(body.stop or ())
        generator = torch.Generator(device=self.device).manual_seed(body.seed)
        eos_token_ids = _configured_eos_token_ids(self.tokenizer, self.model)
        last_event: dict[str, Any] | None = None
        started = time.perf_counter()

        cached_input_ids: Any | None = None
        past_key_values: Any | None = None
        for step in range(min(body.max_tokens, self.config.maximum_new_tokens)):
            if cancellation.is_set():
                yield {"step": step, "cancelled": True, "complete": True, "text_so_far": text_so_far}
                return
            forward_arguments = {
                "input_ids": sequence if cached_input_ids is None else cached_input_ids,
                "use_cache": True,
                "output_hidden_states": body.include_hidden_state_summary,
            }
            if attention_mask is not None:
                forward_arguments["attention_mask"] = attention_mask
            if past_key_values is not None:
                forward_arguments["past_key_values"] = past_key_values
            if self._supports_logits_to_keep:
                forward_arguments["logits_to_keep"] = 1
            with torch.inference_mode():
                output = self.model(**forward_arguments)
            past_key_values = getattr(output, "past_key_values", None)
            if past_key_values is None:
                raise RuntimeError(
                    "The loaded causal language model did not return past_key_values with use_cache=True"
                )
            logits = output.logits[0, -1].float()
            if body.repetition_penalty != 1 and generated:
                for token_id in set(generated):
                    logits[token_id] = (
                        logits[token_id] / body.repetition_penalty
                        if logits[token_id] > 0
                        else logits[token_id] * body.repetition_penalty
                    )
            if len(generated) < body.min_tokens:
                _suppress_token_logits(logits, eos_token_ids)
            sampling_logits = logits if body.temperature == 0 else logits / max(body.temperature, 1e-6)
            probabilities = torch.softmax(sampling_logits, dim=-1)
            probabilities = self._apply_top_p(probabilities, body.top_p)
            if body.temperature == 0:
                selected_id = int(torch.argmax(probabilities).item())
            else:
                selected_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
            if selected_id in eos_token_ids:
                if last_event is not None:
                    last_event["complete"] = True
                return
            effective_top_k = min(body.top_k, probabilities.shape[-1])
            top_probabilities, top_indices = torch.topk(probabilities, effective_top_k)
            token = self.tokenizer.decode([selected_id], clean_up_tokenization_spaces=False)
            generated.append(selected_id)
            text_so_far += token
            cached_input_ids = torch.tensor([[selected_id]], device=self.device, dtype=sequence.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype),
                    ),
                    dim=1,
                )
            complete = len(generated) >= body.min_tokens and any(
                text_so_far.endswith(stop) for stop in stop_sequences
            )
            hidden_summary = None
            if body.include_hidden_state_summary and output.hidden_states:
                hidden = output.hidden_states[-1][0, -1].float()
                hidden_summary = {
                    "shape": list(hidden.shape),
                    "mean": round(float(hidden.mean().item()), 6),
                    "l2_norm": round(float(torch.linalg.vector_norm(hidden).item()), 6),
                }
            event = {
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
            last_event = event
            yield event
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


class _RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Bound JSON request bytes before Pydantic receives a potentially huge payload."""

    def __init__(self, app: Any, maximum_bytes: int = MAX_REQUEST_BYTES) -> None:
        self.app = app
        self.maximum_bytes = maximum_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", ()))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.maximum_bytes:
                    await _error_response(
                        413,
                        "request_too_large",
                        "The JSON request exceeds 4 MiB.",
                    )(scope, receive, send)
                    return
            except ValueError:
                await _error_response(
                    422,
                    "invalid_request",
                    "Content-Length must be an integer.",
                )(scope, receive, send)
                return
        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.maximum_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await _error_response(
                413,
                "request_too_large",
                "The JSON request exceeds 4 MiB.",
            )(scope, receive, send)


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
    app.add_middleware(RequestBodyLimitMiddleware)
    app.state.shutdown_callback = None

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        # Pydantic's error records include the rejected `input`, which may be prompt text.
        # Keep only structural diagnostics in local logs and return a stable safe error body.
        diagnostics = [{"type": item.get("type"), "loc": item.get("loc")} for item in error.errors()]
        LOGGER.info("Request validation failed path=%s diagnostics=%s", request.url.path, diagnostics)
        return _error_response(
            422,
            "invalid_request",
            "The request does not match the local OpenAI chat contract.",
        )

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
    request_id = body.request_id or request.headers.get("x-request-id") or str(uuid.uuid4())
    body.request_id = request_id
    prompt = engine.build_prompt(body)
    validate_token_budget = getattr(engine, "validate_token_budget", None)
    if callable(validate_token_budget):
        try:
            validate_token_budget(prompt, body)
        except ValueError as error:
            LOGGER.info("Request rejected request_id=%s reason=context_token_budget", request_id)
            return _error_response(422, "context_length_exceeded", str(error))
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
    if isinstance(result, JSONResponse):
        return result
    events = result["events"]
    text = events[-1].get("text_so_far", "") if events else ""
    tool_calls, content = _openai_tool_calls(text)
    choice = (
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            },
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }
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


def _model_context_length(model: Any) -> int:
    value = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 256:
        raise RuntimeError("Loaded model does not declare a safe maximum context length")
    return value


def _latest_user_prompt(body: GenerationRequest) -> str:
    if not body.messages:
        return body.prompt or ""
    return next(
        (message.content or "" for message in reversed(body.messages) if message.role == "user"),
        "",
    )


_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _openai_tool_calls(text: str) -> tuple[list[dict[str, Any]], str | None]:
    """Translate Qwen's documented tool-call envelope to OpenAI's response shape."""

    calls: list[dict[str, Any]] = []
    for index, match in enumerate(_TOOL_CALL.finditer(text)):
        try:
            call = json.loads(match.group(1))
            name = str(call["name"])
            arguments = call.get("arguments", {})
        except (KeyError, TypeError, ValueError):
            continue
        calls.append(
            {
                "id": f"call_{index}_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                },
            }
        )
    content = _TOOL_CALL.sub("", text).strip()
    return calls, content or None


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


def _configured_eos_token_ids(tokenizer: Any, model: Any) -> set[int]:
    values = [getattr(tokenizer, "eos_token_id", None)]
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        values.append(getattr(generation_config, "eos_token_id", None))
    token_ids: set[int] = set()
    for value in values:
        candidates = value if isinstance(value, (list, tuple, set)) else [value]
        token_ids.update(
            int(candidate)
            for candidate in candidates
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0
        )
    return token_ids


def _suppress_token_logits(logits: Any, token_ids: set[int]) -> None:
    for token_id in token_ids:
        if token_id < len(logits):
            logits[token_id] = float("-inf")


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


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "param": None,
                "code": code,
            }
        },
    )


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

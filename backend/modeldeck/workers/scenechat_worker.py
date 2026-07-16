from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import hmac
import importlib.metadata
import io
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from modeldeck.contracts.scenechat import (
    CONTRACT_VERSION,
    canonicalise_model_output,
    extract_curated_question,
    system_messages,
)
from modeldeck.protocol import CapabilitySet, GenerationFamily, WorkerState

LOGGER = logging.getLogger("modeldeck.scenechat")
MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
MAX_IMAGE_PIXELS = 16_000_000
SUPPORTED_MIME_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}
SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")
APPROVED_FP32_BUFFER_SUFFIXES = (
    "input_min",
    "input_max",
    "output_min",
    "output_max",
    "inv_timescales",
    "softcap",
    "inv_freq",
    "original_inv_freq",
    "layer_scalar",
    "embed_scale",
    "std_bias",
    "std_scale",
)


@dataclass(frozen=True)
class EngineConfig:
    model_id: str
    revision: str
    cache_root: Path = Path("/mnt/work/models/huggingface/hub")
    dtype: str = "bfloat16"
    context_length: int = 8192
    maximum_new_tokens: int = 256
    generation_timeout_seconds: float = 60.0


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cancelled: bool = False


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_object"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user"]
    content: list[dict[str, Any]] = Field(min_length=2, max_length=2)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=1)
    temperature: Literal[0.1]
    max_tokens: int = Field(ge=1, le=700)
    response_format: ResponseFormat
    stream: Literal[False] = False


class VisionLanguageEngine(Protocol):
    runtime_details: dict[str, Any]

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def generate(
        self,
        *,
        image: Image.Image,
        question: str,
        max_tokens: int,
        cancellation: threading.Event,
    ) -> GenerationResult: ...

    def memory_metrics(self) -> dict[str, int]: ...

    def close(self) -> None: ...


class TransformersSceneChatEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.runtime_details: dict[str, Any] = {}
        self.torch: Any = None
        self.processor: Any = None
        self.model: Any = None
        self.device: Any = None
        self.dtype: Any = None

    @property
    def snapshot_path(self) -> Path:
        organisation, model = self.config.model_id.split("/", maxsplit=1)
        return (
            self.config.cache_root / f"models--{organisation}--{model}" / "snapshots" / self.config.revision
        )

    def load(self) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        snapshot = self._validate_snapshot()
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm PyTorch did not expose an available 'cuda' device")
        if self.config.dtype != "bfloat16":
            raise RuntimeError("The SceneChat Gemma 4 profile requires bfloat16")
        dtype = torch.bfloat16
        device = torch.device("cuda:0")
        try:
            torch.empty(1, device=device, dtype=dtype)
        except Exception as error:
            raise RuntimeError("The detected GPU could not allocate a BF16 tensor") from error

        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        if type(processor).__name__ != "Gemma4Processor":
            raise RuntimeError(f"Expected Gemma4Processor, received {type(processor).__name__}")
        model = AutoModelForMultimodalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        if type(model).__name__ != "Gemma4ForConditionalGeneration":
            raise RuntimeError(f"Expected Gemma4ForConditionalGeneration, received {type(model).__name__}")
        model.to(device)
        model.eval()
        placement_details = self._validate_placement(model, device, dtype)
        self.torch = torch
        self.processor = processor
        self.model = model
        self.device = device
        self.dtype = dtype
        self.runtime_details = {
            "torch_version": str(torch.__version__),
            "hip_version": torch.version.hip,
            "transformers_version": importlib.metadata.version("transformers"),
            "processor_class": type(processor).__name__,
            "model_class": type(model).__name__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "dtype": self.config.dtype,
            "attention_implementation": "sdpa",
            **placement_details,
            "load_seconds": round(time.perf_counter() - started, 4),
            "snapshot_path": str(snapshot),
        }
        LOGGER.info(
            "Model loaded revision=%s duration_seconds=%.4f device=%s",
            self.config.revision,
            self.runtime_details["load_seconds"],
            self.runtime_details["device_name"],
        )

    def _validate_snapshot(self) -> Path:
        snapshot = self.snapshot_path
        if not snapshot.is_dir():
            raise RuntimeError(
                f"Pinned local snapshot is missing for {self.config.model_id} at revision "
                f"{self.config.revision}"
            )
        required = {
            "config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
        }
        missing = sorted(name for name in required if not (snapshot / name).is_file())
        if not list(snapshot.glob("*.safetensors")):
            missing.append("*.safetensors")
        if missing:
            raise RuntimeError(f"Pinned snapshot is incomplete; missing: {', '.join(missing)}")
        return snapshot.resolve(strict=True)

    @staticmethod
    def _validate_placement(model: Any, device: Any, dtype: Any) -> dict[str, Any]:
        unexpected_devices: list[str] = []
        unexpected_dtypes: list[str] = []
        parameter_dtypes: set[str] = set()
        buffer_dtypes: set[str] = set()
        approved_fp32_buffers: list[str] = []
        for name, parameter in model.named_parameters():
            if parameter.device != device:
                unexpected_devices.append(f"parameter {name}={parameter.device}")
            if parameter.is_floating_point():
                parameter_dtypes.add(str(parameter.dtype))
                if parameter.dtype != dtype:
                    unexpected_dtypes.append(f"parameter {name}={parameter.dtype}")
        for name, buffer in model.named_buffers():
            if buffer.device != device:
                unexpected_devices.append(f"buffer {name}={buffer.device}")
            if not buffer.is_floating_point():
                continue
            buffer_dtypes.add(str(buffer.dtype))
            if buffer.dtype == dtype:
                continue
            if str(buffer.dtype) == "torch.float32" and name.endswith(APPROVED_FP32_BUFFER_SUFFIXES):
                approved_fp32_buffers.append(name)
                continue
            unexpected_dtypes.append(f"buffer {name}={buffer.dtype}")
        if unexpected_devices:
            raise RuntimeError(
                "Model contains tensors outside cuda:0: " + ", ".join(sorted(unexpected_devices)[:10])
            )
        if unexpected_dtypes:
            raise RuntimeError(
                "Model contains unexpected floating dtypes: " + ", ".join(sorted(unexpected_dtypes)[:10])
            )
        return {
            "parameter_dtypes": sorted(parameter_dtypes),
            "buffer_dtypes": sorted(buffer_dtypes),
            "approved_fp32_buffer_count": len(approved_fp32_buffers),
        }

    def warmup(self) -> None:
        image = Image.new("RGB", (64, 64), color=(80, 100, 120))
        try:
            self.generate(
                image=image,
                question="Describe the scene.",
                max_tokens=1,
                cancellation=threading.Event(),
            )
        finally:
            image.close()

    def generate(
        self,
        *,
        image: Image.Image,
        question: str,
        max_tokens: int,
        cancellation: threading.Event,
    ) -> GenerationResult:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class CancellationCriteria(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return cancellation.is_set()

        messages = system_messages(question)
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(text=rendered, images=[image], return_tensors="pt")
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        if prompt_tokens + max_tokens > self.config.context_length:
            raise ValueError(
                f"Processed input plus requested output exceeds {self.config.context_length} tokens"
            )
        inputs = inputs.to(self.device, dtype=self.dtype)
        self.torch.cuda.reset_peak_memory_stats(0)
        try:
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, self.config.maximum_new_tokens),
                    do_sample=False,
                    use_cache=True,
                    stopping_criteria=StoppingCriteriaList([CancellationCriteria()]),
                )
            generated = output[0, prompt_tokens:]
            completion_tokens = int(generated.shape[-1])
            decoded = self.processor.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if any(marker in decoded.casefold() for marker in ("<think>", "</think>", "<|channel>")):
                raise ValueError("Model output exposed a reasoning channel")
            return GenerationResult(
                text=decoded,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cancelled=cancellation.is_set(),
            )
        finally:
            del inputs

    def memory_metrics(self) -> dict[str, int]:
        if self.torch is None or not self.torch.cuda.is_available():
            return {}
        return {
            "memory_allocated_bytes": int(self.torch.cuda.memory_allocated(0)),
            "memory_reserved_bytes": int(self.torch.cuda.memory_reserved(0)),
            "peak_memory_allocated_bytes": int(self.torch.cuda.max_memory_allocated(0)),
            "peak_memory_reserved_bytes": int(self.torch.cuda.max_memory_reserved(0)),
        }

    def close(self) -> None:
        self.processor = None
        self.model = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class SceneChatRequestError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


class _RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
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
                        "The JSON request exceeds 12 MiB.",
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
                "The JSON request exceeds 12 MiB.",
            )(scope, receive, send)


def create_app(
    *,
    worker_id: str,
    config: EngineConfig,
    api_key: str = "local",
    engine: VisionLanguageEngine | None = None,
) -> FastAPI:
    runtime = engine or TransformersSceneChatEngine(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.worker_state = WorkerState.LOADING
        app.state.ready = False
        app.state.load_error = None
        app.state.active_request_id = None
        app.state.active_cancellation = None
        app.state.slot_guard = asyncio.Lock()
        app.state.requests = 0
        app.state.successes = 0
        app.state.failures = 0
        app.state.cancelled_requests = 0
        app.state.timed_out_requests = 0
        app.state.concurrent_rejections = 0
        app.state.total_latency_seconds = 0.0
        app.state.load_task = asyncio.create_task(_load_engine(app, runtime))
        try:
            yield
        finally:
            cancellation = app.state.active_cancellation
            if cancellation is not None:
                cancellation.set()
            if not app.state.load_task.done():
                app.state.load_task.cancel()
            await asyncio.to_thread(runtime.close)

    app = FastAPI(
        title=f"ModelDeck SceneChat worker: {worker_id}",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.add_middleware(RequestBodyLimitMiddleware)
    app.state.shutdown_callback = None

    @app.exception_handler(SceneChatRequestError)
    async def scenechat_error(_request: Request, error: SceneChatRequestError) -> JSONResponse:
        return _error_response(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return _error_response(
            422,
            "invalid_request",
            "The request does not match the SceneChat API contract.",
        )

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, error: Exception) -> JSONResponse:
        LOGGER.error("Request failed category=%s", type(error).__name__)
        return _error_response(502, "worker_error", "The local vision worker could not complete the request.")

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        details = runtime.runtime_details
        return {
            "protocol_version": "1",
            "worker_id": worker_id,
            "runtime": "vision-language-transformers-rocm",
            "generation_family": GenerationFamily.VISION_LANGUAGE,
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
            chat="compatibility-only",
            streaming=False,
            cancellation=True,
            image_input=True,
            structured_output=True,
        )
        return {
            "protocol_version": "1",
            "generation_family": GenerationFamily.VISION_LANGUAGE,
            **result.model_dump(),
        }

    @app.get("/metrics")
    async def metrics(request: Request) -> dict[str, Any]:
        state = request.app.state
        average = state.total_latency_seconds / state.successes if state.successes else None
        return {
            **runtime.runtime_details,
            **runtime.memory_metrics(),
            "requests": state.requests,
            "successful_requests": state.successes,
            "failed_requests": state.failures,
            "cancelled_requests": state.cancelled_requests,
            "timed_out_requests": state.timed_out_requests,
            "concurrent_rejections": state.concurrent_rejections,
            "aggregate_latency_seconds": round(state.total_latency_seconds, 4),
            "average_latency_seconds": round(average, 4) if average is not None else None,
            "busy": state.active_request_id is not None,
        }

    @app.get("/model")
    async def model() -> dict[str, Any]:
        return {
            "model_id": config.model_id,
            "revision": config.revision,
            "generation_family": GenerationFamily.VISION_LANGUAGE,
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": config.dtype,
            "contract_version": CONTRACT_VERSION,
        }

    @app.post("/load")
    async def load(request: Request) -> dict[str, Any]:
        return {"ok": request.app.state.load_error is None, "state": request.app.state.worker_state}

    @app.post("/warmup")
    async def warmup(request: Request) -> dict[str, Any]:
        await request.app.state.load_task
        if request.app.state.load_error:
            raise SceneChatRequestError(503, "model_unavailable", request.app.state.load_error)
        request.app.state.worker_state = WorkerState.WARMING
        try:
            await asyncio.to_thread(runtime.warmup)
        except Exception as error:
            request.app.state.worker_state = WorkerState.FAILED
            request.app.state.load_error = f"Warm-up failed: {type(error).__name__}: {error}"
            raise SceneChatRequestError(
                503,
                "model_unavailable",
                "The pinned model failed its local warm-up.",
            ) from error
        request.app.state.ready = True
        request.app.state.worker_state = WorkerState.READY
        return {"ok": True, "ready": True}

    @app.post("/cancel")
    async def cancel(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id", ""))
        if request_id and hmac.compare_digest(request_id, request.app.state.active_request_id or ""):
            request.app.state.active_cancellation.set()
            request.app.state.cancelled_requests += 1
            LOGGER.info("Request cancelled request_id=%s", request_id)
            return {"ok": True, "request_id": request_id}
        return {"ok": False, "request_id": request_id}

    @app.post("/shutdown")
    async def shutdown(request: Request) -> dict[str, bool]:
        request.app.state.worker_state = WorkerState.STOPPING
        if request.app.state.active_cancellation is not None:
            request.app.state.active_cancellation.set()
        if request.app.state.shutdown_callback:
            asyncio.get_running_loop().call_later(0.1, request.app.state.shutdown_callback)
        return {"ok": True}

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, Any]:
        _authorise(request, api_key)
        _ensure_ready(request)
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "google",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request, body: ChatCompletionRequest) -> JSONResponse:
        _authorise(request, api_key)
        _ensure_ready(request)
        if body.model != config.model_id:
            raise SceneChatRequestError(422, "invalid_model", "The model identifier is not allowlisted.")
        image_data_url, prompt = _validate_message(body.messages[0])
        question = _approved_question(prompt)
        image = _decode_image(image_data_url)
        supplied_request_id = request.headers.get("x-request-id")
        if supplied_request_id and not SAFE_REQUEST_ID.fullmatch(supplied_request_id):
            image.close()
            raise SceneChatRequestError(
                422,
                "invalid_request_id",
                "X-Request-ID must use 1 to 80 safe identifier characters.",
            )
        request_id = supplied_request_id or f"chatcmpl-{uuid.uuid4().hex}"
        cancellation = threading.Event()
        claimed = await _claim_slot(request, request_id, cancellation)
        if not claimed:
            image.close()
            request.app.state.concurrent_rejections += 1
            LOGGER.info("Concurrent request rejected")
            raise SceneChatRequestError(
                429,
                "worker_busy",
                "The vision worker is already processing a request.",
            )
        started = time.perf_counter()
        request.app.state.requests += 1
        request.app.state.worker_state = WorkerState.BUSY
        try:
            result = await _run_generation(
                request,
                runtime,
                image=image,
                question=question,
                max_tokens=min(body.max_tokens, config.maximum_new_tokens),
                cancellation=cancellation,
                timeout_seconds=config.generation_timeout_seconds,
            )
            if result.cancelled:
                raise SceneChatRequestError(504, "request_cancelled", "The local generation was cancelled.")
            try:
                content, _analysis = canonicalise_model_output(result.text)
            except ValueError as error:
                raise SceneChatRequestError(
                    502,
                    "invalid_model_output",
                    "The model returned output that did not satisfy the SceneChat contract.",
                ) from error
            latency = time.perf_counter() - started
            request.app.state.successes += 1
            request.app.state.total_latency_seconds += latency
            LOGGER.info(
                "Request completed request_id=%s duration_seconds=%.4f category=success",
                request_id,
                latency,
            )
            payload: dict[str, Any] = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": config.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
            if result.prompt_tokens >= 0 and result.completion_tokens >= 0:
                payload["usage"] = {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                }
            return JSONResponse(payload)
        except SceneChatRequestError:
            request.app.state.failures += 1
            raise
        finally:
            image.close()
            await _release_slot(request, request_id)

    @app.post("/native/vision-language/smoke")
    async def native_smoke(request: Request) -> dict[str, Any]:
        _authorise(request, api_key)
        _ensure_ready(request)
        image = Image.new("RGB", (64, 64), color=(60, 90, 120))
        cancellation = threading.Event()
        request_id = f"smoke-{uuid.uuid4().hex}"
        if not await _claim_slot(request, request_id, cancellation):
            image.close()
            raise SceneChatRequestError(
                429,
                "worker_busy",
                "The vision worker is already processing a request.",
            )
        try:
            result = await asyncio.to_thread(
                runtime.generate,
                image=image,
                question="Describe the scene.",
                max_tokens=config.maximum_new_tokens,
                cancellation=cancellation,
            )
            content, _analysis = canonicalise_model_output(result.text)
            return {
                "ok": True,
                "request_id": request_id,
                "model": config.model_id,
                "revision": config.revision,
                "contract_version": CONTRACT_VERSION,
                "content": content,
                "metrics": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                },
            }
        except ValueError as error:
            raise SceneChatRequestError(
                502,
                "invalid_model_output",
                "The model returned output that did not satisfy the SceneChat contract.",
            ) from error
        finally:
            image.close()
            await _release_slot(request, request_id)

    return app


async def _load_engine(app: FastAPI, engine: VisionLanguageEngine) -> None:
    try:
        await asyncio.to_thread(engine.load)
        app.state.worker_state = WorkerState.WARMING
    except Exception as error:
        app.state.load_error = f"Load failed: {type(error).__name__}: {error}"
        app.state.worker_state = WorkerState.FAILED
        LOGGER.error("Model load failed category=%s", type(error).__name__)


def _authorise(request: Request, api_key: str) -> None:
    supplied = request.headers.get("authorization", "")
    expected = f"Bearer {api_key}"
    if not hmac.compare_digest(supplied, expected):
        raise SceneChatRequestError(401, "unauthorised", "A valid local bearer token is required.")


def _ensure_ready(request: Request) -> None:
    if not request.app.state.ready:
        raise SceneChatRequestError(503, "model_not_ready", "The pinned local model is not ready.")


def _validate_message(message: ChatMessage) -> tuple[str, str]:
    image_part, text_part = message.content
    if set(image_part) != {"type", "image_url"} or image_part.get("type") != "image_url":
        raise SceneChatRequestError(422, "invalid_image", "The first content part must be one image_url.")
    image_url = image_part.get("image_url")
    if not isinstance(image_url, dict) or set(image_url) != {"url"} or not isinstance(image_url["url"], str):
        raise SceneChatRequestError(422, "invalid_image", "image_url must contain only a data URL.")
    if set(text_part) != {"type", "text"} or text_part.get("type") != "text":
        raise SceneChatRequestError(422, "invalid_prompt", "The second content part must be text.")
    prompt = text_part.get("text")
    if not isinstance(prompt, str):
        raise SceneChatRequestError(422, "invalid_prompt", "The SceneChat prompt must be text.")
    return image_url["url"], prompt


def _approved_question(prompt: str) -> str:
    try:
        return extract_curated_question(prompt)
    except ValueError as error:
        raise SceneChatRequestError(
            422,
            "unapproved_prompt",
            "The prompt does not match the pinned SceneChat contract.",
        ) from error


def _decode_image(data_url: str) -> Image.Image:
    if data_url.startswith("data:image/svg+xml"):
        raise SceneChatRequestError(422, "unsupported_image", "SVG images are not supported.")
    header, separator, encoded = data_url.partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        raise SceneChatRequestError(422, "invalid_image", "Only base64 JPEG or PNG data URLs are supported.")
    mime_type = header[5:-7]
    expected_format = SUPPORTED_MIME_TYPES.get(mime_type)
    if expected_format is None:
        raise SceneChatRequestError(422, "unsupported_image", "Only JPEG and PNG images are supported.")
    if not encoded:
        raise SceneChatRequestError(422, "invalid_image", "The image payload is empty.")
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 4:
        raise SceneChatRequestError(413, "image_too_large", "The decoded image exceeds 8 MiB.")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise SceneChatRequestError(
            422,
            "invalid_image",
            "The image payload is not strict base64.",
        ) from error
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise SceneChatRequestError(413, "image_too_large", "The decoded image exceeds 8 MiB.")
    try:
        source = Image.open(io.BytesIO(image_bytes))
        if source.format != expected_format:
            source.close()
            raise SceneChatRequestError(
                422,
                "image_mismatch",
                "The image MIME type does not match its bytes.",
            )
        width, height = source.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION or width * height > MAX_IMAGE_PIXELS:
            source.close()
            raise SceneChatRequestError(
                413,
                "image_dimensions",
                "The image dimensions exceed the worker limits.",
            )
        source.load()
        oriented = ImageOps.exif_transpose(source)
        converted = oriented.convert("RGB")
        if oriented is not source:
            oriented.close()
        source.close()
        return converted
    except SceneChatRequestError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as error:
        raise SceneChatRequestError(422, "invalid_image", "The image could not be decoded safely.") from error
    finally:
        del image_bytes


async def _claim_slot(request: Request, request_id: str, cancellation: threading.Event) -> bool:
    async with request.app.state.slot_guard:
        if request.app.state.active_request_id is not None:
            return False
        request.app.state.active_request_id = request_id
        request.app.state.active_cancellation = cancellation
        return True


async def _release_slot(request: Request, request_id: str) -> None:
    async with request.app.state.slot_guard:
        if request.app.state.active_request_id == request_id:
            request.app.state.active_request_id = None
            request.app.state.active_cancellation = None
            if request.app.state.ready:
                request.app.state.worker_state = WorkerState.READY


async def _run_generation(
    request: Request,
    engine: VisionLanguageEngine,
    *,
    image: Image.Image,
    question: str,
    max_tokens: int,
    cancellation: threading.Event,
    timeout_seconds: float,
) -> GenerationResult:
    task = asyncio.create_task(
        asyncio.to_thread(
            engine.generate,
            image=image,
            question=question,
            max_tokens=max_tokens,
            cancellation=cancellation,
        )
    )
    started = time.monotonic()
    timed_out = False
    disconnected = False
    disconnect_task = asyncio.create_task(request.is_disconnected())
    while not task.done():
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            timed_out = True
            cancellation.set()
            break
        done, _pending = await asyncio.wait(
            {task, disconnect_task},
            timeout=min(0.05, timeout_seconds - elapsed),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            break
        if disconnect_task in done:
            if disconnect_task.result():
                disconnected = True
                cancellation.set()
                break
            await asyncio.sleep(min(0.05, max(0.0, timeout_seconds - elapsed)))
            disconnect_task = asyncio.create_task(request.is_disconnected())
    if not disconnect_task.done():
        disconnect_task.cancel()
    result = await task
    if timed_out:
        request.app.state.timed_out_requests += 1
        LOGGER.info("Request timed out duration_seconds=%.4f", time.monotonic() - started)
        raise SceneChatRequestError(
            504,
            "generation_timeout",
            f"The local generation exceeded {timeout_seconds:g} seconds.",
        )
    if disconnected:
        request.app.state.cancelled_requests += 1
        raise SceneChatRequestError(504, "client_disconnected", "The client disconnected during generation.")
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelDeck SceneChat Gemma 4 worker")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--maximum-new-tokens", type=int, default=700)
    parser.add_argument("--generation-timeout-seconds", type=float, default=60.0)
    arguments = parser.parse_args()
    config = EngineConfig(
        model_id=arguments.model_id,
        revision=arguments.revision,
        cache_root=arguments.cache_root,
        dtype=arguments.dtype,
        context_length=arguments.context_length,
        maximum_new_tokens=arguments.maximum_new_tokens,
        generation_timeout_seconds=arguments.generation_timeout_seconds,
    )
    application = create_app(
        worker_id=arguments.worker_id,
        config=config,
        api_key=os.environ.get("MODELDECK_SCENECHAT_API_KEY", "local"),
    )
    server = uvicorn.Server(
        uvicorn.Config(
            application,
            host="127.0.0.1",
            port=arguments.port,
            access_log=False,
            log_level="info",
        )
    )
    application.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

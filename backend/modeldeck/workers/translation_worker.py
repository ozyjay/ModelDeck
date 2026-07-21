from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import re
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from modeldeck.protocol import CapabilitySet, GenerationFamily, WorkerState
from modeldeck.speechshift import SPEECHSHIFT_MODEL_SPECS, validate_speechshift_snapshot

LOGGER = logging.getLogger("modeldeck.translation")
SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$")
CANCELLATION_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class TranslationConfig:
    model_id: str
    revision: str
    alias: str
    cache_root: Path
    source_language: str
    target_language: str
    maximum_input_characters: int = 4000
    maximum_input_tokens: int = 512
    maximum_new_tokens: int = 512
    generation_timeout_seconds: float = 60.0

    @property
    def snapshot_path(self) -> Path:
        return self.cache_root / f"models--{self.model_id.replace('/', '--')}" / "snapshots" / self.revision


@dataclass
class TranslationResult:
    text: str
    input_tokens: int
    output_tokens: int
    inference_seconds: float
    cancelled: bool = False


class TranslationEngine(Protocol):
    runtime_details: dict[str, Any]

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def translate(self, text: str, cancellation: threading.Event) -> TranslationResult: ...

    def close(self) -> None: ...


@dataclass
class MarianTranslationEngine:
    config: TranslationConfig
    runtime_details: dict[str, Any] = field(default_factory=dict)
    torch: Any = None
    tokenizer: Any = None
    model: Any = None

    def load(self) -> None:
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        error = validate_speechshift_snapshot(
            self.config.snapshot_path, self.config.model_id, self.config.revision
        )
        if error:
            raise RuntimeError(error)
        started = time.perf_counter()
        self.tokenizer = MarianTokenizer.from_pretrained(
            self.config.snapshot_path,
            local_files_only=True,
        )
        self.model = MarianMTModel.from_pretrained(
            self.config.snapshot_path,
            local_files_only=True,
        ).to("cpu")
        self.model.eval()
        self.torch = torch
        self.runtime_details = {
            "device": "cpu",
            "device_name": "CPU",
            "torch_version": str(torch.__version__),
            "transformers_version": __import__("transformers").__version__,
            "load_seconds": round(time.perf_counter() - started, 4),
        }

    def warmup(self) -> None:
        result = self.translate("The local service is ready.", threading.Event())
        if not result.text:
            raise RuntimeError("The pinned translation model returned no warm-up output")

    def translate(self, text: str, cancellation: threading.Event) -> TranslationResult:
        from transformers import StoppingCriteria, StoppingCriteriaList

        if self.model is None or self.tokenizer is None or self.torch is None:
            raise RuntimeError("The translation model is not loaded")

        class CancellationCriteria(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return cancellation.is_set()

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )
        input_tokens = int(encoded["input_ids"].shape[-1])
        if input_tokens > self.config.maximum_input_tokens:
            del encoded
            raise TranslationRequestError(
                422,
                "input_too_long",
                f"Input exceeds {self.config.maximum_input_tokens} tokens.",
            )
        output = None
        decoded = ""
        try:
            started = time.perf_counter()
            with self.torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=self.config.maximum_new_tokens,
                    do_sample=False,
                    stopping_criteria=StoppingCriteriaList([CancellationCriteria()]),
                )
            inference_seconds = time.perf_counter() - started
            output_tokens = int(output[0].shape[-1])
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
            return TranslationResult(
                text=decoded,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                inference_seconds=inference_seconds,
                cancelled=cancellation.is_set(),
            )
        finally:
            del encoded
            if output is not None:
                del output
            decoded = ""

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.torch = None
        gc.collect()


class TranslationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$")
    model: str
    input: str
    source_language: str
    target_language: str


class TranslationRequestError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


def create_app(
    *,
    worker_id: str,
    config: TranslationConfig,
    engine: TranslationEngine | None = None,
) -> FastAPI:
    runtime = engine or MarianTranslationEngine(config)

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
        app.state.last_request = None
        app.state.load_task = asyncio.create_task(_load_engine(app, runtime))
        try:
            yield
        finally:
            if app.state.active_cancellation is not None:
                app.state.active_cancellation.set()
            if not app.state.load_task.done():
                app.state.load_task.cancel()
            await asyncio.to_thread(runtime.close)

    app = FastAPI(title=f"ModelDeck OPUS translation Worker: {worker_id}", lifespan=lifespan)
    app.state.shutdown_callback = None

    @app.exception_handler(TranslationRequestError)
    async def request_error(_request: Request, error: TranslationRequestError) -> JSONResponse:
        return _error_response(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return _error_response(422, "invalid_request", "The request does not match the translation contract.")

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, error: Exception) -> JSONResponse:
        LOGGER.error("Translation failed category=%s", type(error).__name__)
        return _error_response(502, "internal_error", "The local translation Worker failed.")

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        return {
            "protocol_version": "1",
            "worker_id": worker_id,
            "runtime": "marian-transformers-cpu",
            "generation_family": GenerationFamily.TEXT_TRANSLATION,
            "state": request.app.state.worker_state,
            "model_id": config.model_id,
            "model_revision": config.revision,
            "device": "cpu",
            "device_name": "CPU",
            "ready": request.app.state.ready and request.app.state.active_request_id is None,
            "busy": request.app.state.active_request_id is not None,
            "error": request.app.state.load_error,
        }

    @app.get("/capabilities")
    async def capabilities() -> dict[str, Any]:
        return {
            "protocol_version": "1",
            "generation_family": GenerationFamily.TEXT_TRANSLATION,
            **CapabilitySet(streaming=False, cancellation=True, translation=True).model_dump(),
            "source_language": config.source_language,
            "target_language": config.target_language,
        }

    @app.get("/metrics")
    async def metrics(request: Request) -> dict[str, Any]:
        return {
            **runtime.runtime_details,
            "requests": request.app.state.requests,
            "successful_requests": request.app.state.successes,
            "failed_requests": request.app.state.failures,
            "busy": request.app.state.active_request_id is not None,
            "last_request": request.app.state.last_request,
        }

    @app.get("/model")
    async def model() -> dict[str, Any]:
        return {
            "model_id": config.model_id,
            "revision": config.revision,
            "generation_family": GenerationFamily.TEXT_TRANSLATION,
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": "float32",
            "source_language": config.source_language,
            "target_language": config.target_language,
        }

    @app.post("/load")
    async def load(request: Request) -> dict[str, Any]:
        await request.app.state.load_task
        return {"ok": request.app.state.load_error is None, "state": request.app.state.worker_state}

    @app.post("/warmup")
    async def warmup(request: Request) -> dict[str, Any]:
        await request.app.state.load_task
        if request.app.state.load_error:
            raise TranslationRequestError(503, "model_unavailable", "The pinned model failed to load.")
        request.app.state.worker_state = WorkerState.WARMING
        await asyncio.to_thread(runtime.warmup)
        request.app.state.ready = True
        request.app.state.worker_state = WorkerState.READY
        return {"ok": True, "ready": True}

    @app.post("/cancel")
    async def cancel(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id", ""))
        if request_id and request_id == request.app.state.active_request_id:
            request.app.state.active_cancellation.set()
            return {"ok": True, "request_id": request_id, "state": "cancelling"}
        return {"ok": False, "request_id": request_id, "state": "not-found"}

    @app.post("/shutdown")
    async def shutdown(request: Request) -> dict[str, bool]:
        request.app.state.worker_state = WorkerState.STOPPING
        if request.app.state.active_cancellation is not None:
            request.app.state.active_cancellation.set()
        if request.app.state.shutdown_callback:
            asyncio.get_running_loop().call_later(0.05, request.app.state.shutdown_callback)
        return {"ok": True}

    @app.post("/v1/translations")
    async def translate(request: Request, body: TranslationRequest) -> dict[str, Any]:
        _ensure_ready(request)
        if body.model != config.alias:
            raise TranslationRequestError(422, "invalid_model", "The model identifier is not allowlisted.")
        if body.source_language != config.source_language or body.target_language != config.target_language:
            raise TranslationRequestError(
                422,
                "unsupported_direction",
                "The requested language direction does not match this Worker.",
            )
        text = body.input.strip()
        if not text:
            raise TranslationRequestError(422, "empty_input", "Translation input cannot be blank.")
        if len(text) > config.maximum_input_characters:
            raise TranslationRequestError(
                422,
                "input_too_long",
                f"Input exceeds {config.maximum_input_characters} characters.",
            )
        cancellation = threading.Event()
        if not await _claim_slot(request, body.request_id, cancellation):
            raise TranslationRequestError(429, "worker_busy", "The translation Worker is busy.")
        request.app.state.requests += 1
        started = time.perf_counter()
        result: TranslationResult | None = None
        try:
            result = await _run_translation(request, runtime, text, cancellation, config)
            if result.cancelled:
                raise TranslationRequestError(409, "request_cancelled", "Translation was cancelled.")
            request.app.state.successes += 1
            request.app.state.last_request = {
                "request_id": body.request_id,
                "outcome": "success",
                "inference_seconds": round(result.inference_seconds, 6),
                "total_worker_seconds": round(time.perf_counter() - started, 6),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
            return {
                "id": body.request_id,
                "object": "translation",
                "model": config.alias,
                "source_language": config.source_language,
                "target_language": config.target_language,
                "output_text": result.text,
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
            }
        except Exception:
            request.app.state.failures += 1
            raise
        finally:
            text = ""
            if result is not None:
                result.text = ""
            await _release_slot(request, body.request_id)

    @app.post("/native/text-translation/smoke")
    async def smoke(request: Request) -> dict[str, Any]:
        _ensure_ready(request)
        result = await asyncio.to_thread(runtime.translate, "The local service is ready.", threading.Event())
        try:
            return {
                "ok": bool(result.text),
                "source_language": config.source_language,
                "target_language": config.target_language,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
        finally:
            result.text = ""

    return app


async def _load_engine(app: FastAPI, engine: TranslationEngine) -> None:
    try:
        await asyncio.to_thread(engine.load)
        app.state.worker_state = WorkerState.WARMING
    except Exception as error:
        app.state.load_error = f"{type(error).__name__}: {error}"
        app.state.worker_state = WorkerState.FAILED


async def _claim_slot(request: Request, request_id: str, cancellation: threading.Event) -> bool:
    async with request.app.state.slot_guard:
        if request.app.state.active_request_id is not None:
            return False
        request.app.state.active_request_id = request_id
        request.app.state.active_cancellation = cancellation
        request.app.state.worker_state = WorkerState.BUSY
        return True


async def _release_slot(request: Request, request_id: str) -> None:
    async with request.app.state.slot_guard:
        if request.app.state.active_request_id == request_id:
            request.app.state.active_request_id = None
            request.app.state.active_cancellation = None
            request.app.state.worker_state = WorkerState.READY


async def _run_translation(
    request: Request,
    engine: TranslationEngine,
    text: str,
    cancellation: threading.Event,
    config: TranslationConfig,
) -> TranslationResult:
    task = asyncio.create_task(asyncio.to_thread(engine.translate, text, cancellation))
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=config.generation_timeout_seconds)
    except TimeoutError as error:
        cancellation.set()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=CANCELLATION_GRACE_SECONDS)
        except TimeoutError:
            request.app.state.ready = False
            request.app.state.worker_state = WorkerState.FAILED
        raise TranslationRequestError(
            504,
            "generation_timeout",
            f"Translation exceeded {config.generation_timeout_seconds:g} seconds.",
        ) from error


def _ensure_ready(request: Request) -> None:
    if not request.app.state.ready:
        raise TranslationRequestError(503, "model_unavailable", "The translation Worker is not ready.")


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "type": "local_worker_error"}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelDeck pinned OPUS translation Worker")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True, choices=sorted(SPEECHSHIFT_MODEL_SPECS))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-language", choices=("en",), required=True)
    parser.add_argument("--target-language", choices=("fr", "de"), required=True)
    parser.add_argument("--maximum-input-characters", type=int, default=4000)
    parser.add_argument("--maximum-input-tokens", type=int, default=512)
    parser.add_argument("--maximum-new-tokens", type=int, default=512)
    parser.add_argument("--generation-timeout-seconds", type=float, default=60)
    arguments = parser.parse_args()
    spec = SPEECHSHIFT_MODEL_SPECS[arguments.model_id]
    if (
        arguments.revision != spec.revision
        or arguments.source_language != spec.source_language
        or arguments.target_language != spec.target_language
    ):
        raise SystemExit("The model revision and translation direction must match the allowlist")
    application = create_app(
        worker_id=arguments.worker_id,
        config=TranslationConfig(
            model_id=arguments.model_id,
            revision=arguments.revision,
            alias=arguments.alias,
            cache_root=arguments.cache_root,
            source_language=arguments.source_language,
            target_language=arguments.target_language,
            maximum_input_characters=arguments.maximum_input_characters,
            maximum_input_tokens=arguments.maximum_input_tokens,
            maximum_new_tokens=arguments.maximum_new_tokens,
            generation_timeout_seconds=arguments.generation_timeout_seconds,
        ),
    )
    server = uvicorn.Server(
        uvicorn.Config(application, host="127.0.0.1", port=arguments.port, access_log=False)
    )
    application.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

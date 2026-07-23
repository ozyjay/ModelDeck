from __future__ import annotations

import argparse
import asyncio
import gc
import importlib.metadata
import io
import logging
import threading
import time
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from modeldeck.protocol import CapabilitySet, GenerationFamily, WorkerState
from modeldeck.speechshift import (
    QWEN_LANGUAGE_NAMES,
    QWEN_TTS_GENERATION_TIMEOUT_SECONDS,
    QWEN_TTS_LANGUAGES,
    QWEN_TTS_MAXIMUM_CODEC_TOKENS,
    QWEN_TTS_SAMPLE_RATE_HZ,
    QWEN_TTS_SPEAKER_NAMES,
    QWEN_TTS_VOICES,
    SPEECHSHIFT_MODEL_SPECS,
    validate_speechshift_snapshot,
)
from modeldeck.thermal import TemperatureSnapshot, ThermalGuard, ThermalGuardError

LOGGER = logging.getLogger("modeldeck.tts")
CANCELLATION_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class TTSConfig:
    model_id: str
    revision: str
    alias: str
    cache_root: Path
    maximum_input_characters: int = 2000
    maximum_codec_tokens: int = QWEN_TTS_MAXIMUM_CODEC_TOKENS
    maximum_audio_seconds: int = 90
    generation_timeout_seconds: float = QWEN_TTS_GENERATION_TIMEOUT_SECONDS

    @property
    def snapshot_path(self) -> Path:
        return self.cache_root / f"models--{self.model_id.replace('/', '--')}" / "snapshots" / self.revision


@dataclass
class SpeechResult:
    wav_bytes: bytes
    duration_seconds: float
    inference_seconds: float
    codec_tokens: int | None = None
    cancelled: bool = False


class SpeechSynthesisEngine(Protocol):
    runtime_details: dict[str, Any]

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def synthesise(
        self,
        text: str,
        voice: str,
        language: str,
        cancellation: threading.Event,
    ) -> SpeechResult: ...

    def close(self) -> None: ...


@dataclass
class QwenTTSEngine:
    config: TTSConfig
    runtime_details: dict[str, Any] = field(default_factory=dict)
    torch: Any = None
    model: Any = None

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        error = validate_speechshift_snapshot(
            self.config.snapshot_path, self.config.model_id, self.config.revision
        )
        if error:
            raise RuntimeError(error)
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm PyTorch did not expose an available 'cuda' device")
        device = torch.device("cuda:0")
        torch.empty(1, device=device, dtype=torch.bfloat16)
        started = time.perf_counter()
        self.model = Qwen3TTSModel.from_pretrained(
            self.config.snapshot_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self.torch = torch
        self.runtime_details = {
            "device": "cuda:0",
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            "hip_version": torch.version.hip,
            "qwen_tts_version": importlib.metadata.version("qwen-tts"),
            "transformers_version": __import__("transformers").__version__,
            "attention_implementation": "sdpa",
            "load_seconds": round(time.perf_counter() - started, 4),
        }

    def warmup(self) -> None:
        result = self.synthesise("Ready.", "ryan", "en", threading.Event())
        if not result.wav_bytes:
            raise RuntimeError("The pinned Qwen TTS model returned no warm-up audio")
        result.wav_bytes = b""

    def synthesise(
        self,
        text: str,
        voice: str,
        language: str,
        cancellation: threading.Event,
    ) -> SpeechResult:
        from transformers import StoppingCriteria, StoppingCriteriaList

        if self.model is None or self.torch is None:
            raise RuntimeError("The speech synthesis model is not loaded")

        class CancellationCriteria(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return cancellation.is_set()

        waveforms: Any = None
        pcm: Any = None
        try:
            started = time.perf_counter()
            waveforms, sample_rate = self.model.generate_custom_voice(
                text=text,
                language=QWEN_LANGUAGE_NAMES[language],
                speaker=QWEN_TTS_SPEAKER_NAMES[voice],
                instruct=None,
                max_new_tokens=self.config.maximum_codec_tokens,
                do_sample=True,
                subtalker_dosample=True,
                stopping_criteria=StoppingCriteriaList([CancellationCriteria()]),
            )
            inference_seconds = time.perf_counter() - started
            if int(sample_rate) != QWEN_TTS_SAMPLE_RATE_HZ:
                raise RuntimeError("Qwen TTS returned an unexpected sample rate")
            import numpy as np

            pcm = np.asarray(waveforms[0], dtype=np.float32).reshape(-1)
            duration = len(pcm) / QWEN_TTS_SAMPLE_RATE_HZ
            if duration <= 0 or duration > self.config.maximum_audio_seconds:
                raise TTSRequestError(
                    502,
                    "audio_duration_out_of_bounds",
                    "Generated audio duration is outside the configured bounds.",
                )
            wav_bytes = _encode_wav(pcm)
            return SpeechResult(
                wav_bytes=wav_bytes,
                duration_seconds=duration,
                inference_seconds=inference_seconds,
                cancelled=cancellation.is_set(),
            )
        finally:
            waveforms = None
            pcm = None
            if self.torch is not None and self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()

    def close(self) -> None:
        self.model = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self.torch = None
        gc.collect()


class SpeechSynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$")
    model: str
    input: str
    voice: str
    language: str
    response_format: str = "wav"


class TTSRequestError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


def create_app(
    *,
    worker_id: str,
    config: TTSConfig,
    engine: SpeechSynthesisEngine | None = None,
    thermal_guard: ThermalGuard | None = None,
) -> FastAPI:
    runtime = engine or QwenTTSEngine(config)
    guard = thermal_guard or ThermalGuard()

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
        app.state.thermal_rejections = 0
        app.state.thermal_cancellations = 0
        app.state.last_request = None
        app.state.last_temperatures = None
        app.state.load_task = asyncio.create_task(_load_engine(app, runtime))
        try:
            yield
        finally:
            if app.state.active_cancellation is not None:
                app.state.active_cancellation.set()
            if not app.state.load_task.done():
                app.state.load_task.cancel()
            await asyncio.to_thread(runtime.close)

    app = FastAPI(title=f"ModelDeck Qwen TTS Worker: {worker_id}", lifespan=lifespan)
    app.state.shutdown_callback = None

    @app.exception_handler(TTSRequestError)
    async def request_error(_request: Request, error: TTSRequestError) -> JSONResponse:
        return _error_response(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return _error_response(422, "invalid_request", "The request does not match the speech contract.")

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, error: Exception) -> JSONResponse:
        LOGGER.error("Speech synthesis failed category=%s", type(error).__name__)
        return _error_response(502, "internal_error", "The local speech synthesis Worker failed.")

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        details = runtime.runtime_details
        return {
            "protocol_version": "1",
            "worker_id": worker_id,
            "runtime": "qwen3-tts-rocm",
            "generation_family": GenerationFamily.SPEECH_SYNTHESIS,
            "state": request.app.state.worker_state,
            "model_id": config.model_id,
            "model_revision": config.revision,
            "device": details.get("device", "cuda:0"),
            "device_name": details.get("device_name", "AMD GPU"),
            "rocm_version": details.get("hip_version"),
            "ready": request.app.state.ready and request.app.state.active_request_id is None,
            "busy": request.app.state.active_request_id is not None,
            "error": request.app.state.load_error,
        }

    @app.get("/capabilities")
    async def capabilities() -> dict[str, Any]:
        return {
            "protocol_version": "1",
            "generation_family": GenerationFamily.SPEECH_SYNTHESIS,
            **CapabilitySet(
                streaming=False,
                cancellation=True,
                audio_output=True,
                speech_synthesis=True,
            ).model_dump(),
            "voices": list(QWEN_TTS_VOICES),
            "languages": list(QWEN_TTS_LANGUAGES),
            "sample_rate_hz": QWEN_TTS_SAMPLE_RATE_HZ,
            "response_formats": ["wav"],
        }

    @app.get("/metrics")
    async def metrics(request: Request) -> dict[str, Any]:
        return {
            **runtime.runtime_details,
            "requests": request.app.state.requests,
            "successful_requests": request.app.state.successes,
            "failed_requests": request.app.state.failures,
            "thermal_rejections": request.app.state.thermal_rejections,
            "thermal_cancellations": request.app.state.thermal_cancellations,
            "busy": request.app.state.active_request_id is not None,
            "last_temperatures": request.app.state.last_temperatures,
            "last_request": request.app.state.last_request,
        }

    @app.get("/model")
    async def model() -> dict[str, Any]:
        return {
            "model_id": config.model_id,
            "revision": config.revision,
            "generation_family": GenerationFamily.SPEECH_SYNTHESIS,
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "sample_rate_hz": QWEN_TTS_SAMPLE_RATE_HZ,
        }

    @app.post("/load")
    async def load(request: Request) -> dict[str, Any]:
        await request.app.state.load_task
        return {"ok": request.app.state.load_error is None, "state": request.app.state.worker_state}

    @app.post("/warmup")
    async def warmup(request: Request) -> dict[str, Any]:
        await request.app.state.load_task
        if request.app.state.load_error:
            raise TTSRequestError(503, "model_unavailable", "The pinned model failed to load.")
        _require_thermal_start(request, guard)
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

    @app.post("/v1/audio/speech")
    async def synthesise(request: Request, body: SpeechSynthesisRequest) -> Response:
        _ensure_ready(request)
        if body.model != config.alias:
            raise TTSRequestError(422, "invalid_model", "The model identifier is not allowlisted.")
        if body.voice not in QWEN_TTS_VOICES:
            raise TTSRequestError(
                422,
                "unsupported_voice",
                "The requested built-in voice is not allowlisted.",
            )
        if body.language not in QWEN_TTS_LANGUAGES:
            raise TTSRequestError(422, "unsupported_language", "The requested language is not allowlisted.")
        if body.response_format != "wav":
            raise TTSRequestError(422, "unsupported_format", "response_format must be wav.")
        text = body.input.strip()
        if not text:
            raise TTSRequestError(422, "empty_input", "Speech input cannot be blank.")
        if len(text) > config.maximum_input_characters:
            raise TTSRequestError(
                422,
                "input_too_long",
                f"Input exceeds {config.maximum_input_characters} characters.",
            )
        initial_temperature = _require_thermal_start(request, guard)
        cancellation = threading.Event()
        if not await _claim_slot(request, body.request_id, cancellation):
            raise TTSRequestError(429, "worker_busy", "The speech synthesis Worker is busy.")
        request.app.state.requests += 1
        result: SpeechResult | None = None
        started = time.perf_counter()
        try:
            result, thermal_reason, peak = await _run_synthesis(
                request,
                runtime,
                text,
                body.voice,
                body.language,
                cancellation,
                config,
                guard,
                initial_temperature,
            )
            request.app.state.last_temperatures = _temperature_dict(peak)
            if thermal_reason:
                request.app.state.thermal_cancellations += 1
                raise TTSRequestError(
                    503,
                    "thermal_limit_reached",
                    f"Speech synthesis stopped because {thermal_reason.replace('_', ' ')}.",
                )
            if result.cancelled:
                raise TTSRequestError(409, "request_cancelled", "Speech synthesis was cancelled.")
            request.app.state.successes += 1
            request.app.state.last_request = {
                "request_id": body.request_id,
                "outcome": "success",
                "inference_seconds": round(result.inference_seconds, 6),
                "total_worker_seconds": round(time.perf_counter() - started, 6),
                "duration_seconds": round(result.duration_seconds, 6),
                "audio_bytes": len(result.wav_bytes),
            }
            return Response(
                content=result.wav_bytes,
                media_type="audio/wav",
                headers={
                    "x-request-id": body.request_id,
                    "x-modeldeck-sample-rate-hz": str(QWEN_TTS_SAMPLE_RATE_HZ),
                    "x-modeldeck-audio-duration-seconds": f"{result.duration_seconds:.6f}",
                },
            )
        except Exception:
            request.app.state.failures += 1
            raise
        finally:
            text = ""
            if result is not None:
                result.wav_bytes = b""
            await _release_slot(request, body.request_id)

    @app.post("/native/speech-synthesis/smoke")
    async def smoke(request: Request) -> dict[str, Any]:
        _ensure_ready(request)
        _require_thermal_start(request, guard)
        result = await asyncio.to_thread(runtime.synthesise, "Ready.", "ryan", "en", threading.Event())
        try:
            return {
                "ok": bool(result.wav_bytes),
                "sample_rate_hz": QWEN_TTS_SAMPLE_RATE_HZ,
                "channels": 1,
                "duration_seconds": result.duration_seconds,
                "audio_bytes": len(result.wav_bytes),
            }
        finally:
            result.wav_bytes = b""

    return app


def _encode_wav(samples: Any) -> bytes:
    import numpy as np

    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype("<i2", copy=False)
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(QWEN_TTS_SAMPLE_RATE_HZ)
        writer.writeframes(pcm16.tobytes())
    return output.getvalue()


async def _load_engine(app: FastAPI, engine: SpeechSynthesisEngine) -> None:
    try:
        await asyncio.to_thread(engine.load)
        app.state.worker_state = WorkerState.WARMING
    except Exception as error:
        app.state.load_error = f"{type(error).__name__}: {error}"
        app.state.worker_state = WorkerState.FAILED


def _require_thermal_start(request: Request, guard: ThermalGuard) -> TemperatureSnapshot:
    try:
        snapshot = guard.require_start_safe()
        request.app.state.last_temperatures = _temperature_dict(snapshot)
        return snapshot
    except ThermalGuardError as error:
        request.app.state.thermal_rejections += 1
        if error.snapshot is not None:
            request.app.state.last_temperatures = _temperature_dict(error.snapshot)
        raise TTSRequestError(503, error.code, str(error)) from error


async def _run_synthesis(
    request: Request,
    engine: SpeechSynthesisEngine,
    text: str,
    voice: str,
    language: str,
    cancellation: threading.Event,
    config: TTSConfig,
    guard: ThermalGuard,
    initial_temperature: TemperatureSnapshot,
) -> tuple[SpeechResult, str | None, TemperatureSnapshot]:
    task = asyncio.create_task(asyncio.to_thread(engine.synthesise, text, voice, language, cancellation))
    deadline = time.monotonic() + config.generation_timeout_seconds
    peak = initial_temperature
    thermal_reason: str | None = None
    while not task.done():
        if cancellation.is_set():
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            cancellation.set()
            await _await_cancelled_task(request, task)
            raise TTSRequestError(
                504,
                "generation_timeout",
                f"Speech synthesis exceeded {config.generation_timeout_seconds:g} seconds.",
            )
        await asyncio.sleep(min(0.25, remaining))
        try:
            snapshot = guard.sample()
        except ThermalGuardError as error:
            cancellation.set()
            await _await_cancelled_task(request, task)
            raise TTSRequestError(
                503,
                "thermal_monitor_unavailable",
                "Temperature monitoring became unavailable during speech synthesis.",
            ) from error
        peak = TemperatureSnapshot(
            gpu_edge_celsius=max(peak.gpu_edge_celsius, snapshot.gpu_edge_celsius),
            cpu_package_celsius=max(peak.cpu_package_celsius, snapshot.cpu_package_celsius),
        )
        terminal = guard.termination_reason(snapshot)
        if terminal:
            thermal_reason, terminal_snapshot = terminal
            peak = TemperatureSnapshot(
                gpu_edge_celsius=max(peak.gpu_edge_celsius, terminal_snapshot.gpu_edge_celsius),
                cpu_package_celsius=max(peak.cpu_package_celsius, terminal_snapshot.cpu_package_celsius),
            )
            cancellation.set()
            break
    result = await _await_cancelled_task(request, task) if cancellation.is_set() else await task
    return result, thermal_reason, peak


async def _await_cancelled_task(request: Request, task: asyncio.Task[SpeechResult]) -> SpeechResult:
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=CANCELLATION_GRACE_SECONDS)
    except TimeoutError as error:
        request.app.state.ready = False
        request.app.state.worker_state = WorkerState.FAILED
        raise TTSRequestError(
            503,
            "cancellation_unresponsive",
            "Speech synthesis did not stop within the cancellation deadline.",
        ) from error


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
            request.app.state.worker_state = (
                WorkerState.READY if request.app.state.ready else WorkerState.FAILED
            )


def _temperature_dict(snapshot: TemperatureSnapshot) -> dict[str, float]:
    return {
        "gpu_edge_celsius": snapshot.gpu_edge_celsius,
        "cpu_package_celsius": snapshot.cpu_package_celsius,
    }


def _ensure_ready(request: Request) -> None:
    if not request.app.state.ready:
        raise TTSRequestError(503, "model_unavailable", "The speech synthesis Worker is not ready.")


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "type": "local_worker_error"}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelDeck pinned Qwen3-TTS Worker")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--maximum-input-characters", type=int, default=2000)
    parser.add_argument(
        "--maximum-codec-tokens",
        type=int,
        default=QWEN_TTS_MAXIMUM_CODEC_TOKENS,
    )
    parser.add_argument("--maximum-audio-seconds", type=int, default=90)
    parser.add_argument(
        "--generation-timeout-seconds",
        type=float,
        default=QWEN_TTS_GENERATION_TIMEOUT_SECONDS,
    )
    arguments = parser.parse_args()
    spec = SPEECHSHIFT_MODEL_SPECS.get(arguments.model_id)
    if spec is None or arguments.revision != spec.revision or spec.generation_family != "speech-synthesis":
        raise SystemExit("The Qwen TTS model identity is not allowlisted")
    application = create_app(
        worker_id=arguments.worker_id,
        config=TTSConfig(
            model_id=arguments.model_id,
            revision=arguments.revision,
            alias=arguments.alias,
            cache_root=arguments.cache_root,
            maximum_input_characters=arguments.maximum_input_characters,
            maximum_codec_tokens=arguments.maximum_codec_tokens,
            maximum_audio_seconds=arguments.maximum_audio_seconds,
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

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import logging
import os
import signal
import sys
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
from modeldeck.speechshift import (
    SPEECHSHIFT_MODEL_SPECS,
    WHISPER_MAXIMUM_AUDIO_BYTES,
    WHISPER_MAXIMUM_AUDIO_SECONDS,
    WHISPER_SAMPLE_RATE_HZ,
    validate_speechshift_snapshot,
)
from modeldeck.thermal import TemperatureSnapshot, ThermalGuard, ThermalGuardError

LOGGER = logging.getLogger("modeldeck.speech_recognition")
MAXIMUM_ENCODED_AUDIO_CHARACTERS = ((WHISPER_MAXIMUM_AUDIO_BYTES + 2) // 3) * 4
PROCESS_TERMINATION_GRACE_SECONDS = 0.15
THERMAL_POLL_SECONDS = 0.1


@dataclass(frozen=True)
class RecognitionConfig:
    model_id: str
    revision: str
    alias: str
    cache_root: Path
    recognition_timeout_seconds: float = 30.0

    @property
    def snapshot_path(self) -> Path:
        return self.cache_root / f"models--{self.model_id.replace('/', '--')}" / "snapshots" / self.revision


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    inference_seconds: float
    peak_gpu_memory_bytes: int


class RecognitionRunner(Protocol):
    runtime_details: dict[str, Any]

    async def validate(self) -> None: ...

    async def recognise(self, pcm_bytes: bytes) -> RecognitionResult: ...

    async def cancel(self) -> None: ...


@dataclass
class IsolatedWhisperRunner:
    config: RecognitionConfig
    python_executable: Path = field(default_factory=lambda: Path(sys.executable))
    runtime_details: dict[str, Any] = field(default_factory=dict)
    active_process: asyncio.subprocess.Process | None = None
    _process_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def validate(self) -> None:
        error = validate_speechshift_snapshot(
            self.config.snapshot_path, self.config.model_id, self.config.revision
        )
        if error:
            raise RuntimeError(error)
        payload = await self._invoke(b"", probe=True)
        self.runtime_details = {
            "device": "cuda:0",
            "device_name": payload["device_name"],
            "hip_version": payload["hip_version"],
            "process_isolation": "per-request-process-group",
        }

    async def recognise(self, pcm_bytes: bytes) -> RecognitionResult:
        payload = await self._invoke(pcm_bytes, probe=False)
        if (
            not isinstance(payload.get("text"), str)
            or len(payload["text"]) > 4096
            or not isinstance(payload.get("inference_seconds"), int | float)
            or float(payload["inference_seconds"]) < 0
            or not isinstance(payload.get("peak_gpu_memory_bytes"), int)
            or int(payload["peak_gpu_memory_bytes"]) < 0
        ):
            raise RecognitionRequestError(
                502, "invalid_worker_response", "The isolated process returned invalid metadata."
            )
        return RecognitionResult(
            text=payload["text"],
            inference_seconds=float(payload["inference_seconds"]),
            peak_gpu_memory_bytes=int(payload["peak_gpu_memory_bytes"]),
        )

    async def cancel(self) -> None:
        async with self._process_lock:
            process = self.active_process
        if process is not None and process.returncode is None:
            await _terminate_process_group(process)

    async def _invoke(self, pcm_bytes: bytes, *, probe: bool) -> dict[str, Any]:
        command = [
            str(self.python_executable),
            "-m",
            "modeldeck.workers.whisper_inference_child",
            "--snapshot",
            str(self.config.snapshot_path),
        ]
        if probe:
            command.append("--probe")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        async with self._process_lock:
            self.active_process = process
        try:
            stdout, _ = await process.communicate(pcm_bytes)
        finally:
            async with self._process_lock:
                if self.active_process is process:
                    self.active_process = None
        if process.returncode != 0:
            raise RecognitionRequestError(
                502, "inference_failed", "The isolated speech-recognition process failed."
            )
        try:
            payload = json.loads(stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RecognitionRequestError(
                502, "invalid_worker_response", "The isolated process returned invalid metadata."
            ) from error
        if not isinstance(payload, dict) or payload.get("error"):
            raise RecognitionRequestError(
                502, "inference_failed", "The isolated speech-recognition process failed."
            )
        return payload


class TranscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$")
    model: str
    language: str = "en"
    encoding: str = "pcm_s16le"
    sample_rate_hz: int = WHISPER_SAMPLE_RATE_HZ
    channels: int = 1
    audio_base64: str = Field(min_length=4, max_length=MAXIMUM_ENCODED_AUDIO_CHARACTERS)


class RecognitionRequestError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


def create_app(
    *,
    worker_id: str,
    config: RecognitionConfig,
    runner: RecognitionRunner | None = None,
    thermal_guard: ThermalGuard | None = None,
) -> FastAPI:
    runtime = runner or IsolatedWhisperRunner(config)
    guard = thermal_guard or ThermalGuard()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.worker_state = WorkerState.LOADING
        app.state.ready = False
        app.state.load_error = None
        app.state.active_request_id = None
        app.state.cancelled_requests = set()
        app.state.slot_guard = asyncio.Lock()
        app.state.requests = 0
        app.state.successes = 0
        app.state.failures = 0
        app.state.thermal_rejections = 0
        app.state.thermal_cancellations = 0
        app.state.last_request = None
        app.state.last_temperatures = None
        app.state.load_task = asyncio.create_task(_load_runner(app, runtime))
        try:
            yield
        finally:
            await runtime.cancel()
            if not app.state.load_task.done():
                app.state.load_task.cancel()

    app = FastAPI(title=f"ModelDeck Whisper speech-recognition Worker: {worker_id}", lifespan=lifespan)
    app.state.shutdown_callback = None

    @app.exception_handler(RecognitionRequestError)
    async def request_error(_request: Request, error: RecognitionRequestError) -> JSONResponse:
        return _error_response(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return _error_response(422, "invalid_audio", "The request does not match the audio contract.")

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, error: Exception) -> JSONResponse:
        LOGGER.error("Speech recognition failed category=%s", type(error).__name__)
        return _error_response(502, "internal_error", "The local speech-recognition Worker failed.")

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        details = runtime.runtime_details
        return {
            "protocol_version": "1",
            "worker_id": worker_id,
            "runtime": "whisper-small-en-rocm",
            "generation_family": GenerationFamily.SPEECH_RECOGNITION,
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
            "generation_family": GenerationFamily.SPEECH_RECOGNITION,
            **CapabilitySet(
                streaming=False,
                cancellation=True,
                audio_input=True,
                speech_recognition=True,
            ).model_dump(),
            "languages": ["en"],
            "encoding": "pcm_s16le",
            "sample_rate_hz": WHISPER_SAMPLE_RATE_HZ,
            "channels": 1,
            "maximum_audio_seconds": WHISPER_MAXIMUM_AUDIO_SECONDS,
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
            "generation_family": GenerationFamily.SPEECH_RECOGNITION,
            "local_files_only": True,
            "trust_remote_code": False,
            "weights_format": "safetensors",
            "licence": "Apache-2.0",
        }

    @app.post("/load")
    async def load(request: Request) -> dict[str, Any]:
        await request.app.state.load_task
        return {"ok": request.app.state.load_error is None, "state": request.app.state.worker_state}

    @app.post("/warmup")
    async def warmup(request: Request) -> dict[str, Any]:
        await request.app.state.load_task
        if request.app.state.load_error:
            raise RecognitionRequestError(503, "model_unavailable", "The pinned model failed validation.")
        initial_temperature = _require_thermal_start(request, guard)
        request.app.state.worker_state = WorkerState.WARMING
        await _run_recognition(request, runtime, bytes(3_200), config, guard, initial_temperature)
        request.app.state.ready = True
        request.app.state.worker_state = WorkerState.READY
        return {"ok": True, "ready": True}

    @app.post("/cancel")
    async def cancel(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id", ""))
        if request_id and request_id == request.app.state.active_request_id:
            request.app.state.cancelled_requests.add(request_id)
            await runtime.cancel()
            return {"ok": True, "request_id": request_id, "state": "cancelled"}
        return {"ok": False, "request_id": request_id, "state": "not-found"}

    @app.post("/shutdown")
    async def shutdown(request: Request) -> dict[str, bool]:
        request.app.state.worker_state = WorkerState.STOPPING
        await runtime.cancel()
        if request.app.state.shutdown_callback:
            asyncio.get_running_loop().call_later(0.05, request.app.state.shutdown_callback)
        return {"ok": True}

    @app.post("/v1/audio/transcriptions")
    async def transcribe(request: Request, body: TranscriptionRequest) -> dict[str, Any]:
        _ensure_ready(request)
        _validate_metadata(body, config)
        pcm_bytes = _decode_audio(body.audio_base64)
        duration_seconds = len(pcm_bytes) / (WHISPER_SAMPLE_RATE_HZ * 2)
        initial_temperature = _require_thermal_start(request, guard)
        if not await _claim_slot(request, body.request_id):
            raise RecognitionRequestError(429, "worker_busy", "The speech-recognition Worker is busy.")
        request.app.state.requests += 1
        started = time.perf_counter()
        try:
            result, peak = await _run_recognition(
                request, runtime, pcm_bytes, config, guard, initial_temperature
            )
            request.app.state.successes += 1
            request.app.state.last_temperatures = _temperature_dict(peak)
            total_seconds = time.perf_counter() - started
            request.app.state.last_request = {
                "request_id": body.request_id,
                "outcome": "success",
                "audio_seconds": round(duration_seconds, 6),
                "audio_bytes": len(pcm_bytes),
                "inference_seconds": round(result.inference_seconds, 6),
                "total_worker_seconds": round(total_seconds, 6),
                "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
            }
            return {
                "id": body.request_id,
                "object": "audio.transcription",
                "model": body.model,
                "language": "en",
                "text": result.text,
                "metrics": {
                    "audio_seconds": round(duration_seconds, 6),
                    "inference_seconds": round(result.inference_seconds, 6),
                    "total_worker_seconds": round(total_seconds, 6),
                },
            }
        except Exception:
            request.app.state.failures += 1
            raise
        finally:
            pcm_bytes = b""
            await _release_slot(request, body.request_id)

    @app.post("/native/speech-recognition/smoke")
    async def smoke(request: Request) -> dict[str, Any]:
        _ensure_ready(request)
        initial_temperature = _require_thermal_start(request, guard)
        request_id = "modeldeck-worker-smoke"
        if not await _claim_slot(request, request_id):
            raise RecognitionRequestError(429, "worker_busy", "The speech-recognition Worker is busy.")
        try:
            result, _peak = await _run_recognition(
                request, runtime, bytes(3_200), config, guard, initial_temperature
            )
            return {
                "ok": True,
                "output_kind": "transcript",
                "language": "en",
                "mock": False,
                "inference_seconds": result.inference_seconds,
            }
        finally:
            await _release_slot(request, request_id)

    return app


def _validate_metadata(body: TranscriptionRequest, config: RecognitionConfig) -> None:
    if body.model != config.alias:
        raise RecognitionRequestError(422, "invalid_model", "The model identifier is not allowlisted.")
    if body.language != "en":
        raise RecognitionRequestError(422, "unsupported_language", "Only English is supported.")
    if body.encoding != "pcm_s16le" or body.sample_rate_hz != WHISPER_SAMPLE_RATE_HZ or body.channels != 1:
        raise RecognitionRequestError(
            422, "invalid_audio", "Audio must be mono PCM16 little-endian at 16 kHz."
        )


def _decode_audio(encoded: str) -> bytes:
    try:
        pcm_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RecognitionRequestError(422, "invalid_audio", "audio_base64 is not valid base64.") from error
    if not pcm_bytes or len(pcm_bytes) % 2 or len(pcm_bytes) > WHISPER_MAXIMUM_AUDIO_BYTES:
        raise RecognitionRequestError(
            422, "invalid_audio", "Audio must contain between one sample and eight seconds of PCM16."
        )
    return pcm_bytes


async def _run_recognition(
    request: Request,
    runner: RecognitionRunner,
    pcm_bytes: bytes,
    config: RecognitionConfig,
    guard: ThermalGuard,
    initial_temperature: TemperatureSnapshot,
) -> tuple[RecognitionResult, TemperatureSnapshot]:
    task = asyncio.create_task(runner.recognise(pcm_bytes))
    deadline = time.monotonic() + config.recognition_timeout_seconds
    peak = initial_temperature
    while not task.done():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            await runner.cancel()
            await _consume_task(task)
            raise RecognitionRequestError(504, "recognition_timeout", "Speech recognition timed out.")
        await asyncio.sleep(min(THERMAL_POLL_SECONDS, remaining))
        try:
            snapshot = guard.sample()
        except ThermalGuardError as error:
            await runner.cancel()
            await _consume_task(task)
            raise RecognitionRequestError(
                503, "thermal_monitor_unavailable", "Temperature monitoring became unavailable."
            ) from error
        peak = TemperatureSnapshot(
            max(peak.gpu_edge_celsius, snapshot.gpu_edge_celsius),
            max(peak.cpu_package_celsius, snapshot.cpu_package_celsius),
        )
        terminal = guard.termination_reason(snapshot)
        if terminal:
            request.app.state.thermal_cancellations += 1
            await runner.cancel()
            await _consume_task(task)
            raise RecognitionRequestError(
                503, "thermal_limit_reached", "Speech recognition stopped at the thermal safety limit."
            )
    try:
        result = await task
        if request.app.state.active_request_id in request.app.state.cancelled_requests:
            raise RecognitionRequestError(409, "request_cancelled", "Speech recognition was cancelled.")
        return result, peak
    except asyncio.CancelledError as error:
        raise RecognitionRequestError(
            409, "request_cancelled", "Speech recognition was cancelled."
        ) from error
    except RecognitionRequestError as error:
        if request.app.state.active_request_id in request.app.state.cancelled_requests:
            raise RecognitionRequestError(
                409, "request_cancelled", "Speech recognition was cancelled."
            ) from error
        raise
    except Exception as error:
        if request.app.state.active_request_id is not None:
            raise RecognitionRequestError(502, "inference_failed", "Speech recognition failed.") from error
        raise RecognitionRequestError(
            409, "request_cancelled", "Speech recognition was cancelled."
        ) from error


async def _consume_task(task: asyncio.Task[RecognitionResult]) -> None:
    try:
        await task
    except (Exception, asyncio.CancelledError):
        pass


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=PROCESS_TERMINATION_GRACE_SECONDS)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        await process.wait()


async def _load_runner(app: FastAPI, runner: RecognitionRunner) -> None:
    try:
        await runner.validate()
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
        raise RecognitionRequestError(503, error.code, str(error)) from error


async def _claim_slot(request: Request, request_id: str) -> bool:
    async with request.app.state.slot_guard:
        if request.app.state.active_request_id is not None:
            return False
        request.app.state.active_request_id = request_id
        request.app.state.worker_state = WorkerState.BUSY
        return True


async def _release_slot(request: Request, request_id: str) -> None:
    async with request.app.state.slot_guard:
        if request.app.state.active_request_id == request_id:
            request.app.state.active_request_id = None
            request.app.state.cancelled_requests.discard(request_id)
            request.app.state.worker_state = WorkerState.READY


def _temperature_dict(snapshot: TemperatureSnapshot) -> dict[str, float]:
    return {
        "gpu_edge_celsius": snapshot.gpu_edge_celsius,
        "cpu_package_celsius": snapshot.cpu_package_celsius,
    }


def _ensure_ready(request: Request) -> None:
    if not request.app.state.ready:
        raise RecognitionRequestError(503, "model_unavailable", "The speech-recognition Worker is not ready.")


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "type": "local_worker_error"}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelDeck pinned Whisper speech-recognition Worker")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--recognition-timeout-seconds", type=float, default=30)
    arguments = parser.parse_args()
    spec = SPEECHSHIFT_MODEL_SPECS.get(arguments.model_id)
    if spec is None or arguments.revision != spec.revision or spec.generation_family != "speech-recognition":
        raise SystemExit("The Whisper model identity is not allowlisted")
    application = create_app(
        worker_id=arguments.worker_id,
        config=RecognitionConfig(
            model_id=arguments.model_id,
            revision=arguments.revision,
            alias=arguments.alias,
            cache_root=arguments.cache_root,
            recognition_timeout_seconds=arguments.recognition_timeout_seconds,
        ),
    )
    server = uvicorn.Server(
        uvicorn.Config(application, host="127.0.0.1", port=arguments.port, access_log=False)
    )
    application.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

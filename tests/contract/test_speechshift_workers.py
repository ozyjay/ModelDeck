from __future__ import annotations

import asyncio
import base64
import io
import sys
import threading
import time
import wave
from pathlib import Path

import httpx
import pytest
from modeldeck.protocol import WorkerState
from modeldeck.speechshift import SPEECHSHIFT_MODEL_SPECS
from modeldeck.thermal import TemperatureSnapshot, ThermalGuard, ThermalGuardError
from modeldeck.workers.speech_recognition_worker import (
    RecognitionConfig,
    RecognitionResult,
    _terminate_process_group,
)
from modeldeck.workers.speech_recognition_worker import (
    create_app as create_recognition_app,
)
from modeldeck.workers.translation_worker import (
    TranslationConfig,
    TranslationResult,
)
from modeldeck.workers.translation_worker import (
    create_app as create_translation_app,
)
from modeldeck.workers.tts_worker import (
    SpeechResult,
    TTSConfig,
)
from modeldeck.workers.tts_worker import (
    create_app as create_tts_app,
)


class FakeTranslationEngine:
    runtime_details = {"device": "cpu", "transformers_version": "test"}

    def load(self) -> None:
        pass

    def warmup(self) -> None:
        pass

    def translate(self, text: str, cancellation: threading.Event) -> TranslationResult:
        return TranslationResult(f"Traduction: {text}", 4, 5, 0.01, cancellation.is_set())

    def close(self) -> None:
        pass


class FakeTTSEngine:
    runtime_details = {"device": "cuda:0", "torch_version": "test"}

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def load(self) -> None:
        pass

    def warmup(self) -> None:
        pass

    def synthesise(
        self,
        text: str,
        voice: str,
        language: str,
        cancellation: threading.Event,
    ) -> SpeechResult:
        self.calls.append((text, voice, language))
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24_000)
            wav.writeframes(bytes(480))
        return SpeechResult(output.getvalue(), 0.01, 0.01, codec_tokens=4)

    def close(self) -> None:
        pass


class FakeRecognitionRunner:
    runtime_details = {
        "device": "cuda:0",
        "hip_version": "test",
        "process_isolation": "per-request-process-group",
    }

    def __init__(self, *, blocked: bool = False) -> None:
        self.blocked = blocked
        self.cancelled = asyncio.Event()
        self.received_lengths: list[int] = []
        self.gpu_bytes = 0

    async def validate(self) -> None:
        pass

    async def recognise(self, pcm_bytes: bytes) -> RecognitionResult:
        self.received_lengths.append(len(pcm_bytes))
        self.gpu_bytes = 1024
        if self.blocked:
            await self.cancelled.wait()
            self.gpu_bytes = 0
            raise asyncio.CancelledError
        self.gpu_bytes = 0
        return RecognitionResult("The service is ready.", 0.01, 1024)

    async def cancel(self) -> None:
        self.cancelled.set()
        self.gpu_bytes = 0


def mark_recognition_ready(app) -> None:
    app.state.ready = True
    app.state.worker_state = WorkerState.READY
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


def recognition_payload(*, samples: int = 1600, request_id: str = "recognition-1") -> dict[str, object]:
    return {
        "request_id": request_id,
        "model": "speechshift-stt",
        "language": "en",
        "encoding": "pcm_s16le",
        "sample_rate_hz": 16000,
        "channels": 1,
        "audio_base64": base64.b64encode(bytes(samples * 2)).decode("ascii"),
    }


def mark_ready(app, *, tts: bool = False) -> None:
    app.state.ready = True
    app.state.worker_state = WorkerState.READY
    app.state.active_request_id = None
    app.state.active_cancellation = None
    app.state.slot_guard = asyncio.Lock()
    app.state.requests = 0
    app.state.successes = 0
    app.state.failures = 0
    app.state.last_request = None
    if tts:
        app.state.thermal_rejections = 0
        app.state.thermal_cancellations = 0
        app.state.last_temperatures = None


@pytest.mark.asyncio
async def test_translation_contract_is_direction_specific_and_schema_valid(tmp_path: Path) -> None:
    spec = SPEECHSHIFT_MODEL_SPECS["Helsinki-NLP/opus-mt-en-fr"]
    app = create_translation_app(
        worker_id="translation",
        config=TranslationConfig(
            model_id=spec.model_id,
            revision=spec.revision,
            alias="speechshift-en-fr",
            cache_root=tmp_path,
            source_language="en",
            target_language="fr",
        ),
        engine=FakeTranslationEngine(),
    )
    mark_ready(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/translations",
            json={
                "request_id": "translation-1",
                "model": "speechshift-en-fr",
                "input": "The service is ready.",
                "source_language": "en",
                "target_language": "fr",
            },
        )
        wrong_direction = await client.post(
            "/v1/translations",
            json={
                "request_id": "translation-2",
                "model": "speechshift-en-fr",
                "input": "Hallo",
                "source_language": "de",
                "target_language": "en",
            },
        )

    assert response.status_code == 200
    assert response.json()["output_text"] == "Traduction: The service is ready."
    assert response.json()["source_language"] == "en"
    assert response.json()["target_language"] == "fr"
    assert wrong_direction.status_code == 422
    assert wrong_direction.json()["error"]["code"] == "unsupported_direction"


@pytest.mark.asyncio
async def test_tts_contract_returns_allowlisted_24khz_mono_wav(tmp_path: Path) -> None:
    spec = SPEECHSHIFT_MODEL_SPECS["Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"]
    engine = FakeTTSEngine()
    app = create_tts_app(
        worker_id="tts",
        config=TTSConfig(
            model_id=spec.model_id,
            revision=spec.revision,
            alias="speechshift-voice",
            cache_root=tmp_path,
        ),
        engine=engine,
        thermal_guard=ThermalGuard(lambda: TemperatureSnapshot(45, 55)),
    )
    mark_ready(app, tts=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/audio/speech",
            json={
                "request_id": "speech-1",
                "model": "speechshift-voice",
                "input": "The service is ready.",
                "voice": "ryan",
                "language": "en",
                "response_format": "wav",
            },
        )
        invalid_voice = await client.post(
            "/v1/audio/speech",
            json={
                "request_id": "speech-2",
                "model": "speechshift-voice",
                "input": "No arbitrary voices.",
                "voice": "custom",
                "language": "en",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(response.content), "rb") as wav:
        assert wav.getframerate() == 24_000
        assert wav.getnchannels() == 1
    assert engine.calls == [("The service is ready.", "ryan", "en")]
    assert invalid_voice.status_code == 422
    assert invalid_voice.json()["error"]["code"] == "unsupported_voice"


def test_tts_fails_closed_when_start_temperature_is_unsafe() -> None:
    guard = ThermalGuard(lambda: TemperatureSnapshot(56, 60))

    with pytest.raises(ThermalGuardError) as caught:
        guard.require_start_safe()

    assert caught.value.code == "thermal_cooldown_required"


@pytest.mark.asyncio
async def test_recognition_contract_is_bounded_and_content_free_in_metrics(tmp_path: Path) -> None:
    spec = SPEECHSHIFT_MODEL_SPECS["openai/whisper-small.en"]
    runner = FakeRecognitionRunner()
    app = create_recognition_app(
        worker_id="recognition",
        config=RecognitionConfig(spec.model_id, spec.revision, "speechshift-stt", tmp_path),
        runner=runner,
        thermal_guard=ThermalGuard(lambda: TemperatureSnapshot(45, 55)),
    )
    mark_recognition_ready(app)
    payload = recognition_payload()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/audio/transcriptions", json=payload)
        metrics = await client.get("/metrics")
        too_long = await client.post(
            "/v1/audio/transcriptions", json=recognition_payload(samples=128001, request_id="too-long")
        )
        wrong_language = await client.post(
            "/v1/audio/transcriptions",
            json={**recognition_payload(request_id="wrong-language"), "language": "fr"},
        )

    assert response.status_code == 200
    assert response.json()["text"] == "The service is ready."
    assert response.json()["metrics"]["audio_seconds"] == 0.1
    assert runner.received_lengths == [3200]
    assert too_long.status_code == 422
    assert too_long.json()["error"]["code"] == "invalid_audio"
    assert wrong_language.json()["error"]["code"] == "unsupported_language"
    serialised_metrics = str(metrics.json())
    assert "The service is ready." not in serialised_metrics
    assert str(payload["audio_base64"]) not in serialised_metrics
    assert await asyncio.to_thread(lambda: list(tmp_path.iterdir())) == []


@pytest.mark.asyncio
async def test_recognition_cancellation_releases_memory_and_worker_restarts_cleanly(
    tmp_path: Path,
) -> None:
    spec = SPEECHSHIFT_MODEL_SPECS["openai/whisper-small.en"]
    runner = FakeRecognitionRunner(blocked=True)
    app = create_recognition_app(
        worker_id="recognition",
        config=RecognitionConfig(spec.model_id, spec.revision, "speechshift-stt", tmp_path),
        runner=runner,
        thermal_guard=ThermalGuard(lambda: TemperatureSnapshot(45, 55)),
    )
    mark_recognition_ready(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        active = asyncio.create_task(
            client.post("/v1/audio/transcriptions", json=recognition_payload(request_id="cancel-me"))
        )
        for _ in range(100):
            if app.state.active_request_id == "cancel-me":
                break
            await asyncio.sleep(0.001)
        started = time.perf_counter()
        cancellation = await client.post("/cancel", json={"request_id": "cancel-me"})
        cancellation_seconds = time.perf_counter() - started
        cancelled_response = await active
        runner.blocked = False
        runner.cancelled.clear()
        restarted = await client.post(
            "/v1/audio/transcriptions", json=recognition_payload(request_id="restart")
        )

    assert cancellation.status_code == 200
    assert cancellation.json()["state"] == "cancelled"
    assert cancellation_seconds < 0.25
    assert cancelled_response.status_code == 409
    assert cancelled_response.json()["error"]["code"] == "request_cancelled"
    assert runner.gpu_bytes == 0
    assert restarted.status_code == 200


@pytest.mark.asyncio
async def test_recognition_rejects_concurrent_work_and_enforces_timeout(tmp_path: Path) -> None:
    spec = SPEECHSHIFT_MODEL_SPECS["openai/whisper-small.en"]
    runner = FakeRecognitionRunner(blocked=True)
    app = create_recognition_app(
        worker_id="recognition",
        config=RecognitionConfig(
            spec.model_id,
            spec.revision,
            "speechshift-stt",
            tmp_path,
            recognition_timeout_seconds=0.03,
        ),
        runner=runner,
        thermal_guard=ThermalGuard(lambda: TemperatureSnapshot(45, 55)),
    )
    mark_recognition_ready(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        active = asyncio.create_task(
            client.post("/v1/audio/transcriptions", json=recognition_payload(request_id="active"))
        )
        for _ in range(100):
            if app.state.active_request_id == "active":
                break
            await asyncio.sleep(0.001)
        busy = await client.post(
            "/v1/audio/transcriptions", json=recognition_payload(request_id="concurrent")
        )
        timed_out = await active

    assert busy.status_code == 429
    assert busy.json()["error"]["code"] == "worker_busy"
    assert timed_out.status_code == 504
    assert timed_out.json()["error"]["code"] == "recognition_timeout"
    assert runner.gpu_bytes == 0


@pytest.mark.asyncio
async def test_recognition_refuses_unsafe_start_and_terminates_at_thermal_limit(tmp_path: Path) -> None:
    spec = SPEECHSHIFT_MODEL_SPECS["openai/whisper-small.en"]
    unsafe_app = create_recognition_app(
        worker_id="recognition",
        config=RecognitionConfig(spec.model_id, spec.revision, "speechshift-stt", tmp_path),
        runner=FakeRecognitionRunner(),
        thermal_guard=ThermalGuard(lambda: TemperatureSnapshot(56, 60)),
    )
    mark_recognition_ready(unsafe_app)
    readings = iter([TemperatureSnapshot(45, 55), TemperatureSnapshot(80, 60)])
    blocked_runner = FakeRecognitionRunner(blocked=True)
    hot_app = create_recognition_app(
        worker_id="recognition",
        config=RecognitionConfig(spec.model_id, spec.revision, "speechshift-stt", tmp_path),
        runner=blocked_runner,
        thermal_guard=ThermalGuard(lambda: next(readings)),
    )
    mark_recognition_ready(hot_app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=unsafe_app), base_url="http://test"
    ) as client:
        refused = await client.post("/v1/audio/transcriptions", json=recognition_payload())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=hot_app), base_url="http://test"
    ) as client:
        terminated = await client.post("/v1/audio/transcriptions", json=recognition_payload())

    assert refused.status_code == 503
    assert refused.json()["error"]["code"] == "thermal_cooldown_required"
    assert terminated.status_code == 503
    assert terminated.json()["error"]["code"] == "thermal_limit_reached"
    assert blocked_runner.gpu_bytes == 0


@pytest.mark.asyncio
async def test_recognition_process_group_termination_completes_within_250ms() -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        start_new_session=True,
    )
    started = time.perf_counter()
    await _terminate_process_group(process)

    assert process.returncode is not None
    assert time.perf_counter() - started < 0.25

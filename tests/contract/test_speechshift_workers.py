from __future__ import annotations

import asyncio
import io
import threading
import wave
from pathlib import Path

import httpx
import pytest
from modeldeck.protocol import WorkerState
from modeldeck.speechshift import SPEECHSHIFT_MODEL_SPECS
from modeldeck.thermal import TemperatureSnapshot, ThermalGuard, ThermalGuardError
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

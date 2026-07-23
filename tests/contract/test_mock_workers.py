from __future__ import annotations

import base64

import httpx
import pytest
from fastapi.testclient import TestClient
from modeldeck.contracts.scenechat import SceneAnalysis
from modeldeck.mock_templates import MOCK_WORKER_TEMPLATES
from modeldeck.protocol import GenerationFamily
from modeldeck.protocol_contracts import PROTOCOL_CONTRACTS
from modeldeck.speechshift import QWEN_TTS_VOICES
from modeldeck.workers.mock_worker import create_app


def test_every_trusted_contract_has_a_mock_template() -> None:
    assert set(MOCK_WORKER_TEMPLATES) == set(PROTOCOL_CONTRACTS)
    for contract_id, template in MOCK_WORKER_TEMPLATES.items():
        contract = PROTOCOL_CONTRACTS[contract_id]
        assert template.contract.generation_family == contract.generation_family
        assert all(template.capabilities[capability] is True for capability in contract.required_capabilities)


@pytest.mark.asyncio
async def test_autoregressive_contract_includes_top_k_trace() -> None:
    app = create_app(
        worker_id="test-ar",
        model_id="modeldeck/test-ar",
        revision="fixture",
        family=GenerationFamily.AUTOREGRESSIVE,
        startup_delay=0,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            health = (await client.get("/health")).json()
            trace = (
                await client.post(
                    "/native/autoregressive/trace",
                    json={"prompt": "Welcome", "top_k": 3, "seed": 2, "max_tokens": 4},
                )
            ).json()
            await client.post("/cancel", json={"request_id": "cancel-me"})
            cancelled_stream = await client.post(
                "/v1/chat/completions",
                json={"request_id": "cancel-me", "prompt": "private", "stream": True},
            )
    assert health["protocol_version"] == "1"
    assert health["ready"] is True
    assert trace["events"][0]["selected"]["token"]
    assert len(trace["events"][0]["alternatives"]) == 2
    assert "text_so_far" in trace["events"][0]
    assert trace["prompt_tokens"] == ["Welcome"]
    assert len(trace["prompt_token_ids"]) == len(trace["prompt_tokens"])
    assert trace["user_prompt_tokens"] == ["Welcome"]
    assert len(trace["user_prompt_token_ids"]) == len(trace["user_prompt_tokens"])
    assert "event: cancelled" in cancelled_stream.text
    assert "private" not in cancelled_stream.text


@pytest.mark.asyncio
async def test_text_diffusion_contract_is_seeded_and_has_frames() -> None:
    app = create_app(
        worker_id="test-diffusion",
        model_id="modeldeck/test-diffusion",
        revision="fixture",
        family=GenerationFamily.TEXT_DIFFUSION,
        startup_delay=0,
    )
    request = {"prompt": "A robot arrives at university orientation.", "denoising_steps": 4, "seed": 11}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = (await client.post("/v1/refine", json=request)).json()
            second = (await client.post("/v1/refine", json=request)).json()
            wrong_route = await client.post("/v1/chat/completions", json={"prompt": "wrong engine"})
            job = (await client.post("/v1/diffuse", json=request)).json()
            cancelled = await client.post(f"/v1/jobs/{job['job_id']}/cancel")
            events = await client.get(f"/v1/jobs/{job['job_id']}/events")
    assert first == second
    assert first["frames"][0]["complete"] is False
    assert first["frames"][-1]["complete"] is True
    assert first["frames"][-1]["finish_reason"] == "stop"
    assert first["frames"][-1]["seed"] == 11
    assert wrong_route.status_code == 404
    assert cancelled.json()["state"] == "cancelled"
    assert "event: cancelled" in events.text


@pytest.mark.asyncio
async def test_scenechat_contract_returns_labelled_deterministic_structured_output() -> None:
    app = create_app(
        worker_id="test-scenechat",
        model_id="modeldeck/mock-scenechat-vision",
        revision="fixture-v1",
        family=GenerationFamily.VISION_LANGUAGE,
        startup_delay=0,
    )
    request = {
        "model": "scenechat-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    {"type": "text", "text": "What is visible?"},
                ],
            }
        ],
    }
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            capability_response = await client.get("/capabilities")
            first = await client.post("/v1/chat/completions", json=request)
            second = await client.post("/v1/chat/completions", json=request)
            smoke = await client.post("/native/vision-language/smoke")

    capabilities = capability_response.json()
    assert capabilities["generation_family"] == "vision-language"
    assert capabilities["image_input"] is True
    assert capabilities["structured_output"] is True
    assert capabilities["streaming"] is False
    assert first.status_code == 200
    assert (
        first.json()["choices"][0]["message"]["content"] == second.json()["choices"][0]["message"]["content"]
    )
    analysis = SceneAnalysis.model_validate_json(first.json()["choices"][0]["message"]["content"])
    assert analysis.summary.startswith("Mock SceneChat")
    assert "not physical model inference" in analysis.safety_notes[0]
    assert smoke.json() == {
        "ok": True,
        "model_id": "modeldeck/mock-scenechat-vision",
        "mock": True,
        "visual_contract": "scene-analysis-v1",
    }


@pytest.mark.asyncio
async def test_exact_completion_contract_uses_legacy_completion_shape() -> None:
    app = create_app(
        worker_id="test-completions",
        model_id="modeldeck/mock-openai-completions",
        revision="fixture-v1",
        family=GenerationFamily.AUTOREGRESSIVE,
        contract_id="openai-completions-v1",
        startup_delay=0,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            completion = await client.post("/v1/completions", json={"prompt": "Hello"})
            wrong_surface = await client.post("/v1/chat/completions", json={"prompt": "Hello"})

    assert completion.status_code == 200
    assert completion.json()["object"] == "text_completion"
    assert "text" in completion.json()["choices"][0]
    assert "message" not in completion.json()["choices"][0]
    assert wrong_surface.status_code == 404


@pytest.mark.asyncio
async def test_mock_delay_and_request_failure_do_not_affect_health() -> None:
    delayed = create_app(
        worker_id="test-delayed",
        model_id="modeldeck/mock-openai-chat",
        revision="fixture-v1",
        family=GenerationFamily.AUTOREGRESSIVE,
        contract_id="openai-chat-v1",
        scenario="delayed",
        delay_ms=1,
        startup_delay=0,
    )
    failing = create_app(
        worker_id="test-failing",
        model_id="modeldeck/mock-openai-chat",
        revision="fixture-v1",
        family=GenerationFamily.AUTOREGRESSIVE,
        contract_id="openai-chat-v1",
        scenario="request-error",
        startup_delay=0,
    )
    async with delayed.router.lifespan_context(delayed):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=delayed), base_url="http://test"
        ) as client:
            assert (await client.get("/health")).status_code == 200
            assert (await client.post("/v1/chat/completions", json={"prompt": "Hello"})).status_code == 200
    async with failing.router.lifespan_context(failing):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=failing), base_url="http://test"
        ) as client:
            assert (await client.get("/health")).status_code == 200
            response = await client.post("/v1/chat/completions", json={"prompt": "Hello"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "mock_request_failure"


def test_speech_mock_emits_deterministic_transcript_and_pcm() -> None:
    app = create_app(
        worker_id="test-speech",
        model_id="modeldeck/mock-speech-conversation",
        revision="fixture-v1",
        family=GenerationFamily.SPEECH_CONVERSATION,
        contract_id="speech-conversation-v1",
        startup_delay=0,
    )
    with TestClient(app) as client:
        with client.websocket_connect("/v1/speech/conversations") as socket:
            socket.send_json({"model": "speech-mock"})
            assert socket.receive_json()["type"] == "session.ready"
            assert socket.receive_json()["type"] == "response.started"
            assert socket.receive_json() == {
                "type": "transcript.delta",
                "delta": "Mock local speech response.",
            }
            assert socket.receive_bytes() == bytes(640)
            assert socket.receive_json()["type"] == "transcript.final"
            assert socket.receive_json()["type"] == "response.completed"
            socket.send_json({"type": "session.close"})


def test_speech_mock_request_failure_uses_a_fixed_error_event() -> None:
    app = create_app(
        worker_id="test-speech-error",
        model_id="modeldeck/mock-speech-conversation",
        revision="fixture-v1",
        family=GenerationFamily.SPEECH_CONVERSATION,
        contract_id="speech-conversation-v1",
        scenario="request-error",
        startup_delay=0,
    )
    with TestClient(app) as client:
        with client.websocket_connect("/v1/speech/conversations") as socket:
            socket.send_json({"model": "speech-mock"})
            assert socket.receive_json()["code"] == "mock_request_failure"


@pytest.mark.asyncio
async def test_translation_mock_enforces_its_registered_direction() -> None:
    app = create_app(
        worker_id="test-translation",
        model_id="modeldeck/mock-translation-en-fr",
        revision="fixture-v1",
        family=GenerationFamily.TEXT_TRANSLATION,
        contract_id="translation-en-fr-v1",
        startup_delay=0,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/translations",
                json={
                    "request_id": "translation-1",
                    "model": "visitor-translation",
                    "input": "The service is ready.",
                    "source_language": "en",
                    "target_language": "fr",
                },
            )
            wrong = await client.post(
                "/v1/translations",
                json={
                    "request_id": "translation-2",
                    "model": "visitor-translation",
                    "input": "Hallo",
                    "source_language": "en",
                    "target_language": "de",
                },
            )

    assert response.status_code == 200
    assert response.json()["output_text"] == "Bonjour depuis ModelDeck."
    assert wrong.status_code == 422


@pytest.mark.asyncio
async def test_speech_synthesis_mock_returns_deterministic_24khz_wav() -> None:
    app = create_app(
        worker_id="test-tts",
        model_id="modeldeck/mock-speech-synthesis",
        revision="fixture-v1",
        family=GenerationFamily.SPEECH_SYNTHESIS,
        contract_id="speech-synthesis-v1",
        startup_delay=0,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            capabilities = await client.get("/capabilities")
            responses = []
            for voice in QWEN_TTS_VOICES:
                responses.append(
                    await client.post(
                        "/v1/audio/speech",
                        json={
                            "request_id": f"speech-{voice}",
                            "model": "visitor-voice",
                            "input": "Ready.",
                            "voice": voice,
                            "language": "en",
                            "response_format": "wav",
                        },
                    )
                )
            unsupported = await client.post(
                "/v1/audio/speech",
                json={
                    "request_id": "speech-unsupported",
                    "model": "visitor-voice",
                    "input": "Ready.",
                    "voice": "sohee",
                    "language": "en",
                },
            )
            cloning_override = await client.post(
                "/v1/audio/speech",
                json={
                    "request_id": "speech-cloning",
                    "model": "visitor-voice",
                    "input": "Ready.",
                    "voice": "vivian",
                    "language": "en",
                    "reference_audio": "AAAA",
                    "speaker_wav": "/tmp/reference.wav",
                    "instruct": "Change the style.",
                    "temperature": 0.1,
                },
            )

    assert capabilities.json()["voices"] == ["ryan", "aiden", "vivian", "serena"]
    assert capabilities.json()["languages"] == ["en", "fr", "de"]
    assert {response.content for response in responses} == {responses[0].content}
    for response in responses:
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-modeldeck-sample-rate-hz"] == "24000"
    assert unsupported.status_code == 422
    assert cloning_override.status_code == 422


@pytest.mark.asyncio
async def test_speech_recognition_mock_returns_deterministic_transcript() -> None:
    app = create_app(
        worker_id="test-stt",
        model_id="modeldeck/mock-speech-recognition",
        revision="fixture-v1",
        family=GenerationFamily.SPEECH_RECOGNITION,
        contract_id="speech-recognition-v1",
        startup_delay=0,
    )
    payload = {
        "request_id": "recognition-1",
        "model": "speechshift-stt",
        "language": "en",
        "encoding": "pcm_s16le",
        "sample_rate_hz": 16000,
        "channels": 1,
        "audio_base64": base64.b64encode(bytes(3200)).decode("ascii"),
    }
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = await client.post("/v1/audio/transcriptions", json=payload)
            second = await client.post("/v1/audio/transcriptions", json=payload)

    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["text"] == "The local speech recognition Worker is ready."

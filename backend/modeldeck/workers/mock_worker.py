from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import json
import math
import random
import re
import struct
import time
import uuid
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from modeldeck.contracts.scenechat import SceneAnalysis, SceneObject
from modeldeck.mock_templates import MOCK_SCENARIOS, MOCK_WORKER_TEMPLATES
from modeldeck.protocol import CapabilitySet, GenerationFamily, WorkerHealth, WorkerState
from modeldeck.speechshift import QWEN_TTS_LANGUAGES, QWEN_TTS_VOICES


class CompletionRequest(BaseModel):
    request_id: str | None = None
    model: str = "fast-chat"
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    stream: bool = False
    seed: int = 7
    max_tokens: int = Field(default=32, ge=1, le=512)
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


class TranslationRequest(BaseModel):
    request_id: str
    model: str
    input: str = Field(min_length=1, max_length=4000)
    source_language: str
    target_language: str


class SpeechSynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    model: str
    input: str = Field(min_length=1, max_length=2000)
    voice: str
    language: str
    response_format: str = "wav"


class TranscriptionRequest(BaseModel):
    request_id: str
    model: str
    language: str = "en"
    encoding: str = "pcm_s16le"
    sample_rate_hz: int = 16000
    channels: int = 1
    audio_base64: str = Field(min_length=4, max_length=341336)


def capabilities(family: GenerationFamily, contract_id: str | None = None) -> CapabilitySet:
    if contract_id:
        return CapabilitySet.model_validate(MOCK_WORKER_TEMPLATES[contract_id].capabilities)
    if family == GenerationFamily.AUTOREGRESSIVE:
        return CapabilitySet(chat=True, completions=True, logits=True, top_k_trace=True)
    if family == GenerationFamily.VISION_LANGUAGE:
        return CapabilitySet(
            chat="compatibility-only",
            streaming=False,
            image_input=True,
            structured_output=True,
        )
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
    contract_id: str | None = None,
    scenario: str = "success",
    delay_ms: int = 0,
    startup_delay: float = 0.08,
) -> FastAPI:
    if contract_id is not None:
        template = MOCK_WORKER_TEMPLATES.get(contract_id)
        if template is None or template.contract.generation_family != family:
            raise ValueError("Mock contract does not match the configured generation family")
    if scenario not in MOCK_SCENARIOS:
        raise ValueError("Unknown mock scenario")
    if scenario == "delayed" and not 1 <= delay_ms <= 120_000:
        raise ValueError("Delayed mock scenario requires delay_ms from 1 to 120000")

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

    @app.middleware("http")
    async def apply_mock_scenario(request: Request, call_next):
        if _is_mock_request_path(request.url.path):
            if scenario == "delayed":
                await asyncio.sleep(delay_ms / 1000)
            elif scenario == "request-error":
                return JSONResponse(
                    {
                        "error": {
                            "code": "mock_request_failure",
                            "message": "The configured mock Worker returned a deterministic failure.",
                        }
                    },
                    status_code=503,
                )
        return await call_next(request)

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
        response = {
            "protocol_version": "1",
            "generation_family": family,
            "mock_contract_id": contract_id,
            "mock_scenario": scenario,
            **capabilities(family, contract_id).model_dump(),
        }
        if contract_id == "speech-synthesis-v1":
            response["voices"] = list(QWEN_TTS_VOICES)
            response["languages"] = list(QWEN_TTS_LANGUAGES)
        return response

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
        _require_one_of(family, {GenerationFamily.AUTOREGRESSIVE, GenerationFamily.VISION_LANGUAGE})
        _require_contract(contract_id, {"openai-chat-v1", "scene-analysis-v1"})
        if family == GenerationFamily.VISION_LANGUAGE:
            if body.stream:
                raise HTTPException(400, "Mock SceneChat does not stream")
            return _scenechat_completion_response(body, worker_id)
        if body.stream:
            return StreamingResponse(
                _stream_completion(body, worker_id, app.state.cancelled),
                media_type="text/event-stream",
            )
        return _chat_completion_response(body, worker_id)

    @app.post("/native/vision-language/smoke")
    async def vision_language_smoke():
        _require_family(family, GenerationFamily.VISION_LANGUAGE)
        _require_contract(contract_id, {"scene-analysis-v1"})
        return {
            "ok": True,
            "model_id": model_id,
            "mock": True,
            "visual_contract": "scene-analysis-v1",
        }

    @app.post("/v1/completions")
    async def completion(body: CompletionRequest):
        _require_family(family, GenerationFamily.AUTOREGRESSIVE)
        _require_contract(contract_id, {"openai-completions-v1"})
        if body.stream:
            return StreamingResponse(
                _stream_completion(body, worker_id, app.state.cancelled, chat=False),
                media_type="text/event-stream",
            )
        return _legacy_completion_response(body, worker_id)

    @app.post("/native/autoregressive/trace")
    async def trace(body: CompletionRequest):
        _require_family(family, GenerationFamily.AUTOREGRESSIVE)
        _require_contract(contract_id, {"native-ar-trace-v1"})
        prompt = body.prompt or " ".join(message.get("content", "") for message in body.messages or [])
        prompt_token_ids, prompt_tokens = _mock_tokenise(prompt)
        user_prompt = _latest_mock_user_prompt(body)
        user_prompt_token_ids, user_prompt_tokens = _mock_tokenise(user_prompt)
        return {
            "request_id": str(uuid.uuid4()),
            "model": model_id,
            "prompt_token_ids": prompt_token_ids,
            "prompt_tokens": prompt_tokens,
            "user_prompt_token_ids": user_prompt_token_ids,
            "user_prompt_tokens": user_prompt_tokens,
            "events": _trace_events(
                body,
                prompt_token_ids=prompt_token_ids,
                prompt_tokens=prompt_tokens,
                user_prompt_token_ids=user_prompt_token_ids,
                user_prompt_tokens=user_prompt_tokens,
            ),
        }

    @app.post("/v1/refine")
    async def refine(body: DiffusionRequest):
        _require_family(family, GenerationFamily.TEXT_DIFFUSION)
        _require_contract(contract_id, {"text-diffusion-v1"})
        frames = list(_diffusion_frames(body, "sync"))
        return {"model": model_id, "text": frames[-1]["text"], "frames": frames, "seed": body.seed}

    @app.post("/v1/diffuse")
    async def diffuse(body: DiffusionRequest):
        _require_family(family, GenerationFamily.TEXT_DIFFUSION)
        _require_contract(contract_id, {"text-diffusion-v1"})
        job_id = str(uuid.uuid4())
        frames = list(_diffusion_frames(body, job_id))
        app.state.jobs[job_id] = {"state": "complete", "request": body, "frames": frames}
        return {"job_id": job_id, "state": "complete", "events_url": f"/v1/jobs/{job_id}/events"}

    @app.get("/v1/jobs/{job_id}")
    async def job(job_id: str):
        _require_family(family, GenerationFamily.TEXT_DIFFUSION)
        _require_contract(contract_id, {"text-diffusion-v1"})
        if job_id not in app.state.jobs:
            raise HTTPException(404, "Unknown diffusion job")
        job = app.state.jobs[job_id]
        return {"job_id": job_id, "state": job["state"], "frame_count": len(job["frames"])}

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        _require_contract(contract_id, {"text-diffusion-v1"})
        if job_id not in app.state.jobs:
            raise HTTPException(404, "Unknown diffusion job")
        app.state.jobs[job_id]["state"] = "cancelled"
        return {"job_id": job_id, "state": "cancelled"}

    @app.get("/v1/jobs/{job_id}/events")
    async def job_events(job_id: str):
        _require_contract(contract_id, {"text-diffusion-v1"})
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

    @app.post("/smoke")
    async def speech_smoke():
        _require_family(family, GenerationFamily.SPEECH_CONVERSATION)
        _require_contract(contract_id, {"speech-conversation-v1"})
        return {"ok": True, "output_kind": "audio", "mock": True}

    @app.post("/v1/translations")
    async def translation(body: TranslationRequest):
        _require_family(family, GenerationFamily.TEXT_TRANSLATION)
        _require_contract(contract_id, {"translation-en-fr-v1", "translation-en-de-v1"})
        template = MOCK_WORKER_TEMPLATES[str(contract_id)]
        settings = template.fixed_settings or {}
        if body.source_language != settings.get("source_language") or body.target_language != settings.get(
            "target_language"
        ):
            raise HTTPException(422, "The requested language direction does not match this Worker")
        translated = "Bonjour depuis ModelDeck." if body.target_language == "fr" else "Hallo von ModelDeck."
        return {
            "id": body.request_id,
            "object": "translation",
            "model": body.model,
            "source_language": body.source_language,
            "target_language": body.target_language,
            "output_text": translated,
            "usage": {
                "input_tokens": len(body.input.split()),
                "output_tokens": len(translated.split()),
            },
            "mock": True,
        }

    @app.post("/v1/audio/speech")
    async def speech_synthesis(body: SpeechSynthesisRequest):
        _require_family(family, GenerationFamily.SPEECH_SYNTHESIS)
        _require_contract(contract_id, {"speech-synthesis-v1"})
        if body.voice not in QWEN_TTS_VOICES or body.language not in QWEN_TTS_LANGUAGES:
            raise HTTPException(422, "The requested voice or language is not allowlisted")
        if body.response_format != "wav":
            raise HTTPException(422, "Speech synthesis response_format must be wav")
        return Response(
            content=_deterministic_wav(),
            media_type="audio/wav",
            headers={
                "x-request-id": body.request_id,
                "x-modeldeck-mock": "true",
                "x-modeldeck-sample-rate-hz": "24000",
                "x-modeldeck-audio-duration-seconds": "0.25",
            },
        )

    @app.post("/v1/audio/transcriptions")
    async def speech_recognition(body: TranscriptionRequest):
        _require_family(family, GenerationFamily.SPEECH_RECOGNITION)
        _require_contract(contract_id, {"speech-recognition-v1"})
        if (
            body.language != "en"
            or body.encoding != "pcm_s16le"
            or body.sample_rate_hz != 16000
            or body.channels != 1
        ):
            raise HTTPException(422, "Only mono English PCM16 at 16 kHz is supported")
        try:
            pcm = base64.b64decode(body.audio_base64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise HTTPException(422, "Invalid PCM16 audio") from error
        if not pcm or len(pcm) % 2 or len(pcm) > 256000:
            raise HTTPException(422, "PCM16 audio exceeds the eight-second bound")
        duration = len(pcm) / 32000
        pcm = b""
        return {
            "id": body.request_id,
            "object": "audio.transcription",
            "model": body.model,
            "language": "en",
            "text": "The local speech recognition Worker is ready.",
            "metrics": {
                "audio_seconds": round(duration, 6),
                "inference_seconds": 0.001,
                "total_worker_seconds": 0.002,
            },
            "mock": True,
        }

    @app.post("/native/text-translation/smoke")
    async def translation_smoke():
        _require_family(family, GenerationFamily.TEXT_TRANSLATION)
        return {"ok": True, "output_kind": "translation", "mock": True}

    @app.post("/native/speech-synthesis/smoke")
    async def synthesis_smoke():
        _require_family(family, GenerationFamily.SPEECH_SYNTHESIS)
        return {
            "ok": True,
            "output_kind": "audio",
            "sample_rate_hz": 24000,
            "channels": 1,
            "audio_bytes": len(_deterministic_wav()),
            "mock": True,
        }

    @app.post("/native/speech-recognition/smoke")
    async def recognition_smoke():
        _require_family(family, GenerationFamily.SPEECH_RECOGNITION)
        return {"ok": True, "output_kind": "transcript", "language": "en", "mock": True}

    @app.websocket("/v1/speech/conversations")
    async def speech_conversation(client: WebSocket):
        await client.accept()
        try:
            _require_family(family, GenerationFamily.SPEECH_CONVERSATION)
            _require_contract(contract_id, {"speech-conversation-v1"})
            start = json.loads(await client.receive_text())
            if scenario == "delayed":
                await asyncio.sleep(delay_ms / 1000)
            if scenario == "request-error":
                await client.send_json(
                    {
                        "type": "error",
                        "code": "mock_request_failure",
                        "message": "The configured mock Worker returned a deterministic failure.",
                    }
                )
                await client.close(code=1011)
                return
            await client.send_json(
                {
                    "type": "session.ready",
                    "model": start.get("model", model_id),
                    "audio": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1},
                    "voice": "mock",
                    "language": "en",
                    "mock": True,
                }
            )
            await client.send_json({"type": "response.started"})
            await client.send_json({"type": "transcript.delta", "delta": "Mock local speech response."})
            await client.send_bytes(bytes(640))
            await client.send_json({"type": "transcript.final", "text": "Mock local speech response."})
            await client.send_json({"type": "response.completed", "reason": "mock"})
            await client.close()
        except (HTTPException, ValueError, WebSocketDisconnect):
            if client.client_state.name != "DISCONNECTED":
                await client.close(code=1008)

    return app


def _require_family(actual: GenerationFamily, expected: GenerationFamily) -> None:
    if actual != expected:
        raise HTTPException(404, f"Route requires a {expected.value} worker")


def _require_one_of(actual: GenerationFamily, expected: set[GenerationFamily]) -> None:
    if actual not in expected:
        raise HTTPException(
            404, "Route requires one of: " + ", ".join(sorted(item.value for item in expected))
        )


def _require_contract(actual: str | None, expected: set[str]) -> None:
    if actual is not None and actual not in expected:
        raise HTTPException(404, "Route is not supplied by this mock contract")


def _is_mock_request_path(path: str) -> bool:
    return path in {
        "/v1/chat/completions",
        "/v1/completions",
        "/native/autoregressive/trace",
        "/native/vision-language/smoke",
        "/v1/refine",
        "/v1/diffuse",
        "/smoke",
        "/v1/translations",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/native/text-translation/smoke",
        "/native/speech-synthesis/smoke",
        "/native/speech-recognition/smoke",
    } or path.startswith("/v1/jobs/")


def _chat_completion_response(body: CompletionRequest, worker_id: str) -> dict[str, Any]:
    request_id = body.request_id or str(uuid.uuid4())
    prompt = body.prompt or _message_text(body.messages)
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


def _legacy_completion_response(body: CompletionRequest, worker_id: str) -> dict[str, Any]:
    request_id = body.request_id or str(uuid.uuid4())
    prompt = body.prompt or _message_text(body.messages)
    text = f"Mock local response: {prompt.strip() or 'ready'}"[: body.max_tokens * 8]
    return {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": body.model,
        "provider": worker_id,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(text.split())},
    }


def _scenechat_completion_response(body: CompletionRequest, worker_id: str) -> dict[str, Any]:
    request_id = body.request_id or str(uuid.uuid4())
    analysis = SceneAnalysis(
        summary="Mock SceneChat view for local Route rehearsal.",
        objects=[
            SceneObject(
                label="mock object",
                description="A deterministic placeholder produced without inspecting the supplied image.",
                approximate_location="centre",
            )
        ],
        relationships=["The mock object is centred in the placeholder scene."],
        uncertainties=["Mock output does not describe the supplied image."],
        safety_notes=["This response is deterministic mock data, not physical model inference."],
    )
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "provider": worker_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": analysis.model_dump_json()},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(analysis.model_dump_json().split())},
    }


async def _stream_completion(
    body: CompletionRequest,
    worker_id: str,
    cancelled: set[str],
    *,
    chat: bool = True,
) -> AsyncIterator[str]:
    request_id = body.request_id or str(uuid.uuid4())
    prompt = body.prompt or _message_text(body.messages)
    text = f"Mock local response: {prompt.strip() or 'ready'}"[: body.max_tokens * 8]
    for token in text.split(" "):
        if request_id in cancelled:
            payload = {"id": request_id, "object": "chat.completion.cancelled", "provider": worker_id}
            yield f"event: cancelled\ndata: {json.dumps(payload)}\n\n"
            return
        choice = (
            {"index": 0, "delta": {"content": f"{token} "}, "finish_reason": None}
            if chat
            else {"index": 0, "text": f"{token} ", "finish_reason": None}
        )
        payload = {
            "id": request_id,
            "object": "chat.completion.chunk" if chat else "text_completion",
            "model": body.model,
            "provider": worker_id,
            "choices": [choice],
        }
        yield f"event: token\ndata: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0.005)
    yield "event: complete\ndata: [DONE]\n\n"


def _trace_events(
    body: CompletionRequest,
    *,
    prompt_token_ids: list[int],
    prompt_tokens: list[str],
    user_prompt_token_ids: list[int],
    user_prompt_tokens: list[str],
) -> list[dict[str, Any]]:
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
                "prompt_token_ids": prompt_token_ids,
                "prompt_tokens": prompt_tokens if step == 0 else None,
                "user_prompt_token_ids": user_prompt_token_ids if step == 0 else None,
                "user_prompt_tokens": user_prompt_tokens if step == 0 else None,
            }
        )
    return events


def _latest_mock_user_prompt(body: CompletionRequest) -> str:
    if not body.messages:
        return body.prompt or ""
    message = next((message for message in reversed(body.messages) if message.get("role") == "user"), None)
    return _content_text(message.get("content")) if message else ""


def _message_text(messages: list[dict[str, Any]] | None) -> str:
    return " ".join(_content_text(message.get("content")) for message in messages or [])


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _mock_tokenise(text: str) -> tuple[list[int], list[str]]:
    tokens = re.findall(r"\s+|[^\s]+", text)
    return list(range(len(tokens))), tokens


def _deterministic_wav() -> bytes:
    sample_rate = 24_000
    sample_count = sample_rate // 4
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        frames = bytearray()
        for index in range(sample_count):
            sample = round(2400 * math.sin(2 * math.pi * 440 * index / sample_rate))
            frames.extend(struct.pack("<h", sample))
        writer.writeframes(frames)
    return output.getvalue()


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
    parser.add_argument("--contract", choices=tuple(MOCK_WORKER_TEMPLATES))
    parser.add_argument("--scenario", choices=MOCK_SCENARIOS, default="success")
    parser.add_argument("--delay-ms", type=int, default=0)
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
        contract_id=args.contract,
        scenario=args.scenario,
        delay_ms=args.delay_ms,
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    app.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

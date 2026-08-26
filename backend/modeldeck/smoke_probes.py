from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

ProbeSurface = Literal["worker", "gateway"]
TimeoutClass = Literal["default", "diffusion", "translation", "speech-synthesis", "speech-recognition"]


@dataclass(frozen=True)
class ProbeRequest:
    path: str
    body: dict[str, object] | None
    headers: dict[str, str] | None = None


@dataclass(frozen=True)
class ProtocolProbe:
    capability_id: str
    contract_id: str
    worker_request: Callable[[str, str], ProbeRequest]
    gateway_request: Callable[[str, str], ProbeRequest] | None
    validate: Callable[[Mapping[str, object]], bool]
    timeout_class: TimeoutClass = "default"


def _text_chat_request(model: str, _api_key: str) -> ProbeRequest:
    return ProbeRequest(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the word ready."}],
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
        },
    )


def _image_chat_body(model: str) -> dict[str, object]:
    # A fixed local PNG keeps the probe bounded and prevents file or network access.
    image_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42Y"
        "AAAAASUVORK5CYII="
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Reply with the word ready."},
                ],
            }
        ],
        "max_tokens": 4,
        "temperature": 0,
        "stream": False,
    }


def _image_chat_request(model: str, _api_key: str) -> ProbeRequest:
    return ProbeRequest("/v1/chat/completions", _image_chat_body(model))


def _completion_request(model: str, _api_key: str) -> ProbeRequest:
    return ProbeRequest(
        "/v1/completions",
        {
            "model": model,
            "prompt": "Reply with the word ready.",
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
        },
    )


def _embedding_request(model: str, _api_key: str) -> ProbeRequest:
    return ProbeRequest("/v1/embeddings", {"model": model, "input": ["The local Worker is ready."]})


def _trace_worker_request(model: str, _api_key: str) -> ProbeRequest:
    return ProbeRequest(
        "/native/autoregressive/trace",
        {
            "model": model,
            "prompt": "Reply with the word ready.",
            "max_tokens": 4,
            "temperature": 0,
            "top_k": 3,
            "seed": 7,
        },
    )


def _trace_gateway_request(model: str, _api_key: str) -> ProbeRequest:
    request = _trace_worker_request(model, _api_key)
    return ProbeRequest("/native/v1/autoregressive/traces", request.body)


def _scene_worker_request(_model: str, api_key: str) -> ProbeRequest:
    return ProbeRequest(
        "/native/vision-language/smoke",
        None,
        {"Authorization": f"Bearer {api_key}"},
    )


def _diffusion_worker_request(model: str, _api_key: str) -> ProbeRequest:
    return ProbeRequest(
        "/v1/refine",
        {
            "model": model,
            "prompt": "A local Worker is ready.",
            "denoising_steps": 4,
            "seed": 7,
        },
    )


def _diffusion_gateway_request(model: str, api_key: str) -> ProbeRequest:
    request = _diffusion_worker_request(model, api_key)
    return ProbeRequest("/native/v1/text-diffusion/refine", request.body)


def _native_ok_request(path: str) -> Callable[[str, str], ProbeRequest]:
    def build(_model: str, _api_key: str) -> ProbeRequest:
        return ProbeRequest(path, None)

    return build


def _translation_gateway_request(target_language: str) -> Callable[[str, str], ProbeRequest]:
    def build(model: str, _api_key: str) -> ProbeRequest:
        return ProbeRequest(
            "/v1/translations",
            {
                "request_id": "modeldeck-route-smoke",
                "model": model,
                "input": "The local Worker is ready.",
                "source_language": "en",
                "target_language": target_language,
            },
        )

    return build


def _speech_synthesis_gateway_request(model: str, _api_key: str) -> ProbeRequest:
    return ProbeRequest(
        "/v1/audio/speech",
        {
            "request_id": "modeldeck-route-smoke",
            "model": model,
            "input": "The local Worker is ready.",
            "voice": "ryan",
            "language": "en",
            "response_format": "wav",
        },
    )


def _speech_recognition_gateway_request(model: str, _api_key: str) -> ProbeRequest:
    return ProbeRequest(
        "/v1/audio/transcriptions",
        {
            "request_id": "modeldeck-route-smoke",
            "model": model,
            "language": "en",
            "encoding": "pcm_s16le",
            "sample_rate_hz": 16000,
            "channels": 1,
            "audio_base64": "AAAAAA==",
        },
    )


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_chat(payload: Mapping[str, object]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, Mapping):
        return False
    message = first.get("message")
    return isinstance(message, Mapping) and (
        _non_empty_text(message.get("content")) or _non_empty_text(message.get("reasoning_content"))
    )


def _valid_completion(payload: Mapping[str, object]) -> bool:
    choices = payload.get("choices")
    return (
        isinstance(choices, list)
        and bool(choices)
        and isinstance(choices[0], Mapping)
        and _non_empty_text(choices[0].get("text"))
    )


def _valid_embeddings(payload: Mapping[str, object]) -> bool:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return False
    for index, item in enumerate(data):
        if not isinstance(item, Mapping) or item.get("object") != "embedding" or item.get("index") != index:
            return False
        vector = item.get("embedding")
        if not isinstance(vector, list) or len(vector) != 1024:
            return False
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in vector):
            return False
    return True


def _valid_trace(payload: Mapping[str, object]) -> bool:
    events = payload.get("events")
    return (
        isinstance(events, list)
        and bool(events)
        and all(
            isinstance(event, Mapping)
            and isinstance(event.get("token_id"), int)
            and not isinstance(event.get("token_id"), bool)
            for event in events
        )
    )


def _valid_diffusion(payload: Mapping[str, object]) -> bool:
    frames = payload.get("frames")
    return (
        isinstance(frames, list)
        and bool(frames)
        and all(isinstance(frame, Mapping) and _non_empty_text(frame.get("text")) for frame in frames)
        and _non_empty_text(payload.get("text"))
    )


def _valid_ok(payload: Mapping[str, object]) -> bool:
    return payload.get("ok") is True


def _valid_translation(payload: Mapping[str, object]) -> bool:
    return _non_empty_text(payload.get("output_text")) or (
        payload.get("ok") is True
        and payload.get("source_language") == "en"
        and payload.get("target_language") in {"fr", "de"}
    )


def _valid_synthesis(payload: Mapping[str, object]) -> bool:
    audio_bytes = payload.get("audio_bytes")
    return payload.get("audio") is True or (
        payload.get("ok") is True
        and isinstance(audio_bytes, int)
        and not isinstance(audio_bytes, bool)
        and audio_bytes > 0
    )


def _valid_recognition(payload: Mapping[str, object]) -> bool:
    return _non_empty_text(payload.get("text")) or (
        payload.get("ok") is True and payload.get("output_kind") == "transcript"
    )


PROTOCOL_PROBES = {
    probe.contract_id: probe
    for probe in (
        ProtocolProbe("general-chat", "openai-chat-v1", _text_chat_request, _text_chat_request, _valid_chat),
        ProtocolProbe(
            "general-image-chat",
            "openai-image-chat-v1",
            _image_chat_request,
            _image_chat_request,
            _valid_chat,
        ),
        ProtocolProbe(
            "text-completion",
            "openai-completions-v1",
            _completion_request,
            _completion_request,
            _valid_completion,
        ),
        ProtocolProbe(
            "embeddings",
            "openai-embeddings-v1",
            _embedding_request,
            _embedding_request,
            _valid_embeddings,
        ),
        ProtocolProbe(
            "autoregressive-trace",
            "native-ar-trace-v1",
            _trace_worker_request,
            _trace_gateway_request,
            _valid_trace,
        ),
        ProtocolProbe("scene-analysis", "scene-analysis-v1", _scene_worker_request, None, _valid_ok),
        ProtocolProbe(
            "text-refinement",
            "text-diffusion-v1",
            _diffusion_worker_request,
            _diffusion_gateway_request,
            _valid_diffusion,
            "diffusion",
        ),
        ProtocolProbe(
            "speech-conversation",
            "speech-conversation-v1",
            _native_ok_request("/smoke"),
            None,
            _valid_ok,
        ),
        ProtocolProbe(
            "translation-en-fr",
            "translation-en-fr-v1",
            _native_ok_request("/native/text-translation/smoke"),
            _translation_gateway_request("fr"),
            _valid_translation,
            "translation",
        ),
        ProtocolProbe(
            "translation-en-de",
            "translation-en-de-v1",
            _native_ok_request("/native/text-translation/smoke"),
            _translation_gateway_request("de"),
            _valid_translation,
            "translation",
        ),
        ProtocolProbe(
            "speech-synthesis",
            "speech-synthesis-v1",
            _native_ok_request("/native/speech-synthesis/smoke"),
            _speech_synthesis_gateway_request,
            _valid_synthesis,
            "speech-synthesis",
        ),
        ProtocolProbe(
            "speech-recognition",
            "speech-recognition-v1",
            _native_ok_request("/native/speech-recognition/smoke"),
            _speech_recognition_gateway_request,
            _valid_recognition,
            "speech-recognition",
        ),
    )
}

CAPABILITY_PROBES = {probe.capability_id: probe for probe in PROTOCOL_PROBES.values()}


def probe_for_capability(capability_id: str) -> ProtocolProbe:
    try:
        return CAPABILITY_PROBES[capability_id]
    except KeyError as error:
        raise ValueError(f"Capability has no bounded probe adapter: {capability_id}") from error


def probe_for_contract(contract_id: str) -> ProtocolProbe:
    try:
        return PROTOCOL_PROBES[contract_id]
    except KeyError as error:
        raise ValueError(f"Protocol has no bounded probe adapter: {contract_id}") from error


def build_probe_request(
    probe: ProtocolProbe,
    surface: ProbeSurface,
    model: str,
    *,
    api_key: str = "local",
) -> ProbeRequest:
    builder = probe.worker_request if surface == "worker" else probe.gateway_request
    if builder is None:
        raise ValueError(f"{probe.contract_id} requires an interactive {surface} probe")
    return builder(model, api_key)


def validate_probe_response(probe: ProtocolProbe, payload: Mapping[str, object]) -> bool:
    return probe.validate(payload)


def image_chat_probe_body(model: str) -> dict[str, object]:
    return _image_chat_body(model)

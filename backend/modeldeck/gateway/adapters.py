"""Code-owned gateway protocol adapters.

Operators bind published capabilities to these identifiers; they cannot add paths or
change worker-facing protocol behaviour through configuration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolAdapter:
    contract_id: str
    public_surfaces: tuple[str, ...]
    upstream_path: str | None
    timeout_name: str | None = None
    openai_model: bool = False
    native: bool = False


PROTOCOL_ADAPTERS = {
    adapter.contract_id: adapter
    for adapter in (
        ProtocolAdapter(
            "openai-chat-v1",
            ("POST /v1/chat/completions",),
            "/v1/chat/completions",
            "scenechat_timeout_seconds",
            True,
        ),
        ProtocolAdapter(
            "openai-completions-v1", ("POST /v1/completions",), "/v1/completions", openai_model=True
        ),
        ProtocolAdapter(
            "native-ar-trace-v1",
            ("POST /native/v1/autoregressive/traces",),
            "/native/autoregressive/trace",
            native=True,
        ),
        ProtocolAdapter(
            "scene-analysis-v1",
            ("POST /v1/chat/completions", "POST /v1/vision/analyse"),
            "/v1/chat/completions",
            "scenechat_timeout_seconds",
        ),
        ProtocolAdapter(
            "text-diffusion-v1",
            (
                "POST /native/v1/text-diffusion/refine",
                "POST /native/v1/text-diffusion/jobs",
                "GET /native/v1/text-diffusion/jobs/{job_id}",
                "GET /native/v1/text-diffusion/jobs/{job_id}/events",
                "POST /native/v1/text-diffusion/jobs/{job_id}/cancel",
            ),
            "/v1/refine",
            "diffusion_timeout_seconds",
            native=True,
        ),
        ProtocolAdapter("speech-conversation-v1", ("WS /v1/speech/conversations",), None),
        ProtocolAdapter(
            "translation-en-fr-v1",
            ("POST /v1/translations",),
            "/v1/translations",
            "translation_timeout_seconds",
        ),
        ProtocolAdapter(
            "translation-en-de-v1",
            ("POST /v1/translations",),
            "/v1/translations",
            "translation_timeout_seconds",
        ),
        ProtocolAdapter(
            "speech-synthesis-v1",
            ("POST /v1/audio/speech",),
            "/v1/audio/speech",
            "speech_synthesis_timeout_seconds",
            True,
        ),
        ProtocolAdapter(
            "speech-recognition-v1",
            ("POST /v1/audio/transcriptions",),
            "/v1/audio/transcriptions",
            "speech_recognition_timeout_seconds",
            True,
        ),
    )
}


def adapter_ids(*contract_ids: str) -> set[str]:
    return {contract_id for contract_id in contract_ids if contract_id in PROTOCOL_ADAPTERS}

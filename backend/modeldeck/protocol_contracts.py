from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from modeldeck.protocol import GenerationFamily


class ProtocolContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    display_name: str
    generation_family: GenerationFamily
    required_capabilities: tuple[str, ...] = ()
    surfaces: tuple[str, ...]


PROTOCOL_CONTRACTS = {
    contract.id: contract
    for contract in (
        ProtocolContract(
            id="openai-chat-v1",
            display_name="OpenAI-compatible chat",
            generation_family=GenerationFamily.AUTOREGRESSIVE,
            required_capabilities=("chat",),
            surfaces=("POST /v1/chat/completions",),
        ),
        ProtocolContract(
            id="openai-completions-v1",
            display_name="OpenAI-compatible completions",
            generation_family=GenerationFamily.AUTOREGRESSIVE,
            required_capabilities=("completions",),
            surfaces=("POST /v1/completions",),
        ),
        ProtocolContract(
            id="native-ar-trace-v1",
            display_name="Native autoregressive trace",
            generation_family=GenerationFamily.AUTOREGRESSIVE,
            required_capabilities=("top_k_trace",),
            surfaces=("POST /native/autoregressive/trace",),
        ),
        ProtocolContract(
            id="scene-analysis-v1",
            display_name="Scene analysis",
            generation_family=GenerationFamily.VISION_LANGUAGE,
            required_capabilities=("image_input", "structured_output"),
            surfaces=("POST /v1/chat/completions", "POST /v1/vision/analyse"),
        ),
        ProtocolContract(
            id="text-diffusion-v1",
            display_name="Text diffusion",
            generation_family=GenerationFamily.TEXT_DIFFUSION,
            required_capabilities=("iterative_refinement", "intermediate_frames"),
            surfaces=("POST /v1/refine", "POST /v1/diffuse", "GET/POST /v1/jobs/*"),
        ),
        ProtocolContract(
            id="speech-conversation-v1",
            display_name="Speech conversation",
            generation_family=GenerationFamily.SPEECH_CONVERSATION,
            required_capabilities=("audio_input", "audio_output", "full_duplex"),
            surfaces=("WS /v1/speech/conversations",),
        ),
    )
}

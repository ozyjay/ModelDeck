from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modeldeck.protocol import GenerationFamily
from modeldeck.protocol_contracts import PROTOCOL_CONTRACTS, ProtocolContract

MOCK_SCENARIOS = ("success", "delayed", "request-error")


@dataclass(frozen=True)
class MockWorkerTemplate:
    contract_id: str
    model_id: str
    default_name: str
    capabilities: dict[str, bool | str]
    options: tuple[dict[str, Any], ...] = ()

    @property
    def contract(self) -> ProtocolContract:
        return PROTOCOL_CONTRACTS[self.contract_id]

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.contract_id,
            "protocol_contract": self.contract_id,
            "display_name": self.contract.display_name,
            "generation_family": self.contract.generation_family,
            "default_name": self.default_name,
            "scenarios": list(MOCK_SCENARIOS),
            "options": list(self.options),
        }


VISUAL_TOKEN_OPTION = {
    "id": "visual_token_budget",
    "label": "Visual tokens",
    "type": "select",
    "default": 70,
    "choices": [70, 140, 280, 560, 1120],
}

MOCK_WORKER_TEMPLATES = {
    template.contract_id: template
    for template in (
        MockWorkerTemplate(
            "openai-chat-v1",
            "modeldeck/mock-openai-chat",
            "OpenAI chat mock",
            {"chat": True, "streaming": True, "cancellation": True},
        ),
        MockWorkerTemplate(
            "openai-completions-v1",
            "modeldeck/mock-openai-completions",
            "OpenAI completions mock",
            {"completions": True, "streaming": True, "cancellation": True},
        ),
        MockWorkerTemplate(
            "native-ar-trace-v1",
            "modeldeck/mock-autoregressive-trace",
            "Autoregressive trace mock",
            {"top_k_trace": True, "logits": True, "cancellation": True},
        ),
        MockWorkerTemplate(
            "scene-analysis-v1",
            "modeldeck/mock-scenechat-vision",
            "Scene analysis mock",
            {
                "chat": "compatibility-only",
                "streaming": False,
                "cancellation": True,
                "image_input": True,
                "structured_output": True,
            },
            (VISUAL_TOKEN_OPTION,),
        ),
        MockWorkerTemplate(
            "text-diffusion-v1",
            "modeldeck/mock-text-diffusion",
            "Text diffusion mock",
            {
                "iterative_refinement": True,
                "intermediate_frames": True,
                "seeded_generation": True,
                "cancellation": True,
            },
        ),
        MockWorkerTemplate(
            "speech-conversation-v1",
            "modeldeck/mock-speech-conversation",
            "Speech conversation mock",
            {"audio_input": True, "audio_output": True, "full_duplex": True, "cancellation": True},
        ),
    )
}

assert set(MOCK_WORKER_TEMPLATES) == set(PROTOCOL_CONTRACTS)


def legacy_mock_contract(model_id: str, family: GenerationFamily) -> str | None:
    if model_id == "modeldeck/mock-scenechat-vision":
        return "scene-analysis-v1"
    return {
        GenerationFamily.TEXT_DIFFUSION: "text-diffusion-v1",
        GenerationFamily.VISION_LANGUAGE: "scene-analysis-v1",
        GenerationFamily.SPEECH_CONVERSATION: "speech-conversation-v1",
    }.get(family)

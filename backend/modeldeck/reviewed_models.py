from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReviewedModelSpec:
    """Code-owned facts required to attach an exact model to a trusted runtime."""

    model_id: str
    display_name: str
    model_type: str
    architecture: str
    processor_class: str
    configuration_support: str
    capability_ids: tuple[str, ...]
    reviewed_at: str
    quantization: tuple[tuple[str, str], ...] = ()

    def matches_config(self, config: Mapping[str, Any]) -> bool:
        if config.get("model_type") != self.model_type or config.get("architectures") != [self.architecture]:
            return False
        quantization = config.get("quantization_config")
        if not self.quantization:
            return not quantization
        if not isinstance(quantization, Mapping):
            return False
        return all(quantization.get(name) == value for name, value in self.quantization)


_QWEN_CAPABILITIES = (
    "general-chat",
    "text-completion",
    "general-image-chat",
    "video-understanding",
    "scene-analysis",
)


def _qwen_model(
    model_id: str,
    display_name: str,
    *,
    reviewed_at: str = "2026-08-12",
    quantization: tuple[tuple[str, str], ...] = (),
    configuration_support: str = "scenechat-qwen35",
) -> ReviewedModelSpec:
    return ReviewedModelSpec(
        model_id=model_id,
        display_name=display_name,
        model_type="qwen3_5",
        architecture="Qwen3_5ForConditionalGeneration",
        processor_class="Qwen3VLProcessor",
        configuration_support=configuration_support,
        capability_ids=_QWEN_CAPABILITIES,
        reviewed_at=reviewed_at,
        quantization=quantization,
    )


# Adding a reviewed checkpoint in an existing trusted family should normally require one
# entry here and focused validation coverage. A new architecture or quantisation method
# still requires its own worker evidence rather than weakening a generic matcher.
REVIEWED_MODEL_SPECS = {
    spec.model_id: spec
    for spec in (
        _qwen_model("Qwen/Qwen3.5-0.8B", "Qwen3.5 0.8B"),
        _qwen_model("Qwen/Qwen3.5-2B", "Qwen3.5 2B"),
        _qwen_model("Qwen/Qwen3.5-4B", "Qwen3.5 4B"),
        _qwen_model("Qwen/Qwen3.5-9B", "Qwen3.5 9B"),
        _qwen_model(
            "Qwen/Qwen3.8-27B-FP8",
            "Qwen3.8 27B FP8",
            reviewed_at="2026-08-25",
            configuration_support="scenechat-qwen38-fp8",
            quantization=(
                ("quant_method", "fp8"),
                ("activation_scheme", "dynamic"),
                ("fmt", "e4m3"),
            ),
        ),
    )
}


def reviewed_model_spec(model_id: str) -> ReviewedModelSpec | None:
    return REVIEWED_MODEL_SPECS.get(model_id)

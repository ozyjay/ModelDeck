from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from modeldeck.reviewed_models import reviewed_model_spec


@dataclass(frozen=True)
class CapabilityDefinition:
    id: str
    display_name: str
    description: str
    protocol_contract_id: str | None
    traits: tuple[str, ...]
    runtime_template_ids: tuple[str, ...]


CAPABILITY_DEFINITIONS = {
    item.id: item
    for item in (
        CapabilityDefinition(
            "general-chat",
            "General chat",
            "Conversational text generation through the OpenAI-compatible chat contract.",
            "openai-chat-v1",
            ("text-input", "text-output", "chat"),
            (
                "autoregressive-transformers",
                "gpt-oss-llama-vulkan",
                "qwen35-chat-transformers-rocm",
                "qwen38-fp8-chat-transformers-rocm",
                "qwen38-llamacpp-q8-mtp-vulkan",
            ),
        ),
        CapabilityDefinition(
            "text-completion",
            "Text completion",
            "Continuation of a supplied text prompt.",
            "openai-completions-v1",
            ("text-input", "text-output"),
            (
                "autoregressive-transformers",
                "gpt-oss-llama-vulkan",
                "qwen35-chat-transformers-rocm",
                "qwen38-fp8-chat-transformers-rocm",
                "qwen38-llamacpp-q8-mtp-vulkan",
            ),
        ),
        CapabilityDefinition(
            "autoregressive-trace",
            "Autoregressive trace",
            "Native token generation with ranked token trace data.",
            "native-ar-trace-v1",
            ("text-input", "text-output", "token-trace"),
            ("autoregressive-transformers",),
        ),
        CapabilityDefinition(
            "embeddings",
            "Embeddings",
            "Vector embeddings for supplied text.",
            "openai-embeddings-v1",
            ("text-input", "vector-output"),
            ("embedding-transformers",),
        ),
        CapabilityDefinition(
            "general-image-chat",
            "General image chat",
            "Open-ended conversation grounded in one or more images.",
            "openai-image-chat-v1",
            ("text-input", "image-input", "text-output", "chat"),
            ("qwen38-llamacpp-q8-mtp-vulkan",),
        ),
        CapabilityDefinition(
            "video-understanding",
            "Video understanding",
            "Conversation or analysis grounded in video input.",
            None,
            ("text-input", "video-input", "text-output"),
            (),
        ),
        CapabilityDefinition(
            "scene-analysis",
            "Scene analysis",
            "Bounded structured analysis of a supplied scene image.",
            "scene-analysis-v1",
            ("text-input", "image-input", "structured-output"),
            ("scenechat-gemma4", "scenechat-qwen35", "scenechat-qwen38-fp8"),
        ),
        CapabilityDefinition(
            "text-refinement",
            "Text refinement",
            "Iterative text-diffusion refinement with intermediate frames.",
            "text-diffusion-v1",
            ("text-input", "text-output", "iterative-refinement", "intermediate-frames"),
            ("diffusiongemma-transformers", "diffusiongemma-modeldeck-q4"),
        ),
        CapabilityDefinition(
            "speech-conversation",
            "Speech conversation",
            "Full-duplex spoken conversation.",
            "speech-conversation-v1",
            ("audio-input", "audio-output", "streaming"),
            ("moshiko-speech",),
        ),
        CapabilityDefinition(
            "translation-en-fr",
            "English to French translation",
            "Translate English text into French.",
            "translation-en-fr-v1",
            ("text-input", "text-output", "translation"),
            ("opus-translation-cpu",),
        ),
        CapabilityDefinition(
            "translation-en-de",
            "English to German translation",
            "Translate English text into German.",
            "translation-en-de-v1",
            ("text-input", "text-output", "translation"),
            ("opus-translation-cpu",),
        ),
        CapabilityDefinition(
            "speech-synthesis",
            "Speech synthesis",
            "Generate speech audio from text.",
            "speech-synthesis-v1",
            ("text-input", "audio-output"),
            ("qwen3-tts-rocm",),
        ),
        CapabilityDefinition(
            "speech-recognition",
            "Speech recognition",
            "Transcribe speech audio into text.",
            "speech-recognition-v1",
            ("audio-input", "text-output"),
            ("whisper-small-en-rocm",),
        ),
    )
}

CAPABILITY_ID_BY_CONTRACT = {
    definition.protocol_contract_id: definition.id
    for definition in CAPABILITY_DEFINITIONS.values()
    if definition.protocol_contract_id is not None
}

# Reviewed Qwen snapshots can expose text and image capabilities through distinct trusted
# adapters. These are intentionally narrow exceptions to the usual generation-family
# matching rule below; the matcher still requires the exact snapshot and adapter.
QWEN_TEXT_CAPABILITY_TEMPLATES = {
    "qwen38-llamacpp-q8-mtp-vulkan": {
        "general-chat": ("qwen38-llamacpp-q8-mtp-vulkan",),
        "text-completion": ("qwen38-llamacpp-q8-mtp-vulkan",),
        "general-image-chat": ("qwen38-llamacpp-q8-mtp-vulkan",),
    },
    "scenechat-qwen35": {
        "general-chat": ("qwen35-chat-transformers-rocm",),
        "text-completion": ("qwen35-chat-transformers-rocm",),
        "general-image-chat": ("scenechat-qwen35",),
        "scene-analysis": ("scenechat-qwen35",),
    },
    "scenechat-qwen38-fp8": {
        "general-chat": ("qwen35-chat-transformers-rocm", "qwen38-fp8-chat-transformers-rocm"),
        "text-completion": ("qwen35-chat-transformers-rocm", "qwen38-fp8-chat-transformers-rocm"),
        "general-image-chat": ("scenechat-qwen35", "scenechat-qwen38-fp8"),
        "scene-analysis": ("scenechat-qwen35", "scenechat-qwen38-fp8"),
    },
}

FAMILY_CAPABILITIES = {
    "autoregressive": ("general-chat", "text-completion", "autoregressive-trace"),
    "embedding": ("embeddings",),
    "vision-language": ("general-image-chat",),
    "text-diffusion": ("text-refinement",),
    "speech-conversation": ("speech-conversation",),
    "text-translation": (),
    "speech-synthesis": ("speech-synthesis",),
    "speech-recognition": ("speech-recognition",),
}

CAPABILITY_GENERATION_FAMILIES = {
    capability_id: family
    for family, capability_ids in FAMILY_CAPABILITIES.items()
    for capability_id in capability_ids
}
CAPABILITY_GENERATION_FAMILIES.update(
    {
        "scene-analysis": "vision-language",
        "translation-en-fr": "text-translation",
        "translation-en-de": "text-translation",
    }
)


def capability_id_for_contract(protocol_contract: str) -> str | None:
    return CAPABILITY_ID_BY_CONTRACT.get(protocol_contract)


def compatible_runtime_template_ids(
    capability_id: str,
    configuration_support: str | None,
    registrations: Mapping[str, Any],
) -> list[str]:
    """Resolve installed trusted templates without weakening the model's exact matcher."""

    definition = CAPABILITY_DEFINITIONS.get(capability_id)
    if definition is None:
        return []
    qwen_templates = QWEN_TEXT_CAPABILITY_TEMPLATES.get(configuration_support or "", {}).get(
        capability_id, ()
    )
    if qwen_templates:
        return [template_id for template_id in qwen_templates if template_id in registrations]
    if not definition.runtime_template_ids:
        return []
    baseline_registration = registrations.get(configuration_support) if configuration_support else None
    if baseline_registration is None:
        return []
    baseline = baseline_registration.template
    expected_family = CAPABILITY_GENERATION_FAMILIES.get(capability_id)
    required_traits = _required_worker_traits(capability_id)
    result = []
    for template_id, registration in registrations.items():
        template = registration.template
        capabilities = template.capabilities.model_dump(mode="json")
        if (
            template.generation_family.value == expected_family
            and template.generation_family == baseline.generation_family
            and template.cache_setting == baseline.cache_setting
            and template.uses_base_model_identity == baseline.uses_base_model_identity
            and all(capabilities.get(name) is True for name in required_traits)
        ):
            result.append(template_id)
    return sorted(result)


def capability_candidates(
    *,
    model_id: str,
    generation_family: str | None,
    configuration_support: str | None,
    config: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return trusted candidate definitions supported by local facts or reviewed assertions."""

    candidates: dict[str, list[dict[str, Any]]] = {}
    local_detail = _local_detection_detail(config, generation_family)
    for capability_id in FAMILY_CAPABILITIES.get(generation_family, ()):
        candidates.setdefault(capability_id, []).append(
            {
                "kind": "detected",
                "confidence": "inferred",
                "source": "local-cache",
                "detail": local_detail,
            }
        )

    if configuration_support:
        for definition in CAPABILITY_DEFINITIONS.values():
            if configuration_support in definition.runtime_template_ids:
                candidates.setdefault(definition.id, []).append(
                    {
                        "kind": "detected",
                        "confidence": "direct",
                        "source": "trusted-runtime-matcher",
                        "detail": f"The local snapshot matches {configuration_support}.",
                    }
                )

    reviewed = reviewed_model_spec(model_id)
    if reviewed is not None:
        for capability_id in reviewed.capability_ids:
            candidates.setdefault(capability_id, []).append(
                {
                    "kind": "asserted",
                    "confidence": "direct",
                    "source": "reviewed-model-knowledge",
                    "detail": f"The official {reviewed.display_name} checkpoint supports this capability.",
                    "reference": f"https://huggingface.co/{model_id}",
                    "reviewed_at": reviewed.reviewed_at,
                }
            )

    if model_id == "Helsinki-NLP/opus-mt-en-fr":
        candidates.setdefault("translation-en-fr", []).append(_exact_model_evidence(model_id))
    if model_id == "Helsinki-NLP/opus-mt-en-de":
        candidates.setdefault("translation-en-de", []).append(_exact_model_evidence(model_id))

    return [
        {
            "id": definition.id,
            "display_name": definition.display_name,
            "description": definition.description,
            "protocol_contract_id": definition.protocol_contract_id,
            "traits": list(definition.traits),
            "evidence": candidates[definition.id],
            "runtime_template_ids": (
                [configuration_support] if configuration_support in definition.runtime_template_ids else []
            ),
        }
        for definition in CAPABILITY_DEFINITIONS.values()
        if definition.id in candidates
    ]


def capabilities_for_worker(worker: Mapping[str, Any]) -> set[str]:
    template_id = worker.get("runtime_template_id")
    result = {
        definition.id
        for definition in CAPABILITY_DEFINITIONS.values()
        if template_id in definition.runtime_template_ids
        and _worker_matches_capability(worker, definition.id)
    }
    if result:
        return result
    family = str(worker.get("generation_family", ""))
    for capability_id in FAMILY_CAPABILITIES.get(family, ()):
        definition = CAPABILITY_DEFINITIONS[capability_id]
        if definition.protocol_contract_id is None or not _worker_matches_capability(worker, capability_id):
            continue
        result.add(capability_id)
    for capability_id in CAPABILITY_ID_BY_CONTRACT.values():
        definition = CAPABILITY_DEFINITIONS[capability_id]
        expected_family = CAPABILITY_GENERATION_FAMILIES.get(capability_id)
        if expected_family == family and _worker_matches_capability(worker, capability_id):
            result.add(definition.id)
    return result


def worker_cache_identity(worker: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(worker.get("artifact_model_id") or worker.get("model_id")),
        str(worker.get("artifact_revision") or worker.get("revision")),
    )


def worker_configuration_fingerprint(worker: Mapping[str, Any]) -> str:
    fields = (
        "model_id",
        "revision",
        "artifact_model_id",
        "artifact_revision",
        "generation_family",
        "runtime",
        "runtime_template_id",
        "runtime_template_version",
        "dtype",
        "capabilities",
        "settings",
    )
    payload = json.dumps(
        {field: worker.get(field) for field in fields},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def capability_evidence_status(
    worker: Mapping[str, Any], capability_id: str, tests: Iterable[Mapping[str, Any]]
) -> tuple[str, int | None]:
    fingerprint = worker_configuration_fingerprint(worker)
    legacy_worker = worker.get("capability_policy_version") is None
    stale = False
    failed: int | None = None
    for test in tests:
        evidence = test.get("evidence", {})
        if not isinstance(evidence, Mapping):
            continue
        evidence_capability = evidence.get("capability_id")
        if legacy_worker and evidence_capability is None:
            legacy_match = (
                evidence.get("model_id") == worker.get("model_id")
                and evidence.get("model_revision") == worker.get("revision")
                and evidence.get("runtime") == worker.get("runtime")
            )
            if legacy_match and test.get("result") == "tested-working":
                return "legacy", int(test.get("id", 0)) or None
            continue
        if evidence.get("worker_id") != worker.get("id"):
            continue
        if evidence_capability == capability_id:
            if evidence.get("worker_configuration_fingerprint") != fingerprint:
                stale = True
                continue
            if test.get("result") == "tested-working":
                return "qualified", int(test["id"])
            failed = int(test["id"])
    if failed is not None:
        return "failed", failed
    return ("stale", None) if stale else ("not-tested", None)


def _local_detection_detail(config: Mapping[str, Any] | None, family: str | None) -> str:
    if config:
        model_type = config.get("model_type")
        architectures = ", ".join(str(item) for item in config.get("architectures") or ())
        values = [
            value for value in (f"model_type={model_type}" if model_type else "", architectures) if value
        ]
        if values:
            return "Local config.json: " + "; ".join(values)
    return f"Local cache artefacts identify the {family or 'unknown'} model family."


def _exact_model_evidence(model_id: str) -> dict[str, str]:
    return {
        "kind": "detected",
        "confidence": "direct",
        "source": "exact-model-matcher",
        "detail": f"The cached model identity exactly matches {model_id}.",
    }


def _required_worker_traits(capability_id: str) -> tuple[str, ...]:
    return {
        "general-chat": ("chat",),
        "text-completion": ("completions",),
        "autoregressive-trace": ("top_k_trace",),
        "embeddings": ("embeddings",),
        "general-image-chat": ("chat", "image_input"),
        "scene-analysis": ("image_input", "structured_output"),
        "text-refinement": ("iterative_refinement", "intermediate_frames"),
        "speech-conversation": ("audio_input", "audio_output", "full_duplex"),
        "translation-en-fr": ("translation",),
        "translation-en-de": ("translation",),
        "speech-synthesis": ("speech_synthesis", "audio_output"),
        "speech-recognition": ("speech_recognition", "audio_input"),
    }.get(capability_id, ())


def _worker_matches_capability(worker: Mapping[str, Any], capability_id: str) -> bool:
    capabilities = worker.get("capabilities", {})
    if not isinstance(capabilities, Mapping):
        return False
    if not all(capabilities.get(name) is True for name in _required_worker_traits(capability_id)):
        return False
    settings = worker.get("settings", {})
    if not isinstance(settings, Mapping):
        return False
    required_settings = {
        "translation-en-fr": {"source_language": "en", "target_language": "fr"},
        "translation-en-de": {"source_language": "en", "target_language": "de"},
    }.get(capability_id, {})
    return all(settings.get(name) == expected for name, expected in required_settings.items())

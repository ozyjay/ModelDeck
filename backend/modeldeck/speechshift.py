from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LanguageCode = Literal["en", "fr", "de"]


@dataclass(frozen=True)
class SpeechShiftModelSpec:
    model_id: str
    revision: str
    architecture: str
    model_type: str
    generation_family: str
    configuration_support: str
    required_files: frozenset[str]
    source_language: LanguageCode | None = None
    target_language: LanguageCode | None = None


OPUS_REQUIRED_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "pytorch_model.bin",
        "source.spm",
        "target.spm",
        "tokenizer_config.json",
        "vocab.json",
    }
)
QWEN_TTS_REQUIRED_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "preprocessor_config.json",
        "speech_tokenizer/config.json",
        "speech_tokenizer/configuration.json",
        "speech_tokenizer/model.safetensors",
        "speech_tokenizer/preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)

SPEECHSHIFT_MODEL_SPECS = {
    spec.model_id: spec
    for spec in (
        SpeechShiftModelSpec(
            model_id="Helsinki-NLP/opus-mt-en-fr",
            revision="dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18",
            architecture="MarianMTModel",
            model_type="marian",
            generation_family="text-translation",
            configuration_support="opus-translation-cpu",
            required_files=OPUS_REQUIRED_FILES,
            source_language="en",
            target_language="fr",
        ),
        SpeechShiftModelSpec(
            model_id="Helsinki-NLP/opus-mt-en-de",
            revision="6183067f769a302e3861815543b9f312c71b0ca4",
            architecture="MarianMTModel",
            model_type="marian",
            generation_family="text-translation",
            configuration_support="opus-translation-cpu",
            required_files=OPUS_REQUIRED_FILES,
            source_language="en",
            target_language="de",
        ),
        SpeechShiftModelSpec(
            model_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            revision="85e237c12c027371202489a0ec509ded67b5e4b5",
            architecture="Qwen3TTSForConditionalGeneration",
            model_type="qwen3_tts",
            generation_family="speech-synthesis",
            configuration_support="qwen3-tts-rocm",
            required_files=QWEN_TTS_REQUIRED_FILES,
        ),
    )
}

QWEN_TTS_VOICES = ("ryan", "aiden")
QWEN_TTS_LANGUAGES: tuple[LanguageCode, ...] = ("en", "fr", "de")
QWEN_LANGUAGE_NAMES: dict[LanguageCode, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
}
QWEN_TTS_SAMPLE_RATE_HZ = 24_000


def validate_speechshift_snapshot(snapshot: Path, model_id: str, revision: str) -> str | None:
    spec = SPEECHSHIFT_MODEL_SPECS.get(model_id)
    if spec is None or revision != spec.revision:
        return "The repository or revision is not in the SpeechShift allowlist."
    missing = sorted(name for name in spec.required_files if not (snapshot / name).is_file())
    if missing:
        return "The pinned snapshot is incomplete: missing " + ", ".join(missing) + "."
    try:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "The pinned snapshot has no readable Transformers configuration."
    if config.get("architectures") != [spec.architecture] or config.get("model_type") != spec.model_type:
        return "The pinned snapshot does not declare the allowlisted architecture."
    if config.get("auto_map"):
        return "The pinned snapshot unexpectedly requires remote code."
    return None

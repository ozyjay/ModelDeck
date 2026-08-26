import hashlib
import json

import pytest
from modeldeck.profiles import LocalProfileRequest, create_local_profile
from modeldeck.registry import (
    install_runtime_manifest,
    runtime_template_registrations,
    runtime_templates,
)
from modeldeck.speechshift import (
    QWEN_TTS_GENERATION_TIMEOUT_SECONDS,
    QWEN_TTS_MAXIMUM_CODEC_TOKENS,
    SPEECHSHIFT_MODEL_SPECS,
)


def test_packaged_runtime_registry_is_versioned(tmp_path) -> None:
    templates = runtime_templates()
    registrations = runtime_template_registrations()

    assert set(templates) == {
        "autoregressive-transformers",
        "embedding-transformers",
        "scenechat-gemma4",
        "scenechat-qwen35",
        "qwen35-chat-transformers-rocm",
        "scenechat-qwen38-fp8",
        "qwen38-fp8-chat-transformers-rocm",
        "diffusiongemma-transformers",
        "diffusiongemma-modeldeck-q4",
        "gpt-oss-llama-vulkan",
        "qwen35-llamacpp-q8-vulkan",
        "qwen35-llamacpp-q8-vulkan-adaptive",
        "qwen38-llamacpp-q8-mtp-vulkan",
        "qwen38-llamacpp-q8-mtp-vulkan-no-thinking",
        "moshiko-speech",
        "opus-translation-cpu",
        "qwen3-tts-rocm",
        "whisper-small-en-rocm",
    }
    assert registrations["autoregressive-transformers"].package.id == "modeldeck-core"
    assert registrations["scenechat-qwen35"].package.version == "0.8.0"
    assert registrations["autoregressive-transformers"].source == "packaged"


def test_repartee_profiles_are_created_from_allowlisted_templates(tmp_path) -> None:
    gguf = tmp_path / "gpt-oss-120b-mxfp4-00001-of-00003.gguf"
    gguf.write_bytes(b"gguf")
    strong = create_local_profile(
        LocalProfileRequest(
            model_id="ggml-org/gpt-oss-120b-GGUF",
            revision="a" * 40,
            alias="repartee-strong",
            profile_name="repartee-gpt-oss-120b",
            artifact_id="gpt-oss-120b-mxfp4",
            context_length=8192,
            maximum_new_tokens=256,
        ),
        cache_root=tmp_path,
        artifact_path=gguf,
        port=8630,
        configuration_support="gpt-oss-llama-vulkan",
    )
    speech = create_local_profile(
        LocalProfileRequest(
            model_id="kyutai/moshiko-pytorch-bf16",
            revision="b" * 40,
            alias="repartee-speech",
            profile_name="repartee-moshiko",
        ),
        cache_root=tmp_path,
        port=8631,
        configuration_support="moshiko-speech",
    )

    assert strong.id == "local-repartee-gpt-oss-120b"
    assert strong.preferred_runtime == "llama-vulkan"
    assert strong.capabilities.top_k_trace is False
    assert strong.settings["artifact_path"] == str(gguf)
    assert speech.id == "local-repartee-moshiko"
    assert speech.generation_family == "speech-conversation"
    assert speech.capabilities.full_duplex is True


def test_qwen38_llamacpp_profile_keeps_quantised_identity_separate_from_fp8(tmp_path) -> None:
    gguf = tmp_path / "Qwen3.8-27B-Q8_0.gguf"
    gguf.write_bytes(b"gguf")

    profile = create_local_profile(
        LocalProfileRequest(
            model_id="ggml-org/Qwen3.8-27B-GGUF",
            revision="97c30c65c8d9a3e73f9fdfb50f1d1a669e9a2827",
            alias="qwen38-llamacpp",
            artifact_id="qwen38-27b-q8-mtp",
            runtime_template_id="qwen38-llamacpp-q8-mtp-vulkan",
            context_length=8192,
            maximum_new_tokens=512,
        ),
        cache_root=tmp_path,
        artifact_path=gguf,
        port=8630,
        configuration_support="qwen38-llamacpp-q8-mtp-vulkan",
    )

    assert profile.preferred_runtime == "qwen38-llamacpp-vulkan"
    assert profile.runtime_template_version == "0.8.0"
    assert profile.dtype == "q8_0"
    assert profile.settings["runtime_profile"] == "qwen38-q8-mtp-vulkan"
    assert profile.settings["thinking_mode"] == "adaptive"


def test_qwen38_llamacpp_disabled_thinking_is_a_distinct_worker_template(tmp_path) -> None:
    gguf = tmp_path / "Qwen3.8-27B-Q8_0.gguf"
    gguf.write_bytes(b"gguf")

    profile = create_local_profile(
        LocalProfileRequest(
            model_id="ggml-org/Qwen3.8-27B-GGUF",
            revision="97c30c65c8d9a3e73f9fdfb50f1d1a669e9a2827",
            alias="qwen38-no-thinking",
            artifact_id="qwen38-27b-q8-mtp",
            runtime_template_id="qwen38-llamacpp-q8-mtp-vulkan-no-thinking",
            context_length=8192,
            maximum_new_tokens=512,
        ),
        cache_root=tmp_path,
        artifact_path=gguf,
        port=8630,
        configuration_support="qwen38-llamacpp-q8-mtp-vulkan-no-thinking",
    )

    assert profile.preferred_runtime == "qwen38-llamacpp-vulkan"
    assert profile.runtime_template_version == "0.8.0"
    assert profile.settings["thinking_mode"] == "disabled"
    assert profile.capabilities.reasoning is False
    assert profile.capabilities.mtp is True
    assert profile.capabilities.image_input is True


def test_qwen35_llamacpp_profile_is_text_only_with_thinking_disabled(tmp_path) -> None:
    gguf = tmp_path / "Qwen_Qwen3.5-4B-Q8_0.gguf"
    gguf.write_bytes(b"gguf")

    profile = create_local_profile(
        LocalProfileRequest(
            model_id="bartowski/Qwen_Qwen3.5-4B-GGUF",
            revision="4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
            alias="wayfinder-fast-qwen35-4b",
            artifact_id="qwen35-4b-q8",
            runtime_template_id="qwen35-llamacpp-q8-vulkan",
            context_length=8192,
            maximum_new_tokens=256,
        ),
        cache_root=tmp_path,
        artifact_path=gguf,
        port=8630,
        configuration_support="qwen35-llamacpp-q8-vulkan",
    )

    assert profile.preferred_runtime == "qwen35-llamacpp-vulkan"
    assert profile.runtime_template_version == "0.8.0"
    assert profile.dtype == "q8_0"
    assert profile.capabilities.image_input is False
    assert profile.settings["runtime_profile"] == "qwen35-4b-q8-vulkan"
    assert profile.settings["thinking_mode"] == "disabled"


def test_qwen35_llamacpp_adaptive_thinking_is_a_distinct_worker_template(tmp_path) -> None:
    gguf = tmp_path / "Qwen_Qwen3.5-4B-Q8_0.gguf"
    gguf.write_bytes(b"gguf")

    profile = create_local_profile(
        LocalProfileRequest(
            model_id="bartowski/Qwen_Qwen3.5-4B-GGUF",
            revision="4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
            alias="qwen35-adaptive",
            artifact_id="qwen35-4b-q8",
            runtime_template_id="qwen35-llamacpp-q8-vulkan-adaptive",
            context_length=8192,
            maximum_new_tokens=1024,
        ),
        cache_root=tmp_path,
        artifact_path=gguf,
        port=8630,
        configuration_support="qwen35-llamacpp-q8-vulkan-adaptive",
    )

    assert profile.preferred_runtime == "qwen35-llamacpp-vulkan"
    assert profile.runtime_template_version == "0.8.0"
    assert profile.settings["thinking_mode"] == "adaptive"
    assert profile.settings["maximum_new_tokens"] == 1024
    assert profile.capabilities.reasoning is True


def test_local_profile_is_instantiated_from_runtime_template(tmp_path) -> None:
    profile = create_local_profile(
        LocalProfileRequest(
            model_id="google/gemma-4-26B-A4B-it",
            revision="a" * 40,
            alias="gemma-26b-vision",
            dtype="float16",
            context_length=4096,
            maximum_new_tokens=256,
        ),
        cache_root=tmp_path,
        port=8630,
        configuration_support="scenechat-gemma4",
    )

    assert profile.preferred_runtime == "vision-language-transformers-rocm"
    assert profile.generation_family.value == "vision-language"
    assert profile.capabilities.image_input is True
    assert profile.capabilities.structured_output is True
    assert profile.dtype == "bfloat16"
    assert profile.settings["cache_root"] == str(tmp_path)
    assert profile.settings["maximum_new_tokens"] == 256


def test_scenechat_runtime_declares_safe_creation_defaults() -> None:
    template = runtime_templates()["scenechat-gemma4"]

    assert template.dtype == "bfloat16"
    assert template.settings["context_length"] == 8192
    assert template.settings["maximum_new_tokens"] == 512
    assert template.settings["visual_token_budget"] == 280


def test_qwen35_scenechat_runtime_is_dedicated_and_requires_hardware_verification() -> None:
    template = runtime_templates()["scenechat-qwen35"]

    assert template.runtime == "qwen35-vision-language-transformers-rocm"
    assert template.dtype == "bfloat16"
    assert template.settings["context_length"] == 8192
    assert template.settings["maximum_new_tokens"] == 1024
    assert template.settings["visual_token_budget"] == 140
    assert template.settings["hardware_verification_required"] is True


def test_qwen35_chat_runtime_is_dedicated_and_requires_hardware_verification() -> None:
    template = runtime_templates()["qwen35-chat-transformers-rocm"]

    assert template.runtime == "qwen35-chat-transformers-rocm"
    assert template.generation_family.value == "autoregressive"
    assert template.capabilities.chat is True
    assert template.capabilities.completions is True
    assert template.settings["thinking_mode"] == "disabled"
    assert template.settings["hardware_verification_required"] is True


def test_qwen38_native_fp8_runtimes_are_separate_and_hardware_gated() -> None:
    scene = runtime_templates()["scenechat-qwen38-fp8"]
    chat = runtime_templates()["qwen38-fp8-chat-transformers-rocm"]

    assert scene.runtime == "qwen38-fp8-vision-language-transformers-rocm"
    assert chat.runtime == "qwen38-fp8-chat-transformers-rocm"
    assert scene.dtype == chat.dtype == "bfloat16"
    assert scene.settings["hardware_verification_required"] is True
    assert chat.settings["hardware_verification_required"] is True
    assert chat.settings["thinking_mode"] == "disabled"


def test_speechshift_runtimes_are_allowlisted_with_bounded_defaults() -> None:
    translation = runtime_templates()["opus-translation-cpu"]
    synthesis = runtime_templates()["qwen3-tts-rocm"]
    recognition = runtime_templates()["whisper-small-en-rocm"]

    assert translation.runtime == "marian-transformers-cpu"
    assert translation.generation_family.value == "text-translation"
    assert translation.capabilities.translation is True
    assert translation.dtype == "float32"
    assert translation.settings["maximum_input_characters"] == 4_000
    assert synthesis.runtime == "qwen3-tts-rocm"
    assert synthesis.generation_family.value == "speech-synthesis"
    assert synthesis.capabilities.speech_synthesis is True
    assert synthesis.capabilities.cancellation is True
    assert synthesis.settings["sample_rate_hz"] == 24_000
    assert synthesis.settings["maximum_codec_tokens"] == QWEN_TTS_MAXIMUM_CODEC_TOKENS
    assert synthesis.settings["generation_timeout_seconds"] == QWEN_TTS_GENERATION_TIMEOUT_SECONDS
    assert synthesis.settings["hardware_verification_required"] is True
    assert recognition.runtime == "whisper-small-en-rocm"
    assert recognition.generation_family.value == "speech-recognition"
    assert recognition.capabilities.speech_recognition is True
    assert recognition.capabilities.audio_input is True
    assert recognition.settings["sample_rate_hz"] == 16_000
    assert recognition.settings["channels"] == 1
    assert recognition.settings["maximum_audio_seconds"] == 8
    assert recognition.settings["hardware_verification_required"] is True
    whisper = SPEECHSHIFT_MODEL_SPECS["openai/whisper-small.en"]
    assert whisper.revision == "e8727524f962ee844a7319d92be39ac1bd25655a"
    assert whisper.licence == "Apache-2.0"
    assert whisper.licence_review == "approved-for-governed-local-inference"


def test_unknown_runtime_template_cannot_create_a_profile(tmp_path) -> None:
    request = LocalProfileRequest(
        model_id="example/model",
        revision="a" * 40,
        alias="unknown-model",
    )

    try:
        create_local_profile(
            request,
            cache_root=tmp_path,
            port=8630,
            configuration_support="arbitrary-command",
        )
    except ValueError as error:
        assert "allowlisted local worker" in str(error)
    else:
        raise AssertionError("unknown templates must be rejected")


def test_operator_can_install_a_versioned_template_for_a_trusted_implementation(tmp_path) -> None:
    document = {
        "format": "modeldeck-runtime-templates",
        "version": 1,
        "package": {
            "id": "open-day-presets",
            "version": "1.2.0",
            "display_name": "Open Day runtime presets",
            "publisher": "Local operator",
        },
        "templates": [
            {
                "id": "autoregressive-long-context",
                "display_name": "Autoregressive long context",
                "runtime": "transformers-rocm",
                "generation_family": "autoregressive",
                "capabilities": {"chat": True, "completions": True},
                "settings": {"context_length": 8192, "maximum_new_tokens": 256},
                "cache_setting": "cache_root",
            }
        ],
    }
    source = tmp_path / "incoming.json"
    content = (json.dumps(document, indent=2) + "\n").encode()
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    installed = install_runtime_manifest(source, tmp_path / "data", digest)
    registrations = runtime_template_registrations(tmp_path / "data")
    registration = registrations["autoregressive-long-context"]

    assert installed.name == "open-day-presets-1.2.0.json"
    assert registration.source == "trusted-local"
    assert registration.package.version == "1.2.0"
    profile = create_local_profile(
        LocalProfileRequest(
            model_id="example/model",
            revision="a" * 40,
            alias="example-model",
            runtime_template_id="autoregressive-long-context",
        ),
        cache_root=tmp_path / "cache",
        port=8630,
        configuration_support="autoregressive-long-context",
        template_registrations=registrations,
    )
    assert profile.preferred_runtime == "transformers-rocm"
    assert profile.runtime_template_id == "autoregressive-long-context"
    assert profile.runtime_template_version == "1.2.0"


def test_runtime_manifest_installation_requires_exact_operator_approved_digest(tmp_path) -> None:
    source = tmp_path / "runtime.json"
    source.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="operator-approved digest"):
        install_runtime_manifest(source, tmp_path / "data", "0" * 64)


def test_trusted_manifest_cannot_define_a_launch_implementation_or_unknown_settings(tmp_path) -> None:
    document = {
        "format": "modeldeck-runtime-templates",
        "version": 1,
        "package": {
            "id": "unsafe",
            "version": "1.0.0",
            "display_name": "Unsafe",
            "publisher": "Unknown",
        },
        "templates": [
            {
                "id": "unsafe-template",
                "display_name": "Unsafe template",
                "runtime": "arbitrary-python",
                "generation_family": "autoregressive",
                "capabilities": {"chat": True},
                "settings": {"executable_path": "/tmp/run-me"},
                "cache_setting": "cache_root",
            }
        ],
    }
    source = tmp_path / "unsafe.json"
    content = json.dumps(document).encode()
    source.write_bytes(content)

    with pytest.raises(ValueError, match="trusted worker implementation"):
        install_runtime_manifest(source, tmp_path / "data", hashlib.sha256(content).hexdigest())

    document["templates"][0]["runtime"] = "transformers-rocm"
    source.write_text(json.dumps(document), encoding="utf-8")
    content = source.read_bytes()
    with pytest.raises(ValueError, match="settings not accepted"):
        install_runtime_manifest(source, tmp_path / "data", hashlib.sha256(content).hexdigest())

    document["templates"][0]["settings"] = {}
    document["templates"][0]["capabilities"] = {"audio_output": True}
    source.write_text(json.dumps(document), encoding="utf-8")
    content = source.read_bytes()
    with pytest.raises(ValueError, match="capabilities not provided"):
        install_runtime_manifest(source, tmp_path / "data", hashlib.sha256(content).hexdigest())


def test_tampered_installed_runtime_manifest_fails_closed(tmp_path) -> None:
    document = {
        "format": "modeldeck-runtime-templates",
        "version": 1,
        "package": {
            "id": "operator-presets",
            "version": "1.0.0",
            "display_name": "Operator presets",
            "publisher": "Local operator",
        },
        "templates": [
            {
                "id": "operator-autoregressive",
                "display_name": "Operator autoregressive",
                "runtime": "transformers-rocm",
                "generation_family": "autoregressive",
                "capabilities": {"chat": True},
                "settings": {},
                "cache_setting": "cache_root",
            }
        ],
    }
    source = tmp_path / "operator.json"
    content = json.dumps(document).encode()
    source.write_bytes(content)
    installed = install_runtime_manifest(
        source,
        tmp_path / "data",
        hashlib.sha256(content).hexdigest(),
    )
    installed.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="digest changed"):
        runtime_template_registrations(tmp_path / "data")

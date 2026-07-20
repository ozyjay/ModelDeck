import hashlib
import json

import pytest
from modeldeck.profiles import LocalProfileRequest, create_local_profile
from modeldeck.registry import (
    install_runtime_manifest,
    runtime_template_registrations,
    runtime_templates,
)


def test_packaged_runtime_registry_is_versioned(tmp_path) -> None:
    templates = runtime_templates()
    registrations = runtime_template_registrations()

    assert set(templates) == {
        "autoregressive-transformers",
        "scenechat-gemma4",
        "diffusiongemma-transformers",
        "diffusiongemma-modeldeck-q4",
        "gpt-oss-llama-vulkan",
        "moshiko-speech",
    }
    assert registrations["autoregressive-transformers"].package.id == "modeldeck-core"
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

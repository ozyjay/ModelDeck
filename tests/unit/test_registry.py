from modeldeck.profiles import LocalProfileRequest, create_local_profile, default_model_profiles
from modeldeck.registry import reserved_aliases, runtime_templates


def test_packaged_registries_are_versioned_and_cross_referenced(tmp_path) -> None:
    profiles = {profile.id: profile for profile in default_model_profiles()}
    templates = runtime_templates()
    aliases = reserved_aliases()

    assert len(profiles) == 8
    assert set(templates) == {
        "autoregressive-transformers",
        "scenechat-gemma4",
        "diffusiongemma-transformers",
        "diffusiongemma-modeldeck-q4",
        "gpt-oss-llama-vulkan",
        "moshiko-speech",
    }
    assert aliases["scenechat-vision"].default_provider == "scenechat-gemma4-e2b-rocm"
    assert aliases["scenechat-vision"].accepts(profiles["scenechat-gemma4-e2b-rocm"])
    assert not aliases["scenechat-vision"].accepts(profiles["qwen-small-rocm"])
    assert aliases["repartee-strong"].providers == []
    assert aliases["repartee-speech"].required_generation_family == "speech-conversation"


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
            dtype="bfloat16",
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
    assert profile.settings["cache_root"] == str(tmp_path)
    assert profile.settings["maximum_new_tokens"] == 256


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

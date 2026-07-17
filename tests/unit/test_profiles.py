from __future__ import annotations

from pathlib import Path

import pytest
from modeldeck.profiles import (
    LocalAutoregressiveProfileRequest,
    LocalProfileRequest,
    ModelProfile,
    create_local_autoregressive_profile,
    create_local_profile,
    default_model_profiles,
)
from modeldeck.workers.scenechat_worker import EngineConfig
from pydantic import ValidationError


def test_default_profiles_keep_generation_engines_separate() -> None:
    profiles = {profile.id: profile for profile in default_model_profiles()}
    ar = profiles["mock-ar"]
    diffusion = profiles["mock-diffusion"]
    assert ar.generation_family == "autoregressive"
    assert ar.capabilities.top_k_trace is True
    assert ar.capabilities.iterative_refinement is False
    assert diffusion.generation_family == "text-diffusion"
    assert diffusion.capabilities.iterative_refinement is True
    assert diffusion.capabilities.intermediate_frames is True
    live = profiles["qwen-small-rocm"]
    assert live.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert live.local_files_only is True
    assert live.trust_remote_code is False
    assert live.preferred_runtime == "transformers-rocm"
    diffusion_live = profiles["diffusiongemma-rocm"]
    assert diffusion_live.alias == "text-diffusion-bf16"
    assert diffusion_live.revision == "52de6b914ee1749a7d4933202505ddf5b414ec43"
    assert diffusion_live.generation_family == "text-diffusion"
    assert diffusion_live.preferred_runtime == "text-diffusion-transformers-rocm"
    assert diffusion_live.settings["hsa_preload_evidence"] is False
    assert diffusion_live.settings["cache_root"] == "/mnt/work/models/huggingface/hub"
    diffusion_q4 = profiles["diffusiongemma-q4-rocm"]
    assert diffusion_q4.alias == "text-diffusion"
    assert diffusion_q4.port == 8622
    assert diffusion_q4.preferred_runtime == "text-diffusion-gptq-rocm"
    assert "cache_root" not in diffusion_q4.settings
    assert diffusion_q4.settings["q4_checkpoint_dir"].endswith("gptq-q4-g32")
    scenechat = profiles["scenechat-gemma4-e2b-rocm"]
    assert scenechat.model_id == "google/gemma-4-E2B-it"
    assert scenechat.revision == "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf"
    assert scenechat.alias == "scenechat-vision"
    assert scenechat.generation_family == "vision-language"
    assert scenechat.preferred_runtime == "vision-language-transformers-rocm"
    assert scenechat.lifecycle == "on-demand"
    assert scenechat.port == 8000
    assert scenechat.capabilities.chat == "compatibility-only"
    assert scenechat.capabilities.streaming is False
    assert scenechat.capabilities.image_input is True
    assert scenechat.capabilities.structured_output is True
    assert scenechat.settings["generation_timeout_seconds"] == 60
    assert scenechat.settings["maximum_new_tokens"] == 512
    worker_defaults = EngineConfig(model_id=scenechat.model_id, revision=scenechat.revision)
    assert worker_defaults.maximum_new_tokens == scenechat.settings["maximum_new_tokens"]
    assert worker_defaults.generation_timeout_seconds == scenechat.settings["generation_timeout_seconds"]


def test_qwen_workers_are_distinct_pinned_local_profiles() -> None:
    profiles = {profile.id: profile for profile in default_model_profiles()}
    expected = {
        "qwen-small-rocm": (
            "Qwen/Qwen2.5-0.5B-Instruct",
            "7ae557604adf67be50417f59c2c2f167def9a775",
            8620,
        ),
        "qwen-1-5b-rocm": (
            "Qwen/Qwen2.5-1.5B-Instruct",
            "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            8623,
        ),
        "qwen-3b-rocm": (
            "Qwen/Qwen2.5-3B-Instruct",
            "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            8624,
        ),
    }

    for profile_id, (model_id, revision, port) in expected.items():
        profile = profiles[profile_id]
        assert profile.model_id == model_id
        assert profile.revision == revision
        assert profile.port == port
        assert profile.preferred_runtime == "transformers-rocm"
        assert profile.local_files_only is True
        assert profile.trust_remote_code is False
        assert profile.settings["cache_root"] == "/mnt/work/models/huggingface/hub"

    assert len({profiles[profile_id].port for profile_id in expected}) == len(expected)


def test_qwen_json_manifests_match_the_active_allowlist() -> None:
    profiles = {profile.id: profile for profile in default_model_profiles()}
    profile_root = Path(__file__).parents[2] / "profiles/models"

    for filename in ("qwen-small-rocm.json", "qwen-1-5b-rocm.json", "qwen-3b-rocm.json"):
        document = ModelProfile.model_validate_json((profile_root / filename).read_text())
        assert document == profiles[document.id]


def test_scenechat_json_manifest_matches_the_active_allowlist() -> None:
    profiles = {profile.id: profile for profile in default_model_profiles()}
    profile_root = Path(__file__).parents[2] / "profiles/models"
    document = ModelProfile.model_validate_json((profile_root / "scenechat-gemma4-e2b-rocm.json").read_text())

    assert document == profiles[document.id]


def test_profile_rejects_unallowlisted_runtime() -> None:
    document = next(profile for profile in default_model_profiles() if profile.id == "mock-ar").model_dump()
    document["preferred_runtime"] = "shell-command"
    with pytest.raises(ValidationError, match="not allowlisted"):
        ModelProfile.model_validate(document)


def test_profile_rejects_diffusion_capability_on_ar_model() -> None:
    document = next(profile for profile in default_model_profiles() if profile.id == "mock-ar").model_dump()
    document["capabilities"]["iterative_refinement"] = True
    with pytest.raises(ValidationError, match="cannot advertise"):
        ModelProfile.model_validate(document)


def test_local_autoregressive_profile_has_a_fixed_safe_manifest(tmp_path) -> None:
    request = LocalAutoregressiveProfileRequest(
        model_id="Example/LocalModel",
        revision="revision-1",
        alias="local-example",
        dtype="bfloat16",
        lifecycle="on-demand",
        context_length=4096,
        maximum_new_tokens=96,
    )

    profile = create_local_autoregressive_profile(request, cache_root=tmp_path, port=8630)

    assert profile.id == "local-local-example"
    assert profile.preferred_runtime == "transformers-rocm"
    assert profile.local_files_only is True
    assert profile.trust_remote_code is False
    assert profile.settings["cache_root"] == str(tmp_path)
    assert profile.settings["context_length"] == 4096
    assert profile.capabilities.top_k_trace is True


def test_local_profile_request_rejects_unsafe_or_unbounded_fields() -> None:
    with pytest.raises(ValidationError):
        LocalAutoregressiveProfileRequest(
            model_id="Example/LocalModel",
            revision="revision-1",
            alias="unsafe alias; echo",
        )
    with pytest.raises(ValidationError):
        LocalAutoregressiveProfileRequest(
            model_id="Example/LocalModel",
            revision="revision-1",
            alias="safe-alias",
            context_length=100_000,
        )


@pytest.mark.parametrize(
    ("support", "family", "runtime", "lifecycle"),
    [
        (
            "scenechat-gemma4",
            "vision-language",
            "vision-language-transformers-rocm",
            "on-demand",
        ),
        (
            "diffusiongemma-transformers",
            "text-diffusion",
            "text-diffusion-transformers-rocm",
            "exclusive",
        ),
    ],
)
def test_local_family_profiles_use_dedicated_allowlisted_workers(
    tmp_path, support, family, runtime, lifecycle
) -> None:
    request = LocalProfileRequest(
        model_id="google/supported-model",
        revision="revision-1",
        alias="family-model",
        dtype="bfloat16",
        lifecycle="on-demand",
        maximum_new_tokens=256,
        maximum_denoising_steps=24,
    )

    profile = create_local_profile(
        request,
        cache_root=tmp_path,
        port=8630,
        configuration_support=support,
    )

    assert profile.generation_family == family
    assert profile.preferred_runtime == runtime
    assert profile.lifecycle == lifecycle
    assert profile.trust_remote_code is False
    assert profile.settings["cache_root"] == str(tmp_path)


def test_local_q4_profile_separates_release_and_base_model_identity(tmp_path) -> None:
    request = LocalProfileRequest(
        model_id="ozyjay/diffusiongemma-modeldeck-q4",
        revision="release-revision",
        alias="local-diffusion-q4",
        maximum_new_tokens=128,
        maximum_denoising_steps=24,
    )
    checkpoint_dir = tmp_path / "snapshots" / "release-revision"

    profile = create_local_profile(
        request,
        cache_root=tmp_path,
        port=8630,
        configuration_support="diffusiongemma-modeldeck-q4",
        checkpoint_dir=checkpoint_dir,
        base_model_id="google/diffusiongemma-26B-A4B-it",
        base_model_revision="52de6b914ee1749a7d4933202505ddf5b414ec43",
    )

    assert profile.model_id == "google/diffusiongemma-26B-A4B-it"
    assert profile.artifact_model_id == request.model_id
    assert profile.artifact_revision == request.revision
    assert profile.preferred_runtime == "text-diffusion-gptq-rocm"
    assert profile.lifecycle == "exclusive"
    assert profile.dtype == "bfloat16"
    assert profile.settings["q4_checkpoint_dir"] == str(checkpoint_dir)

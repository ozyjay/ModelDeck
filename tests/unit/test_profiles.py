from __future__ import annotations

import pytest
from modeldeck.profiles import ModelProfile, default_model_profiles
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

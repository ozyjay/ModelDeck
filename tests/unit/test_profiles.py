from __future__ import annotations

import pytest
from modeldeck.profiles import ModelProfile, default_model_profiles
from pydantic import ValidationError


def test_default_profiles_keep_generation_engines_separate() -> None:
    ar, diffusion = default_model_profiles()
    assert ar.generation_family == "autoregressive"
    assert ar.capabilities.top_k_trace is True
    assert ar.capabilities.iterative_refinement is False
    assert diffusion.generation_family == "text-diffusion"
    assert diffusion.capabilities.iterative_refinement is True
    assert diffusion.capabilities.intermediate_frames is True


def test_profile_rejects_unallowlisted_runtime() -> None:
    document = default_model_profiles()[0].model_dump()
    document["preferred_runtime"] = "shell-command"
    with pytest.raises(ValidationError, match="not allowlisted"):
        ModelProfile.model_validate(document)


def test_profile_rejects_diffusion_capability_on_ar_model() -> None:
    document = default_model_profiles()[0].model_dump()
    document["capabilities"]["iterative_refinement"] = True
    with pytest.raises(ValidationError, match="cannot advertise"):
        ModelProfile.model_validate(document)

from __future__ import annotations

from modeldeck.profiles import ModelProfile

SCENECHAT_ALIAS = "scenechat-vision"
DEFAULT_SCENECHAT_PROVIDER_ID = "scenechat-gemma4-e2b-rocm"


def scenechat_provider_compatible(profile: ModelProfile) -> bool:
    return (
        profile.generation_family.value == "vision-language"
        and profile.capabilities.image_input is True
        and profile.capabilities.structured_output is True
    )

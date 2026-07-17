from __future__ import annotations

from modeldeck.profiles import ModelProfile
from modeldeck.registry import ReservedAlias, reserved_aliases

SCENECHAT_ALIAS = "scenechat-vision"
DEFAULT_SCENECHAT_PROVIDER_ID = "scenechat-gemma4-e2b-rocm"


def scenechat_provider_compatible(profile: ModelProfile) -> bool:
    return provider_compatible(reserved_aliases()[SCENECHAT_ALIAS], profile)


def selectable_aliases() -> dict[str, ReservedAlias]:
    return {
        alias: contract for alias, contract in reserved_aliases().items() if contract.selection == "explicit"
    }


def provider_compatible(contract: ReservedAlias, profile: ModelProfile) -> bool:
    return contract.accepts(profile)

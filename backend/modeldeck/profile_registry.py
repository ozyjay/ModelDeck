from __future__ import annotations

from pathlib import Path

from modeldeck.compatibility import CompatibilityStore
from modeldeck.profiles import ModelProfile


def load_local_profiles(data_dir: Path) -> list[ModelProfile]:
    store = CompatibilityStore(data_dir / "modeldeck.sqlite3")
    profiles = []
    for document in store.list_model_profiles():
        try:
            profiles.append(ModelProfile.model_validate(document))
        except ValueError:
            continue
    return profiles


def profile_uses_huggingface_cache(profile: ModelProfile) -> bool:
    return bool(profile.settings.get("cache_root"))


def profile_cache_identity(profile: ModelProfile) -> tuple[str, str]:
    return (
        profile.artifact_model_id or profile.model_id,
        profile.artifact_revision or profile.revision,
    )


def profile_allowed(profile: ModelProfile, policy: dict[tuple[str, str], bool]) -> bool:
    if not profile_uses_huggingface_cache(profile):
        return True
    return policy.get(profile_cache_identity(profile), True)

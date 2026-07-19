from __future__ import annotations

from pathlib import Path

from modeldeck.compatibility import CompatibilityStore
from modeldeck.profiles import ModelProfile

DEPLOYMENT_SEED_VERSION = "1"


def ensure_seeded_profiles(data_dir: Path, seeds: list[ModelProfile]) -> list[ModelProfile]:
    store = CompatibilityStore(data_dir / "modeldeck.sqlite3")
    store.initialise()
    if store.configuration_value("deployment_seed_version") is None:
        existing_ids = {str(document.get("id")) for document in store.list_model_profiles()}
        for profile in seeds:
            if profile.id not in existing_ids:
                store.save_model_profile(profile.model_dump(mode="json"), origin="seed")
        store.set_configuration_value("deployment_seed_version", DEPLOYMENT_SEED_VERSION)
    return load_local_profiles(data_dir)


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


def profile_verified(profile: ModelProfile, tests: list[dict]) -> bool:
    if profile.settings.get("hardware_verification_required") is not True:
        return True
    return any(
        test.get("result") == "tested-working"
        and test.get("evidence", {}).get("model_id") == profile.model_id
        and test.get("evidence", {}).get("model_revision") == profile.revision
        and test.get("evidence", {}).get("runtime") == profile.preferred_runtime
        for test in tests
    )

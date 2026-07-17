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

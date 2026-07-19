from pathlib import Path

from modeldeck.profiles import ModelProfile


def default_model_profiles() -> list[ModelProfile]:
    """Worker launch fixtures used only by supervisor and gateway tests."""
    root = Path(__file__).parent / "fixtures/model_profiles"
    return [
        ModelProfile.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*.json"))
    ]

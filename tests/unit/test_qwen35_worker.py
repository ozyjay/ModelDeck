from __future__ import annotations

import json

import pytest
from modeldeck.workers.qwen35_worker import (
    QWEN35_MODEL_IDS,
    EngineConfig,
    TransformersQwen35Engine,
    _configure_qwen35_image_processor,
    _qwen35_visual_token_count,
)


class FakeImageProcessor:
    patch_size = 16
    merge_size = 2
    size = {"shortest_edge": 65_536, "longest_edge": 16_777_216}


@pytest.mark.parametrize("budget,maximum_pixels", [(140, 143_360), (280, 286_720)])
def test_qwen35_visual_token_budget_bounds_processor_pixels(budget: int, maximum_pixels: int) -> None:
    processor = FakeImageProcessor()

    _configure_qwen35_image_processor(processor, budget)

    assert processor.size == {"shortest_edge": 65_536, "longest_edge": maximum_pixels}


def test_qwen35_visual_tokens_are_derived_from_processor_grid() -> None:
    assert _qwen35_visual_token_count({"image_grid_thw": [[1, 20, 28]]}) == 140


@pytest.mark.parametrize("model_id", sorted(QWEN35_MODEL_IDS))
def test_qwen35_snapshot_validation_accepts_only_complete_official_models(tmp_path, model_id) -> None:
    organisation, model_name = model_id.split("/", maxsplit=1)
    snapshot = tmp_path / f"models--{organisation}--{model_name}" / "snapshots" / "pinned"
    snapshot.mkdir(parents=True)
    for filename in (
        "chat_template.jinja",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (snapshot / filename).write_text("{}", encoding="utf-8")
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")
    engine = TransformersQwen35Engine(EngineConfig(model_id=model_id, revision="pinned", cache_root=tmp_path))

    assert engine._validate_snapshot() == snapshot.resolve()


def test_qwen35_snapshot_validation_rejects_third_party_fork(tmp_path) -> None:
    engine = TransformersQwen35Engine(
        EngineConfig(model_id="Example/Qwen3.5-4B", revision="pinned", cache_root=tmp_path)
    )

    with pytest.raises(RuntimeError, match="not an allowlisted"):
        engine._validate_snapshot()

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
from modeldeck.workers.qwen35_worker import (
    QWEN35_MODEL_IDS,
    EngineConfig,
    TransformersQwen35Engine,
    _configure_qwen35_image_processor,
    _qwen35_visual_token_count,
)
from PIL import Image


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


def test_qwen35_generation_retains_deterministic_cached_profile(monkeypatch) -> None:
    transformers = ModuleType("transformers")

    class StoppingCriteria:
        pass

    transformers.StoppingCriteria = StoppingCriteria
    transformers.StoppingCriteriaList = list
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    class Inputs(dict):
        def to(self, _device, dtype):
            assert dtype == "bfloat16"
            return self

    class Generated:
        shape = (220,)

    class Output:
        def __getitem__(self, _key):
            return Generated()

    calls = {}

    class Processor:
        def apply_chat_template(self, _messages, **kwargs):
            calls["template"] = kwargs
            return "rendered"

        def __call__(self, **_kwargs):
            return Inputs(
                input_ids=SimpleNamespace(shape=(1, 400)),
                image_grid_thw=[[1, 20, 28]],
            )

        def decode(self, _generated, **_kwargs):
            return json.dumps(
                {
                    "summary": "A fixed synthetic scene.",
                    "objects": [],
                    "relationships": [],
                    "uncertainties": [],
                    "safety_notes": [],
                }
            )

    class Model:
        def generate(self, **kwargs):
            calls["generation"] = kwargs
            return Output()

    engine = TransformersQwen35Engine(
        EngineConfig(
            model_id="Qwen/Qwen3.5-0.8B",
            revision="pinned",
            maximum_new_tokens=1024,
            visual_token_budget=140,
        )
    )
    engine.processor = Processor()
    engine.model = Model()
    engine.device = "cuda:0"
    engine.dtype = "bfloat16"
    engine.torch = SimpleNamespace(
        cuda=SimpleNamespace(reset_peak_memory_stats=lambda _device: None),
        inference_mode=nullcontext,
    )
    image = Image.new("RGB", (64, 64))
    try:
        result = engine.generate(
            image=image,
            question="Describe the scene.",
            max_tokens=1024,
            cancellation=__import__("threading").Event(),
        )
    finally:
        image.close()

    assert calls["template"]["enable_thinking"] is False
    assert calls["generation"]["max_new_tokens"] == 1024
    assert calls["generation"]["do_sample"] is False
    assert calls["generation"]["use_cache"] is True
    assert result.prompt_tokens == 400
    assert result.completion_tokens == 220
    assert result.visual_tokens == 140

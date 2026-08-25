from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
from modeldeck.reviewed_models import REVIEWED_MODEL_SPECS
from modeldeck.workers.qwen35_worker import (
    QWEN35_MODEL_IDS,
    EngineConfig,
    TransformersQwen35Engine,
    _configure_qwen35_image_processor,
    _is_complete_json_output,
    _qwen35_visual_token_count,
    _qwen_quantization_load_config,
)
from PIL import Image


class FakeImageProcessor:
    patch_size = 16
    merge_size = 2
    size = {"shortest_edge": 65_536, "longest_edge": 16_777_216}


def test_qwen3_8_fp8_is_dequantized_for_offline_bf16_execution() -> None:
    class FakeFineGrainedFP8Config:
        def __init__(self, **settings) -> None:
            self.settings = settings

    spec = REVIEWED_MODEL_SPECS["Qwen/Qwen3.8-27B-FP8"]
    result = _qwen_quantization_load_config(
        spec,
        {
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            }
        },
        FakeFineGrainedFP8Config,
    )

    assert result.settings == {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "dequantize": True,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('{"summary":"complete"}', True),
        ('```json\n{"summary":"complete"}\n```', True),
        ('```json\n{"summary":"complete"}', False),
        ('{"summary":"incomplete"', False),
        ('{"summary":"complete"} trailing', False),
    ],
)
def test_complete_json_output_requires_the_whole_raw_or_fenced_value(
    value: str,
    expected: bool,
) -> None:
    assert _is_complete_json_output(value) is expected


@pytest.mark.parametrize("budget,maximum_pixels", [(140, 143_360), (280, 286_720)])
def test_qwen35_visual_token_budget_bounds_processor_pixels(budget: int, maximum_pixels: int) -> None:
    processor = FakeImageProcessor()

    _configure_qwen35_image_processor(processor, budget)

    assert processor.size == {"shortest_edge": 65_536, "longest_edge": maximum_pixels}


def test_qwen35_visual_tokens_are_derived_from_processor_grid() -> None:
    assert _qwen35_visual_token_count({"image_grid_thw": [[1, 20, 28]]}) == 140


@pytest.mark.parametrize("model_id", sorted(QWEN35_MODEL_IDS))
def test_qwen35_snapshot_validation_accepts_only_complete_official_models(tmp_path, model_id) -> None:
    spec = REVIEWED_MODEL_SPECS[model_id]
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
    config = {"architectures": [spec.architecture], "model_type": spec.model_type}
    if spec.quantization:
        config["quantization_config"] = dict(spec.quantization)
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    engine = TransformersQwen35Engine(EngineConfig(model_id=model_id, revision="pinned", cache_root=tmp_path))

    assert engine._validate_snapshot() == snapshot.resolve()


def test_qwen3_8_snapshot_validation_rejects_unreviewed_fp8_format(tmp_path) -> None:
    model_id = "Qwen/Qwen3.8-27B-FP8"
    snapshot = tmp_path / "models--Qwen--Qwen3.8-27B-FP8" / "snapshots" / "pinned"
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
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "fmt": "e5m2",
                },
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")
    engine = TransformersQwen35Engine(EngineConfig(model_id=model_id, revision="pinned", cache_root=tmp_path))

    with pytest.raises(RuntimeError, match="architecture and quantisation"):
        engine._validate_snapshot()


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

    class InputIds:
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
            calls["complete_json_stops"] = kwargs["stopping_criteria"][1](
                InputIds(),
                scores=None,
            )
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
    assert len(calls["generation"]["stopping_criteria"]) == 2
    assert calls["complete_json_stops"] is True
    assert result.prompt_tokens == 400
    assert result.completion_tokens == 220
    assert result.visual_tokens == 140

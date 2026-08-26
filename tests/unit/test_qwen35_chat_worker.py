from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from modeldeck.workers.autoregressive_worker import EngineConfig, GenerationRequest
from modeldeck.workers.qwen35_chat_worker import TransformersQwen35ChatEngine


def test_qwen35_chat_engine_uses_the_processor_text_tokenizer(monkeypatch) -> None:
    class Qwen3VLProcessor:
        tokenizer = object()

    class Qwen3_5ForConditionalGeneration:
        config = SimpleNamespace(
            max_position_embeddings=None,
            text_config=SimpleNamespace(max_position_embeddings=262_144),
        )

        def to(self, device):
            assert device == "cuda:0"

        def eval(self):
            return None

        def named_parameters(self):
            return []

        def named_buffers(self):
            return []

    processor = Qwen3VLProcessor()
    model = Qwen3_5ForConditionalGeneration()
    torch = SimpleNamespace(
        bfloat16="bfloat16",
        __version__="test",
        version=SimpleNamespace(hip="7.2"),
        device=lambda value: value,
        empty=lambda *args, **kwargs: None,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "AMD test GPU",
        ),
    )
    transformers = SimpleNamespace(
        FineGrainedFP8Config=object,
        AutoConfig=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: SimpleNamespace(
                to_dict=lambda: {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                }
            )
        ),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
        AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr("importlib.metadata.version", lambda package: "test")

    engine = TransformersQwen35ChatEngine(
        EngineConfig(
            model_id="Qwen/Qwen3.5-0.8B",
            revision="a" * 40,
            dtype="bfloat16",
            context_length=8192,
        )
    )
    engine.load()

    assert engine.tokenizer is processor.tokenizer
    assert engine.runtime_details["thinking_mode"] == "disabled"


def test_qwen35_chat_template_explicitly_disables_thinking() -> None:
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "rendered"

    engine = TransformersQwen35ChatEngine(
        EngineConfig(model_id="Qwen/Qwen3.5-4B", revision="a" * 40, dtype="bfloat16")
    )
    engine.tokenizer = Tokenizer()
    request = GenerationRequest(
        messages=[{"role": "user", "content": "Be brief."}],
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        tool_choice="auto",
    )

    assert engine.build_prompt(request) == "rendered"
    assert calls[0][1]["enable_thinking"] is False
    assert calls[0][1]["tool_choice"] == "auto"
    assert calls[0][1]["tools"][0]["name"] == "lookup"


def test_qwen35_chat_engine_rejects_any_other_thinking_mode() -> None:
    with pytest.raises(ValueError, match="thinking_mode=disabled"):
        TransformersQwen35ChatEngine(
            EngineConfig(model_id="Qwen/Qwen3.5-4B", revision="a" * 40, dtype="bfloat16"),
            thinking_mode="adaptive",
        )

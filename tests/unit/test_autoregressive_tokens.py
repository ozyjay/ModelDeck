from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest
from modeldeck.workers.autoregressive_worker import (
    EngineConfig,
    GenerationRequest,
    TransformersAutoregressiveEngine,
    _configured_eos_token_ids,
    _decode_tokens,
    _latest_user_prompt,
    _suppress_token_logits,
    _tokenise_without_special_tokens,
)


class FakeTokenizer:
    tokens = {0: "<bos>", 1: "hello", 2: "  ", 3: "world", 4: "<eos>"}

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        assert text == "hello  world"
        return {"input_ids": [1, 2, 3]}

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return self.tokens[token_ids[0]]


def test_readable_tokens_preserve_order_special_tokens_and_whitespace() -> None:
    tokenizer = FakeTokenizer()

    assert _decode_tokens(tokenizer, [0, 1, 2, 3, 4]) == [
        "<bos>",
        "hello",
        "  ",
        "world",
        "<eos>",
    ]
    assert _tokenise_without_special_tokens(tokenizer, "hello  world") == [1, 2, 3]


def test_latest_user_prompt_excludes_system_wrappers_and_earlier_messages() -> None:
    body = GenerationRequest(
        messages=[
            {"role": "system", "content": "hidden instruction"},
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "hello  world"},
        ]
    )

    assert _latest_user_prompt(body) == "hello  world"


def test_plain_prompt_is_the_displayable_user_prompt() -> None:
    assert _latest_user_prompt(GenerationRequest(prompt="hello  world")) == "hello  world"


def test_all_configured_eos_tokens_are_suppressed_during_minimum_generation() -> None:
    tokenizer = type("Tokenizer", (), {"eos_token_id": 4})()
    generation_config = type("GenerationConfig", (), {"eos_token_id": [4, 5]})()
    model = type("Model", (), {"generation_config": generation_config})()
    logits = [0.0, 1.0, 2.0, 3.0, 9.0, 8.0]

    eos_token_ids = _configured_eos_token_ids(tokenizer, model)
    _suppress_token_logits(logits, eos_token_ids)

    assert eos_token_ids == {4, 5}
    assert logits[:4] == [0.0, 1.0, 2.0, 3.0]
    assert logits[4:] == [float("-inf"), float("-inf")]


def test_token_budget_validation_reports_counts_without_prompt_contents() -> None:
    engine = TransformersAutoregressiveEngine(
        EngineConfig(model_id="Qwen/test", revision="pinned", context_length=32)
    )
    engine.tokenizer = lambda *_args, **_kwargs: {
        "input_ids": SimpleNamespace(shape=(1, 30)),
    }
    prompt = "sensitive prompt content must not appear in the error"

    with pytest.raises(ValueError) as error:
        engine.validate_token_budget(prompt, GenerationRequest(prompt=prompt, max_tokens=3))

    assert str(error.value) == (
        "Token budget exceeds worker context: prompt_tokens=30, requested_output_tokens=3, context_length=32."
    )
    assert prompt not in str(error.value)


def test_cached_decoding_uses_only_new_tokens_after_the_prompt_pass() -> None:
    torch = pytest.importorskip("torch")

    class TraceTokenizer:
        eos_token_id = 0

        def __call__(self, _text: str, *, return_tensors=None, add_special_tokens: bool):
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor([[11, 12, 13]]),
                    "attention_mask": torch.ones((1, 3), dtype=torch.long),
                }
            assert add_special_tokens is False
            return {"input_ids": [11]}

        def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
            return {
                0: "<eos>",
                1: "one",
                2: "two",
                3: "three",
                11: "prompt",
                12: "wrapper",
                13: "token",
            }[token_ids[0]]

    class RecordingModel:
        generation_config = SimpleNamespace(eos_token_id=0)

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def __call__(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            selected_id = len(self.calls)
            logits = torch.full((1, 1, 4), -100.0)
            logits[0, 0, selected_id] = 100.0
            return SimpleNamespace(
                logits=logits,
                past_key_values=("cache", len(self.calls)),
                hidden_states=None,
            )

    engine = TransformersAutoregressiveEngine(
        EngineConfig(model_id="Qwen/test", revision="pinned", context_length=32, maximum_new_tokens=3)
    )
    engine.torch = torch
    engine.tokenizer = TraceTokenizer()
    engine.model = RecordingModel()
    engine.device = torch.device("cpu")
    engine._supports_logits_to_keep = True

    events = list(
        engine.trace(
            prompt="long wrapper prompt",
            body=GenerationRequest(prompt="user prompt", max_tokens=3, temperature=0, top_k=2),
            cancellation=threading.Event(),
        )
    )

    assert [call["input_ids"].shape[-1] for call in engine.model.calls] == [3, 1, 1]
    assert "past_key_values" not in engine.model.calls[0]
    assert engine.model.calls[1]["past_key_values"] == ("cache", 1)
    assert engine.model.calls[2]["past_key_values"] == ("cache", 2)
    assert all(call["use_cache"] is True for call in engine.model.calls)
    assert all(call["logits_to_keep"] == 1 for call in engine.model.calls)
    assert [event["selected"]["token_id"] for event in events] == [1, 2, 3]


def test_cached_decoding_honours_cancellation_before_another_forward() -> None:
    torch = pytest.importorskip("torch")

    class Tokenizer:
        eos_token_id = 0

        def __call__(self, _text: str, *, return_tensors=None, add_special_tokens: bool):
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.ones((1, 1))}
            return {"input_ids": [1]}

        def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
            return {0: "<eos>", 1: "one"}[token_ids[0]]

    class Model:
        generation_config = SimpleNamespace(eos_token_id=0)

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, **_kwargs: Any) -> Any:
            self.calls += 1
            return SimpleNamespace(
                logits=torch.tensor([[[-100.0, 100.0]]]),
                past_key_values=("cache", self.calls),
                hidden_states=None,
            )

    engine = TransformersAutoregressiveEngine(
        EngineConfig(model_id="Qwen/test", revision="pinned", context_length=32, maximum_new_tokens=2)
    )
    engine.torch = torch
    engine.tokenizer = Tokenizer()
    engine.model = Model()
    engine.device = torch.device("cpu")
    engine._supports_logits_to_keep = True
    cancellation = threading.Event()
    iterator = engine.trace(
        prompt="prompt",
        body=GenerationRequest(prompt="prompt", max_tokens=2, temperature=0),
        cancellation=cancellation,
    )

    first = next(iterator)
    cancellation.set()
    cancelled = next(iterator)

    assert first["cancelled"] is False
    assert {key: cancelled[key] for key in ("step", "cancelled", "complete", "text_so_far")} == {
        "step": 1,
        "cancelled": True,
        "complete": True,
        "text_so_far": "one",
    }
    assert cancelled["prefix_cache"]["status"] == "bypass"
    assert engine.model.calls == 1

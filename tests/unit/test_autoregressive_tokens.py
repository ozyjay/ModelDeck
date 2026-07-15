from __future__ import annotations

from typing import Any

from modeldeck.workers.autoregressive_worker import (
    GenerationRequest,
    _decode_tokens,
    _latest_user_prompt,
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

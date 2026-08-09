from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest
from modeldeck.prefix_cache import (
    PREFIX_CACHE_MAX_BYTES,
    PREFIX_CACHE_MAX_TOKENS,
    WAYFINDER_PREFIX_CACHE_MODEL_IDS,
    stable_model_configuration_fingerprint,
    supports_wayfinder_prefix_cache,
)
from modeldeck.workers.autoregressive_worker import (
    EngineConfig,
    GenerationRequest,
    PrefixCacheEntry,
    TransformersAutoregressiveEngine,
    _cache_tensor_bytes,
    _clone_qwen_dynamic_cache,
)
from pydantic import ValidationError

QWEN_SMALL = "Qwen/Qwen2.5-0.5B-Instruct"


class _Row:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return list(self.values)


class _Ids:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.shape = (1, len(values))

    def __getitem__(self, index: int) -> _Row:
        assert index == 0
        return _Row(self.values)


class PrefixTokenizer:
    chat_template = "test-template"
    init_kwargs: dict[str, Any] = {}
    special_tokens_map = {"eos_token": "</s>"}

    def __init__(self, *, mismatch: bool = False, fail_stable_render: bool = False) -> None:
        self.mismatch = mismatch
        self.fail_stable_render = fail_stable_render
        self.full_tokenisations = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: Any,
        tool_choice: Any,
    ) -> str:
        assert tokenize is False
        if not add_generation_prompt and self.fail_stable_render:
            raise RuntimeError("deliberate stable-render failure")
        return "full" if add_generation_prompt else "stable"

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, _Ids]:
        assert return_tensors == "pt"
        assert add_special_tokens is True
        if text == "full":
            self.full_tokenisations += 1
            values = [1, 2, 3, 4]
        else:
            values = [1, 9] if self.mismatch else [1, 2]
        return {"input_ids": _Ids(values), "attention_mask": _Ids([1] * len(values))}


def _request(*, profile_version: str = "wayfinder-agent-v1", tools: Any = None) -> GenerationRequest:
    return GenerationRequest.model_validate(
        {
            "messages": [
                {"role": "system", "content": "stable application instructions"},
                {"role": "user", "content": "dynamic request"},
            ],
            "tools": tools,
            "modeldeck": {
                "prefix_cache": {
                    "stable_message_count": 1,
                    "profile_version": profile_version,
                }
            },
        }
    )


def _engine(tokenizer: PrefixTokenizer | None = None) -> TransformersAutoregressiveEngine:
    engine = TransformersAutoregressiveEngine(
        EngineConfig(model_id=QWEN_SMALL, revision="pinned", prefix_cache_enabled=True)
    )
    engine.tokenizer = tokenizer or PrefixTokenizer()
    engine._configuration_fingerprint = "configuration-a"
    engine._load_epoch = "epoch-a"
    return engine


@pytest.mark.parametrize("model_id", sorted(WAYFINDER_PREFIX_CACHE_MODEL_IDS))
def test_only_dedicated_wayfinder_models_are_allowlisted(model_id: str) -> None:
    assert supports_wayfinder_prefix_cache(model_id)
    assert not supports_wayfinder_prefix_cache("Qwen/Qwen2.5-1.5B-Instruct")
    with pytest.raises(ValueError, match="allowlisted only"):
        EngineConfig(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            revision="pinned",
            prefix_cache_enabled=True,
        )


def test_request_hint_requires_safe_leading_system_messages_and_dynamic_content() -> None:
    base = {
        "modeldeck": {"prefix_cache": {"stable_message_count": 1, "profile_version": "wayfinder-agent-v1"}}
    }
    with pytest.raises(ValidationError, match="message-based"):
        GenerationRequest.model_validate({**base, "prompt": "plain prompt"})
    with pytest.raises(ValidationError, match="leading system"):
        GenerationRequest.model_validate(
            {
                **base,
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": "second"},
                ],
            }
        )
    with pytest.raises(ValidationError, match="leave at least one dynamic"):
        GenerationRequest.model_validate({**base, "messages": [{"role": "system", "content": "only"}]})
    with pytest.raises(ValidationError):
        GenerationRequest.model_validate(
            {
                "messages": [{"role": "system", "content": "stable"}, {"role": "user", "content": "dynamic"}],
                "modeldeck": {"prefix_cache": {"stable_message_count": 1, "profile_version": "unsafe value"}},
            }
        )


def test_complete_prompt_is_tokenised_once_and_exact_prefix_is_required() -> None:
    tokenizer = PrefixTokenizer()
    prepared = _engine(tokenizer).prepare_prompt(_request())

    assert tokenizer.full_tokenisations == 1
    assert prepared.prompt_ids == [1, 2, 3, 4]
    assert prepared.prefix_ids == (1, 2)
    assert prepared.prefix_bypass_reason == "eligible"

    mismatch = _engine(PrefixTokenizer(mismatch=True)).prepare_prompt(_request())
    assert mismatch.prefix_ids is None
    assert mismatch.prefix_bypass_reason == "rendered_token_mismatch"


def test_cache_specific_render_failure_is_a_bounded_bypass() -> None:
    prepared = _engine(PrefixTokenizer(fail_stable_render=True)).prepare_prompt(_request())

    assert prepared.prompt_ids == [1, 2, 3, 4]
    assert prepared.prefix_ids is None
    assert prepared.prefix_bypass_reason == "render_error"


def test_identity_covers_profile_tools_configuration_and_load_epoch() -> None:
    engine = _engine()
    baseline = engine.prepare_prompt(_request()).prefix_identity
    assert baseline
    assert engine.prepare_prompt(_request()).prefix_identity == baseline
    assert engine.prepare_prompt(_request(profile_version="wayfinder-agent-v2")).prefix_identity != baseline
    tools_identity = engine.prepare_prompt(
        _request(tools=[{"type": "function", "function": {"name": "x"}}])
    ).prefix_identity
    assert tools_identity != baseline
    engine._configuration_fingerprint = "configuration-b"
    assert engine.prepare_prompt(_request()).prefix_identity != baseline
    engine._configuration_fingerprint = "configuration-a"
    engine._load_epoch = "epoch-b"
    assert engine.prepare_prompt(_request()).prefix_identity != baseline


def test_clear_reports_only_entry_count_and_released_bytes() -> None:
    engine = _engine()
    engine._prefix_cache_entry = PrefixCacheEntry("opaque", (1, 2), object(), 1234)

    assert engine.clear_prefix_cache() == {"cleared_entries": 1, "released_bytes": 1234}
    assert engine.clear_prefix_cache() == {"cleared_entries": 0, "released_bytes": 0}
    assert engine.prefix_cache_metrics()["prefix_cache_clear_events"] == 2


def test_public_limits_and_configuration_fingerprint_are_stable() -> None:
    assert PREFIX_CACHE_MAX_TOKENS == 8192
    assert PREFIX_CACHE_MAX_BYTES == 512 * 1024 * 1024
    arguments = {
        "model_id": QWEN_SMALL,
        "revision": "pinned",
        "runtime": "transformers-rocm",
        "dtype": "float16",
        "context_length": 32768,
        "runtime_template_version": "2",
    }
    assert stable_model_configuration_fingerprint(**arguments) == stable_model_configuration_fingerprint(
        **arguments
    )
    assert "pinned" not in stable_model_configuration_fingerprint(**arguments)


def test_qwen_cache_branch_has_distinct_storage_and_does_not_mutate_base() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    DynamicCache = transformers.DynamicCache
    base = DynamicCache()
    keys = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    values = keys + 100
    base.update(keys, values, 0)
    base_length = base.get_seq_length()

    branch = _clone_qwen_dynamic_cache(base, expected_tokens=3)

    assert base.get_seq_length() == base_length == 3
    assert branch.get_seq_length() == 3
    assert branch.layers[0].keys.data_ptr() != base.layers[0].keys.data_ptr()
    assert branch.layers[0].values.data_ptr() != base.layers[0].values.data_ptr()
    assert _cache_tensor_bytes(branch) == _cache_tensor_bytes(base)
    branch.layers[0].keys.add_(1)
    assert not torch.equal(branch.layers[0].keys, base.layers[0].keys)


def test_cache_miss_hit_and_bypass_have_equivalent_incremental_traces() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    DynamicCache = transformers.DynamicCache

    class Tokenizer:
        eos_token_id = 0
        chat_template = "test-template"
        init_kwargs: dict[str, Any] = {}
        special_tokens_map: dict[str, Any] = {}

        def apply_chat_template(
            self,
            _messages: list[dict[str, Any]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            tools: Any,
            tool_choice: Any,
        ) -> str:
            assert tokenize is False
            return "full" if add_generation_prompt else "stable"

        def __call__(
            self,
            text: str,
            *,
            return_tensors: str | None = None,
            add_special_tokens: bool,
        ) -> dict[str, Any]:
            values = {"full": [1, 2, 3, 4], "stable": [1, 2]}.get(text, [3, 4])
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor([values]),
                    "attention_mask": torch.ones((1, len(values)), dtype=torch.long),
                }
            assert add_special_tokens is False
            return {"input_ids": values}

        def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
            return {0: "<eos>", 1: "one", 2: "two", 3: "three", 4: "four"}.get(token_ids[0], "selected")

    engine = TransformersAutoregressiveEngine(
        EngineConfig(
            model_id=QWEN_SMALL,
            revision="pinned",
            context_length=32,
            maximum_new_tokens=2,
            prefix_cache_enabled=True,
        )
    )

    class Model:
        generation_config = SimpleNamespace(eos_token_id=0)

        def __init__(self) -> None:
            self.input_lengths: list[int] = []
            self.used_past: list[bool] = []

        def __call__(self, **kwargs: Any) -> Any:
            input_ids = kwargs["input_ids"]
            self.input_lengths.append(int(input_ids.shape[-1]))
            cache = kwargs.get("past_key_values")
            self.used_past.append(cache is not None)
            if cache is None:
                cache = DynamicCache()
            elif engine._prefix_cache_entry is not None:
                assert (
                    cache.layers[0].keys.data_ptr()
                    != engine._prefix_cache_entry.cache.layers[0].keys.data_ptr()
                )
            token_count = int(input_ids.shape[-1])
            keys = torch.ones((1, 1, token_count, 2))
            cache.update(keys, keys + 1, 0)
            logits = torch.full((1, 1, 6), -100.0)
            logits[0, 0, 5] = 100.0
            return SimpleNamespace(logits=logits, past_key_values=cache, hidden_states=None)

    engine.torch = torch
    engine.tokenizer = Tokenizer()
    engine.model = Model()
    engine.device = torch.device("cpu")
    engine._supports_logits_to_keep = True
    engine._configuration_fingerprint = "configuration"
    engine._load_epoch = "epoch"
    hinted = _request().model_copy(update={"max_tokens": 1, "temperature": 0, "top_k": 2})
    bypass = GenerationRequest(
        messages=[
            {"role": "system", "content": "stable application instructions"},
            {"role": "user", "content": "dynamic request"},
        ],
        max_tokens=1,
        temperature=0,
        top_k=2,
    )

    miss = list(
        engine.trace(
            prompt=engine.prepare_prompt(hinted),
            body=hinted,
            cancellation=threading.Event(),
        )
    )
    assert engine._prefix_cache_entry is not None
    base_length = engine._prefix_cache_entry.cache.get_seq_length()
    hit = list(
        engine.trace(
            prompt=engine.prepare_prompt(hinted),
            body=hinted,
            cancellation=threading.Event(),
        )
    )
    bypass_events = list(
        engine.trace(
            prompt=engine.prepare_prompt(bypass),
            body=bypass,
            cancellation=threading.Event(),
        )
    )

    assert engine.model.input_lengths == [2, 2, 2, 4]
    assert engine.model.used_past == [False, True, True, False]
    assert base_length == engine._prefix_cache_entry.cache.get_seq_length() == 2
    assert miss[0]["prefix_cache"]["status"] == "miss"
    assert hit[0]["prefix_cache"]["status"] == "hit"
    assert bypass_events[0]["prefix_cache"]["status"] == "bypass"
    for candidate in (miss, hit):
        assert candidate[0]["selected"] == bypass_events[0]["selected"]
        assert candidate[0]["alternatives"] == bypass_events[0]["alternatives"]
        assert candidate[0]["text_so_far"] == bypass_events[0]["text_so_far"]

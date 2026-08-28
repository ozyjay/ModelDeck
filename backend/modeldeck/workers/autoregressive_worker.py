from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modeldeck.async_execution import iterate_in_isolated_thread, run_in_isolated_thread
from modeldeck.prefix_cache import (
    PREFIX_CACHE_MAX_BYTES,
    PREFIX_CACHE_MAX_TOKENS,
    supports_application_managed_prefix_cache,
)
from modeldeck.protocol import CapabilitySet, GenerationFamily, WorkerState
from modeldeck.registry import MAXIMUM_NEW_TOKENS_LIMIT

LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class EngineConfig:
    model_id: str
    revision: str
    dtype: str = "float16"
    context_length: int = 2048
    maximum_new_tokens: int = 128
    prefix_cache_enabled: bool = False

    def __post_init__(self) -> None:
        if self.context_length > 32_768:
            raise ValueError("Autoregressive worker context length cannot exceed 32,768 tokens")
        if self.prefix_cache_enabled and not supports_application_managed_prefix_cache(self.model_id):
            raise ValueError("Application-managed prefix caching is not qualified for this Worker")


@dataclass(frozen=True)
class PreparedPrompt:
    text: str
    input_ids: Any
    attention_mask: Any | None
    prompt_ids: list[int]
    prefix_ids: tuple[int, ...] | None = None
    prefix_identity: str | None = None
    prefix_bypass_reason: str = "hint_absent"


@dataclass(frozen=True)
class PrefixCacheEntry:
    identity: str
    token_ids: tuple[int, ...]
    cache: Any
    bytes: int


class PrefixCacheHint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stable_message_count: int = Field(ge=1, le=64)
    profile_version: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")


class ModelDeckRequestExtensions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prefix_cache: PrefixCacheHint | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(pattern=r"^(system|user|assistant|tool)$")
    content: str | None = None
    name: str | None = Field(default=None, max_length=128)
    tool_call_id: str | None = Field(default=None, max_length=256)
    tool_calls: list[dict[str, Any]] | None = None

    @field_validator("content", mode="before")
    @classmethod
    def normalise_openai_text_content_parts(cls, value: Any) -> Any:
        """Accept OpenAI text content parts for the text-only local backends."""

        if not isinstance(value, list):
            return value
        text_parts: list[str] = []
        for index, part in enumerate(value):
            if not isinstance(part, dict) or part.get("type") != "text":
                raise ValueError(f"content part {index} must be a text part; this local backend is text-only")
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError(f"content text part {index} requires a string text value")
            text_parts.append(text)
        return "".join(text_parts)

    @model_validator(mode="after")
    def openai_message_shape(self) -> ChatMessage:
        if self.role == "tool" and (not self.tool_call_id or self.content is None):
            raise ValueError("tool messages require tool_call_id and content")
        if self.role == "assistant" and self.content is None and not self.tool_calls:
            raise ValueError("assistant messages require content or tool_calls")
        if self.role in {"system", "user"} and self.content is None:
            raise ValueError(f"{self.role} messages require content")
        return self


class GenerationRequest(BaseModel):
    request_id: str | None = None
    model: str = "local-worker"
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    stream: bool = False
    seed: int = 7
    max_tokens: int = Field(default=32, ge=1, le=MAXIMUM_NEW_TOKENS_LIMIT)
    min_tokens: int = Field(default=0, ge=0, le=MAXIMUM_NEW_TOKENS_LIMIT)
    temperature: float = Field(default=0.2, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    top_k: int = Field(default=5, ge=1, le=50)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=3)
    stop: str | list[str] | None = None
    include_hidden_state_summary: bool = False
    modeldeck: ModelDeckRequestExtensions | None = None

    @model_validator(mode="after")
    def prompt_or_messages(self) -> GenerationRequest:
        if not self.prompt and not self.messages:
            raise ValueError("prompt or messages is required")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens cannot exceed max_tokens")
        hint = self.modeldeck.prefix_cache if self.modeldeck else None
        if hint is not None:
            if not self.messages:
                raise ValueError("prefix caching requires a message-based chat request")
            if hint.stable_message_count >= len(self.messages):
                raise ValueError("stable_message_count must leave at least one dynamic message")
            if any(message.role != "system" for message in self.messages[: hint.stable_message_count]):
                raise ValueError("prefix caching permits only leading system messages in the stable prefix")
        return self


class AutoregressiveEngine(Protocol):
    runtime_details: dict[str, Any]

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def build_prompt(self, body: GenerationRequest) -> str: ...

    def memory_metrics(self) -> dict[str, int]: ...

    def trace(
        self,
        *,
        prompt: str,
        body: GenerationRequest,
        cancellation: threading.Event,
    ) -> Iterator[dict[str, Any]]: ...


class TransformersAutoregressiveEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.runtime_details: dict[str, Any] = {}
        self.torch: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: Any = None
        self._supports_logits_to_keep = False
        self._load_epoch = uuid.uuid4().hex
        self._configuration_fingerprint = ""
        self._prefix_cache_entry: PrefixCacheEntry | None = None
        self._prefix_cache_counters = {
            "hits": 0,
            "misses": 0,
            "bypasses": 0,
            "evictions": 0,
            "clear_events": 0,
        }

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("ROCm PyTorch did not expose an available 'cuda' device")
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self.config.dtype)
        if dtype is None:
            raise RuntimeError(f"Unsupported dtype: {self.config.dtype}")
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
        )
        supported_context_length = _model_context_length(model)
        if self.config.context_length > supported_context_length:
            raise RuntimeError(
                "Configured context length "
                f"{self.config.context_length} exceeds the model-supported limit {supported_context_length}"
            )
        device = torch.device("cuda:0")
        model.to(device)
        model.eval()
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self._supports_logits_to_keep = "logits_to_keep" in inspect.signature(model.forward).parameters
        self._load_epoch = uuid.uuid4().hex
        self._configuration_fingerprint = _configuration_fingerprint(
            config=self.config,
            model=model,
            tokenizer=tokenizer,
            transformers_version=importlib.metadata.version("transformers"),
        )
        self.clear_prefix_cache(count_clear=False)
        self.runtime_details = {
            "torch_version": str(torch.__version__),
            "hip_version": torch.version.hip,
            "transformers_version": importlib.metadata.version("transformers"),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "load_seconds": round(time.perf_counter() - started, 4),
            "dtype": self.config.dtype,
            "model_max_context_tokens": supported_context_length,
            "last_token_logits_only": self._supports_logits_to_keep,
            "configuration_fingerprint": self._configuration_fingerprint,
            "load_epoch": self._load_epoch,
            "prefix_caching": (
                "application-managed"
                if supports_application_managed_prefix_cache(self.config.model_id)
                else "unsupported"
            ),
            "prefix_cache_enabled": self.config.prefix_cache_enabled,
        }

    def warmup(self) -> None:
        body = GenerationRequest(prompt="Hello", max_tokens=1, temperature=0, top_k=1)
        list(self.trace(prompt="Hello", body=body, cancellation=threading.Event()))

    def build_prompt(self, body: GenerationRequest) -> str:
        if body.messages:
            messages = _qwen_chat_messages(body.messages)
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                # Hugging Face Qwen chat templates consume native function schemas,
                # not OpenAI's {"type": "function", "function": ...} envelopes.
                # Keep OpenAI at the boundary and render the template's documented
                # tool shape inside the Worker.
                tools=_qwen_tool_schemas(body.tools),
                tool_choice=body.tool_choice,
            )
        return body.prompt or ""

    def prepare_prompt(self, body: GenerationRequest) -> PreparedPrompt:
        prompt = self.build_prompt(body)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_ids = [int(token_id) for token_id in encoded["input_ids"][0].tolist()]
        hint = body.modeldeck.prefix_cache if body.modeldeck else None
        prefix_ids: tuple[int, ...] | None = None
        prefix_identity: str | None = None
        bypass_reason = "hint_absent"
        if hint is not None and not self.config.prefix_cache_enabled:
            bypass_reason = "disabled"
        elif hint is not None and not supports_application_managed_prefix_cache(self.config.model_id):
            bypass_reason = "unsupported_model"
        elif hint is not None and body.messages:
            stable_messages = _qwen_chat_messages(body.messages[: hint.stable_message_count])
            try:
                stable_prompt = self.tokenizer.apply_chat_template(
                    stable_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    tools=_qwen_tool_schemas(body.tools),
                    tool_choice=body.tool_choice,
                )
                stable_encoded = self.tokenizer(
                    stable_prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                candidate = tuple(int(token_id) for token_id in stable_encoded["input_ids"][0].tolist())
            except Exception as error:
                LOGGER.warning("Stable prefix rendering bypassed error=%s", type(error).__name__)
                candidate = ()
                bypass_reason = "render_error"
            if bypass_reason != "render_error":
                if not candidate:
                    bypass_reason = "empty_prefix"
                elif len(candidate) > PREFIX_CACHE_MAX_TOKENS:
                    bypass_reason = "prefix_token_limit"
                elif tuple(prompt_ids[: len(candidate)]) != candidate:
                    bypass_reason = "rendered_token_mismatch"
                elif len(candidate) >= len(prompt_ids):
                    bypass_reason = "dynamic_suffix_empty"
                else:
                    prefix_ids = candidate
                    prefix_identity = self._prefix_identity(candidate, hint.profile_version, body)
                    bypass_reason = "eligible"
        return PreparedPrompt(
            text=prompt,
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            prompt_ids=prompt_ids,
            prefix_ids=prefix_ids,
            prefix_identity=prefix_identity,
            prefix_bypass_reason=bypass_reason,
        )

    def memory_metrics(self) -> dict[str, int]:
        if self.torch is None or not self.torch.cuda.is_available():
            return {}
        return {
            "memory_allocated_bytes": int(self.torch.cuda.memory_allocated(0)),
            "memory_reserved_bytes": int(self.torch.cuda.memory_reserved(0)),
            "peak_memory_allocated_bytes": int(self.torch.cuda.max_memory_allocated(0)),
            "peak_memory_reserved_bytes": int(self.torch.cuda.max_memory_reserved(0)),
        }

    def prefix_cache_metrics(self) -> dict[str, Any]:
        entry = self._prefix_cache_entry
        return {
            "prefix_caching": (
                "application-managed"
                if supports_application_managed_prefix_cache(self.config.model_id)
                else "unsupported"
            ),
            "prefix_cache_enabled": self.config.prefix_cache_enabled,
            "prefix_cache_entries": 1 if entry else 0,
            "prefix_cache_bytes": entry.bytes if entry else 0,
            "prefix_cache_tokens": len(entry.token_ids) if entry else 0,
            **{f"prefix_cache_{name}": value for name, value in self._prefix_cache_counters.items()},
        }

    def clear_prefix_cache(self, *, count_clear: bool = True) -> dict[str, int]:
        entry = self._prefix_cache_entry
        self._prefix_cache_entry = None
        if count_clear:
            self._prefix_cache_counters["clear_events"] += 1
        return {
            "cleared_entries": 1 if entry else 0,
            "released_bytes": entry.bytes if entry else 0,
        }

    def _prefix_identity(
        self,
        token_ids: tuple[int, ...],
        profile_version: str,
        body: GenerationRequest,
    ) -> str:
        payload = {
            "configuration_fingerprint": self._configuration_fingerprint,
            "load_epoch": self._load_epoch,
            "profile_version": profile_version,
            "token_ids": token_ids,
            "tools": body.tools,
            "tool_choice": body.tool_choice,
            "adapter": None,
        }
        return _sha256_document(payload)

    def validate_token_budget(self, prompt: str | PreparedPrompt, body: GenerationRequest) -> None:
        """Reject requests that cannot fit before allocating inference tensors."""

        encoded = (
            prompt.input_ids
            if isinstance(prompt, PreparedPrompt)
            else self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        )
        prompt_tokens = int(encoded.shape[-1])
        if prompt_tokens + body.max_tokens > self.config.context_length:
            raise ValueError(
                "Token budget exceeds worker context: "
                f"prompt_tokens={prompt_tokens}, requested_output_tokens={body.max_tokens}, "
                f"context_length={self.config.context_length}."
            )

    def trace(
        self,
        *,
        prompt: str | PreparedPrompt,
        body: GenerationRequest,
        cancellation: threading.Event,
    ) -> Iterator[dict[str, Any]]:
        torch = self.torch
        prepared = prompt if isinstance(prompt, PreparedPrompt) else self.prepare_prompt(body)
        prompt_ids = prepared.prompt_ids
        prompt_tokens = _decode_tokens(self.tokenizer, prompt_ids)
        user_prompt = _latest_user_prompt(body)
        user_prompt_ids = _tokenise_without_special_tokens(self.tokenizer, user_prompt)
        user_prompt_tokens = _decode_tokens(self.tokenizer, user_prompt_ids)
        self.validate_token_budget(prepared, body)
        sequence = prepared.input_ids.to(self.device)
        attention_mask = prepared.attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        generated: list[int] = []
        text_so_far = ""
        stop_sequences = [body.stop] if isinstance(body.stop, str) else list(body.stop or ())
        generator = torch.Generator(device=self.device).manual_seed(body.seed)
        eos_token_ids = _configured_eos_token_ids(self.tokenizer, self.model)
        last_event: dict[str, Any] | None = None
        started = time.perf_counter()
        if cancellation.is_set():
            yield {
                "step": 0,
                "cancelled": True,
                "complete": True,
                "text_so_far": text_so_far,
                "prefix_cache": _prefix_cache_observation(
                    status="bypass",
                    reason="cancelled_before_prefill",
                    total_tokens=len(prompt_ids),
                ),
            }
            return
        try:
            output, past_key_values, cache_observation = self._initial_forward(
                prepared,
                body,
                sequence,
                attention_mask,
                cancellation,
            )
        except _GenerationCancelled:
            yield {
                "step": 0,
                "cancelled": True,
                "complete": True,
                "text_so_far": text_so_far,
                "prefix_cache": _prefix_cache_observation(
                    status="bypass",
                    reason="cancelled_during_prefill",
                    total_tokens=len(prompt_ids),
                ),
            }
            return
        cached_input_ids: Any | None = None
        for step in range(min(body.max_tokens, self.config.maximum_new_tokens)):
            if cancellation.is_set():
                yield {
                    "step": step,
                    "cancelled": True,
                    "complete": True,
                    "text_so_far": text_so_far,
                    "prefix_cache": cache_observation,
                }
                return
            if step > 0:
                output = self._model_forward(
                    input_ids=cached_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=body.include_hidden_state_summary,
                )
                past_key_values = _required_past_key_values(output)
            logits = output.logits[0, -1].float()
            if body.repetition_penalty != 1 and generated:
                for token_id in set(generated):
                    logits[token_id] = (
                        logits[token_id] / body.repetition_penalty
                        if logits[token_id] > 0
                        else logits[token_id] * body.repetition_penalty
                    )
            if len(generated) < body.min_tokens:
                _suppress_token_logits(logits, eos_token_ids)
            sampling_logits = logits if body.temperature == 0 else logits / max(body.temperature, 1e-6)
            probabilities = torch.softmax(sampling_logits, dim=-1)
            probabilities = self._apply_top_p(probabilities, body.top_p)
            if body.temperature == 0:
                selected_id = int(torch.argmax(probabilities).item())
            else:
                selected_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
            if selected_id in eos_token_ids:
                if last_event is not None:
                    last_event["complete"] = True
                return
            effective_top_k = min(body.top_k, probabilities.shape[-1])
            top_probabilities, top_indices = torch.topk(probabilities, effective_top_k)
            token = self.tokenizer.decode([selected_id], clean_up_tokenization_spaces=False)
            generated.append(selected_id)
            text_so_far += token
            cached_input_ids = torch.tensor([[selected_id]], device=self.device, dtype=sequence.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype),
                    ),
                    dim=1,
                )
            complete = len(generated) >= body.min_tokens and any(
                text_so_far.endswith(stop) for stop in stop_sequences
            )
            hidden_summary = None
            if body.include_hidden_state_summary and output.hidden_states:
                hidden = output.hidden_states[-1][0, -1].float()
                hidden_summary = {
                    "shape": list(hidden.shape),
                    "mean": round(float(hidden.mean().item()), 6),
                    "l2_norm": round(float(torch.linalg.vector_norm(hidden).item()), 6),
                }
            event = {
                "step": step,
                "selected": {
                    "token_id": selected_id,
                    "token": token,
                    "probability": round(float(probabilities[selected_id].item()), 8),
                },
                "alternatives": [
                    {
                        "token_id": int(token_id),
                        "token": self.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False),
                        "probability": round(float(probability), 8),
                    }
                    for probability, token_id in zip(
                        top_probabilities.tolist(), top_indices.tolist(), strict=True
                    )
                ],
                "prompt_token_ids": prompt_ids if step == 0 else None,
                "prompt_tokens": prompt_tokens if step == 0 else None,
                "user_prompt_token_ids": user_prompt_ids if step == 0 else None,
                "user_prompt_tokens": user_prompt_tokens if step == 0 else None,
                "generated_token_ids": list(generated),
                "text_so_far": text_so_far,
                "timestamp": time.time(),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "hidden_state_summary": hidden_summary,
                "prefix_cache": cache_observation,
                "cancelled": False,
                "complete": complete,
            }
            last_event = event
            yield event
            if complete:
                return

    def _initial_forward(
        self,
        prepared: PreparedPrompt,
        body: GenerationRequest,
        sequence: Any,
        attention_mask: Any | None,
        cancellation: threading.Event,
    ) -> tuple[Any, Any, dict[str, Any]]:
        total_tokens = len(prepared.prompt_ids)
        if prepared.prefix_ids is None or prepared.prefix_identity is None:
            self._prefix_cache_counters["bypasses"] += 1
            started = time.perf_counter()
            output = self._model_forward(
                input_ids=sequence,
                attention_mask=attention_mask,
                output_hidden_states=body.include_hidden_state_summary,
            )
            return (
                output,
                _required_past_key_values(output),
                _prefix_cache_observation(
                    status="bypass",
                    reason=prepared.prefix_bypass_reason,
                    total_tokens=total_tokens,
                    suffix_prefill_seconds=time.perf_counter() - started,
                ),
            )

        prefix_tokens = len(prepared.prefix_ids)
        status = "hit"
        prefix_prefill_seconds = 0.0
        entry = self._prefix_cache_entry
        if (
            entry is None
            or entry.identity != prepared.prefix_identity
            or entry.token_ids != prepared.prefix_ids
        ):
            status = "miss"
            self._prefix_cache_counters["misses"] += 1
            if entry is not None:
                self._evict_prefix_cache()
            if cancellation.is_set():
                raise _GenerationCancelled
            prefix_sequence = sequence[:, :prefix_tokens]
            prefix_attention = attention_mask[:, :prefix_tokens] if attention_mask is not None else None
            prefix_started = time.perf_counter()
            try:
                prefix_output = self._model_forward(
                    input_ids=prefix_sequence,
                    attention_mask=prefix_attention,
                    output_hidden_states=False,
                )
                prefix_cache = _required_past_key_values(prefix_output)
                prefix_bytes = _cache_tensor_bytes(prefix_cache)
                if prefix_bytes > PREFIX_CACHE_MAX_BYTES:
                    self._prefix_cache_counters["bypasses"] += 1
                    return self._full_prefill_fallback(
                        sequence,
                        attention_mask,
                        body,
                        total_tokens=total_tokens,
                        prefix_tokens=prefix_tokens,
                        reason="prefix_byte_limit",
                    )
                _validate_qwen_dynamic_cache(prefix_cache, expected_tokens=prefix_tokens)
                self._prefix_cache_entry = PrefixCacheEntry(
                    identity=prepared.prefix_identity,
                    token_ids=prepared.prefix_ids,
                    cache=prefix_cache,
                    bytes=prefix_bytes,
                )
                entry = self._prefix_cache_entry
                prefix_prefill_seconds = time.perf_counter() - prefix_started
            except _AcceleratorOutOfMemory:
                raise
            except Exception as error:
                LOGGER.warning("Prefix cache prefill failed error=%s", type(error).__name__)
                self.clear_prefix_cache()
                self._prefix_cache_counters["bypasses"] += 1
                return self._full_prefill_fallback(
                    sequence,
                    attention_mask,
                    body,
                    total_tokens=total_tokens,
                    prefix_tokens=prefix_tokens,
                    reason="cache_error",
                )
        else:
            self._prefix_cache_counters["hits"] += 1

        if cancellation.is_set():
            raise _GenerationCancelled
        try:
            branch = _clone_qwen_dynamic_cache(entry.cache, expected_tokens=prefix_tokens)
        except Exception as error:
            LOGGER.warning("Prefix cache branch failed error=%s", type(error).__name__)
            self.clear_prefix_cache()
            self._prefix_cache_counters["bypasses"] += 1
            return self._full_prefill_fallback(
                sequence,
                attention_mask,
                body,
                total_tokens=total_tokens,
                prefix_tokens=prefix_tokens,
                reason="branch_error",
            )
        if cancellation.is_set():
            del branch
            raise _GenerationCancelled
        suffix_started = time.perf_counter()
        suffix = sequence[:, prefix_tokens:]
        try:
            output = self._model_forward(
                input_ids=suffix,
                attention_mask=attention_mask,
                past_key_values=branch,
                output_hidden_states=body.include_hidden_state_summary,
            )
        except _AcceleratorOutOfMemory:
            del branch
            raise
        if cancellation.is_set():
            del output
            raise _GenerationCancelled
        return (
            output,
            _required_past_key_values(output),
            _prefix_cache_observation(
                status=status,
                reason="exact_prefix",
                prefix_tokens=prefix_tokens,
                total_tokens=total_tokens,
                prefix_prefill_seconds=prefix_prefill_seconds,
                suffix_prefill_seconds=time.perf_counter() - suffix_started,
                cache_bytes=entry.bytes,
            ),
        )

    def _full_prefill_fallback(
        self,
        sequence: Any,
        attention_mask: Any | None,
        body: GenerationRequest,
        *,
        total_tokens: int,
        prefix_tokens: int,
        reason: str,
    ) -> tuple[Any, Any, dict[str, Any]]:
        started = time.perf_counter()
        output = self._model_forward(
            input_ids=sequence,
            attention_mask=attention_mask,
            output_hidden_states=body.include_hidden_state_summary,
        )
        return (
            output,
            _required_past_key_values(output),
            _prefix_cache_observation(
                status="bypass",
                reason=reason,
                prefix_tokens=prefix_tokens,
                total_tokens=total_tokens,
                suffix_prefill_seconds=time.perf_counter() - started,
            ),
        )

    def _model_forward(
        self,
        *,
        input_ids: Any,
        attention_mask: Any | None,
        output_hidden_states: bool,
        past_key_values: Any | None = None,
    ) -> Any:
        arguments = {
            "input_ids": input_ids,
            "use_cache": True,
            "output_hidden_states": output_hidden_states,
        }
        if attention_mask is not None:
            arguments["attention_mask"] = attention_mask
        if past_key_values is not None:
            arguments["past_key_values"] = past_key_values
        if self._supports_logits_to_keep:
            arguments["logits_to_keep"] = 1
        try:
            with self.torch.inference_mode():
                return self.model(**arguments)
        except getattr(self.torch, "OutOfMemoryError", MemoryError) as error:
            self.clear_prefix_cache()
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            raise _AcceleratorOutOfMemory from error

    def _evict_prefix_cache(self) -> None:
        if self._prefix_cache_entry is not None:
            self._prefix_cache_entry = None
            self._prefix_cache_counters["evictions"] += 1

    def _apply_top_p(self, probabilities: Any, top_p: float) -> Any:
        if top_p >= 1:
            return probabilities
        torch = self.torch
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities > top_p
        sorted_probabilities[remove] = 0
        filtered = torch.zeros_like(probabilities).scatter(0, sorted_indices, sorted_probabilities)
        return filtered / filtered.sum()


class _GenerationCancelled(Exception):
    pass


class _AcceleratorOutOfMemory(Exception):
    pass


def _sha256_document(document: Any) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _configuration_fingerprint(
    *,
    config: EngineConfig,
    model: Any,
    tokenizer: Any,
    transformers_version: str,
) -> str:
    model_config = getattr(model, "config", None)
    model_document = model_config.to_dict() if hasattr(model_config, "to_dict") else str(model_config)
    return _sha256_document(
        {
            "model_id": config.model_id,
            "revision": config.revision,
            "dtype": config.dtype,
            "context_length": config.context_length,
            "transformers_version": transformers_version,
            "model_config": model_document,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_init": getattr(tokenizer, "init_kwargs", {}),
            "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
            "chat_template": getattr(tokenizer, "chat_template", None),
            "adapter": None,
        }
    )


def _required_past_key_values(output: Any) -> Any:
    past_key_values = getattr(output, "past_key_values", None)
    if past_key_values is None:
        raise RuntimeError(
            "The loaded causal language model did not return past_key_values with use_cache=True"
        )
    return past_key_values


def _cache_tensor_bytes(cache: Any) -> int:
    total = 0
    for layer in getattr(cache, "layers", ()):  # Qwen uses one DynamicLayer per decoder layer.
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if tensor is not None and hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
                total += int(tensor.numel()) * int(tensor.element_size())
    return total


def _validate_qwen_dynamic_cache(cache: Any, *, expected_tokens: int) -> None:
    from transformers.cache_utils import DynamicCache, DynamicLayer

    if not isinstance(cache, DynamicCache) or not cache.layers:
        raise TypeError("Qwen prefix caching requires a populated Transformers DynamicCache")
    if any(not isinstance(layer, DynamicLayer) for layer in cache.layers):
        raise TypeError("Qwen prefix caching supports only full-attention DynamicLayer entries")
    if int(cache.get_seq_length()) != expected_tokens:
        raise ValueError("Qwen prefix cache sequence length does not match the rendered prefix")
    if _cache_tensor_bytes(cache) <= 0:
        raise ValueError("Qwen prefix cache contains no key/value tensors")


def _clone_qwen_dynamic_cache(cache: Any, *, expected_tokens: int) -> Any:
    from transformers.cache_utils import DynamicCache, DynamicLayer

    _validate_qwen_dynamic_cache(cache, expected_tokens=expected_tokens)
    branch = DynamicCache()
    for source in cache.layers:
        target = DynamicLayer()
        target.lazy_initialization(source.keys, source.values)
        target.keys = source.keys.clone()
        target.values = source.values.clone()
        if (
            target.keys.data_ptr() == source.keys.data_ptr()
            or target.values.data_ptr() == source.values.data_ptr()
        ):
            raise RuntimeError("Qwen prefix cache branch shares tensor storage with the immutable base")
        branch.layers.append(target)
    _validate_qwen_dynamic_cache(branch, expected_tokens=expected_tokens)
    if int(cache.get_seq_length()) != expected_tokens:
        raise RuntimeError("Qwen prefix cache base was mutated while creating a request branch")
    return branch


def _prefix_cache_observation(
    *,
    status: str,
    reason: str,
    total_tokens: int,
    prefix_tokens: int = 0,
    prefix_prefill_seconds: float = 0.0,
    suffix_prefill_seconds: float = 0.0,
    cache_bytes: int = 0,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "prefix_tokens": prefix_tokens,
        "total_input_tokens": total_tokens,
        "prefix_prefill_seconds": round(prefix_prefill_seconds, 6),
        "suffix_prefill_seconds": round(suffix_prefill_seconds, 6),
        "cache_bytes": cache_bytes,
    }


class _RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Bound JSON request bytes before Pydantic receives a potentially huge payload."""

    def __init__(self, app: Any, maximum_bytes: int = MAX_REQUEST_BYTES) -> None:
        self.app = app
        self.maximum_bytes = maximum_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", ()))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.maximum_bytes:
                    await _error_response(
                        413,
                        "request_too_large",
                        "The JSON request exceeds 4 MiB.",
                    )(scope, receive, send)
                    return
            except ValueError:
                await _error_response(
                    422,
                    "invalid_request",
                    "Content-Length must be an integer.",
                )(scope, receive, send)
                return
        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.maximum_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await _error_response(
                413,
                "request_too_large",
                "The JSON request exceeds 4 MiB.",
            )(scope, receive, send)


def create_app(
    *,
    worker_id: str,
    config: EngineConfig,
    engine: AutoregressiveEngine | None = None,
) -> FastAPI:
    runtime = engine or TransformersAutoregressiveEngine(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.worker_state = WorkerState.LOADING
        app.state.ready = False
        app.state.load_error = None
        app.state.requests = 0
        app.state.cancelled_requests = 0
        app.state.cancellations = {}
        app.state.generation_lock = asyncio.Lock()
        app.state.load_task = asyncio.create_task(_load_engine(app, runtime))
        yield
        if not app.state.load_task.done():
            app.state.load_task.cancel()

    app = FastAPI(title=f"ModelDeck Transformers worker: {worker_id}", lifespan=lifespan)
    app.add_middleware(RequestBodyLimitMiddleware)
    app.state.shutdown_callback = None

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        # Pydantic's error records include the rejected `input`, which may be prompt text.
        # Keep only structural diagnostics in local logs and return a stable safe error body.
        diagnostics = [{"type": item.get("type"), "loc": item.get("loc")} for item in error.errors()]
        LOGGER.info("Request validation failed path=%s diagnostics=%s", request.url.path, diagnostics)
        return _error_response(
            422,
            "invalid_request",
            "The request does not match the local OpenAI chat contract.",
        )

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        details = runtime.runtime_details
        return {
            "protocol_version": "1",
            "worker_id": worker_id,
            "runtime": "transformers-rocm",
            "generation_family": GenerationFamily.AUTOREGRESSIVE,
            "state": request.app.state.worker_state,
            "model_id": config.model_id,
            "model_revision": config.revision,
            "device": details.get("device", "cuda:0"),
            "device_name": details.get("device_name", "AMD GPU"),
            "rocm_version": details.get("hip_version"),
            "ready": request.app.state.ready,
            "error": request.app.state.load_error,
            "configuration_fingerprint": details.get("configuration_fingerprint"),
            "prefix_caching": details.get("prefix_caching", "unsupported"),
            "prefix_cache_enabled": details.get("prefix_cache_enabled", False),
        }

    @app.get("/capabilities")
    async def capabilities() -> dict[str, Any]:
        result = CapabilitySet(
            chat=True,
            completions=True,
            streaming=True,
            cancellation=True,
            logits=True,
            top_k_trace=True,
            hidden_states="optional",
            seeded_generation=True,
            prefix_caching=(
                "application-managed"
                if supports_application_managed_prefix_cache(config.model_id)
                else "unsupported"
            ),
            prefix_cache_enabled=config.prefix_cache_enabled,
        )
        return {
            "protocol_version": "1",
            "generation_family": GenerationFamily.AUTOREGRESSIVE,
            **result.model_dump(),
        }

    @app.get("/metrics")
    async def metrics(request: Request) -> dict[str, Any]:
        memory_metrics = getattr(runtime, "memory_metrics", lambda: {})()
        prefix_metrics = getattr(runtime, "prefix_cache_metrics", lambda: {})()
        return {
            **runtime.runtime_details,
            **memory_metrics,
            **prefix_metrics,
            "requests": request.app.state.requests,
            "cancelled_requests": request.app.state.cancelled_requests,
            "busy": request.app.state.generation_lock.locked(),
        }

    @app.get("/model")
    async def model() -> dict[str, Any]:
        return {
            "model_id": config.model_id,
            "revision": config.revision,
            "generation_family": GenerationFamily.AUTOREGRESSIVE,
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": config.dtype,
        }

    @app.post("/load")
    async def load(request: Request) -> dict[str, Any]:
        return {"ok": request.app.state.load_error is None, "state": request.app.state.worker_state}

    @app.post("/warmup")
    async def warmup(request: Request) -> dict[str, Any]:
        await request.app.state.load_task
        if request.app.state.load_error:
            raise HTTPException(503, request.app.state.load_error)
        request.app.state.worker_state = WorkerState.WARMING
        try:
            await run_in_isolated_thread(runtime.warmup)
        except Exception as error:
            request.app.state.worker_state = WorkerState.FAILED
            request.app.state.load_error = f"Warmup failed: {type(error).__name__}: {error}"
            raise HTTPException(500, request.app.state.load_error) from error
        request.app.state.ready = True
        request.app.state.worker_state = WorkerState.READY
        return {"ok": True, "ready": True}

    @app.post("/cancel")
    async def cancel(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id", ""))
        cancellation = request.app.state.cancellations.get(request_id)
        if cancellation:
            cancellation.set()
            request.app.state.cancelled_requests += 1
        return {"ok": bool(cancellation), "request_id": request_id}

    @app.post("/shutdown")
    async def shutdown(request: Request) -> dict[str, bool]:
        request.app.state.worker_state = WorkerState.STOPPING
        for cancellation in request.app.state.cancellations.values():
            cancellation.set()
        clear_prefix_cache = getattr(runtime, "clear_prefix_cache", None)
        if callable(clear_prefix_cache):
            clear_prefix_cache()
        if request.app.state.shutdown_callback:
            asyncio.get_running_loop().call_later(0.05, request.app.state.shutdown_callback)
        return {"ok": True}

    @app.post("/prefix-cache/clear")
    async def clear_prefix_cache(request: Request) -> dict[str, Any]:
        if request.app.state.generation_lock.locked():
            raise HTTPException(409, "Wait for active generation before clearing the prefix cache")
        clear = getattr(runtime, "clear_prefix_cache", None)
        result = clear() if callable(clear) else {"cleared_entries": 0, "released_bytes": 0}
        return {"ok": True, **result}

    @app.post("/v1/chat/completions")
    async def chat(request: Request, body: GenerationRequest):
        return await _generate_response(request, body, runtime, chat=True)

    @app.post("/v1/completions")
    async def completions(request: Request, body: GenerationRequest):
        return await _generate_response(request, body, runtime, chat=False)

    @app.post("/native/autoregressive/trace")
    async def trace(request: Request, body: GenerationRequest):
        return await _trace_response(request, body, runtime)

    return app


async def _load_engine(app: FastAPI, engine: AutoregressiveEngine) -> None:
    try:
        await run_in_isolated_thread(engine.load)
        app.state.worker_state = WorkerState.WARMING
    except Exception as error:
        app.state.load_error = f"Load failed: {type(error).__name__}: {error}"
        app.state.worker_state = WorkerState.FAILED


async def _trace_response(request: Request, body: GenerationRequest, engine: AutoregressiveEngine):
    _ensure_ready(request)
    request_id = body.request_id or request.headers.get("x-request-id") or str(uuid.uuid4())
    body.request_id = request_id
    prepare_prompt = getattr(engine, "prepare_prompt", None)
    prompt = prepare_prompt(body) if callable(prepare_prompt) else engine.build_prompt(body)
    validate_token_budget = getattr(engine, "validate_token_budget", None)
    if callable(validate_token_budget):
        try:
            validate_token_budget(prompt, body)
        except ValueError as error:
            LOGGER.info("Request rejected request_id=%s reason=context_token_budget", request_id)
            return _error_response(422, "context_length_exceeded", str(error))
    cancellation = threading.Event()
    request.app.state.cancellations[request_id] = cancellation
    if body.stream:
        return StreamingResponse(
            _stream_trace(request, body, engine, prompt, cancellation),
            media_type="text/event-stream",
        )
    async with request.app.state.generation_lock:
        request.app.state.worker_state = WorkerState.BUSY
        started = time.perf_counter()
        try:
            try:
                events = await run_in_isolated_thread(
                    lambda: list(engine.trace(prompt=prompt, body=body, cancellation=cancellation))
                )
            except _AcceleratorOutOfMemory:
                LOGGER.error("Inference failed request_id=%s reason=accelerator_out_of_memory", request_id)
                return _error_response(
                    503,
                    "inference_memory_exhausted",
                    "Local inference stopped because accelerator memory was exhausted.",
                )
            token_metadata = _trace_token_metadata(events)
            request.app.state.requests += 1
            return {
                "request_id": request_id,
                "model": body.model,
                **token_metadata,
                "events": events,
                "metrics": _request_metrics(events, started),
            }
        finally:
            request.app.state.cancellations.pop(request_id, None)
            request.app.state.worker_state = WorkerState.READY


async def _generate_response(
    request: Request,
    body: GenerationRequest,
    engine: AutoregressiveEngine,
    *,
    chat: bool,
):
    try:
        _validate_tool_request(body)
    except ToolCallProtocolError as error:
        return _error_response(422, error.code, error.message)
    if body.stream and (body.tool_choice == "required" or isinstance(body.tool_choice, dict)):
        return _error_response(
            409,
            "tool_calling_streaming_unsupported",
            "Required tool calling is not available with this Worker's streaming response protocol.",
        )
    try:
        if body.stream:
            trace_response = await _trace_response(request, body, engine)
            return trace_response
        result = await _trace_response(request, body, engine)
    except ToolCallProtocolError as error:
        return _error_response(422, error.code, error.message)
    if isinstance(result, JSONResponse):
        return result
    events = result["events"]
    text = events[-1].get("text_so_far", "") if events else ""
    try:
        tool_calls, content = _openai_tool_calls(text)
        _normalise_qwen_tool_arguments(tool_calls, body.tools)
        _enforce_tool_choice(body, tool_calls)
    except ToolCallProtocolError as error:
        return _error_response(422, error.code, error.message)
    choice = (
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            },
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }
        if chat
        else {"index": 0, "text": text, "finish_reason": "stop"}
    )
    return {
        "id": result["request_id"],
        "object": "chat.completion" if chat else "text_completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [choice],
        "metrics": result["metrics"],
    }


async def _stream_trace(
    request: Request,
    body: GenerationRequest,
    engine: AutoregressiveEngine,
    prompt: str,
    cancellation: threading.Event,
) -> AsyncIterator[str]:
    request_id = body.request_id or "unknown"
    async with request.app.state.generation_lock:
        request.app.state.worker_state = WorkerState.BUSY
        try:
            try:
                async for event in iterate_in_isolated_thread(
                    lambda: engine.trace(prompt=prompt, body=body, cancellation=cancellation)
                ):
                    name = "cancelled" if event.get("cancelled") else "token"
                    payload = {"request_id": request_id, **event}
                    yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"
            except _AcceleratorOutOfMemory:
                LOGGER.error(
                    "Inference failed request_id=%s reason=accelerator_out_of_memory",
                    request_id,
                )
                payload = {
                    "request_id": request_id,
                    "error": {
                        "code": "inference_memory_exhausted",
                        "message": "Local inference stopped because accelerator memory was exhausted.",
                    },
                }
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                return
            yield "event: complete\ndata: [DONE]\n\n"
            request.app.state.requests += 1
        finally:
            cancellation.set()
            request.app.state.cancellations.pop(request_id, None)
            request.app.state.worker_state = WorkerState.READY


def _model_context_length(model: Any) -> int:
    config = getattr(model, "config", None)
    candidates = (
        getattr(config, "max_position_embeddings", None),
        # Qwen3.5 conditional-generation models keep the text context declaration
        # within their nested text configuration rather than at the top level.
        getattr(getattr(config, "text_config", None), "max_position_embeddings", None),
    )
    for value in candidates:
        if not isinstance(value, bool) and isinstance(value, int) and value >= 256:
            return value
    raise RuntimeError("Loaded model does not declare a safe maximum context length")


def _latest_user_prompt(body: GenerationRequest) -> str:
    if not body.messages:
        return body.prompt or ""
    return next(
        (message.content or "" for message in reversed(body.messages) if message.role == "user"),
        "",
    )


_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_OPEN_TOOL_CALL = re.compile(r"<tool_call>\s*(.*)", re.DOTALL)
_QWEN_FUNCTION_CALL = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^\s>]+)>\s*(?P<body>.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_QWEN_FUNCTION_PARAMETER = re.compile(
    r"<parameter=(?P<name>[^\s>]+)>\s*(?P<value>.*?)\s*</parameter>", re.DOTALL
)


class ToolCallProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _qwen_chat_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert OpenAI tool-call history to the shape consumed by Qwen templates."""

    rendered_messages: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        rendered = message.model_dump(exclude_none=True)
        tool_calls = rendered.get("tool_calls")
        if tool_calls is None:
            rendered_messages.append(rendered)
            continue
        native_calls: list[dict[str, Any]] = []
        for call_index, call in enumerate(tool_calls):
            if not isinstance(call, dict) or call.get("type") != "function":
                raise ToolCallProtocolError(
                    "invalid_tool_history",
                    f"Assistant message {message_index} tool call {call_index} must be a function call.",
                )
            function = call.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                raise ToolCallProtocolError(
                    "invalid_tool_history",
                    f"Assistant message {message_index} tool call {call_index} requires a function name.",
                )
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise ToolCallProtocolError(
                        "malformed_tool_arguments",
                        (
                            f"Assistant message {message_index} tool call {call_index} "
                            "contains malformed JSON arguments."
                        ),
                    ) from error
            if not isinstance(arguments, dict):
                raise ToolCallProtocolError(
                    "malformed_tool_arguments",
                    (
                        f"Assistant message {message_index} tool call {call_index} "
                        "arguments must decode to a JSON object."
                    ),
                )
            native_calls.append(
                {
                    **call,
                    "function": {
                        **function,
                        "arguments": arguments,
                    },
                }
            )
        rendered_messages.append({**rendered, "tool_calls": native_calls})
    return rendered_messages


def _qwen_tool_schemas(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert OpenAI function definitions to Qwen's native template input."""

    if tools is None:
        return None
    schemas: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ToolCallProtocolError(
                "invalid_tool_definition", f"Tool {index} must be an OpenAI function definition."
            )
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ToolCallProtocolError(
                "invalid_tool_definition", f"Tool {index} must define a function name."
            )
        schemas.append(dict(function))
    return schemas


def _openai_tool_calls(text: str) -> tuple[list[dict[str, Any]], str | None]:
    """Translate Qwen's documented tool-call envelope to OpenAI's response shape."""

    qwen_calls = list(_QWEN_FUNCTION_CALL.finditer(text))
    if qwen_calls:
        calls = []
        for index, match in enumerate(qwen_calls):
            body = match.group("body")
            parameters = {
                parameter.group("name"): parameter.group("value").strip()
                for parameter in _QWEN_FUNCTION_PARAMETER.finditer(body)
            }
            if _QWEN_FUNCTION_PARAMETER.sub("", body).strip():
                raise ToolCallProtocolError(
                    "malformed_tool_call", "The local model returned malformed Qwen tool-call parameters."
                )
            calls.append(_openai_tool_call({"name": match.group("name"), "arguments": parameters}, index))
        # Qwen3.5 may emit its reasoning channel before a tool call. It is neither
        # user-facing completion text nor part of the function protocol.
        content = _QWEN_FUNCTION_CALL.sub("", text)
        if "</think>" in content:
            content = content.split("</think>", 1)[1]
        return calls, content.strip() or None

    calls: list[dict[str, Any]] = []
    matches = list(_TOOL_CALL.finditer(text))
    if "<tool_call" in text and not matches:
        # Qwen templates may terminate generation immediately after the JSON object
        # rather than emitting a closing XML-like marker. It remains a complete native
        # tool call when the remaining text is exactly one valid object.
        open_match = _OPEN_TOOL_CALL.search(text)
        if open_match is None:
            raise ToolCallProtocolError(
                "malformed_tool_call", "The local model returned an incomplete tool-call envelope."
            )
        try:
            call, end = json.JSONDecoder().raw_decode(open_match.group(1).lstrip())
            if open_match.group(1).lstrip()[end:].strip():
                raise ValueError("unexpected text after tool-call JSON")
            matches = [None]
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ToolCallProtocolError(
                "malformed_tool_call", "The local model returned malformed tool-call JSON."
            ) from error
        calls.append(_openai_tool_call(call, 0))
        return calls, None
    for index, match in enumerate(matches):
        try:
            call = json.loads(match.group(1))
        except json.JSONDecodeError as error:
            raise ToolCallProtocolError(
                "malformed_tool_call", "The local model returned malformed tool-call JSON."
            ) from error
        calls.append(_openai_tool_call(call, index))
    content = _TOOL_CALL.sub("", text).strip()
    return calls, content or None


def _normalise_qwen_tool_arguments(calls: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> None:
    """Unwrap Qwen's occasional XML ``json`` parameter for argument-free tools."""

    schemas = {
        function["name"]: function.get("parameters", {})
        for tool in tools or []
        if isinstance(tool, dict)
        and isinstance((function := tool.get("function")), dict)
        and isinstance(function.get("name"), str)
    }
    for call in calls:
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        parameters = schemas.get(function.get("name"))
        if not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict) or "json" in properties:
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            continue
        try:
            decoded = json.loads(arguments)
            inner = json.loads(decoded["json"]) if set(decoded) == {"json"} else None
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(inner, dict):
            function["arguments"] = json.dumps(inner)


def _openai_tool_call(call: Any, index: int) -> dict[str, Any]:
    try:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ValueError("tool call must contain a function name")
        arguments = call.get("arguments", {})
        arguments_json = arguments if isinstance(arguments, str) else json.dumps(arguments)
        json.loads(arguments_json)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ToolCallProtocolError(
            "malformed_tool_call", "The local model returned malformed tool-call JSON."
        ) from error
    return {
        "id": f"call_{index}_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {"name": call["name"], "arguments": arguments_json},
    }


def _enforce_tool_choice(body: GenerationRequest, calls: list[dict[str, Any]]) -> None:
    """Reject protocol violations before a text result can escape as a tool result."""

    _validate_tool_request(body)
    if not body.tools:
        return
    if body.tool_choice == "required" and not calls:
        raise ToolCallProtocolError(
            "tool_choice_not_honoured",
            "The local model returned text instead of the required tool call.",
        )
    if not isinstance(body.tool_choice, dict):
        return
    function = body.tool_choice.get("function")
    required_name = function.get("name") if isinstance(function, dict) else None
    if not calls:
        raise ToolCallProtocolError(
            "tool_choice_not_honoured",
            "The local model returned text instead of the required named tool call.",
        )
    if any(call["function"]["name"] != required_name for call in calls):
        raise ToolCallProtocolError(
            "tool_choice_not_honoured",
            "The local model called a function other than the named required tool.",
        )


def _validate_tool_request(body: GenerationRequest) -> None:
    """Validate the bounded OpenAI tool input before template rendering."""

    if body.messages:
        _qwen_chat_messages(body.messages)
    if not body.tools:
        if body.tool_choice is not None:
            raise ToolCallProtocolError("invalid_tool_choice", "tool_choice requires at least one tool.")
        return
    _qwen_tool_schemas(body.tools)
    if body.tool_choice is None or body.tool_choice in ("auto", "required", "none"):
        return
    if not isinstance(body.tool_choice, dict):
        raise ToolCallProtocolError(
            "invalid_tool_choice",
            "tool_choice must be 'auto', 'required', 'none', or a named function choice.",
        )
    function = body.tool_choice.get("function")
    required_name = function.get("name") if isinstance(function, dict) else None
    if body.tool_choice.get("type") != "function" or not isinstance(required_name, str):
        raise ToolCallProtocolError(
            "invalid_tool_choice",
            "tool_choice must be 'auto', 'required', 'none', or a named function choice.",
        )


def _tokenise_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [
        str(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        for token_id in token_ids
    ]


def _configured_eos_token_ids(tokenizer: Any, model: Any) -> set[int]:
    values = [getattr(tokenizer, "eos_token_id", None)]
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        values.append(getattr(generation_config, "eos_token_id", None))
    token_ids: set[int] = set()
    for value in values:
        candidates = value if isinstance(value, (list, tuple, set)) else [value]
        token_ids.update(
            int(candidate)
            for candidate in candidates
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0
        )
    return token_ids


def _suppress_token_logits(logits: Any, token_ids: set[int]) -> None:
    for token_id in token_ids:
        if token_id < len(logits):
            logits[token_id] = float("-inf")


def _trace_token_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    first = events[0] if events else {}
    metadata = {
        "prompt_token_ids": first.get("prompt_token_ids", []),
        "prompt_tokens": first.get("prompt_tokens", []),
        "user_prompt_token_ids": first.get("user_prompt_token_ids", []),
        "user_prompt_tokens": first.get("user_prompt_tokens", []),
    }
    error = _token_metadata_error(metadata)
    if error:
        raise HTTPException(500, f"Worker produced invalid trace token metadata: {error}")
    return metadata


def _token_metadata_error(metadata: dict[str, Any]) -> str | None:
    prompt_ids = metadata.get("prompt_token_ids")
    prompt_tokens = metadata.get("prompt_tokens")
    user_ids = metadata.get("user_prompt_token_ids")
    user_tokens = metadata.get("user_prompt_tokens")
    if not isinstance(prompt_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in prompt_ids
    ):
        return "prompt_token_ids must be an array of integers"
    if not isinstance(prompt_tokens, list) or not all(isinstance(token, str) for token in prompt_tokens):
        return "prompt_tokens must be an array of strings"
    if len(prompt_tokens) != len(prompt_ids):
        return "prompt_tokens must contain one entry for every prompt_token_ids entry"
    if not isinstance(user_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in user_ids
    ):
        return "user_prompt_token_ids must be an array of integers"
    if not isinstance(user_tokens, list) or not all(isinstance(token, str) for token in user_tokens):
        return "user_prompt_tokens must be an array of strings"
    if len(user_tokens) != len(user_ids):
        return "user_prompt_tokens must contain one entry for every user_prompt_token_ids entry"
    return None


def _request_metrics(events: list[dict[str, Any]], started: float) -> dict[str, Any]:
    total = time.perf_counter() - started
    generated = len([event for event in events if event.get("selected")])
    first = events[0].get("elapsed_seconds") if events else None
    cache_metrics = events[0].get("prefix_cache", {}) if events else {}
    return {
        "first_token_seconds": first,
        "total_seconds": round(total, 6),
        "decode_seconds": round(max(total - float(first or 0), 0), 6) if first is not None else None,
        "generated_tokens": generated,
        "tokens_per_second": round(generated / total, 4) if total else None,
        "output_tokens_per_second": round(generated / total, 4) if total else None,
        "cancelled": any(event.get("cancelled") for event in events),
        "prefix_cache": cache_metrics,
    }


def _ensure_ready(request: Request) -> None:
    if not request.app.state.ready:
        raise HTTPException(503, "Worker is not ready")


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "param": None,
                "code": code,
            }
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ModelDeck autoregressive ROCm worker")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--maximum-new-tokens", type=int, default=128)
    parser.add_argument("--prefix-cache-enabled", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EngineConfig(
        model_id=args.model_id,
        revision=args.revision,
        dtype=args.dtype,
        context_length=args.context_length,
        maximum_new_tokens=args.maximum_new_tokens,
        prefix_cache_enabled=args.prefix_cache_enabled,
    )
    app = create_app(worker_id=args.worker_id, config=config)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)
    )
    app.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

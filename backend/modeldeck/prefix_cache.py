from __future__ import annotations

import hashlib
import json
from typing import Any

APPLICATION_MANAGED_PREFIX_CACHE_MODEL_IDS = frozenset(
    {
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    }
)
PREFIX_CACHE_MAX_TOKENS = 8_192
PREFIX_CACHE_MAX_BYTES = 512 * 1024 * 1024


def supports_application_managed_prefix_cache(model_id: str) -> bool:
    return model_id in APPLICATION_MANAGED_PREFIX_CACHE_MODEL_IDS


def stable_model_configuration_fingerprint(
    *,
    model_id: str,
    revision: str,
    runtime: str,
    dtype: str,
    context_length: int | None,
    runtime_template_version: str | None,
) -> str:
    payload: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "runtime": runtime,
        "dtype": dtype,
        "context_length": context_length,
        "runtime_template_version": runtime_template_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

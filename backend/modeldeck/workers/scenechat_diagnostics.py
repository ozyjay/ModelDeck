"""Content-free generation receipts shared by the dedicated SceneChat engines."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any


def repetition_diagnostics(text: str) -> dict[str, int]:
    words = text.split()
    grams = Counter(tuple(words[index : index + 4]) for index in range(max(0, len(words) - 3)))
    return {
        "words": len(words),
        "characters": len(text),
        "repeated_four_grams": sum(count - 1 for count in grams.values()),
        "maximum_four_gram_occurrences": max(grams.values(), default=0),
    }


def generation_receipt(
    engine: Any,
    rendered: str,
    generated: Any,
    *,
    limit: int,
    cancelled: bool,
    complete_json: bool = False,
) -> dict[str, Any]:
    config = getattr(engine.model, "generation_config", None)
    eos = getattr(config, "eos_token_id", None)
    eos_ids = eos if isinstance(eos, list) else [eos] if isinstance(eos, int) else []
    count = int(generated.shape[-1]) if hasattr(generated, "shape") else len(generated)
    final_id = int(generated[-1]) if count and eos_ids else None
    # Record overlapping conditions rather than attributing EOS to every short output.
    conditions = {
        "cancelled": cancelled,
        "complete_json": complete_json,
        "eos": final_id is not None and final_id in eos_ids,
        "token_limit": count >= limit,
    }
    return {
        "stopping_cause": next((name for name, hit in conditions.items() if hit), "unknown"),
        "stopping_conditions": conditions,
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "effective_generation": {
            "do_sample": False,
            "use_cache": True,
            "max_new_tokens": limit,
            "eos_token_id": eos,
            "pad_token_id": getattr(config, "pad_token_id", None),
            "cache_implementation": getattr(config, "cache_implementation", None),
            "resolved_cache_class": None,
            "resolved_cache_dtype": None,
            "cross_request_cache_reuse": False,
            "attention_implementation": getattr(
                getattr(engine.model, "config", None), "_attn_implementation", None
            ),
            "resolved_attention_kernel": None,
        },
        "stage_timings": {
            "vision_encoding_seconds": None,
            "text_prefill_seconds": None,
            "first_token_seconds": None,
            "generation_seconds": None,
            "unavailable_reason": "Vision, prefill and generation are fused in the generate measurement.",
        },
    }

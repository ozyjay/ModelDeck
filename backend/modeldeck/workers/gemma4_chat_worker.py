"""Allowlisted general text and image chat worker for local Gemma 4 snapshots."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from modeldeck.gemma4_settings import ALLOWED_VISUAL_TOKEN_BUDGETS, DEFAULT_VISUAL_TOKEN_BUDGET
from modeldeck.workers.scenechat_worker import EngineConfig, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelDeck Gemma 4 general chat worker")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--maximum-new-tokens", type=int, default=512)
    parser.add_argument("--generation-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--thinking-mode", choices=("disabled", "adaptive"), default="disabled")
    parser.add_argument(
        "--visual-token-budget",
        type=int,
        choices=ALLOWED_VISUAL_TOKEN_BUDGETS,
        default=DEFAULT_VISUAL_TOKEN_BUDGET,
    )
    arguments = parser.parse_args()
    config = EngineConfig(
        model_id=arguments.model_id,
        revision=arguments.revision,
        cache_root=arguments.cache_root,
        dtype=arguments.dtype,
        context_length=arguments.context_length,
        maximum_new_tokens=arguments.maximum_new_tokens,
        generation_timeout_seconds=arguments.generation_timeout_seconds,
        visual_token_budget=arguments.visual_token_budget,
        general_chat=True,
        thinking_mode=arguments.thinking_mode,
    )
    app = create_app(
        worker_id=arguments.worker_id,
        config=config,
        api_key=os.environ.get("MODELDECK_SCENECHAT_API_KEY", "local"),
        worker_label="Gemma 4 general chat",
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=arguments.port, access_log=False, log_level="info")
    )
    app.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

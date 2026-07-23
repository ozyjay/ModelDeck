from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from PIL import Image

from modeldeck.contracts.scenechat import system_messages
from modeldeck.gemma4_settings import ALLOWED_VISUAL_TOKEN_BUDGETS, DEFAULT_VISUAL_TOKEN_BUDGET
from modeldeck.workers.scenechat_worker import (
    EngineConfig,
    GenerationResult,
    TransformersSceneChatEngine,
    create_app,
)

QWEN35_MODEL_IDS = frozenset(
    {
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
    }
)
QWEN35_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
QWEN35_MODEL_TYPE = "qwen3_5"
QWEN35_PROCESSOR_CLASS = "Qwen3VLProcessor"
QWEN35_PATCH_SIZE = 16
QWEN35_SPATIAL_MERGE_SIZE = 2
QWEN35_MINIMUM_PIXELS = 65_536
QWEN35_DEFAULT_MAXIMUM_NEW_TOKENS = 1024


def _is_complete_json_output(value: str) -> bool:
    candidate = value.strip()
    fenced = candidate.startswith("```json")
    if fenced:
        candidate = candidate[7:].lstrip()
    try:
        _, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return False
    remainder = candidate[end:].strip()
    return remainder == "```" if fenced else not remainder


class TransformersQwen35Engine(TransformersSceneChatEngine):
    def load(self) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        snapshot = self._validate_snapshot()
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm PyTorch did not expose an available 'cuda' device")
        if self.config.dtype != "bfloat16":
            raise RuntimeError("The SceneChat Qwen3.5 profile requires bfloat16")
        dtype = torch.bfloat16
        device = torch.device("cuda:0")
        try:
            torch.empty(1, device=device, dtype=dtype)
        except Exception as error:
            raise RuntimeError("The detected GPU could not allocate a BF16 tensor") from error

        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        if type(processor).__name__ != QWEN35_PROCESSOR_CLASS:
            raise RuntimeError(f"Expected {QWEN35_PROCESSOR_CLASS}, received {type(processor).__name__}")
        _configure_qwen35_image_processor(
            getattr(processor, "image_processor", None),
            self.config.visual_token_budget,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        if type(model).__name__ != QWEN35_ARCHITECTURE:
            raise RuntimeError(f"Expected {QWEN35_ARCHITECTURE}, received {type(model).__name__}")
        model.to(device)
        model.eval()
        placement_details = self._validate_placement(model, device, dtype)
        self.torch = torch
        self.processor = processor
        self.model = model
        self.device = device
        self.dtype = dtype
        self.runtime_details = {
            "torch_version": str(torch.__version__),
            "hip_version": torch.version.hip,
            "transformers_version": importlib.metadata.version("transformers"),
            "processor_class": type(processor).__name__,
            "model_class": type(model).__name__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "dtype": self.config.dtype,
            "attention_implementation": "sdpa",
            "visual_token_budget": self.config.visual_token_budget,
            "image_patch_size": QWEN35_PATCH_SIZE,
            "image_spatial_merge_size": QWEN35_SPATIAL_MERGE_SIZE,
            **placement_details,
            "load_seconds": round(time.perf_counter() - started, 4),
            "snapshot_path": str(snapshot),
        }

    def _validate_snapshot(self) -> Path:
        if self.config.model_id not in QWEN35_MODEL_IDS:
            raise RuntimeError("The requested model is not an allowlisted Qwen3.5 checkpoint")
        snapshot = self.snapshot_path
        if not snapshot.is_dir():
            raise RuntimeError(
                f"Pinned local snapshot is missing for {self.config.model_id} at revision "
                f"{self.config.revision}"
            )
        required = {
            "chat_template.jinja",
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
        missing = sorted(name for name in required if not (snapshot / name).is_file())
        if not list(snapshot.glob("*.safetensors")):
            missing.append("*.safetensors")
        if missing:
            raise RuntimeError(f"Pinned snapshot is incomplete; missing: {', '.join(missing)}")
        try:
            model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("The pinned Qwen3.5 configuration is unreadable") from error
        if model_config.get("model_type") != QWEN35_MODEL_TYPE or model_config.get("architectures") != [
            QWEN35_ARCHITECTURE
        ]:
            raise RuntimeError("The pinned snapshot does not declare the allowlisted Qwen3.5 architecture")
        if model_config.get("quantization_config"):
            raise RuntimeError("Quantised Qwen3.5 snapshots require a dedicated tested runtime")
        return snapshot.resolve(strict=True)

    def generate(
        self,
        *,
        image: Image.Image,
        question: str,
        max_tokens: int,
        cancellation: threading.Event,
    ) -> GenerationResult:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class CancellationCriteria(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return cancellation.is_set()

        class CompleteJsonCriteria(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                generated = input_ids[0, prompt_tokens:]
                candidate = self_processor.decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                return _is_complete_json_output(candidate)

        rendered = self.processor.apply_chat_template(
            system_messages(question),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        preprocessing_started = time.perf_counter()
        inputs = self.processor(text=rendered, images=[image], return_tensors="pt")
        preprocessing_seconds = time.perf_counter() - preprocessing_started
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        visual_tokens = _qwen35_visual_token_count(inputs)
        if visual_tokens is not None and visual_tokens > self.config.visual_token_budget:
            raise ValueError("Processed image exceeds the configured visual token budget")
        if prompt_tokens + max_tokens > self.config.context_length:
            raise ValueError(
                f"Processed input plus requested output exceeds {self.config.context_length} tokens"
            )
        inputs = inputs.to(self.device, dtype=self.dtype)
        self_processor = self.processor
        self.torch.cuda.reset_peak_memory_stats(0)
        try:
            inference_started = time.perf_counter()
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, self.config.maximum_new_tokens),
                    do_sample=False,
                    use_cache=True,
                    stopping_criteria=StoppingCriteriaList([CancellationCriteria(), CompleteJsonCriteria()]),
                )
            inference_seconds = time.perf_counter() - inference_started
            generated = output[0, prompt_tokens:]
            completion_tokens = int(generated.shape[-1])
            decoded = self.processor.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if any(marker in decoded.casefold() for marker in ("<think>", "</think>", "<|channel>")):
                raise ValueError("Model output exposed a reasoning channel")
            return GenerationResult(
                text=decoded,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cancelled=cancellation.is_set(),
                preprocessing_seconds=preprocessing_seconds,
                inference_seconds=inference_seconds,
                visual_tokens=visual_tokens,
            )
        finally:
            del inputs


def _configure_qwen35_image_processor(image_processor: Any, visual_token_budget: int) -> None:
    if image_processor is None:
        raise RuntimeError("The allowlisted Qwen3.5 processor has no image processor")
    if getattr(image_processor, "patch_size", None) != QWEN35_PATCH_SIZE:
        raise RuntimeError("The Qwen3.5 image processor must use patch size 16")
    if getattr(image_processor, "merge_size", None) != QWEN35_SPATIAL_MERGE_SIZE:
        raise RuntimeError("The Qwen3.5 image processor must use spatial merge size 2")
    if visual_token_budget not in ALLOWED_VISUAL_TOKEN_BUDGETS:
        raise RuntimeError("The Qwen3.5 visual token budget is not allowlisted")
    maximum_pixels = visual_token_budget * QWEN35_PATCH_SIZE**2 * QWEN35_SPATIAL_MERGE_SIZE**2
    image_processor.size["shortest_edge"] = min(QWEN35_MINIMUM_PIXELS, maximum_pixels)
    image_processor.size["longest_edge"] = maximum_pixels


def _qwen35_visual_token_count(inputs: Any) -> int | None:
    grid = inputs.get("image_grid_thw") if hasattr(inputs, "get") else None
    if grid is None:
        return None
    try:
        rows = grid.tolist() if hasattr(grid, "tolist") else grid
        return sum(
            int(temporal) * int(height) * int(width) // QWEN35_SPATIAL_MERGE_SIZE**2
            for temporal, height, width in rows
        )
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelDeck SceneChat Qwen3.5 worker")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True, choices=sorted(QWEN35_MODEL_IDS))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument(
        "--maximum-new-tokens",
        type=int,
        default=QWEN35_DEFAULT_MAXIMUM_NEW_TOKENS,
    )
    parser.add_argument("--generation-timeout-seconds", type=float, default=60.0)
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
    )
    application = create_app(
        worker_id=arguments.worker_id,
        config=config,
        api_key=os.environ.get("MODELDECK_SCENECHAT_API_KEY", "local"),
        engine=TransformersQwen35Engine(config),
        worker_label="SceneChat Qwen3.5",
        model_owner="Qwen",
        vision_settings={
            "image_patch_size": QWEN35_PATCH_SIZE,
            "image_pooling_kernel_size": QWEN35_SPATIAL_MERGE_SIZE,
        },
    )
    server = uvicorn.Server(
        uvicorn.Config(
            application,
            host="127.0.0.1",
            port=arguments.port,
            access_log=False,
            log_level="info",
        )
    )
    application.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

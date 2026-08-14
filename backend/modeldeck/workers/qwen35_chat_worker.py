from __future__ import annotations

import argparse
import importlib.metadata
import time
import uuid

import uvicorn

from modeldeck.workers.autoregressive_worker import (
    EngineConfig,
    TransformersAutoregressiveEngine,
    _configuration_fingerprint,
    _model_context_length,
    create_app,
)
from modeldeck.workers.qwen35_worker import QWEN35_ARCHITECTURE, QWEN35_MODEL_IDS, QWEN35_PROCESSOR_CLASS


class TransformersQwen35ChatEngine(TransformersAutoregressiveEngine):
    """Text-only chat adapter for the reviewed Qwen3.5 multimodal checkpoints."""

    def load(self) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if self.config.model_id not in QWEN35_MODEL_IDS:
            raise RuntimeError("The requested model is not an allowlisted Qwen3.5 checkpoint")
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm PyTorch did not expose an available 'cuda' device")
        if self.config.dtype != "bfloat16":
            raise RuntimeError("The Qwen3.5 text chat profile requires bfloat16")
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        if type(processor).__name__ != QWEN35_PROCESSOR_CLASS:
            raise RuntimeError(f"Expected {QWEN35_PROCESSOR_CLASS}, received {type(processor).__name__}")
        model = AutoModelForMultimodalLM.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        if type(model).__name__ != QWEN35_ARCHITECTURE:
            raise RuntimeError(f"Expected {QWEN35_ARCHITECTURE}, received {type(model).__name__}")
        supported_context_length = _model_context_length(model)
        if self.config.context_length > supported_context_length:
            raise RuntimeError(
                "Configured context length "
                f"{self.config.context_length} exceeds the model-supported limit {supported_context_length}"
            )
        device = torch.device("cuda:0")
        try:
            torch.empty(1, device=device, dtype=torch.bfloat16)
        except Exception as error:
            raise RuntimeError("The detected GPU could not allocate a BF16 tensor") from error
        model.to(device)
        model.eval()
        self.torch = torch
        # The processor treats its first positional argument as image input. The shared
        # autoregressive protocol passes text positionally, so use the processor's
        # underlying tokenizer for the text-only chat Worker.
        self.tokenizer = processor.tokenizer
        self.model = model
        self.device = device
        self._supports_logits_to_keep = False
        self._load_epoch = uuid.uuid4().hex
        self._configuration_fingerprint = _configuration_fingerprint(
            config=self.config,
            model=model,
            tokenizer=self.tokenizer,
            transformers_version=importlib.metadata.version("transformers"),
        )
        self.clear_prefix_cache(count_clear=False)
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
            "model_max_context_tokens": supported_context_length,
            "configuration_fingerprint": self._configuration_fingerprint,
            "load_epoch": self._load_epoch,
            "prefix_caching": "unsupported",
            "prefix_cache_enabled": False,
            "load_seconds": round(time.perf_counter() - started, 4),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelDeck Qwen3.5 text chat worker")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True, choices=sorted(QWEN35_MODEL_IDS))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--maximum-new-tokens", type=int, default=512)
    arguments = parser.parse_args()
    config = EngineConfig(
        model_id=arguments.model_id,
        revision=arguments.revision,
        dtype=arguments.dtype,
        context_length=arguments.context_length,
        maximum_new_tokens=arguments.maximum_new_tokens,
    )
    application = create_app(
        worker_id=arguments.worker_id,
        config=config,
        engine=TransformersQwen35ChatEngine(config),
    )
    server = uvicorn.Server(
        uvicorn.Config(application, host="127.0.0.1", port=arguments.port, access_log=False, log_level="info")
    )
    application.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

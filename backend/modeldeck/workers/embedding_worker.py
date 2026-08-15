from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modeldeck.async_execution import run_in_isolated_thread
from modeldeck.protocol import CapabilitySet, GenerationFamily, WorkerState

EMBEDDING_DIMENSIONS = 1024


@dataclass(frozen=True)
class EmbeddingEngineConfig:
    model_id: str
    revision: str
    dtype: str = "float16"
    maximum_input_tokens: int = 8192


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="local-worker", min_length=1, max_length=128)
    input: str | list[str]

    @model_validator(mode="after")
    def valid_input(self) -> EmbeddingRequest:
        inputs = self.inputs
        if not inputs:
            raise ValueError("input must contain at least one text string")
        if len(inputs) > 128:
            raise ValueError("input supports at most 128 text strings")
        if any(not value.strip() for value in inputs):
            raise ValueError("input text strings cannot be empty")
        if any(len(value) > 32_000 for value in inputs):
            raise ValueError("input text strings cannot exceed 32000 characters")
        return self

    @property
    def inputs(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else self.input


class EmbeddingEngine(Protocol):
    runtime_details: dict[str, Any]

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def embed(self, inputs: list[str]) -> list[list[float]]: ...

    def memory_metrics(self) -> dict[str, int]: ...


class TransformersEmbeddingEngine:
    def __init__(self, config: EmbeddingEngineConfig) -> None:
        self.config = config
        self.runtime_details: dict[str, Any] = {}
        self.torch: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("ROCm PyTorch did not expose an available 'cuda' device")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
            self.config.dtype
        )
        if dtype is None:
            raise RuntimeError(f"Unsupported dtype: {self.config.dtype}")
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
        )
        self.device = torch.device("cuda:0")
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch
        self.runtime_details = {
            "torch_version": str(torch.__version__),
            "hip_version": torch.version.hip,
            "transformers_version": importlib.metadata.version("transformers"),
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(0),
            "load_seconds": round(time.perf_counter() - started, 4),
            "dtype": self.config.dtype,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
        }

    def warmup(self) -> None:
        self.embed(["ModelDeck embedding warmup"])

    def embed(self, inputs: list[str]) -> list[list[float]]:
        if self.torch is None or self.tokenizer is None or self.model is None or self.device is None:
            raise RuntimeError("Embedding engine is not loaded")
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config.maximum_input_tokens,
            return_tensors="pt",
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        attention_mask = encoded["attention_mask"]
        with self.torch.inference_mode():
            hidden_states = self.model(**encoded).last_hidden_state
        if bool(self.torch.all(attention_mask[:, -1])):
            pooled = hidden_states[:, -1]
        else:
            positions = attention_mask.sum(dim=1) - 1
            pooled = hidden_states[self.torch.arange(hidden_states.shape[0], device=self.device), positions]
        vectors = self.torch.nn.functional.normalize(pooled.float(), p=2, dim=1).cpu().tolist()
        if any(len(vector) != EMBEDDING_DIMENSIONS for vector in vectors):
            dimensions = sorted({len(vector) for vector in vectors})
            raise RuntimeError(
                f"Loaded model returned embedding dimensions {dimensions}; expected {EMBEDDING_DIMENSIONS}"
            )
        return [[float(value) for value in vector] for vector in vectors]

    def memory_metrics(self) -> dict[str, int]:
        if self.torch is None or not self.torch.cuda.is_available():
            return {}
        return {
            "memory_allocated_bytes": int(self.torch.cuda.memory_allocated(0)),
            "memory_reserved_bytes": int(self.torch.cuda.memory_reserved(0)),
            "peak_memory_allocated_bytes": int(self.torch.cuda.max_memory_allocated(0)),
            "peak_memory_reserved_bytes": int(self.torch.cuda.max_memory_reserved(0)),
        }


def create_app(
    *,
    worker_id: str,
    config: EmbeddingEngineConfig,
    engine: EmbeddingEngine | None = None,
) -> FastAPI:
    runtime = engine or TransformersEmbeddingEngine(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.worker_state = WorkerState.LOADING
        app.state.ready = False
        app.state.load_error = None
        app.state.requests = 0
        app.state.embedding_lock = asyncio.Lock()
        app.state.load_task = asyncio.create_task(_load_engine(app, runtime))
        yield
        if not app.state.load_task.done():
            app.state.load_task.cancel()

    app = FastAPI(title=f"ModelDeck embeddings worker: {worker_id}", lifespan=lifespan)
    app.state.shutdown_callback = None

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        details = runtime.runtime_details
        return {
            "protocol_version": "1",
            "worker_id": worker_id,
            "runtime": "transformers-rocm",
            "generation_family": GenerationFamily.EMBEDDING,
            "state": request.app.state.worker_state,
            "model_id": config.model_id,
            "model_revision": config.revision,
            "device": details.get("device", "cuda:0"),
            "device_name": details.get("device_name", "AMD GPU"),
            "rocm_version": details.get("hip_version"),
            "ready": request.app.state.ready,
            "error": request.app.state.load_error,
        }

    @app.get("/capabilities")
    async def capabilities() -> dict[str, Any]:
        return {
            "protocol_version": "1",
            "generation_family": GenerationFamily.EMBEDDING,
            **CapabilitySet(embeddings=True, streaming=False, cancellation=True).model_dump(),
        }

    @app.get("/metrics")
    async def metrics(request: Request) -> dict[str, Any]:
        return {
            **runtime.runtime_details,
            **runtime.memory_metrics(),
            "requests": request.app.state.requests,
            "busy": request.app.state.embedding_lock.locked(),
        }

    @app.get("/model")
    async def model() -> dict[str, Any]:
        return {
            "model_id": config.model_id,
            "revision": config.revision,
            "generation_family": GenerationFamily.EMBEDDING,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
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

    @app.post("/shutdown")
    async def shutdown(request: Request) -> dict[str, bool]:
        request.app.state.worker_state = WorkerState.STOPPING
        if request.app.state.shutdown_callback:
            asyncio.get_running_loop().call_later(0.05, request.app.state.shutdown_callback)
        return {"ok": True}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request, body: EmbeddingRequest) -> dict[str, Any]:
        if not request.app.state.ready:
            raise HTTPException(503, "Worker is not ready")
        async with request.app.state.embedding_lock:
            request.app.state.worker_state = WorkerState.BUSY
            try:
                vectors = await run_in_isolated_thread(runtime.embed, body.inputs)
                if len(vectors) != len(body.inputs):
                    raise RuntimeError(
                        "Embedding engine returned a vector count different from the input count"
                    )
                request.app.state.requests += 1
                return {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "embedding": vector, "index": index}
                        for index, vector in enumerate(vectors)
                    ],
                    "model": body.model,
                }
            finally:
                request.app.state.worker_state = WorkerState.READY

    return app


async def _load_engine(app: FastAPI, engine: EmbeddingEngine) -> None:
    try:
        await run_in_isolated_thread(engine.load)
        app.state.worker_state = WorkerState.WARMING
    except Exception as error:
        app.state.load_error = f"Load failed: {type(error).__name__}: {error}"
        app.state.worker_state = WorkerState.FAILED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ModelDeck embeddings ROCm worker")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--maximum-input-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EmbeddingEngineConfig(
        model_id=args.model_id,
        revision=args.revision,
        dtype=args.dtype,
        maximum_input_tokens=args.maximum_input_tokens,
    )
    app = create_app(worker_id=args.worker_id, config=config)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)
    )
    app.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from modeldeck.protocol import CapabilitySet, GenerationFamily, WorkerHealth, WorkerState

LOGGER = logging.getLogger("uvicorn.error")
SYSTEM_INSTRUCTION = (
    "Give a concise, accurate final answer. Do not expose private reasoning or thinking. "
    "Finish the answer within the available response length."
)


@dataclass(frozen=True)
class EngineConfig:
    model_id: str
    revision: str
    cache_root: str | None = None
    q4_checkpoint_dir: str | None = None
    dtype: str = "bfloat16"
    maximum_new_tokens: int = 256
    maximum_denoising_steps: int = 48


class DiffusionRequest(BaseModel):
    model: str = "text-diffusion"
    prompt: str = Field(min_length=1, max_length=16_000)
    max_length: int = Field(default=256, ge=8, le=256)
    denoising_steps: int = Field(default=48, ge=1, le=48)
    block_length: int = Field(default=256, ge=1, le=256)
    temperature: float = Field(default=0.8, gt=0, le=2)
    seed: int = 11
    stream_intermediate_frames: bool = True


class DiffusionEngine(Protocol):
    runtime_details: dict[str, Any]

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def memory_metrics(self) -> dict[str, int]: ...

    def refine(
        self,
        body: DiffusionRequest,
        cancellation: threading.Event,
        frame_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]: ...


class FrameStreamer:
    """Collect native denoising drafts without printing prompts or generated text."""

    _takes_logits = False

    def __init__(
        self,
        tokenizer: Any,
        total_steps: int,
        cancellation: threading.Event,
        frame_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.total_steps = total_steps
        self.cancellation = cancellation
        self.frame_callback = frame_callback
        self.frames: list[dict[str, Any]] = []
        self._previous: list[int] = []
        self._prompt_seen = False

    def put_draft(self, value: Any, **_: Any) -> None:
        if self.cancellation.is_set():
            return
        token_ids = value[0].tolist() if len(value.shape) > 1 else value.tolist()
        stable = sum(a == b for a, b in zip(self._previous, token_ids, strict=False))
        self._previous = token_ids
        frame = {
            "step": len(self.frames) + 1,
            "total_steps": self.total_steps,
            "text": _decode_response(self.tokenizer, token_ids),
            "masked_tokens": None,
            "stable_tokens": stable,
            "complete": False,
        }
        self.frames.append(frame)
        if self.frame_callback and frame["step"] < self.total_steps:
            self.frame_callback(frame)

    def put(self, value: Any) -> None:
        # The first confirmed value is the prompt. Confirmed canvases are represented
        # by the final frame assembled from generate()'s returned sequence.
        self._prompt_seen = True

    def end(self) -> None:
        return


class CancellationCriteria:
    def __init__(self, cancellation: threading.Event) -> None:
        self.cancellation = cancellation

    def __call__(self, input_ids: Any, scores: Any, **_: Any) -> Any:
        return input_ids.new_full(
            (input_ids.shape[0],), self.cancellation.is_set(), dtype=self._bool_dtype(input_ids)
        )

    @staticmethod
    def _bool_dtype(input_ids: Any) -> Any:
        import torch

        return torch.bool


class TransformersDiffusionEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.runtime_details: dict[str, Any] = {}
        self.torch: Any = None
        self.processor: Any = None
        self.model: Any = None
        self.device: Any = None
        self.q4_layers: list[Any] = []

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

        if not torch.cuda.is_available():
            raise RuntimeError("ROCm PyTorch did not expose an available 'cuda' device")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(self.config.dtype)
        if dtype is None:
            raise RuntimeError(f"Unsupported dtype: {self.config.dtype}")
        started = time.perf_counter()
        device = torch.device("cuda:0")
        q4_details: dict[str, Any] = {}
        if self.config.q4_checkpoint_dir:
            from pathlib import Path

            from modeldeck.workers.diffusiongemma_q4 import load_diffusiongemma_q4

            if not self.config.cache_root:
                raise RuntimeError("The Q4 runtime requires --cache-root")
            loaded = load_diffusiongemma_q4(
                model_id=self.config.model_id,
                revision=self.config.revision,
                cache_root=Path(self.config.cache_root),
                checkpoint_dir=Path(self.config.q4_checkpoint_dir),
                device=device,
                dtype=dtype,
            )
            processor = loaded.processor
            model = loaded.model
            self.q4_layers = loaded.q4_layers
            q4_details = loaded.details
        else:
            processor = AutoProcessor.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                local_files_only=True,
                trust_remote_code=False,
            )
            model = DiffusionGemmaForBlockDiffusion.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                local_files_only=True,
                trust_remote_code=False,
                dtype=dtype,
            )
            model.to(device)
            model.eval()
        self.torch, self.processor, self.model, self.device = torch, processor, model, device
        self.runtime_details = {
            "torch_version": str(torch.__version__),
            "hip_version": torch.version.hip,
            "transformers_version": importlib.metadata.version("transformers"),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "load_seconds": round(time.perf_counter() - started, 4),
            "dtype": self.config.dtype,
            "runtime": "text-diffusion-transformers-rocm",
            **q4_details,
        }

    def warmup(self) -> None:
        self.refine(
            DiffusionRequest(prompt="Reply with ready.", max_length=8, denoising_steps=1, seed=7),
            threading.Event(),
        )

    def memory_metrics(self) -> dict[str, int]:
        if self.torch is None or not self.torch.cuda.is_available():
            return {}
        metrics = {
            "memory_allocated_bytes": int(self.torch.cuda.memory_allocated(0)),
            "memory_reserved_bytes": int(self.torch.cuda.memory_reserved(0)),
            "peak_memory_allocated_bytes": int(self.torch.cuda.max_memory_allocated(0)),
            "peak_memory_reserved_bytes": int(self.torch.cuda.max_memory_reserved(0)),
        }
        if self.q4_layers:
            from modeldeck.workers.diffusiongemma_q4 import q4_invocation_metrics

            metrics.update(q4_invocation_metrics(self.q4_layers))
        return metrics

    def refine(
        self,
        body: DiffusionRequest,
        cancellation: threading.Event,
        frame_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        from transformers import StoppingCriteriaList

        torch = self.torch
        torch.manual_seed(body.seed)
        inputs = self.processor.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": body.prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        prompt_length = int(inputs["input_ids"].shape[-1])
        streamer = FrameStreamer(
            self.processor.tokenizer,
            body.denoising_steps,
            cancellation,
            frame_callback,
        )
        output = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=min(body.max_length, self.config.maximum_new_tokens),
            max_denoising_steps=min(body.denoising_steps, self.config.maximum_denoising_steps),
            t_max=body.temperature,
            t_min=min(0.4, body.temperature),
            disable_compile=True,
            stopping_criteria=StoppingCriteriaList([CancellationCriteria(cancellation)]),
        )
        if cancellation.is_set():
            return _finalise_frames(
                streamer.frames,
                body.denoising_steps,
                cancelled=True,
                finish_reason="cancelled",
            )
        sequences = output.sequences if hasattr(output, "sequences") else output
        generated_tokens = sequences[0, prompt_length:]
        text = _decode_response(self.processor.tokenizer, generated_tokens)
        finish_reason = _finish_reason(
            generated_tokens,
            self.model.generation_config.eos_token_id,
        )
        return _finalise_frames(
            streamer.frames,
            body.denoising_steps,
            cancelled=False,
            text=text,
            finish_reason=finish_reason,
        )


def _decode_response(tokenizer: Any, token_ids: Any) -> str:
    raw_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    parser = getattr(tokenizer, "parse_response", None)
    if callable(parser):
        try:
            parsed = parser(raw_text)
            if isinstance(parsed, dict) and "content" in parsed:
                content = str(parsed.get("content") or "")
                content_ids = tokenizer.encode(content, add_special_tokens=False)
                return tokenizer.decode(content_ids, skip_special_tokens=True).strip()
        except (TypeError, ValueError):
            pass

    if re.match(r"^\s*<\|channel>thought(?:\n|$)", raw_text):
        if "<channel|>" not in raw_text:
            return ""
        raw_text = raw_text.split("<channel|>", 1)[1]
        token_ids = tokenizer.encode(raw_text, add_special_tokens=False)
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return re.sub(r"^\s*thought\s*(?:\n|$)", "", text, count=1).strip()


def _finish_reason(token_ids: Any, eos_token_ids: int | list[int] | tuple[int, ...] | None) -> str:
    values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    if not values:
        return "stop"
    eos = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or ())
    return "stop" if any(int(value) in eos for value in values) else "length"


def _finalise_frames(
    frames: list[dict[str, Any]],
    total_steps: int,
    *,
    cancelled: bool,
    finish_reason: str,
    text: str | None = None,
) -> list[dict[str, Any]]:
    terminal = _terminal_frame(
        frames,
        total_steps,
        cancelled,
        finish_reason=finish_reason,
        text=text,
    )
    if frames and len(frames) >= total_steps:
        return [*frames[: total_steps - 1], terminal]
    return [*frames, terminal]


def _terminal_frame(
    frames: list[dict[str, Any]],
    total_steps: int,
    cancelled: bool,
    *,
    finish_reason: str,
    text: str | None = None,
) -> dict[str, Any]:
    return {
        "step": min(len(frames) + 1, total_steps),
        "total_steps": total_steps,
        "text": text if text is not None else (frames[-1]["text"] if frames else ""),
        "masked_tokens": 0 if not cancelled else None,
        "stable_tokens": frames[-1]["stable_tokens"] if frames else 0,
        "complete": True,
        "cancelled": cancelled,
        "finish_reason": finish_reason,
    }


def create_app(*, worker_id: str, config: EngineConfig, engine: DiffusionEngine | None = None) -> FastAPI:
    runtime = engine or TransformersDiffusionEngine(config)
    run_in_thread = True

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.worker_state = WorkerState.LOADING
        app.state.ready = False
        app.state.load_error = None
        app.state.jobs = {}
        app.state.job_tasks = {}
        app.state.job_events = {}
        app.state.cancellations = {}
        app.state.generation_lock = asyncio.Lock()
        app.state.run_in_thread = run_in_thread
        app.state.load_task = asyncio.create_task(_load_engine(app, runtime, threaded=run_in_thread))
        yield
        if not app.state.load_task.done():
            app.state.load_task.cancel()

    app = FastAPI(title=f"ModelDeck text-diffusion worker: {worker_id}", lifespan=lifespan)
    app.state.shutdown_callback = None

    @app.get("/health", response_model=None)
    async def health(request: Request) -> dict[str, Any]:
        details = runtime.runtime_details
        payload = WorkerHealth(
            worker_id=worker_id,
            runtime=details.get("runtime", "text-diffusion-transformers-rocm"),
            generation_family=GenerationFamily.TEXT_DIFFUSION,
            state=request.app.state.worker_state,
            model_id=config.model_id,
            model_revision=config.revision,
            device=details.get("device", "cuda:0"),
            device_name=details.get("device_name", "ROCm device"),
            rocm_version=details.get("hip_version"),
            ready=request.app.state.ready,
        ).model_dump(mode="json")
        if request.app.state.load_error:
            payload["error"] = request.app.state.load_error
        return payload

    @app.get("/capabilities")
    async def capabilities() -> dict[str, Any]:
        return {
            "protocol_version": "1",
            "generation_family": GenerationFamily.TEXT_DIFFUSION,
            **CapabilitySet(
                iterative_refinement=True,
                intermediate_frames=True,
                seeded_generation=True,
                logits="model-specific",
            ).model_dump(),
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return {**runtime.runtime_details, **runtime.memory_metrics(), "jobs": len(app.state.jobs)}

    @app.get("/model")
    async def model() -> dict[str, Any]:
        return {
            "model_id": config.model_id,
            "revision": config.revision,
            "generation_family": GenerationFamily.TEXT_DIFFUSION,
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": config.dtype,
            "quantization": runtime.runtime_details.get("quantization", "none"),
            "q4_checkpoint_dir": runtime.runtime_details.get("q4_checkpoint_dir"),
        }

    @app.post("/load")
    async def load(request: Request) -> dict[str, Any]:
        return {"ok": request.app.state.load_error is None, "state": request.app.state.worker_state}

    @app.post("/warmup")
    async def warmup(request: Request) -> dict[str, Any]:
        if request.app.state.load_error:
            raise HTTPException(503, request.app.state.load_error)
        request.app.state.worker_state = WorkerState.WARMING
        try:
            if request.app.state.run_in_thread:
                await asyncio.to_thread(runtime.warmup)
            else:
                runtime.warmup()
        except Exception as error:
            request.app.state.worker_state = WorkerState.FAILED
            raise HTTPException(500, f"Warmup failed: {type(error).__name__}: {error}") from error
        request.app.state.ready = True
        request.app.state.worker_state = WorkerState.READY
        return {"ok": True, "ready": True}

    async def run_refinement(
        body: DiffusionRequest,
        job_id: str,
        cancellation: threading.Event | None = None,
    ) -> dict[str, Any]:
        _ensure_ready(app)
        cancellation = cancellation or threading.Event()
        app.state.cancellations[job_id] = cancellation
        completion_event = app.state.job_events.setdefault(job_id, asyncio.Event())
        app.state.jobs[job_id] = {"job_id": job_id, "state": "running", "frames": []}
        loop = asyncio.get_running_loop()

        def publish_frame(frame: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(_record_job_frame, app, job_id, frame)

        async with app.state.generation_lock:
            app.state.worker_state = WorkerState.BUSY
            started = time.perf_counter()
            try:
                if app.state.run_in_thread:
                    frames = await asyncio.to_thread(runtime.refine, body, cancellation, publish_frame)
                else:
                    frames = runtime.refine(body, cancellation, publish_frame)
                await asyncio.sleep(0)
                result = {
                    "job_id": job_id,
                    "state": "cancelled" if frames[-1].get("cancelled") else "complete",
                    "model": body.model,
                    "text": frames[-1]["text"],
                    "frames": frames,
                    "seed": body.seed,
                    "metrics": {"total_seconds": round(time.perf_counter() - started, 4)},
                }
                app.state.jobs[job_id] = result
                completion_event.set()
                return result
            except Exception as error:
                app.state.jobs[job_id] = {
                    "job_id": job_id,
                    "state": "failed",
                    "frames": [],
                    "error": f"{type(error).__name__}: {error}",
                }
                completion_event.set()
                raise
            finally:
                app.state.cancellations.pop(job_id, None)
                app.state.worker_state = WorkerState.READY

    @app.post("/v1/refine")
    async def refine(body: DiffusionRequest) -> dict[str, Any]:
        return await run_refinement(body, str(uuid.uuid4()))

    @app.post("/v1/diffuse")
    async def diffuse(body: DiffusionRequest) -> dict[str, Any]:
        _ensure_ready(app)
        job_id = str(uuid.uuid4())
        cancellation = threading.Event()
        app.state.cancellations[job_id] = cancellation
        app.state.job_events[job_id] = asyncio.Event()
        app.state.jobs[job_id] = {"job_id": job_id, "state": "queued", "frames": []}
        task = asyncio.create_task(run_refinement(body, job_id, cancellation))
        app.state.job_tasks[job_id] = task
        task.add_done_callback(lambda _: app.state.job_tasks.pop(job_id, None))
        return {"job_id": job_id, "state": "queued", "events_url": f"/v1/jobs/{job_id}/events"}

    @app.get("/v1/jobs/{job_id}")
    async def job(job_id: str) -> dict[str, Any]:
        if job_id not in app.state.jobs:
            raise HTTPException(404, "Unknown diffusion job")
        result = app.state.jobs[job_id]
        return {**result, "frame_count": len(result["frames"])}

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        cancellation = app.state.cancellations.get(job_id)
        if cancellation:
            cancellation.set()
        state = "cancelling" if cancellation else app.state.jobs.get(job_id, {}).get("state", "complete")
        return {"job_id": job_id, "state": state}

    @app.get("/v1/jobs/{job_id}/events")
    async def job_events(job_id: str):
        if job_id not in app.state.jobs:
            raise HTTPException(404, "Unknown diffusion job")

        async def events() -> AsyncIterator[str]:
            sent = 0
            while True:
                result = app.state.jobs[job_id]
                frames = result["frames"]
                for frame in frames[sent:]:
                    yield f"event: frame\ndata: {json.dumps(frame)}\n\n"
                    sent += 1
                if result["state"] in {"complete", "cancelled"}:
                    return
                if result["state"] == "failed":
                    yield f"event: error\ndata: {json.dumps({'error': result['error']})}\n\n"
                    return
                update = app.state.job_events[job_id]
                update.clear()
                current = app.state.jobs[job_id]
                if len(current["frames"]) == sent and current["state"] not in {
                    "complete",
                    "cancelled",
                    "failed",
                }:
                    await update.wait()

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/cancel")
    async def cancel(payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id", ""))
        cancellation = app.state.cancellations.get(request_id)
        if cancellation:
            cancellation.set()
        return {"ok": bool(cancellation), "request_id": request_id}

    @app.post("/shutdown")
    async def shutdown() -> dict[str, bool]:
        app.state.worker_state = WorkerState.STOPPING
        for cancellation in app.state.cancellations.values():
            cancellation.set()
        if app.state.shutdown_callback:
            asyncio.get_running_loop().call_later(0.05, app.state.shutdown_callback)
        return {"ok": True}

    return app


def _record_job_frame(app: FastAPI, job_id: str, frame: dict[str, Any]) -> None:
    job = app.state.jobs.get(job_id)
    if job is None or job["state"] not in {"queued", "running"}:
        return
    job["frames"].append(frame)
    app.state.job_events[job_id].set()


async def _load_engine(app: FastAPI, engine: DiffusionEngine, *, threaded: bool) -> None:
    try:
        if threaded:
            await asyncio.to_thread(engine.load)
        else:
            engine.load()
        app.state.worker_state = WorkerState.WARMING
    except Exception as error:
        app.state.load_error = f"Load failed: {type(error).__name__}: {error}"
        app.state.worker_state = WorkerState.FAILED
        LOGGER.exception("Diffusion engine load failed: %s", error)


def _ensure_ready(app: FastAPI) -> None:
    if not app.state.ready:
        raise HTTPException(503, app.state.load_error or "Worker is not ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--cache-root")
    parser.add_argument("--q4-checkpoint-dir")
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument("--maximum-denoising-steps", type=int, default=48)
    args = parser.parse_args()
    app = create_app(
        worker_id=args.worker_id,
        config=EngineConfig(
            model_id=args.model_id,
            revision=args.revision,
            cache_root=args.cache_root,
            q4_checkpoint_dir=args.q4_checkpoint_dir,
            dtype=args.dtype,
            maximum_new_tokens=args.maximum_new_tokens,
            maximum_denoising_steps=args.maximum_denoising_steps,
        ),
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=args.port,
            log_level="info",
            access_log=False,
        )
    )
    app.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

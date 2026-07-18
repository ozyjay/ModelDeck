from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from modeldeck.protocol import GenerationFamily

REASONING_MARKERS = re.compile(
    r"<\|(?:analysis|reasoning|channel)[^>]*\>.*?<\|(?:end|final)[^>]*\>", re.DOTALL
)
AMD_VENDOR_ID = "0x1002"


def fixed_llama_server() -> Path:
    return Path(".runtime-tools/llama.cpp/bin/llama-server").resolve()


def llama_command(*, model: Path, port: int, context_length: int, preset: str) -> list[str]:
    if preset not in {"vulkan-full", "vulkan-cpu-moe"}:
        raise ValueError("Unknown allowlisted GPT-OSS execution preset")
    executable = fixed_llama_server()
    if not executable.is_file():
        raise ValueError(
            "Pinned llama.cpp Vulkan runtime is missing; run "
            "pwsh -NoProfile -File scripts/setup_llama_vulkan.ps1"
        )
    allowed_names = {
        "gpt-oss-120b-MXFP4.gguf",
        "gpt-oss-120b-mxfp4-00001-of-00003.gguf",
    }
    if not model.is_file() or model.name not in allowed_names:
        raise ValueError("The allowlisted GPT-OSS MXFP4 GGUF artefact is missing")
    command = [
        str(executable),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        str(model),
        "--ctx-size",
        str(context_length),
        "--parallel",
        "1",
        "--n-gpu-layers",
        "999",
        "--flash-attn",
        "--jinja",
    ]
    if preset == "vulkan-cpu-moe":
        command.extend(["--n-cpu-moe", "20"])
    return command


def remove_reasoning(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: remove_reasoning(item)
            for key, item in value.items()
            if key not in {"reasoning", "reasoning_content", "analysis"}
        }
    if isinstance(value, list):
        return [remove_reasoning(item) for item in value]
    if isinstance(value, str):
        return REASONING_MARKERS.sub("", value)
    return value


def amd_gpu_memory_metrics() -> dict[str, int]:
    """Read whole-device AMD memory counters from the fixed Linux DRM sysfs interface."""
    for device in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        try:
            if (device / "vendor").read_text(encoding="utf-8").strip().lower() != AMD_VENDOR_ID:
                continue
            values = {}
            for source, key in (
                ("mem_info_gtt_used", "system_gtt_used_bytes"),
                ("mem_info_gtt_total", "system_gtt_total_bytes"),
                ("mem_info_vram_used", "system_vram_used_bytes"),
                ("mem_info_vram_total", "system_vram_total_bytes"),
            ):
                values[key] = int((device / source).read_text(encoding="utf-8").strip())
            return values
        except (OSError, ValueError):
            continue
    return {}


class LlamaProcess:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.internal_port = args.port + 1000
        # Keep the catalogue-approved snapshot filename for the strict GGUF allowlist.
        # Hugging Face snapshots are symlinks whose resolved blob names are opaque hashes.
        self.artifact_path = Path(args.artifact_path).absolute()
        self.process: asyncio.subprocess.Process | None = None
        self.memory_task: asyncio.Task[None] | None = None
        self.peak_gtt_used_bytes: int | None = None
        self.started = time.monotonic()
        self.load_seconds: float | None = None

    async def start(self) -> None:
        command = llama_command(
            model=self.artifact_path,
            port=self.internal_port,
            context_length=self.args.context_length,
            preset=self.args.execution_preset,
        )
        environment = dict(os.environ)
        environment["GGML_VK_VISIBLE_DEVICES"] = "0"
        self.process = await asyncio.create_subprocess_exec(*command, env=environment)
        self.memory_task = asyncio.create_task(self._sample_gpu_memory())

    async def stop(self) -> None:
        try:
            if self.process is None or self.process.returncode is not None:
                return
            self.process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self.process.wait(), timeout=8)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        finally:
            if self.memory_task is not None:
                self.memory_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.memory_task
                self.memory_task = None

    def memory_metrics(self) -> dict[str, int]:
        metrics = amd_gpu_memory_metrics()
        current = metrics.get("system_gtt_used_bytes")
        if current is not None:
            self.peak_gtt_used_bytes = max(self.peak_gtt_used_bytes or current, current)
        if self.peak_gtt_used_bytes is not None:
            metrics["system_gtt_peak_used_bytes"] = self.peak_gtt_used_bytes
        return metrics

    async def _sample_gpu_memory(self) -> None:
        while True:
            self.memory_metrics()
            await asyncio.sleep(0.1)

    async def ready(self) -> bool:
        if self.process is None or self.process.returncode is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=0.5) as client:
                response = await client.get(f"http://127.0.0.1:{self.internal_port}/health")
            if response.is_success and self.load_seconds is None:
                self.load_seconds = round(time.monotonic() - self.started, 4)
            return response.is_success
        except httpx.HTTPError:
            return False


def create_app(args: argparse.Namespace) -> FastAPI:
    runtime = LlamaProcess(args)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await runtime.start()
        yield
        await runtime.stop()

    app = FastAPI(title="ModelDeck GPT-OSS Vulkan worker", lifespan=lifespan)
    app.state.shutdown_callback = None

    @app.get("/health")
    async def health():
        ready = await runtime.ready()
        return {
            "protocol_version": "1",
            "worker_id": args.worker_id,
            "runtime": "llama-vulkan",
            "generation_family": GenerationFamily.AUTOREGRESSIVE,
            "state": "warming" if ready else "loading",
            "model_id": args.model_id,
            "model_revision": args.revision,
            "device": "vulkan:0",
            "device_name": "AMD Vulkan",
            "rocm_version": None,
            "ready": ready,
        }

    @app.post("/warmup")
    async def warmup():
        if not await runtime.ready():
            return JSONResponse({"ready": False}, status_code=503)
        payload = {"prompt": "Hello", "n_predict": 1, "temperature": 0}
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"http://127.0.0.1:{runtime.internal_port}/completion", json=payload)
        return JSONResponse({"ready": response.is_success}, status_code=200 if response.is_success else 503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": args.model_id, "object": "model"}]}

    @app.get("/model")
    async def model():
        return {
            "model_id": args.model_id,
            "revision": args.revision,
            "generation_family": GenerationFamily.AUTOREGRESSIVE,
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": "mxfp4",
            "quantization": "mxfp4",
        }

    @app.get("/metrics")
    async def metrics():
        return {
            "runtime": "llama-vulkan",
            "execution_preset": args.execution_preset,
            "load_seconds": runtime.load_seconds,
            **runtime.memory_metrics(),
        }

    async def proxy(request: Request, path: str):
        body = await request.json()
        body["model"] = args.model_id
        client = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=1))
        try:
            response = await client.send(
                client.build_request("POST", f"http://127.0.0.1:{runtime.internal_port}{path}", json=body),
                stream=bool(body.get("stream")),
            )
        except httpx.HTTPError:
            await client.aclose()
            return JSONResponse({"error": {"code": "llama_runtime_unavailable"}}, status_code=503)
        if body.get("stream"):

            async def filtered_stream():
                try:
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                line = "data: " + json.dumps(remove_reasoning(json.loads(line[6:])))
                            except json.JSONDecodeError:
                                continue
                        yield (line + "\n").encode()
                finally:
                    await response.aclose()
                    await client.aclose()

            return StreamingResponse(filtered_stream(), media_type="text/event-stream")
        try:
            return JSONResponse(remove_reasoning(response.json()), status_code=response.status_code)
        finally:
            await response.aclose()
            await client.aclose()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await proxy(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await proxy(request, "/v1/completions")

    @app.post("/cancel")
    async def cancel():
        return {"ok": False, "reason": "Cancellation is driven by closing the streaming request"}

    @app.post("/shutdown")
    async def shutdown():
        if app.state.shutdown_callback:
            asyncio.get_running_loop().call_later(0.05, app.state.shutdown_callback)
        return {"ok": True}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument(
        "--execution-preset",
        choices=("vulkan-full", "vulkan-cpu-moe"),
        default="vulkan-full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)
    )
    app.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

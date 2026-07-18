from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import time
from contextlib import asynccontextmanager
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
    if not model.is_file() or not model.name.endswith("-00001-of-00003.gguf"):
        raise ValueError("The allowlisted first GPT-OSS MXFP4 GGUF shard is missing")
    command = [
        str(executable), "--host", "127.0.0.1", "--port", str(port), "--model", str(model),
        "--ctx-size", str(context_length), "--parallel", "1", "--n-gpu-layers", "999",
        "--flash-attn", "on", "--jinja",
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


class LlamaProcess:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.internal_port = args.port + 1000
        self.artifact_path = Path(args.artifact_path).resolve()
        self.process: asyncio.subprocess.Process | None = None
        self.started = time.monotonic()

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

    async def stop(self) -> None:
        if self.process is None or self.process.returncode is not None:
            return
        self.process.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(self.process.wait(), timeout=8)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()

    async def ready(self) -> bool:
        if self.process is None or self.process.returncode is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=0.5) as client:
                response = await client.get(f"http://127.0.0.1:{self.internal_port}/health")
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

    @app.get("/health")
    async def health():
        ready = await runtime.ready()
        return {
            "protocol_version": "1", "worker_id": args.worker_id, "runtime": "llama-vulkan",
            "generation_family": GenerationFamily.AUTOREGRESSIVE, "state": "warming" if ready else "loading",
            "model_id": args.model_id, "model_revision": args.revision, "device": "vulkan:0",
            "device_name": "AMD Vulkan", "rocm_version": None, "ready": ready,
        }

    @app.post("/warmup")
    async def warmup():
        if not await runtime.ready():
            return JSONResponse({"ready": False}, status_code=503)
        payload = {"prompt": "Hello", "n_predict": 1, "temperature": 0}
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"http://127.0.0.1:{runtime.internal_port}/completion", json=payload
            )
        return JSONResponse({"ready": response.is_success}, status_code=200 if response.is_success else 503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": args.model_id, "object": "model"}]}

    @app.get("/metrics")
    async def metrics():
        return {
            "runtime": "llama-vulkan", "execution_preset": args.execution_preset,
            "load_seconds": round(time.monotonic() - runtime.started, 4),
        }

    async def proxy(request: Request, path: str):
        body = await request.json()
        body["model"] = args.model_id
        client = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=1))
        try:
            response = await client.send(
                client.build_request(
                    "POST", f"http://127.0.0.1:{runtime.internal_port}{path}", json=body
                ),
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
    uvicorn.run(create_app(args), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

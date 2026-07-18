from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from modeldeck.protocol import GenerationFamily

SAMPLE_RATE_HZ = 24_000
CHANNELS = 1
MAX_PCM_FRAME_BYTES = SAMPLE_RATE_HZ * 2


class MoshiProcess:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.internal_port = args.port + 1000
        self.process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        snapshot = self._snapshot()
        environment = dict(os.environ)
        environment.update({"HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": str(self.args.cache_root)})
        self.process = await asyncio.create_subprocess_exec(
            os.sys.executable,
            "-m",
            "moshi.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.internal_port),
            "--static",
            "none",
            "--hf-repo",
            self.args.model_id,
            "--moshi-weight",
            str(snapshot / "model.safetensors"),
            "--tokenizer",
            str(snapshot / "tokenizer_spm_32k_3.model"),
            "--device",
            "cuda",
            env=environment,
        )

    def _snapshot(self) -> Path:
        organisation, model = self.args.model_id.split("/", maxsplit=1)
        snapshot = (
            Path(self.args.cache_root) / f"models--{organisation}--{model}" / "snapshots" / self.args.revision
        ).resolve()
        required = {
            "model.safetensors",
            "tokenizer_spm_32k_3.model",
            "tokenizer-e351c8d8-checkpoint125.safetensors",
        }
        missing = sorted(name for name in required if not (snapshot / name).is_file())
        if missing:
            raise RuntimeError("Pinned Moshiko snapshot is incomplete: " + ", ".join(missing))
        return snapshot

    async def ready(self) -> bool:
        if self.process is None or self.process.returncode is not None:
            return False
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", self.internal_port), timeout=0.4
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (OSError, TimeoutError):
            return False

    async def stop(self) -> None:
        if self.process is None or self.process.returncode is not None:
            return
        self.process.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(self.process.wait(), timeout=8)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()


def validate_start(message: object, model_alias: str) -> None:
    if not isinstance(message, dict) or message.get("type") != "session.start":
        raise ValueError("The first message must be session.start")
    audio = message.get("audio")
    if message.get("model") != model_alias:
        raise ValueError("The session model must match the selected worker alias")
    if audio != {"encoding": "pcm_s16le", "sample_rate_hz": SAMPLE_RATE_HZ, "channels": CHANNELS}:
        raise ValueError("Moshiko requires PCM16 mono audio at 24 kHz")


def speech_control_type(message: str) -> str:
    try:
        event = json.loads(message)
    except json.JSONDecodeError as error:
        raise ValueError("Speech session controls must be JSON objects") from error
    event_type = event.get("type") if isinstance(event, dict) else None
    if event_type not in {"session.close", "response.cancel"}:
        raise ValueError("Unknown speech session control message")
    return str(event_type)


def create_app(args: argparse.Namespace) -> FastAPI:
    runtime = MoshiProcess(args)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await runtime.start()
        yield
        await runtime.stop()

    app = FastAPI(title="ModelDeck Moshiko speech worker", lifespan=lifespan)

    @app.get("/health")
    async def health():
        ready = await runtime.ready()
        return {
            "protocol_version": "1",
            "worker_id": args.worker_id,
            "runtime": "moshiko-rocm",
            "generation_family": GenerationFamily.SPEECH_CONVERSATION,
            "state": "warming" if ready else "loading",
            "model_id": args.model_id,
            "model_revision": args.revision,
            "device": "cuda:0",
            "device_name": "ROCm GPU",
            "rocm_version": None,
            "ready": ready,
        }

    @app.post("/warmup")
    async def warmup():
        return {"ready": await runtime.ready()}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": args.model_id, "object": "model"}]}

    @app.get("/model")
    async def model():
        return {
            "model_id": args.model_id,
            "revision": args.revision,
            "generation_family": GenerationFamily.SPEECH_CONVERSATION,
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "quantization": "none",
        }

    @app.get("/metrics")
    async def metrics():
        import importlib.metadata

        import torch

        return {
            "runtime": "moshiko-rocm",
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "maximum_sessions": 1,
            "torch_version": str(torch.__version__),
            "moshi_version": importlib.metadata.version("moshi"),
        }

    @app.post("/smoke")
    async def smoke():
        import aiohttp
        import numpy as np
        import sphn

        started = time.perf_counter()
        frame = np.zeros(int(SAMPLE_RATE_HZ * 0.08), dtype=np.float32)
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"http://127.0.0.1:{runtime.internal_port}/api/chat") as upstream:
                writer = sphn.OpusStreamWriter(SAMPLE_RATE_HZ)

                async def send_silence() -> None:
                    for _ in range(50):
                        encoded = writer.append_pcm(frame)
                        if encoded:
                            await upstream.send_bytes(b"\x01" + encoded)
                        await asyncio.sleep(0.08)

                async def wait_for_output() -> str:
                    async for message in upstream:
                        if message.type == aiohttp.WSMsgType.BINARY and message.data:
                            if message.data[0] in {1, 2}:
                                return "audio" if message.data[0] == 1 else "text"
                    raise RuntimeError("Moshiko closed without output")

                sender = asyncio.create_task(send_silence())
                try:
                    output_kind = await asyncio.wait_for(wait_for_output(), timeout=10)
                finally:
                    sender.cancel()
                    with suppress(asyncio.CancelledError):
                        await sender
                    await upstream.close()
        first_output = round(time.perf_counter() - started, 4)
        return {"ok": True, "output_kind": output_kind, "metrics": {"first_output_seconds": first_output}}

    @app.websocket("/v1/speech/conversations")
    async def conversation(client: WebSocket):
        await client.accept()
        try:
            start = await asyncio.wait_for(client.receive_json(), timeout=5)
            validate_start(start, args.alias)
        except (ValueError, TimeoutError, WebSocketDisconnect) as error:
            with suppress(RuntimeError):
                await client.send_json({"type": "error", "code": "invalid_session", "message": str(error)})
                await client.close(code=1008)
            return

        import aiohttp
        import numpy as np
        import sphn

        transcript: list[str] = []
        response_started = False
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"http://127.0.0.1:{runtime.internal_port}/api/chat") as upstream:
                opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE_HZ)
                opus_reader = sphn.OpusStreamReader(SAMPLE_RATE_HZ)
                await client.send_json(
                    {
                        "type": "session.ready",
                        "model": args.alias,
                        "audio": {"encoding": "pcm_s16le", "sample_rate_hz": SAMPLE_RATE_HZ, "channels": 1},
                        "voice": "moshiko",
                        "language": "en",
                    }
                )

                async def client_input() -> str:
                    while True:
                        message = await client.receive()
                        if message["type"] == "websocket.disconnect":
                            return "disconnect"
                        payload = message.get("bytes")
                        if payload is not None:
                            if not payload or len(payload) > MAX_PCM_FRAME_BYTES or len(payload) % 2:
                                raise ValueError(
                                    "PCM frames must contain at most one second of complete samples"
                                )
                            pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
                            encoded = opus_writer.append_pcm(pcm)
                            if encoded:
                                await upstream.send_bytes(b"\x01" + encoded)
                            continue
                        control = message.get("text")
                        if control:
                            event_type = speech_control_type(control)
                            if event_type == "session.close":
                                return "closed"
                            if event_type == "response.cancel":
                                return "cancelled"

                async def model_output() -> None:
                    nonlocal response_started
                    async for message in upstream:
                        if message.type != aiohttp.WSMsgType.BINARY or not message.data:
                            continue
                        kind, payload = message.data[0], message.data[1:]
                        if kind == 0:
                            continue
                        if kind == 2:
                            token = payload.decode("utf-8", errors="replace")
                            transcript.append(token)
                            await client.send_json({"type": "transcript.delta", "delta": token})
                        elif kind == 1:
                            pcm = opus_reader.append_bytes(payload)
                            if pcm.shape[-1] == 0:
                                continue
                            if not response_started:
                                response_started = True
                                await client.send_json({"type": "response.started"})
                            output = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
                            await client.send_bytes(output)

                input_task = asyncio.create_task(client_input())
                output_task = asyncio.create_task(model_output())
                done, pending = await asyncio.wait(
                    {input_task, output_task}, return_when=asyncio.FIRST_COMPLETED
                )
                completion_reason = "upstream_closed"
                for task in pending:
                    task.cancel()
                for task in done:
                    error = task.exception()
                    if error is not None:
                        completion_reason = "stream_error"
                        await client.send_json(
                            {"type": "error", "code": "stream_error", "message": str(error)}
                        )
                    elif task is input_task:
                        completion_reason = task.result()
                await upstream.close()
        with suppress(WebSocketDisconnect, RuntimeError):
            await client.send_json({"type": "transcript.final", "text": "".join(transcript).strip()})
            await client.send_json(
                {
                    "type": "response.completed",
                    "cancelled": completion_reason in {"cancelled", "disconnect", "stream_error"},
                }
            )
            await client.close()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cache-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(create_app(args), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

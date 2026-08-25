from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp


async def benchmark(url: str, duration_seconds: float) -> dict[str, object]:
    frame_seconds = 0.08
    pcm_frame = bytes(int(24_000 * frame_seconds) * 2)
    started = time.perf_counter()
    ready_at: float | None = None
    first_text_at: float | None = None
    first_audio_at: float | None = None
    audio_bytes = 0
    text = []
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url, max_msg_size=96_000) as socket:
            await socket.send_json(
                {
                    "type": "session.start",
                    "model": "repartee-speech",
                    "audio": {
                        "encoding": "pcm_s16le",
                        "sample_rate_hz": 24_000,
                        "channels": 1,
                    },
                }
            )
            message = await asyncio.wait_for(socket.receive(), timeout=5)
            if message.type != aiohttp.WSMsgType.TEXT:
                raise RuntimeError("Moshiko did not return the session.ready event")
            event = json.loads(message.data)
            if event.get("type") != "session.ready":
                raise RuntimeError(f"Unexpected first event: {event}")
            ready_at = time.perf_counter()

            async def send_audio() -> None:
                deadline = time.perf_counter() + duration_seconds
                while time.perf_counter() < deadline:
                    await socket.send_bytes(pcm_frame)
                    await asyncio.sleep(frame_seconds)
                await socket.send_json({"type": "session.close"})

            sender = asyncio.create_task(send_audio())
            deadline = time.perf_counter() + duration_seconds + 10
            while time.perf_counter() < deadline:
                try:
                    message = await asyncio.wait_for(socket.receive(), timeout=1)
                except TimeoutError:
                    if sender.done():
                        break
                    continue
                now = time.perf_counter()
                if message.type == aiohttp.WSMsgType.BINARY:
                    first_audio_at = first_audio_at or now
                    audio_bytes += len(message.data)
                elif message.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(message.data)
                    if event.get("type") == "transcript.delta":
                        first_text_at = first_text_at or now
                        text.append(str(event.get("delta", "")))
                    elif event.get("type") in {"response.completed", "error"}:
                        break
                else:
                    break
            await sender
    return {
        "session_ready_seconds": round((ready_at - started), 4) if ready_at else None,
        "first_text_seconds": round((first_text_at - started), 4) if first_text_at else None,
        "first_audio_seconds": round((first_audio_at - started), 4) if first_audio_at else None,
        "audio_bytes": audio_bytes,
        "transcript": "".join(text).strip(),
        "input_duration_seconds": duration_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8681/v1/speech/conversations")
    parser.add_argument("--duration-seconds", type=float, default=5)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(benchmark(args.url, args.duration_seconds)), indent=2))


if __name__ == "__main__":
    main()

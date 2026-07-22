from __future__ import annotations

import argparse
import json
import sys
import time
from array import array
from pathlib import Path
from typing import Any


def _require_rocm(torch: Any) -> None:
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("ROCm PyTorch did not expose an available GPU device")
    torch.empty(1, device="cuda:0", dtype=torch.float16)


def _transcribe(snapshot: Path, pcm_bytes: bytes) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    _require_rocm(torch)
    samples = array("h")
    samples.frombytes(pcm_bytes)
    if sys.byteorder != "little":
        samples.byteswap()
    audio = torch.tensor(samples, dtype=torch.float32).div_(32768.0).numpy()
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        snapshot,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda:0")
    model.eval()
    inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
    input_features = inputs.input_features.to("cuda:0", dtype=torch.float16)
    with torch.inference_mode():
        generated = model.generate(input_features, do_sample=False)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    torch.cuda.synchronize()
    return {
        "text": text,
        "inference_seconds": round(time.perf_counter() - started, 6),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "device_name": torch.cuda.get_device_name(0),
        "hip_version": str(torch.version.hip),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated ModelDeck Whisper inference child")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    arguments = parser.parse_args()
    try:
        import torch

        _require_rocm(torch)
        if arguments.probe:
            result = {
                "ok": True,
                "device_name": torch.cuda.get_device_name(0),
                "hip_version": str(torch.version.hip),
            }
        else:
            result = _transcribe(arguments.snapshot, sys.stdin.buffer.read())
        sys.stdout.write(json.dumps(result, ensure_ascii=True))
        sys.stdout.flush()
    except Exception as error:
        # Never include audio, transcript content, paths, or third-party exception text.
        sys.stdout.write(
            json.dumps(
                {
                    "error": {
                        "code": "inference_failed",
                        "category": type(error).__name__,
                    }
                }
            )
        )
        sys.stdout.flush()
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

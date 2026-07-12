from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from q4_direct_load_smoke import load_manifest, load_q4_layers
from q4_expert_probe import synchronise
from q4_hybrid_smoke import decode_generated, generate, make_inputs
from q4_inventory import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    find_local_snapshot,
)

GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the known-good BF16 DiffusionGemma model normally, stream "
            "the exported Q4 checkpoint over its shared expert layers, release "
            "the BF16 experts, and run a generation to isolate checkpoint "
            "restoration from meta-model reconstruction."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("MODELDECK_HF_HUB_CACHE", DEFAULT_CACHE_ROOT)),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("var/diffusiongemma-26b-a4b-it-gptq-q4-g32"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--denoising-steps", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--prompt",
        default="Explain why the sky appears blue in three concise sentences.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("var/q4-checkpoint-replace-smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This smoke test requires ROCm through torch.cuda")

    snapshot = find_local_snapshot(args.cache_root, args.model_id, args.revision)
    manifest = load_manifest(args.checkpoint_dir / "q4-manifest.json", args)
    print(f"Base snapshot: {snapshot}")
    print(f"Q4 checkpoint: {args.checkpoint_dir}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Torch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    synchronise(device)
    bf16_load_seconds = time.perf_counter() - load_started
    bf16_memory = int(torch.cuda.memory_allocated(device))
    print(f"BF16 load: {bf16_load_seconds:.3f} s")
    print(f"Allocated before replacement: {bf16_memory / GIB:.3f} GiB")

    replace_started = time.perf_counter()
    q4_layers, q4_bytes = load_q4_layers(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
        manifest=manifest,
        device=device,
    )
    torch.cuda.empty_cache()
    synchronise(device)
    replace_seconds = time.perf_counter() - replace_started
    q4_memory = int(torch.cuda.memory_allocated(device))
    print(f"Checkpoint replacement: {replace_seconds:.3f} s")
    print(f"Loaded packed Q4 experts: {q4_bytes / GIB:.3f} GiB")
    print(f"Allocated after replacement: {q4_memory / GIB:.3f} GiB")

    inputs = make_inputs(processor, args.prompt, device)
    output, generation_seconds = generate(
        model=model,
        inputs=inputs,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        device=device,
    )
    _, text = decode_generated(processor, inputs, output)
    gate_calls = sum(layer.q4_gate_calls for layer in q4_layers)
    down_calls = sum(layer.q4_down_calls for layer in q4_layers)
    routed_tokens = sum(layer.q4_tokens for layer in q4_layers)
    print(f"Generation: {generation_seconds:.3f} s")
    print(f"Generated text: {text!r}")
    print(
        "Q4 invocation: "
        f"gate_calls={gate_calls}, down_calls={down_calls}, "
        f"tokens={routed_tokens}"
    )

    payload = {
        "model_id": args.model_id,
        "revision": args.revision,
        "checkpoint_dir": str(args.checkpoint_dir),
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name(device),
        "bf16_load_seconds": bf16_load_seconds,
        "bf16_memory_bytes": bf16_memory,
        "replacement_seconds": replace_seconds,
        "q4_bytes": q4_bytes,
        "q4_memory_bytes": q4_memory,
        "generation_seconds": generation_seconds,
        "text": text,
        "q4_gate_calls": gate_calls,
        "q4_down_calls": down_calls,
        "q4_tokens": routed_tokens,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"JSON results: {args.json_output}")
    print(f"Peak memory: {payload['peak_memory_bytes'] / GIB:.3f} GiB")


if __name__ == "__main__":
    main()

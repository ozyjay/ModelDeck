from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from q4_expert_probe import synchronise
from q4_inventory import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    find_local_snapshot,
)

GIB = 1024**3

DEFAULT_PROMPTS = [
    "Explain why the sky appears blue in three concise sentences.",
    "Write a short Python function that returns the prime numbers below n.",
    "Compare photosynthesis and cellular respiration for a high-school student.",
    "Plan a three-day visit to Cairns with indoor and outdoor alternatives.",
    "A train travels 180 kilometres in 2.5 hours. Explain how to find its average speed.",
    "Write a brief imaginative scene about a robot discovering rain for the first time.",
    "Explain one benefit and one risk of using artificial intelligence in education.",
    "Translate 'The library opens at nine tomorrow morning' into French and explain one grammar choice.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure DiffusionGemma MoE routing coverage across a diverse local "
            "prompt suite before full Q4 expert calibration."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("MODELDECK_HF_HUB_CACHE", DEFAULT_CACHE_ROOT)),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--seed-repeats",
        type=int,
        default=1,
        help="Repeat the prompt suite with distinct deterministic canvas seeds.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="Optional JSON file containing a list of prompt strings.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("var/q4-calibration-coverage.json"),
    )
    return parser.parse_args()


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise SystemExit("--prompts-file must contain a non-empty JSON list")
    prompts = [str(item).strip() for item in payload]
    if any(not prompt for prompt in prompts):
        raise SystemExit("calibration prompts must be non-empty strings")
    return prompts


def counting_hook(
    layer: int,
    counts: dict[int, torch.Tensor],
    calls: dict[int, int],
):
    def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        if len(args) < 2:
            raise RuntimeError(f"Layer {layer} hook did not receive routing indices")
        top_k_index = args[1]
        counts[layer].add_(
            torch.bincount(
                top_k_index.reshape(-1),
                minlength=counts[layer].numel(),
            )
        )
        calls[layer] += 1

    return hook


def layer_summary(layer: int, values: torch.Tensor) -> dict[str, Any]:
    sample_counts = [int(value) for value in values.tolist()]
    return {
        "layer": layer,
        "total_assignments": sum(sample_counts),
        "experts_hit": sum(value > 0 for value in sample_counts),
        "minimum": min(sample_counts),
        "median": median(sample_counts),
        "mean": mean(sample_counts),
        "maximum": max(sample_counts),
        "at_least_32": sum(value >= 32 for value in sample_counts),
        "at_least_128": sum(value >= 128 for value in sample_counts),
        "at_least_512": sum(value >= 512 for value in sample_counts),
        "at_least_1024": sum(value >= 1024 for value in sample_counts),
        "counts": sample_counts,
    }


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_file)
    if args.max_new_tokens < 1 or args.denoising_steps < 1:
        raise SystemExit("token and denoising limits must be positive")
    if args.seed_repeats < 1:
        raise SystemExit("--seed-repeats must be at least 1")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This probe requires ROCm through torch.cuda")

    snapshot = find_local_snapshot(
        args.cache_root,
        args.model_id,
        args.revision,
    )
    print(f"Snapshot: {snapshot}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Torch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
    total_runs = len(prompts) * args.seed_repeats
    print(f"Prompts: {len(prompts)}")
    print(f"Seed repetitions: {args.seed_repeats}")
    print(f"Total generation runs: {total_runs}")
    print(f"Denoising steps per prompt: {args.denoising_steps}")

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
    load_seconds = time.perf_counter() - load_started
    print(f"Model load: {load_seconds:.3f} s")

    decoder_layers = model.model.decoder.layers
    encoder_layers = model.model.encoder.language_model.layers
    if len(encoder_layers) != len(decoder_layers):
        raise SystemExit("Encoder and decoder layer counts do not match")
    num_experts = decoder_layers[0].experts.num_experts
    decoder_counts = {
        layer: torch.zeros(num_experts, dtype=torch.int64, device=device)
        for layer in range(len(decoder_layers))
    }
    encoder_counts = {
        layer: torch.zeros(num_experts, dtype=torch.int64, device=device)
        for layer in range(len(encoder_layers))
    }
    decoder_calls = {
        layer: 0 for layer in range(len(decoder_layers))
    }
    encoder_calls = {
        layer: 0 for layer in range(len(encoder_layers))
    }
    handles = [
        layer.experts.register_forward_pre_hook(
            counting_hook(index, decoder_counts, decoder_calls)
        )
        for index, layer in enumerate(decoder_layers)
    ]
    handles.extend(
        layer.experts.register_forward_pre_hook(
            counting_hook(index, encoder_counts, encoder_calls)
        )
        for index, layer in enumerate(encoder_layers)
    )

    generation_seconds = 0.0
    try:
        run_index = 0
        for repeat in range(args.seed_repeats):
            for prompt_index, prompt in enumerate(prompts):
                run_index += 1
                inputs = processor.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(device)
                run_seed = args.seed + repeat * 10_000 + prompt_index
                torch.manual_seed(run_seed)
                started = time.perf_counter()
                with torch.inference_mode():
                    model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        max_denoising_steps=args.denoising_steps,
                        t_max=args.temperature,
                        t_min=min(0.4, args.temperature),
                        disable_compile=True,
                    )
                synchronise(device)
                elapsed = time.perf_counter() - started
                generation_seconds += elapsed
                print(
                    f"Run {run_index}/{total_runs}: "
                    f"prompt={prompt_index + 1}, repeat={repeat + 1}, "
                    f"seed={run_seed}, {elapsed:.3f} s"
                )
    finally:
        for handle in handles:
            handle.remove()

    decoder_summaries = [
        layer_summary(layer, decoder_counts[layer].to("cpu"))
        for layer in range(len(decoder_layers))
    ]
    encoder_summaries = [
        layer_summary(layer, encoder_counts[layer].to("cpu"))
        for layer in range(len(encoder_layers))
    ]
    summaries = [
        layer_summary(
            layer,
            (
                decoder_counts[layer]
                + encoder_counts[layer]
            ).to("cpu"),
        )
        for layer in range(len(decoder_layers))
    ]
    worst = sorted(
        (
            {
                "layer": summary["layer"],
                "expert": expert,
                "samples": samples,
            }
            for summary in summaries
            for expert, samples in enumerate(summary["counts"])
        ),
        key=lambda item: item["samples"],
    )

    print()
    print(
        "Layer  EncCalls DecCalls EncHit DecHit Union  Min  "
        "Median    Mean    Max  >=128  >=512"
    )
    print("-" * 92)
    for summary in summaries:
        layer = summary["layer"]
        print(
            f"{layer:>5}  "
            f"{encoder_calls[layer]:>8} "
            f"{decoder_calls[layer]:>8} "
            f"{encoder_summaries[layer]['experts_hit']:>6} "
            f"{decoder_summaries[layer]['experts_hit']:>6} "
            f"{summary['experts_hit']:>5}  "
            f"{summary['minimum']:>3}  "
            f"{summary['median']:>6.1f}  "
            f"{summary['mean']:>7.1f}  "
            f"{summary['maximum']:>5}  "
            f"{summary['at_least_128']:>5}  "
            f"{summary['at_least_512']:>5}"
        )

    print()
    print("Ten least-covered layer/expert pairs:")
    for item in worst[:10]:
        print(
            f"  layer={item['layer']}, expert={item['expert']}, "
            f"samples={item['samples']}"
        )

    total_pairs = len(decoder_layers) * num_experts
    global_summary = {
        "layer_expert_pairs": total_pairs,
        "all_experts_hit": all(item["samples"] > 0 for item in worst),
        "minimum_samples": worst[0]["samples"],
        "pairs_at_least_32": sum(item["samples"] >= 32 for item in worst),
        "pairs_at_least_128": sum(item["samples"] >= 128 for item in worst),
        "pairs_at_least_512": sum(item["samples"] >= 512 for item in worst),
        "pairs_at_least_1024": sum(item["samples"] >= 1024 for item in worst),
    }
    print()
    print(json.dumps(global_summary, indent=2))

    payload = {
        "model_id": args.model_id,
        "revision": args.revision,
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name(device),
        "prompt_count": len(prompts),
        "seed_repeats": args.seed_repeats,
        "generation_runs": total_runs,
        "prompts": prompts,
        "max_new_tokens": args.max_new_tokens,
        "denoising_steps": args.denoising_steps,
        "seed": args.seed,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "encoder_calls_per_layer": encoder_calls,
        "decoder_calls_per_layer": decoder_calls,
        "global_summary": global_summary,
        "least_covered": worst[:50],
        "combined_layers": summaries,
        "encoder_layers": encoder_summaries,
        "decoder_layers": decoder_summaries,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"JSON results: {args.json_output}")
    print(
        "Peak memory: "
        f"{torch.cuda.max_memory_allocated(device) / GIB:.3f} GiB"
    )


if __name__ == "__main__":
    main()

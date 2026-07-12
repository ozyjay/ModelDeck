from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from q4_expert_probe import (
    dequantize_symmetric_int4,
    quantize_symmetric_int4,
    synchronise,
    tensor_metrics,
)
from q4_inventory import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    find_local_snapshot,
)

GIB = 1024**3


def integer_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("values must be non-negative integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture real DiffusionGemma denoising activations and compare BF16 "
            "expert outputs with reconstructed symmetric INT4 expert outputs."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("HF_HUB_CACHE", DEFAULT_CACHE_ROOT)),
    )
    parser.add_argument("--layers", type=integer_list, default=integer_list("0,14,29"))
    parser.add_argument("--group-sizes", type=integer_list, default=integer_list("32,64,128"))
    parser.add_argument("--experts-per-layer", type=int, default=3)
    parser.add_argument("--max-samples-per-expert", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--denoising-steps", type=int, default=4)
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
        default=Path("var/q4-activation-probe.json"),
    )
    return parser.parse_args()


def capture_hook(
    layer: int,
    captures: dict[int, list[dict[str, torch.Tensor]]],
):
    def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        if len(args) < 3:
            raise RuntimeError(
                f"Layer {layer} expert hook expected hidden states, indices and weights"
            )
        hidden_states, top_k_index, top_k_weights = args[:3]
        captures[layer].append(
            {
                "hidden_states": hidden_states.detach().to("cpu"),
                "top_k_index": top_k_index.detach().to("cpu"),
                "top_k_weights": top_k_weights.detach().to("cpu"),
            }
        )

    return hook


def routed_samples(
    layer_captures: list[dict[str, torch.Tensor]],
    expert: int,
    maximum: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    states: list[torch.Tensor] = []
    route_weights: list[torch.Tensor] = []

    for capture in layer_captures:
        positions = torch.nonzero(capture["top_k_index"] == expert, as_tuple=False)
        if positions.numel() == 0:
            continue
        token_index = positions[:, 0]
        top_k_position = positions[:, 1]
        states.append(capture["hidden_states"][token_index])
        route_weights.append(
            capture["top_k_weights"][token_index, top_k_position]
        )

    if not states:
        raise RuntimeError(f"Expert {expert} received no captured tokens")

    combined_states = torch.cat(states, dim=0)
    combined_weights = torch.cat(route_weights, dim=0)
    return combined_states[:maximum], combined_weights[:maximum]


def route_counts(
    layer_captures: list[dict[str, torch.Tensor]],
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for capture in layer_captures:
        counts.update(int(value) for value in capture["top_k_index"].reshape(-1).tolist())
    return counts


def expert_forward(
    *,
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate, up = F.linear(hidden_states, gate_up_weight).chunk(2, dim=-1)
    intermediate = module.act_fn(gate) * up
    output = F.linear(intermediate, down_weight)
    weighted_output = output * route_weights[:, None]
    return gate, weighted_output


def evaluate_expert(
    *,
    layer: int,
    expert: int,
    hits: int,
    module: torch.nn.Module,
    hidden_states_cpu: torch.Tensor,
    route_weights_cpu: torch.Tensor,
    group_sizes: list[int],
    device: torch.device,
) -> list[dict[str, float | int]]:
    hidden_states = hidden_states_cpu.to(device=device, dtype=torch.bfloat16)
    route_weights = route_weights_cpu.to(device=device, dtype=torch.bfloat16)
    gate_up_weight = module.gate_up_proj[expert]
    down_weight = module.down_proj[expert]

    reference_gate, reference_output = expert_forward(
        module=module,
        hidden_states=hidden_states,
        route_weights=route_weights,
        gate_up_weight=gate_up_weight,
        down_weight=down_weight,
    )
    synchronise(device)

    bf16_bytes = (gate_up_weight.numel() + down_weight.numel()) * 2
    records: list[dict[str, float | int]] = []

    for group_size in group_sizes:
        quantized_gate_up = quantize_symmetric_int4(gate_up_weight, group_size)
        quantized_down = quantize_symmetric_int4(down_weight, group_size)
        reconstructed_gate_up = dequantize_symmetric_int4(
            quantized_gate_up,
            gate_up_weight.dtype,
        )
        reconstructed_down = dequantize_symmetric_int4(
            quantized_down,
            down_weight.dtype,
        )

        candidate_gate, candidate_output = expert_forward(
            module=module,
            hidden_states=hidden_states,
            route_weights=route_weights,
            gate_up_weight=reconstructed_gate_up,
            down_weight=reconstructed_down,
        )
        synchronise(device)

        gate_metrics = tensor_metrics(reference_gate, candidate_gate)
        output_metrics = tensor_metrics(reference_output, candidate_output)
        q4_bytes = (
            quantized_gate_up.packed.numel()
            + quantized_gate_up.scales.numel() * 2
            + quantized_down.packed.numel()
            + quantized_down.scales.numel() * 2
        )

        if not torch.isfinite(candidate_output).all():
            raise SystemExit(
                f"Layer {layer}, expert {expert}, group {group_size}: "
                "non-finite expert output"
            )

        records.append(
            {
                "layer": layer,
                "expert": expert,
                "route_hits": hits,
                "samples": hidden_states.shape[0],
                "group_size": group_size,
                "storage_ratio": q4_bytes / bf16_bytes,
                "gate_cosine": gate_metrics["cosine_similarity"],
                "gate_nrmse": gate_metrics["normalized_rmse"],
                "expert_output_cosine": output_metrics["cosine_similarity"],
                "expert_output_nrmse": output_metrics["normalized_rmse"],
                "expert_output_maximum_absolute_error": output_metrics[
                    "maximum_absolute_error"
                ],
            }
        )

        del (
            quantized_gate_up,
            quantized_down,
            reconstructed_gate_up,
            reconstructed_down,
            candidate_gate,
            candidate_output,
        )

    del hidden_states, route_weights, reference_gate, reference_output
    return records


def print_summary(records: list[dict[str, float | int]]) -> None:
    grouped: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for record in records:
        grouped[int(record["group_size"])].append(record)

    print()
    print(
        "Group  Storage  Gate cosine mean/min  Gate NRMSE mean/max  "
        "Expert cosine mean/min  Expert NRMSE mean/max"
    )
    print("-" * 111)

    for group_size in sorted(grouped):
        items = grouped[group_size]
        storage = mean(float(item["storage_ratio"]) for item in items)
        gate_cosines = [float(item["gate_cosine"]) for item in items]
        gate_nrmse = [float(item["gate_nrmse"]) for item in items]
        output_cosines = [float(item["expert_output_cosine"]) for item in items]
        output_nrmse = [float(item["expert_output_nrmse"]) for item in items]
        worst = max(items, key=lambda item: float(item["expert_output_nrmse"]))

        print(
            f"{group_size:>5}  "
            f"{storage:>7.2%}  "
            f"{mean(gate_cosines):.6f}/{min(gate_cosines):.6f}       "
            f"{mean(gate_nrmse):.6f}/{max(gate_nrmse):.6f}       "
            f"{mean(output_cosines):.6f}/{min(output_cosines):.6f}       "
            f"{mean(output_nrmse):.6f}/{max(output_nrmse):.6f}"
        )
        print(
            "       worst expert output: "
            f"layer={int(worst['layer'])}, expert={int(worst['expert'])}, "
            f"hits={int(worst['route_hits'])}, "
            f"NRMSE={float(worst['expert_output_nrmse']):.6f}, "
            f"cosine={float(worst['expert_output_cosine']):.6f}"
        )


def main() -> None:
    args = parse_args()
    if args.experts_per_layer < 1:
        raise SystemExit("--experts-per-layer must be at least 1")
    if args.max_samples_per_expert < 1:
        raise SystemExit("--max-samples-per-expert must be at least 1")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1")
    if args.denoising_steps < 1:
        raise SystemExit("--denoising-steps must be at least 1")
    if any(group_size < 1 for group_size in args.group_sizes):
        raise SystemExit("group sizes must be at least 1")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This probe requires the ROCm device exposed through torch.cuda")

    snapshot = find_local_snapshot(args.cache_root, args.model_id, args.revision)
    print(f"Snapshot: {snapshot}")
    print(f"Torch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
    print(f"Device: {device}")
    print(f"Device name: {torch.cuda.get_device_name(device)}")
    print(f"Layers: {args.layers}")
    print(f"Group sizes: {args.group_sizes}")
    print(f"Denoising steps: {args.denoising_steps}")
    print(f"Prompt: {args.prompt}")

    torch.manual_seed(args.seed)
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
    print(
        "Allocated after load: "
        f"{torch.cuda.memory_allocated(device) / GIB:.3f} GiB"
    )

    decoder_layers = model.model.decoder.layers
    for layer in args.layers:
        if not 0 <= layer < len(decoder_layers):
            raise SystemExit(
                f"Layer {layer} is outside the valid range 0..{len(decoder_layers) - 1}"
            )

    captures: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(list)
    handles = [
        decoder_layers[layer].experts.register_forward_pre_hook(
            capture_hook(layer, captures)
        )
        for layer in args.layers
    ]

    try:
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        generation_started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                max_denoising_steps=args.denoising_steps,
                t_max=args.temperature,
                t_min=min(0.4, args.temperature),
                disable_compile=True,
            )
        synchronise(device)
        generation_seconds = time.perf_counter() - generation_started
    finally:
        for handle in handles:
            handle.remove()

    sequences = output.sequences if hasattr(output, "sequences") else output
    prompt_length = int(inputs["input_ids"].shape[-1])
    generated_text = processor.decode(
        sequences[0, prompt_length:],
        skip_special_tokens=True,
    )
    print(f"Generation: {generation_seconds:.3f} s")
    print(f"Generated text: {generated_text!r}")

    records: list[dict[str, float | int]] = []
    selected_experts: dict[str, list[dict[str, int]]] = {}

    with torch.inference_mode():
        for layer in args.layers:
            counts = route_counts(captures[layer])
            busiest = counts.most_common(args.experts_per_layer)
            selected_experts[str(layer)] = [
                {"expert": expert, "hits": hits} for expert, hits in busiest
            ]
            print(f"Layer {layer} captured calls: {len(captures[layer])}")
            print(f"Layer {layer} busiest experts: {busiest}")

            module = decoder_layers[layer].experts
            for expert, hits in busiest:
                states, weights = routed_samples(
                    captures[layer],
                    expert,
                    args.max_samples_per_expert,
                )
                records.extend(
                    evaluate_expert(
                        layer=layer,
                        expert=expert,
                        hits=hits,
                        module=module,
                        hidden_states_cpu=states,
                        route_weights_cpu=weights,
                        group_sizes=args.group_sizes,
                        device=device,
                    )
                )

    print_summary(records)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": args.model_id,
        "revision": args.revision,
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name(device),
        "layers": args.layers,
        "group_sizes": args.group_sizes,
        "denoising_steps": args.denoising_steps,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "prompt": args.prompt,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "generated_text": generated_text,
        "selected_experts": selected_experts,
        "records": records,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    args.json_output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"JSON results: {args.json_output}")
    print(
        "Peak probe memory: "
        f"{torch.cuda.max_memory_allocated(device) / GIB:.3f} GiB"
    )


if __name__ == "__main__":
    main()

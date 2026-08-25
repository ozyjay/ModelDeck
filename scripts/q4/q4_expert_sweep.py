from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

from q4_expert_probe import (
    dequantize_symmetric_int4,
    load_expert_slice,
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
            "Compare DiffusionGemma packed-expert INT4 group sizes across a "
            "representative sample of layers and experts."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-root", type=Path, default=Path(DEFAULT_CACHE_ROOT))
    parser.add_argument("--layers", type=integer_list, default=integer_list("0,14,29"))
    parser.add_argument("--experts", type=integer_list, default=integer_list("0,63,127"))
    parser.add_argument("--group-sizes", type=integer_list, default=integer_list("32,64,128"))
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("var/q4-expert-sweep.json"),
    )
    return parser.parse_args()


def evaluate_projection(
    *,
    name: str,
    weight_cpu: torch.Tensor,
    device: torch.device,
    group_sizes: list[int],
    tokens: int,
    seed: int,
) -> list[dict[str, float | int | str | list[int]]]:
    weight = weight_cpu.to(device=device, dtype=torch.bfloat16)
    torch.manual_seed(seed)
    inputs = torch.randn(
        tokens,
        weight.shape[-1],
        dtype=torch.bfloat16,
        device=device,
    )
    reference_output = F.linear(inputs, weight)
    synchronise(device)

    bf16_bytes = weight.numel() * 2
    records: list[dict[str, float | int | str | list[int]]] = []

    for group_size in group_sizes:
        quantized = quantize_symmetric_int4(weight, group_size)
        reconstructed = dequantize_symmetric_int4(quantized, weight.dtype)
        candidate_output = F.linear(inputs, reconstructed)
        synchronise(device)

        weight_results = tensor_metrics(weight, reconstructed)
        output_results = tensor_metrics(reference_output, candidate_output)
        q4_bytes = quantized.packed.numel() + quantized.scales.numel() * 2

        if not torch.isfinite(candidate_output).all():
            raise SystemExit(
                f"{name}, group {group_size}: Q4 output contains non-finite values"
            )

        records.append(
            {
                "projection": name,
                "shape": list(weight.shape),
                "parameters": weight.numel(),
                "group_size": group_size,
                "storage_ratio": q4_bytes / bf16_bytes,
                "weight_cosine": weight_results["cosine_similarity"],
                "weight_nrmse": weight_results["normalized_rmse"],
                "output_cosine": output_results["cosine_similarity"],
                "output_nrmse": output_results["normalized_rmse"],
                "output_maximum_absolute_error": output_results[
                    "maximum_absolute_error"
                ],
            }
        )

        del quantized, reconstructed, candidate_output

    del weight, inputs, reference_output
    return records


def print_summary(records: list[dict[str, float | int | str | list[int]]]) -> None:
    grouped: dict[int, list[dict[str, float | int | str | list[int]]]] = defaultdict(list)
    for record in records:
        grouped[int(record["group_size"])].append(record)

    print()
    print(
        "Group  Storage  Weight cosine mean/min  Weight NRMSE mean/max  "
        "Output cosine mean/min  Output NRMSE mean/max"
    )
    print("-" * 119)

    for group_size in sorted(grouped):
        items = grouped[group_size]
        storage = mean(float(item["storage_ratio"]) for item in items)
        weight_cosines = [float(item["weight_cosine"]) for item in items]
        weight_nrmse = [float(item["weight_nrmse"]) for item in items]
        output_cosines = [float(item["output_cosine"]) for item in items]
        output_nrmse = [float(item["output_nrmse"]) for item in items]

        print(
            f"{group_size:>5}  "
            f"{storage:>7.2%}  "
            f"{mean(weight_cosines):.6f}/{min(weight_cosines):.6f}       "
            f"{mean(weight_nrmse):.6f}/{max(weight_nrmse):.6f}       "
            f"{mean(output_cosines):.6f}/{min(output_cosines):.6f}       "
            f"{mean(output_nrmse):.6f}/{max(output_nrmse):.6f}"
        )

        worst = max(items, key=lambda item: float(item["output_nrmse"]))
        print(
            "       worst output: "
            f"{worst['projection']}, "
            f"NRMSE={float(worst['output_nrmse']):.6f}, "
            f"cosine={float(worst['output_cosine']):.6f}, "
            f"max_abs={float(worst['output_maximum_absolute_error']):.6f}"
        )


def main() -> None:
    args = parse_args()
    if args.tokens < 1:
        raise SystemExit("--tokens must be at least 1")
    if any(group_size < 1 for group_size in args.group_sizes):
        raise SystemExit("group sizes must be at least 1")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ROCm/CUDA device requested but torch.cuda.is_available() is false")

    snapshot = find_local_snapshot(args.cache_root, args.model_id, args.revision)
    print(f"Snapshot: {snapshot}")
    print(f"Torch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device)}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Layers: {args.layers}")
    print(f"Experts: {args.experts}")
    print(f"Group sizes: {args.group_sizes}")
    print(f"Test tokens: {args.tokens}")

    records: list[dict[str, float | int | str | list[int]]] = []
    for layer in args.layers:
        for expert in args.experts:
            prefix = f"model.decoder.layers.{layer}.experts"
            projections = (
                ("gate_up_proj", f"{prefix}.gate_up_proj"),
                ("down_proj", f"{prefix}.down_proj"),
            )
            for projection_index, (_, tensor_name) in enumerate(projections):
                print(f"Testing layer {layer}, expert {expert}, {tensor_name.rsplit('.', 1)[-1]}")
                weight = load_expert_slice(snapshot, tensor_name, expert)
                records.extend(
                    evaluate_projection(
                        name=tensor_name,
                        weight_cpu=weight,
                        device=device,
                        group_sizes=args.group_sizes,
                        tokens=args.tokens,
                        seed=11 + layer * 10_000 + expert * 10 + projection_index,
                    )
                )
                del weight

    print_summary(records)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": args.model_id,
        "revision": args.revision,
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "layers": args.layers,
        "experts": args.experts,
        "group_sizes": args.group_sizes,
        "tokens": args.tokens,
        "records": records,
    }
    args.json_output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"JSON results: {args.json_output}")
    if device.type == "cuda":
        print(
            "Peak sweep memory: "
            f"{torch.cuda.max_memory_allocated(device) / 1024**3:.3f} GiB"
        )


if __name__ == "__main__":
    main()

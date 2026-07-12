from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from q4_expert_probe import load_expert_slice, synchronise, tensor_metrics
from q4_gptq_probe import GPTQLinearResult, quantize_linear_gptq
from q4_inventory import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    find_local_snapshot,
)

MIB = 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack one real DiffusionGemma expert projection into GPTQModel's "
            "standard format and validate Torch/Triton Q4 runtimes on ROCm."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("MODELDECK_HF_HUB_CACHE", DEFAULT_CACHE_ROOT)),
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=103)
    parser.add_argument(
        "--projection",
        choices=("gate_up_proj", "down_proj"),
        default="gate_up_proj",
    )
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("var/q4-kernel-probe.json"),
    )
    return parser.parse_args()


def make_runtime(
    runtime_class: type[TorchLinear],
    result: GPTQLinearResult,
    device: torch.device,
) -> TorchLinear:
    source = nn.Linear(
        result.weight.shape[1],
        result.weight.shape[0],
        bias=False,
        device=device,
        dtype=result.weight.dtype,
    )
    source.weight.data.copy_(result.weight)

    runtime = runtime_class(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=result.weight.shape[1],
        out_features=result.weight.shape[0],
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=False,
        dtype=torch.bfloat16,
    )
    runtime.pack(
        source,
        result.scales.to("cpu"),
        result.zeros.to("cpu"),
        result.group_index.to("cpu", dtype=torch.int32),
        workers=1,
    )
    # pack_block emits direct logical zero points. Mark that layout as v2
    # without applying the legacy v1 + 1 correction a second time.
    runtime.qzero_format(format=2)
    runtime.to(device)
    runtime.eval()
    runtime.bias = None

    # The pure Torch result should not silently use Triton dequantisation.
    if type(runtime) is TorchLinear:
        runtime._triton_dequant_enabled = False
        runtime._cache_enabled = False
        runtime._init_wf_unsqueeze_buffers()

    del source
    return runtime


def logical_zero_points(module: TorchLinear) -> list[int]:
    shifts = torch.arange(
        0,
        32,
        4,
        dtype=torch.int64,
        device=module.qzeros.device,
    )
    unpacked = torch.bitwise_and(
        torch.bitwise_right_shift(
            module.qzeros.to(torch.int64).unsqueeze(-1),
            shifts,
        ),
        0x0F,
    )
    return [int(value) for value in torch.unique(unpacked).tolist()]


def packed_storage_bytes(module: TorchLinear) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            module.qweight,
            module.qzeros,
            module.scales,
            module.g_idx,
        )
    )


def benchmark(
    operation: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[torch.Tensor, float]:
    output = operation()
    for _ in range(warmup):
        output = operation()
    synchronise(device)

    started = time.perf_counter()
    for _ in range(iterations):
        output = operation()
    synchronise(device)
    milliseconds = (time.perf_counter() - started) * 1000 / iterations
    return output, milliseconds


def comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    values = tensor_metrics(reference, candidate)
    return {
        "cosine": values["cosine_similarity"],
        "nrmse": values["normalized_rmse"],
        "mean_absolute_error": values["mean_absolute_error"],
        "maximum_absolute_error": values["maximum_absolute_error"],
    }


def runtime_probe(
    *,
    name: str,
    operation: Callable[[], torch.Tensor],
    reference: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    try:
        output, milliseconds = benchmark(
            operation,
            device=device,
            warmup=warmup,
            iterations=iterations,
        )
        return {
            "runtime": name,
            "success": True,
            "milliseconds": milliseconds,
            "comparison": comparison(reference, output),
        }
    except Exception as error:
        return {
            "runtime": name,
            "success": False,
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> None:
    args = parse_args()
    if args.layer < 0 or args.expert < 0:
        raise SystemExit("--layer and --expert must be non-negative")
    if args.group_size != 32:
        raise SystemExit("This first kernel probe is pinned to --group-size 32")
    if min(
        args.calibration_samples,
        args.tokens,
        args.iterations,
        args.warmup + 1,
    ) < 1:
        raise SystemExit("sample, token, iteration and warmup values must be positive")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This probe requires the ROCm device exposed through torch.cuda")

    snapshot = find_local_snapshot(args.cache_root, args.model_id, args.revision)
    tensor_name = (
        f"model.decoder.layers.{args.layer}.experts.{args.projection}"
    )
    print(f"Snapshot: {snapshot}")
    print(f"Tensor: {tensor_name}")
    print(f"Expert: {args.expert}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Torch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")

    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    weight_cpu = load_expert_slice(snapshot, tensor_name, args.expert)
    weight = weight_cpu.to(device=device, dtype=torch.bfloat16)
    calibration_inputs = torch.randn(
        args.calibration_samples,
        weight.shape[1],
        dtype=torch.bfloat16,
        device=device,
    )

    print("Calibrating projection with GPTQ...")
    result = quantize_linear_gptq(
        weight=weight,
        calibration_inputs=calibration_inputs,
        group_size=args.group_size,
        calibration_batch_size=128,
        block_size=128,
        damp_percent=0.01,
    )
    print(
        "GPTQ calibration: "
        f"{result.metadata['wall_seconds']:.4f} s, "
        f"samples={result.metadata['samples']}"
    )

    inputs = torch.randn(
        args.tokens,
        result.weight.shape[1],
        dtype=torch.bfloat16,
        device=device,
    )
    reference, reference_ms = benchmark(
        lambda: F.linear(inputs, result.weight),
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    print("Packing standard GPTQ TorchLinear...")
    torch_runtime = make_runtime(TorchLinear, result, device)
    storage_bytes = packed_storage_bytes(torch_runtime)
    torch_zero_points = logical_zero_points(torch_runtime)
    torch_result = runtime_probe(
        name="torch",
        operation=lambda: torch_runtime(inputs),
        reference=reference,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    print("Packing standard GPTQ TritonV2Linear...")
    triton_runtime = make_runtime(TritonV2Linear, result, device)
    triton_zero_points = logical_zero_points(triton_runtime)
    triton_result = runtime_probe(
        name="triton_v2",
        operation=lambda: triton_runtime(inputs),
        reference=reference,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    print()
    print(f"Reference BF16 linear: {reference_ms:.4f} ms")
    print(
        "Standard packed storage: "
        f"{storage_bytes / MIB:.2f} MiB "
        f"({storage_bytes / (result.weight.numel() * 2):.2%} of BF16)"
    )
    print(
        "Logical zero points: "
        f"torch={torch_zero_points}, triton={triton_zero_points}"
    )
    for value in (torch_result, triton_result):
        if value["success"]:
            metrics = value["comparison"]
            print(
                f"{value['runtime']}: {value['milliseconds']:.4f} ms, "
                f"cosine={metrics['cosine']:.8f}, "
                f"NRMSE={metrics['nrmse']:.6f}, "
                f"max_abs={metrics['maximum_absolute_error']:.6f}"
            )
        else:
            print(f"{value['runtime']}: FAILED: {value['error']}")

    payload = {
        "model_id": args.model_id,
        "revision": args.revision,
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name(device),
        "layer": args.layer,
        "expert": args.expert,
        "projection": args.projection,
        "shape": list(result.weight.shape),
        "group_size": args.group_size,
        "tokens": args.tokens,
        "iterations": args.iterations,
        "reference_milliseconds": reference_ms,
        "packed_storage_bytes": storage_bytes,
        "packed_storage_ratio": storage_bytes / (result.weight.numel() * 2),
        "torch_zero_points": torch_zero_points,
        "triton_zero_points": triton_zero_points,
        "gptq_metadata": result.metadata,
        "runtimes": [torch_result, triton_result],
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"JSON results: {args.json_output}")
    print(
        "Peak probe memory: "
        f"{torch.cuda.max_memory_allocated(device) / MIB:.2f} MiB"
    )


if __name__ == "__main__":
    main()

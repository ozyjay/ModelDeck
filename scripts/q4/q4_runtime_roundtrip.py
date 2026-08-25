from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from safetensors.torch import load_file, save_file

from q4_expert_probe import load_expert_slice, synchronise, tensor_metrics
from q4_gptq_probe import quantize_linear_gptq
from q4_inventory import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    find_local_snapshot,
)
from q4_kernel_probe import make_runtime

MIB = 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save one packed DiffusionGemma Triton Q4 projection to "
            "safetensors, reconstruct it without a BF16 source weight, and "
            "verify its ROCm output against the original packed module."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("MODELDECK_HF_HUB_CACHE", DEFAULT_CACHE_ROOT)),
    )
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--expert", type=int, default=50)
    parser.add_argument(
        "--projection",
        choices=("gate_up_proj", "down_proj"),
        default="gate_up_proj",
    )
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("var/q4-runtime-roundtrip.safetensors"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("var/q4-runtime-roundtrip.json"),
    )
    return parser.parse_args()


def runtime_constructor(
    *,
    in_features: int,
    out_features: int,
) -> TritonV2Linear:
    runtime = TritonV2Linear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=False,
        dtype=torch.bfloat16,
    )
    runtime.bias = None
    return runtime


def checkpoint_tensors(module: TritonV2Linear) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for name, tensor in module.state_dict().items():
        values[name] = tensor.detach().to("cpu").contiguous()
    if not values:
        raise RuntimeError("Packed runtime produced an empty state_dict")
    return values


def restore_runtime(
    tensors: dict[str, torch.Tensor],
    *,
    in_features: int,
    out_features: int,
    device: torch.device,
) -> TritonV2Linear:
    runtime = runtime_constructor(
        in_features=in_features,
        out_features=out_features,
    )
    for name, tensor in tensors.items():
        if "." in name:
            raise RuntimeError(f"Unexpected nested packed state key: {name}")
        if hasattr(runtime, name):
            delattr(runtime, name)
        runtime.register_buffer(name, tensor.to(device))
    runtime.qzero_format(format=2)
    runtime.eval()
    return runtime


def comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    values = tensor_metrics(reference, candidate)
    return {
        "cosine": values["cosine_similarity"],
        "nrmse": values["normalized_rmse"],
        "mean_absolute_error": values["mean_absolute_error"],
        "maximum_absolute_error": values["maximum_absolute_error"],
    }


def tensor_inventory(values: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes": tensor.numel() * tensor.element_size(),
        }
        for name, tensor in sorted(values.items())
    ]


def main() -> None:
    args = parse_args()
    if min(args.calibration_samples, args.tokens) < 1:
        raise SystemExit("--calibration-samples and --tokens must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This probe requires ROCm through torch.cuda")

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
    calibration = torch.randn(
        args.calibration_samples,
        weight.shape[1],
        device=device,
        dtype=torch.bfloat16,
    )
    print("Calibrating projection with GPTQ...")
    result = quantize_linear_gptq(
        weight=weight,
        calibration_inputs=calibration,
        group_size=32,
        calibration_batch_size=128,
        block_size=128,
        damp_percent=0.01,
    )
    print("Packing original Triton runtime...")
    original = make_runtime(TritonV2Linear, result, device)

    inputs = torch.randn(
        args.tokens,
        result.weight.shape[1],
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        dequantized_reference = F.linear(inputs, result.weight)
        original_output = original(inputs)
    synchronise(device)

    tensors = checkpoint_tensors(original)
    inventory = tensor_inventory(tensors)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.checkpoint),
        metadata={
            "format": "modeldeck-diffusiongemma-gptq-v1",
            "bits": "4",
            "group_size": "32",
            "sym": "true",
            "desc_act": "false",
            "qzero_format": "2",
            "in_features": str(result.weight.shape[1]),
            "out_features": str(result.weight.shape[0]),
        },
    )
    checkpoint_bytes = args.checkpoint.stat().st_size
    print(f"Checkpoint tensors: {[value['name'] for value in inventory]}")
    print(f"Checkpoint size: {checkpoint_bytes / MIB:.3f} MiB")

    del original, tensors
    torch.cuda.empty_cache()
    loaded = load_file(str(args.checkpoint), device="cpu")
    restored = restore_runtime(
        loaded,
        in_features=result.weight.shape[1],
        out_features=result.weight.shape[0],
        device=device,
    )
    with torch.inference_mode():
        restored_output = restored(inputs)
    synchronise(device)

    packed_roundtrip = comparison(original_output, restored_output)
    dequantized_comparison = comparison(
        dequantized_reference,
        restored_output,
    )
    tensor_exact = all(
        torch.equal(loaded[name], restored.state_dict()[name].to("cpu"))
        for name in loaded
    )
    print(
        "Packed round-trip: "
        f"cosine={packed_roundtrip['cosine']:.8f}, "
        f"NRMSE={packed_roundtrip['nrmse']:.8f}, "
        f"max_abs={packed_roundtrip['maximum_absolute_error']:.8f}"
    )
    print(
        "Restored vs dequantized GPTQ: "
        f"cosine={dequantized_comparison['cosine']:.8f}, "
        f"NRMSE={dequantized_comparison['nrmse']:.8f}"
    )
    print(f"Checkpoint tensors exact: {tensor_exact}")

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
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": checkpoint_bytes,
        "tensors": inventory,
        "packed_roundtrip": packed_roundtrip,
        "restored_vs_dequantized_gptq": dequantized_comparison,
        "checkpoint_tensors_exact": tensor_exact,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"JSON results: {args.json_output}")
    print(f"Peak memory: {payload['peak_memory_bytes'] / MIB:.2f} MiB")


if __name__ == "__main__":
    main()

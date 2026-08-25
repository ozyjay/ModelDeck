from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from q4_inventory import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    find_local_snapshot,
)

GIB = 1024**3


@dataclass(frozen=True)
class QuantizedTensor:
    packed: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    group_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one packed DiffusionGemma expert, apply symmetric group-wise "
            "INT4 RTN quantisation, validate nibble packing, and compare BF16 "
            "and reconstructed-Q4 linear outputs."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-root", type=Path, default=Path(DEFAULT_CACHE_ROOT))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_expert_slice(snapshot: Path, tensor_name: str, expert: int) -> torch.Tensor:
    for shard in sorted(snapshot.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as weights:
            if tensor_name not in weights.keys():
                continue
            tensor_slice = weights.get_slice(tensor_name)
            shape = tuple(tensor_slice.get_shape())
            if len(shape) != 3:
                raise SystemExit(f"{tensor_name} is not a packed 3D expert tensor: {shape}")
            if not 0 <= expert < shape[0]:
                raise SystemExit(
                    f"Expert {expert} is outside the valid range 0..{shape[0] - 1}"
                )
            return tensor_slice[expert].contiguous()
    raise SystemExit(f"Tensor was not found in the local checkpoint: {tensor_name}")


def pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1).to(torch.int16)
    unsigned = torch.bitwise_and(flat, 0x0F).to(torch.uint8)
    if unsigned.numel() % 2:
        unsigned = F.pad(unsigned, (0, 1))
    low = unsigned[0::2]
    high = torch.bitwise_left_shift(unsigned[1::2], 4)
    return torch.bitwise_or(low, high).contiguous()


def unpack_signed_int4(packed: torch.Tensor, count: int) -> torch.Tensor:
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.int16, device=packed.device)
    unpacked[0::2] = torch.bitwise_and(packed, 0x0F).to(torch.int16)
    unpacked[1::2] = torch.bitwise_right_shift(packed, 4).to(torch.int16)
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    return unpacked[:count].to(torch.int8)


def quantize_symmetric_int4(weight: torch.Tensor, group_size: int) -> QuantizedTensor:
    if group_size < 1:
        raise ValueError("group_size must be at least 1")

    original_shape = tuple(weight.shape)
    columns = original_shape[-1]
    rows = math.prod(original_shape[:-1])
    groups_per_row = math.ceil(columns / group_size)
    padded_columns = groups_per_row * group_size

    matrix = weight.float().reshape(rows, columns)
    if padded_columns != columns:
        matrix = F.pad(matrix, (0, padded_columns - columns))

    grouped = matrix.reshape(rows, groups_per_row, group_size)
    maximum = grouped.abs().amax(dim=-1, keepdim=True)
    scales = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 7)
    scales = scales.to(torch.bfloat16)

    quantized = torch.round(grouped / scales.float()).clamp(-8, 7).to(torch.int8)
    unpadded = quantized.reshape(rows, padded_columns)[:, :columns]
    packed = pack_signed_int4(unpadded)

    return QuantizedTensor(
        packed=packed,
        scales=scales.squeeze(-1).contiguous(),
        shape=original_shape,
        group_size=group_size,
    )


def dequantize_symmetric_int4(value: QuantizedTensor, dtype: torch.dtype) -> torch.Tensor:
    columns = value.shape[-1]
    rows = math.prod(value.shape[:-1])
    groups_per_row = math.ceil(columns / value.group_size)
    padded_columns = groups_per_row * value.group_size

    quantized = unpack_signed_int4(value.packed, rows * columns)
    matrix = quantized.reshape(rows, columns)
    if padded_columns != columns:
        matrix = F.pad(matrix, (0, padded_columns - columns))

    grouped = matrix.reshape(rows, groups_per_row, value.group_size)
    restored = grouped.float() * value.scales.float().unsqueeze(-1)
    return restored.reshape(rows, padded_columns)[:, :columns].reshape(value.shape).to(dtype)


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    difference = candidate_f32 - reference_f32
    rmse = difference.square().mean().sqrt()
    reference_rms = reference_f32.square().mean().sqrt()
    cosine = F.cosine_similarity(
        reference_f32.reshape(1, -1),
        candidate_f32.reshape(1, -1),
    ).item()
    return {
        "cosine_similarity": cosine,
        "rmse": rmse.item(),
        "normalized_rmse": (rmse / reference_rms.clamp_min(1e-12)).item(),
        "mean_absolute_error": difference.abs().mean().item(),
        "maximum_absolute_error": difference.abs().max().item(),
    }


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def probe_projection(
    *,
    name: str,
    weight_cpu: torch.Tensor,
    device: torch.device,
    group_size: int,
    tokens: int,
) -> None:
    weight = weight_cpu.to(device=device, dtype=torch.bfloat16)
    synchronise(device)
    started = time.perf_counter()
    quantized = quantize_symmetric_int4(weight, group_size)
    reconstructed = dequantize_symmetric_int4(quantized, weight.dtype)
    synchronise(device)
    quantization_seconds = time.perf_counter() - started

    input_values = torch.randn(
        tokens,
        weight.shape[-1],
        dtype=torch.bfloat16,
        device=device,
    )
    reference_output = F.linear(input_values, weight)
    candidate_output = F.linear(input_values, reconstructed)
    synchronise(device)

    weight_results = tensor_metrics(weight, reconstructed)
    output_results = tensor_metrics(reference_output, candidate_output)
    bf16_bytes = weight.numel() * 2
    q4_bytes = quantized.packed.numel() + quantized.scales.numel() * 2

    print()
    print(name)
    print(f"  shape: {tuple(weight.shape)}")
    print(f"  parameters: {weight.numel():,}")
    print(f"  BF16 storage: {bf16_bytes / 1024**2:.2f} MiB")
    print(f"  packed Q4 storage: {q4_bytes / 1024**2:.2f} MiB")
    print(f"  storage ratio: {q4_bytes / bf16_bytes:.2%}")
    print(f"  quantize + pack + unpack: {quantization_seconds:.4f} s")
    print(
        "  weight: "
        f"cosine={weight_results['cosine_similarity']:.8f}, "
        f"NRMSE={weight_results['normalized_rmse']:.6f}, "
        f"max_abs={weight_results['maximum_absolute_error']:.6f}"
    )
    print(
        "  linear output: "
        f"cosine={output_results['cosine_similarity']:.8f}, "
        f"NRMSE={output_results['normalized_rmse']:.6f}, "
        f"max_abs={output_results['maximum_absolute_error']:.6f}"
    )

    if not torch.isfinite(reconstructed).all():
        raise SystemExit(f"{name}: reconstructed weights contain non-finite values")
    if not torch.isfinite(candidate_output).all():
        raise SystemExit(f"{name}: Q4 linear output contains non-finite values")


def main() -> None:
    args = parse_args()
    if args.layer < 0:
        raise SystemExit("--layer must be non-negative")
    if args.expert < 0:
        raise SystemExit("--expert must be non-negative")
    if args.tokens < 1:
        raise SystemExit("--tokens must be at least 1")
    if args.group_size < 1:
        raise SystemExit("--group-size must be at least 1")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ROCm/CUDA device requested but torch.cuda.is_available() is false")

    snapshot = find_local_snapshot(args.cache_root, args.model_id, args.revision)
    prefix = f"model.decoder.layers.{args.layer}.experts"
    gate_up_name = f"{prefix}.gate_up_proj"
    down_name = f"{prefix}.down_proj"

    print(f"Snapshot: {snapshot}")
    print(f"Torch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device)}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    print(f"Layer: {args.layer}")
    print(f"Expert: {args.expert}")
    print(f"Group size: {args.group_size}")
    print(f"Test tokens: {args.tokens}")

    torch.manual_seed(11)
    gate_up = load_expert_slice(snapshot, gate_up_name, args.expert)
    down = load_expert_slice(snapshot, down_name, args.expert)

    probe_projection(
        name=gate_up_name,
        weight_cpu=gate_up,
        device=device,
        group_size=args.group_size,
        tokens=args.tokens,
    )
    probe_projection(
        name=down_name,
        weight_cpu=down,
        device=device,
        group_size=args.group_size,
        tokens=args.tokens,
    )

    if device.type == "cuda":
        print()
        print(
            "Peak probe memory: "
            f"{torch.cuda.max_memory_allocated(device) / GIB:.3f} GiB"
        )


if __name__ == "__main__":
    main()

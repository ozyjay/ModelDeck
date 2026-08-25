from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from q4_activation_probe import (
    capture_hook,
    expert_forward,
    route_counts,
    routed_samples,
)
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


@dataclass(frozen=True)
class GPTQLinearResult:
    weight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    group_index: torch.Tensor
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RTN Q4 with low-level Hessian-calibrated GPTQ for one "
            "DiffusionGemma packed expert using real denoising activations."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(DEFAULT_CACHE_ROOT),
    )
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--expert", type=int, default=85)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--calibration-batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--damp-percent", type=float, default=0.01)
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
        default=Path("var/q4-gptq-probe.json"),
    )
    return parser.parse_args()


def quantize_linear_gptq(
    *,
    weight: torch.Tensor,
    calibration_inputs: torch.Tensor,
    group_size: int,
    calibration_batch_size: int,
    block_size: int,
    damp_percent: float,
) -> GPTQLinearResult:
    linear = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        device=weight.device,
        dtype=weight.dtype,
    )
    linear.weight.data.copy_(weight)

    config = QuantizeConfig(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        act_group_aware=False,
        static_groups=False,
        damp_percent=damp_percent,
        fallback=None,
        offload_to_disk=False,
    )
    engine = GPTQ(linear, qcfg=config)
    # GPTQModel's whole-model loop normally performs this configuration.
    # The packed-expert probe intentionally calls the low-level API directly.
    engine.quantizer.configure(perchannel=True)

    for start in range(0, calibration_inputs.shape[0], calibration_batch_size):
        batch = calibration_inputs[start : start + calibration_batch_size]
        engine.add_batch(batch, None)

    started = time.perf_counter()
    (
        quantized_weight,
        scales,
        zeros,
        group_index,
        duration,
        average_loss,
        used_damp,
        sample_count,
    ) = engine.quantize(blocksize=block_size)
    synchronise(weight.device)
    wall_seconds = time.perf_counter() - started

    result = quantized_weight.detach().clone()
    stored_scales = scales.detach().clone()
    stored_zeros = zeros.detach().clone()
    stored_group_index = group_index.detach().to(torch.int64).clone()
    unique_zero_points = torch.unique(
        stored_zeros.round().clamp(0, 15).to(torch.uint8)
    )
    metadata = {
        "samples": int(sample_count),
        "duration_seconds": float(duration),
        "wall_seconds": wall_seconds,
        "average_loss": str(average_loss),
        "used_damp": float(used_damp),
        "scale_shape": list(scales.shape),
        "zero_shape": list(zeros.shape),
        "group_index_shape": list(group_index.shape),
        "zero_points": [int(value) for value in unique_zero_points.tolist()],
    }
    engine.free()
    del linear, engine, scales, zeros, group_index
    return GPTQLinearResult(
        weight=result,
        scales=stored_scales,
        zeros=stored_zeros,
        group_index=stored_group_index,
        metadata=metadata,
    )


def pack_unsigned_int4(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1).to(torch.uint8)
    if flat.numel() % 2:
        flat = torch.nn.functional.pad(flat, (0, 1))
    return torch.bitwise_or(
        flat[0::2],
        torch.bitwise_left_shift(flat[1::2], 4),
    ).contiguous()


def unpack_unsigned_int4(packed: torch.Tensor, count: int) -> torch.Tensor:
    unpacked = torch.empty(
        packed.numel() * 2,
        dtype=torch.uint8,
        device=packed.device,
    )
    unpacked[0::2] = torch.bitwise_and(packed, 0x0F)
    unpacked[1::2] = torch.bitwise_right_shift(packed, 4)
    return unpacked[:count]


def pack_gptq_result(
    value: GPTQLinearResult,
) -> tuple[torch.Tensor, dict[str, Any]]:
    group_index = value.group_index.to(torch.long)
    scale_by_column = value.scales.float()[:, group_index]
    zero_codes = value.zeros.round().clamp(0, 15).to(torch.uint8)
    zero_by_column = zero_codes[:, group_index].float()

    integer_codes = torch.round(
        value.weight.float() / scale_by_column + zero_by_column
    ).clamp(0, 15).to(torch.uint8)
    packed = pack_unsigned_int4(integer_codes)
    restored_codes = unpack_unsigned_int4(
        packed,
        value.weight.numel(),
    ).reshape(value.weight.shape)

    stored_scales = value.scales.to(torch.bfloat16)
    stored_scale_by_column = stored_scales.float()[:, group_index]
    reconstructed = (
        restored_codes.float() - zero_by_column
    ) * stored_scale_by_column
    reconstructed = reconstructed.to(value.weight.dtype)

    unique_zero_points = torch.unique(zero_codes)
    constant_zero = unique_zero_points.numel() == 1
    zero_storage_bytes = (
        0 if constant_zero else (zero_codes.numel() + 1) // 2
    )
    storage_bytes = (
        packed.numel()
        + stored_scales.numel() * stored_scales.element_size()
        + zero_storage_bytes
    )
    bf16_bytes = value.weight.numel() * 2
    drift = tensor_metrics(value.weight, reconstructed)

    metadata = {
        "packed_shape": list(packed.shape),
        "scale_shape": list(stored_scales.shape),
        "zero_shape": list(zero_codes.shape),
        "zero_points": [int(item) for item in unique_zero_points.tolist()],
        "constant_zero_point": constant_zero,
        "zero_storage_bytes": int(zero_storage_bytes),
        "storage_bytes": int(storage_bytes),
        "storage_ratio": storage_bytes / bf16_bytes,
        "weight_roundtrip_cosine": drift["cosine_similarity"],
        "weight_roundtrip_nrmse": drift["normalized_rmse"],
        "weight_roundtrip_maximum_absolute_error": drift[
            "maximum_absolute_error"
        ],
    }
    return reconstructed, metadata


def output_comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    metrics = tensor_metrics(reference, candidate)
    return {
        "cosine": metrics["cosine_similarity"],
        "nrmse": metrics["normalized_rmse"],
        "mean_absolute_error": metrics["mean_absolute_error"],
        "maximum_absolute_error": metrics["maximum_absolute_error"],
    }


def main() -> None:
    args = parse_args()
    if args.layer < 0 or args.expert < 0:
        raise SystemExit("--layer and --expert must be non-negative")
    if args.group_size < 1:
        raise SystemExit("--group-size must be at least 1")
    if args.calibration_samples < 1 or args.calibration_batch_size < 1:
        raise SystemExit("calibration sample and batch sizes must be at least 1")
    if not 0 < args.damp_percent < 1:
        raise SystemExit("--damp-percent must be between 0 and 1")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This probe requires the ROCm device exposed through torch.cuda")

    snapshot = find_local_snapshot(args.cache_root, args.model_id, args.revision)
    print(f"Snapshot: {snapshot}")
    print(f"Torch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Layer: {args.layer}")
    print(f"Expert: {args.expert}")
    print(f"Group size: {args.group_size}")

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
    print(f"Allocated after load: {torch.cuda.memory_allocated(device) / GIB:.3f} GiB")

    decoder_layers = model.model.decoder.layers
    if args.layer >= len(decoder_layers):
        raise SystemExit(
            f"Layer {args.layer} is outside the valid range 0..{len(decoder_layers) - 1}"
        )
    expert_module = decoder_layers[args.layer].experts
    if args.expert >= expert_module.num_experts:
        raise SystemExit(
            f"Expert {args.expert} is outside the valid range "
            f"0..{expert_module.num_experts - 1}"
        )

    captures: dict[int, list[dict[str, torch.Tensor]]] = {args.layer: []}
    handle = expert_module.register_forward_pre_hook(
        capture_hook(args.layer, captures)
    )

    try:
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
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
    finally:
        handle.remove()

    counts = route_counts(captures[args.layer])
    print(f"Captured calls: {len(captures[args.layer])}")
    print(f"Busiest experts: {counts.most_common(10)}")
    if counts[args.expert] == 0:
        raise SystemExit(
            f"Expert {args.expert} received no tokens. Choose one of the listed experts."
        )

    hidden_states_cpu, route_weights_cpu = routed_samples(
        captures[args.layer],
        args.expert,
        args.calibration_samples,
    )
    hidden_states = hidden_states_cpu.to(device=device, dtype=torch.bfloat16)
    route_weights = route_weights_cpu.to(device=device, dtype=torch.bfloat16)
    print(f"Calibration samples: {hidden_states.shape[0]}")

    gate_up_weight = expert_module.gate_up_proj[args.expert]
    down_weight = expert_module.down_proj[args.expert]

    with torch.inference_mode():
        reference_gate, reference_output = expert_forward(
            module=expert_module,
            hidden_states=hidden_states,
            route_weights=route_weights,
            gate_up_weight=gate_up_weight,
            down_weight=down_weight,
        )

        rtn_gate_value = quantize_symmetric_int4(gate_up_weight, args.group_size)
        rtn_down_value = quantize_symmetric_int4(down_weight, args.group_size)
        rtn_gate_weight = dequantize_symmetric_int4(
            rtn_gate_value,
            gate_up_weight.dtype,
        )
        rtn_down_weight = dequantize_symmetric_int4(
            rtn_down_value,
            down_weight.dtype,
        )
        rtn_gate, rtn_output = expert_forward(
            module=expert_module,
            hidden_states=hidden_states,
            route_weights=route_weights,
            gate_up_weight=rtn_gate_weight,
            down_weight=rtn_down_weight,
        )

        print("Calibrating gate_up_proj with GPTQ...")
        gptq_gate_result = quantize_linear_gptq(
            weight=gate_up_weight,
            calibration_inputs=hidden_states,
            group_size=args.group_size,
            calibration_batch_size=args.calibration_batch_size,
            block_size=args.block_size,
            damp_percent=args.damp_percent,
        )
        gptq_gate_weight = gptq_gate_result.weight
        gate_metadata = gptq_gate_result.metadata

        gptq_gate, _ = expert_forward(
            module=expert_module,
            hidden_states=hidden_states,
            route_weights=route_weights,
            gate_up_weight=gptq_gate_weight,
            down_weight=down_weight,
        )
        gate_values, up_values = torch.nn.functional.linear(
            hidden_states,
            gptq_gate_weight,
        ).chunk(2, dim=-1)
        down_calibration_inputs = expert_module.act_fn(gate_values) * up_values

        print("Calibrating down_proj with GPTQ...")
        gptq_down_result = quantize_linear_gptq(
            weight=down_weight,
            calibration_inputs=down_calibration_inputs,
            group_size=args.group_size,
            calibration_batch_size=args.calibration_batch_size,
            block_size=args.block_size,
            damp_percent=args.damp_percent,
        )
        gptq_down_weight = gptq_down_result.weight
        down_metadata = gptq_down_result.metadata

        _, gptq_output = expert_forward(
            module=expert_module,
            hidden_states=hidden_states,
            route_weights=route_weights,
            gate_up_weight=gptq_gate_weight,
            down_weight=gptq_down_weight,
        )

        packed_gate_weight, gate_pack_metadata = pack_gptq_result(
            gptq_gate_result
        )
        packed_down_weight, down_pack_metadata = pack_gptq_result(
            gptq_down_result
        )
        packed_gate, packed_output = expert_forward(
            module=expert_module,
            hidden_states=hidden_states,
            route_weights=route_weights,
            gate_up_weight=packed_gate_weight,
            down_weight=packed_down_weight,
        )
        synchronise(device)

    results = {
        "rtn_gate": output_comparison(reference_gate, rtn_gate),
        "rtn_expert_output": output_comparison(reference_output, rtn_output),
        "gptq_gate": output_comparison(reference_gate, gptq_gate),
        "gptq_expert_output": output_comparison(reference_output, gptq_output),
        "packed_gptq_gate": output_comparison(reference_gate, packed_gate),
        "packed_gptq_expert_output": output_comparison(
            reference_output,
            packed_output,
        ),
        "packing_drift_expert_output": output_comparison(
            gptq_output,
            packed_output,
        ),
    }

    print()
    print("Method       Stage          Cosine      NRMSE       Mean abs     Max abs")
    print("-" * 75)
    for method, stage, key in (
        ("RTN", "gate", "rtn_gate"),
        ("RTN", "expert output", "rtn_expert_output"),
        ("GPTQ", "gate", "gptq_gate"),
        ("GPTQ", "expert output", "gptq_expert_output"),
        ("GPTQ-pack", "gate", "packed_gptq_gate"),
        ("GPTQ-pack", "expert output", "packed_gptq_expert_output"),
    ):
        value = results[key]
        print(
            f"{method:<12} {stage:<14} "
            f"{value['cosine']:.8f}  "
            f"{value['nrmse']:.6f}  "
            f"{value['mean_absolute_error']:.6f}  "
            f"{value['maximum_absolute_error']:.6f}"
        )

    rtn_nrmse = results["rtn_expert_output"]["nrmse"]
    gptq_nrmse = results["gptq_expert_output"]["nrmse"]
    improvement = (rtn_nrmse - gptq_nrmse) / rtn_nrmse
    print()
    print(f"GPTQ expert-output NRMSE improvement: {improvement:.2%}")
    packing_drift = results["packing_drift_expert_output"]
    print(
        "Packing-only expert-output drift: "
        f"NRMSE={packing_drift['nrmse']:.8f}, "
        f"cosine={packing_drift['cosine']:.8f}"
    )
    print(
        "Packed storage ratio: "
        f"gate={gate_pack_metadata['storage_ratio']:.2%}, "
        f"down={down_pack_metadata['storage_ratio']:.2%}"
    )
    print(
        "Zero points: "
        f"gate={gate_pack_metadata['zero_points']}, "
        f"down={down_pack_metadata['zero_points']}"
    )

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": args.model_id,
        "revision": args.revision,
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name(device),
        "layer": args.layer,
        "expert": args.expert,
        "route_hits": counts[args.expert],
        "calibration_samples": hidden_states.shape[0],
        "group_size": args.group_size,
        "load_seconds": load_seconds,
        "results": results,
        "gptq_gate_metadata": gate_metadata,
        "gptq_down_metadata": down_metadata,
        "gptq_gate_pack_metadata": gate_pack_metadata,
        "gptq_down_pack_metadata": down_pack_metadata,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    args.json_output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"JSON results: {args.json_output}")
    print(f"Peak probe memory: {torch.cuda.max_memory_allocated(device) / GIB:.3f} GiB")


if __name__ == "__main__":
    main()

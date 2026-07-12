from __future__ import annotations

import argparse
import json
import os
import time
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
        default=Path(os.environ.get("HF_HUB_CACHE", DEFAULT_CACHE_ROOT)),
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
) -> tuple[torch.Tensor, dict[str, Any]]:
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
    metadata = {
        "samples": int(sample_count),
        "duration_seconds": float(duration),
        "wall_seconds": wall_seconds,
        "average_loss": str(average_loss),
        "used_damp": float(used_damp),
        "scale_shape": list(scales.shape),
        "zero_shape": list(zeros.shape),
        "group_index_shape": list(group_index.shape),
    }
    engine.free()
    del linear, engine, scales, zeros, group_index
    return result, metadata


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
        gptq_gate_weight, gate_metadata = quantize_linear_gptq(
            weight=gate_up_weight,
            calibration_inputs=hidden_states,
            group_size=args.group_size,
            calibration_batch_size=args.calibration_batch_size,
            block_size=args.block_size,
            damp_percent=args.damp_percent,
        )

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
        gptq_down_weight, down_metadata = quantize_linear_gptq(
            weight=down_weight,
            calibration_inputs=down_calibration_inputs,
            group_size=args.group_size,
            calibration_batch_size=args.calibration_batch_size,
            block_size=args.block_size,
            damp_percent=args.damp_percent,
        )

        _, gptq_output = expert_forward(
            module=expert_module,
            hidden_states=hidden_states,
            route_weights=route_weights,
            gate_up_weight=gptq_gate_weight,
            down_weight=gptq_down_weight,
        )
        synchronise(device)

    results = {
        "rtn_gate": output_comparison(reference_gate, rtn_gate),
        "rtn_expert_output": output_comparison(reference_output, rtn_output),
        "gptq_gate": output_comparison(reference_gate, gptq_gate),
        "gptq_expert_output": output_comparison(reference_output, gptq_output),
    }

    print()
    print("Method  Stage          Cosine      NRMSE       Mean abs     Max abs")
    print("-" * 70)
    for method, stage, key in (
        ("RTN", "gate", "rtn_gate"),
        ("RTN", "expert output", "rtn_expert_output"),
        ("GPTQ", "gate", "gptq_gate"),
        ("GPTQ", "expert output", "gptq_expert_output"),
    ):
        value = results[key]
        print(
            f"{method:<7} {stage:<14} "
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

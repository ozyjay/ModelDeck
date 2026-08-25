from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from q4_calibration_coverage import load_prompts
from q4_expert_probe import synchronise, tensor_metrics
from q4_gptq_probe import quantize_linear_gptq
from q4_hybrid_smoke import (
    decode_generated,
    generate,
    make_inputs,
    token_agreement,
)
from q4_inventory import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    find_local_snapshot,
)
from q4_kernel_probe import make_runtime, packed_storage_bytes

GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate every expert in one shared DiffusionGemma layer, replace "
            "both its encoder and decoder expert weights with packed Triton Q4 "
            "modules, verify memory release, and run an end-to-end smoke test."
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
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--calibration-denoising-steps", type=int, default=8)
    parser.add_argument("--smoke-denoising-steps", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("var/q4-full-layer-smoke.json"),
    )
    return parser.parse_args()


def activation_capture(bucket: list[dict[str, torch.Tensor]]):
    def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if len(args) < 3:
            raise RuntimeError("Expert hook did not receive hidden states and routing data")
        bucket.append(
            {
                "hidden_states": args[0].detach().to("cpu"),
                "top_k_index": args[1].detach().to("cpu"),
            }
        )

    return hook


def routed_calibration(
    captures: list[dict[str, torch.Tensor]],
    expert: int,
    maximum: int,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    remaining = maximum
    for capture in captures:
        positions = torch.nonzero(
            capture["top_k_index"] == expert,
            as_tuple=False,
        )
        if positions.numel() == 0:
            continue
        selected = capture["hidden_states"][positions[:, 0]]
        selected = selected[:remaining]
        values.append(selected)
        remaining -= int(selected.shape[0])
        if remaining == 0:
            break
    if not values:
        return torch.empty(
            0,
            captures[0]["hidden_states"].shape[-1],
            dtype=captures[0]["hidden_states"].dtype,
        )
    return torch.cat(values, dim=0)


def evenly_spaced_rows(values: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        return values[:0]
    if values.shape[0] <= count:
        repeats = (count + values.shape[0] - 1) // values.shape[0]
        return values.repeat((repeats, 1))[:count]
    indices = torch.linspace(
        0,
        values.shape[0] - 1,
        steps=count,
        dtype=torch.float64,
    ).round().to(torch.long)
    return values[indices]


def combined_calibration(
    captures: list[dict[str, torch.Tensor]],
    pooled: torch.Tensor,
    expert: int,
    maximum: int,
) -> tuple[torch.Tensor, int, int]:
    routed = routed_calibration(captures, expert, maximum)
    routed_count = int(routed.shape[0])
    fallback_count = maximum - routed_count
    if fallback_count:
        routed = torch.cat(
            [routed, evenly_spaced_rows(pooled, fallback_count)],
            dim=0,
        )
    return routed, routed_count, fallback_count


class FullQ4Experts(nn.Module):
    def __init__(
        self,
        *,
        gate_up: list[nn.Module],
        down: list[nn.Module],
        act_fn: nn.Module,
    ) -> None:
        super().__init__()
        self.gate_up = nn.ModuleList(gate_up)
        self.down = nn.ModuleList(down)
        self.act_fn = act_fn
        self.num_experts = len(gate_up)
        self.q4_gate_calls = 0
        self.q4_down_calls = 0
        self.q4_tokens = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(
                top_k_index,
                num_classes=self.num_experts,
            ).permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)),
                0,
            ).nonzero()

        for expert_tensor in expert_hit:
            expert = int(expert_tensor[0].item())
            top_k_position, token_index = torch.where(expert_mask[expert])
            current_state = hidden_states[token_index]
            gate, up = self.gate_up[expert](current_state).chunk(2, dim=-1)
            intermediate = self.act_fn(gate) * up
            current_hidden_states = self.down[expert](intermediate)
            current_hidden_states = (
                current_hidden_states
                * top_k_weights[token_index, top_k_position, None]
            )
            final_hidden_states.index_add_(
                0,
                token_index,
                current_hidden_states.to(final_hidden_states.dtype),
            )
            self.q4_gate_calls += 1
            self.q4_down_calls += 1
            self.q4_tokens += int(current_state.shape[0])

        return final_hidden_states


def memory_bytes(device: torch.device) -> int:
    synchronise(device)
    return int(torch.cuda.memory_allocated(device))


def main() -> None:
    args = parse_args()
    if args.layer < 0:
        raise SystemExit("--layer must be non-negative")
    if args.group_size != 32:
        raise SystemExit("This first full-layer smoke is pinned to --group-size 32")
    if args.calibration_samples < 32:
        raise SystemExit("--calibration-samples must be at least 32")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This smoke test requires ROCm through torch.cuda")

    prompts = load_prompts(args.prompts_file)
    snapshot = find_local_snapshot(args.cache_root, args.model_id, args.revision)
    print(f"Snapshot: {snapshot}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Torch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
    print(f"Layer: {args.layer}")
    print(f"Calibration prompts: {len(prompts)}")

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
    load_seconds = time.perf_counter() - load_started
    loaded_memory = memory_bytes(device)
    print(f"Model load: {load_seconds:.3f} s")
    print(f"Allocated after load: {loaded_memory / GIB:.3f} GiB")

    encoder_layers = model.model.encoder.language_model.layers
    decoder_layers = model.model.decoder.layers
    if args.layer >= min(len(encoder_layers), len(decoder_layers)):
        raise SystemExit("--layer is outside the encoder/decoder layer range")
    encoder_experts = encoder_layers[args.layer].experts
    decoder_experts = decoder_layers[args.layer].experts
    if encoder_experts.num_experts != decoder_experts.num_experts:
        raise RuntimeError("Encoder and decoder expert counts differ")
    num_experts = int(decoder_experts.num_experts)
    shared_gate_storage = (
        encoder_experts.gate_up_proj.data_ptr()
        == decoder_experts.gate_up_proj.data_ptr()
    )
    shared_down_storage = (
        encoder_experts.down_proj.data_ptr()
        == decoder_experts.down_proj.data_ptr()
    )
    print(
        "Shared BF16 storage: "
        f"gate={shared_gate_storage}, down={shared_down_storage}"
    )

    captures: list[dict[str, torch.Tensor]] = []
    handles = [
        encoder_experts.register_forward_pre_hook(activation_capture(captures)),
        decoder_experts.register_forward_pre_hook(activation_capture(captures)),
    ]
    calibration_seconds: list[float] = []
    try:
        for index, prompt in enumerate(prompts):
            inputs = make_inputs(processor, prompt, device)
            _, elapsed = generate(
                model=model,
                inputs=inputs,
                seed=args.seed + index,
                max_new_tokens=args.max_new_tokens,
                denoising_steps=args.calibration_denoising_steps,
                temperature=args.temperature,
                device=device,
            )
            calibration_seconds.append(elapsed)
            print(
                f"Calibration run {index + 1}/{len(prompts)}: "
                f"{elapsed:.3f} s"
            )
    finally:
        for handle in handles:
            handle.remove()

    pooled = torch.cat(
        [capture["hidden_states"] for capture in captures],
        dim=0,
    )
    print(f"Captured calls: {len(captures)}")
    print(f"Pooled hidden states: {pooled.shape[0]}")

    smoke_inputs = make_inputs(processor, prompts[0], device)
    baseline_output, baseline_seconds = generate(
        model=model,
        inputs=smoke_inputs,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        denoising_steps=args.smoke_denoising_steps,
        temperature=args.temperature,
        device=device,
    )
    baseline_tokens, baseline_text = decode_generated(
        processor,
        smoke_inputs,
        baseline_output,
    )
    print(f"Baseline smoke: {baseline_seconds:.3f} s")
    print(f"Baseline text: {baseline_text!r}")

    q4_gate: list[nn.Module] = []
    q4_down: list[nn.Module] = []
    expert_results: list[dict[str, Any]] = []
    quantize_started = time.perf_counter()
    for expert in range(num_experts):
        calibration_cpu, routed_count, fallback_count = combined_calibration(
            captures,
            pooled,
            expert,
            args.calibration_samples,
        )
        calibration = calibration_cpu.to(device=device, dtype=torch.bfloat16)
        gate_weight = decoder_experts.gate_up_proj[expert]
        down_weight = decoder_experts.down_proj[expert]

        gate_result = quantize_linear_gptq(
            weight=gate_weight,
            calibration_inputs=calibration,
            group_size=args.group_size,
            calibration_batch_size=128,
            block_size=128,
            damp_percent=0.01,
        )
        with torch.inference_mode():
            gate_values, up_values = F.linear(
                calibration,
                gate_result.weight,
            ).chunk(2, dim=-1)
            down_calibration = decoder_experts.act_fn(gate_values) * up_values

        down_result = quantize_linear_gptq(
            weight=down_weight,
            calibration_inputs=down_calibration,
            group_size=args.group_size,
            calibration_batch_size=128,
            block_size=128,
            damp_percent=0.01,
        )
        gate_runtime = make_runtime(TritonV2Linear, gate_result, device)
        down_runtime = make_runtime(TritonV2Linear, down_result, device)

        validation_rows = min(32, int(calibration.shape[0]))
        with torch.inference_mode():
            gate_reference = F.linear(
                calibration[:validation_rows],
                gate_result.weight,
            )
            gate_candidate = gate_runtime(calibration[:validation_rows])
            down_reference = F.linear(
                down_calibration[:validation_rows],
                down_result.weight,
            )
            down_candidate = down_runtime(down_calibration[:validation_rows])
        gate_metrics = tensor_metrics(gate_reference, gate_candidate)
        down_metrics = tensor_metrics(down_reference, down_candidate)

        q4_gate.append(gate_runtime)
        q4_down.append(down_runtime)
        expert_results.append(
            {
                "expert": expert,
                "routed_samples": routed_count,
                "fallback_samples": fallback_count,
                "gate_wall_seconds": gate_result.metadata["wall_seconds"],
                "down_wall_seconds": down_result.metadata["wall_seconds"],
                "gate_runtime_nrmse": gate_metrics["normalized_rmse"],
                "down_runtime_nrmse": down_metrics["normalized_rmse"],
                "packed_bytes": (
                    packed_storage_bytes(gate_runtime)
                    + packed_storage_bytes(down_runtime)
                ),
            }
        )
        del (
            calibration_cpu,
            calibration,
            gate_weight,
            down_weight,
            gate_values,
            up_values,
            down_calibration,
            gate_result,
            down_result,
            gate_reference,
            gate_candidate,
            down_reference,
            down_candidate,
        )
        if (expert + 1) % 8 == 0 or expert + 1 == num_experts:
            print(
                f"Packed experts: {expert + 1}/{num_experts}; "
                f"last routed={routed_count}, fallback={fallback_count}"
            )

    quantize_seconds = time.perf_counter() - quantize_started
    packed_memory = memory_bytes(device)
    packed_bytes = sum(value["packed_bytes"] for value in expert_results)
    print(f"Full-layer calibration + packing: {quantize_seconds:.3f} s")
    print(f"Packed layer storage: {packed_bytes / GIB:.3f} GiB")

    q4_experts = FullQ4Experts(
        gate_up=q4_gate,
        down=q4_down,
        act_fn=decoder_experts.act_fn,
    )
    encoder_layers[args.layer].experts = q4_experts
    decoder_layers[args.layer].experts = q4_experts
    del q4_gate, q4_down, encoder_experts, decoder_experts
    gc.collect()
    torch.cuda.empty_cache()
    converted_memory = memory_bytes(device)
    memory_saved = loaded_memory - converted_memory
    print(f"Allocated with BF16 + packed layer: {packed_memory / GIB:.3f} GiB")
    print(f"Allocated after replacement: {converted_memory / GIB:.3f} GiB")
    print(f"Net allocated saving vs load: {memory_saved / GIB:.3f} GiB")

    q4_smoke_inputs = make_inputs(processor, prompts[0], device)
    q4_output, q4_seconds = generate(
        model=model,
        inputs=q4_smoke_inputs,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        denoising_steps=args.smoke_denoising_steps,
        temperature=args.temperature,
        device=device,
    )
    q4_tokens, q4_text = decode_generated(
        processor,
        q4_smoke_inputs,
        q4_output,
    )
    agreement = token_agreement(baseline_tokens, q4_tokens)
    print(f"Q4 smoke: {q4_seconds:.3f} s")
    print(f"Q4 text: {q4_text!r}")
    print(
        "Token agreement: "
        f"{agreement['matching_tokens']}/{agreement['shared_tokens']} "
        f"({agreement['agreement']:.2%}), exact={agreement['exact_match']}"
    )
    print(
        "Q4 invocation: "
        f"gate_calls={q4_experts.q4_gate_calls}, "
        f"down_calls={q4_experts.q4_down_calls}, "
        f"tokens={q4_experts.q4_tokens}"
    )

    payload = {
        "model_id": args.model_id,
        "revision": args.revision,
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name(device),
        "layer": args.layer,
        "num_experts": num_experts,
        "group_size": args.group_size,
        "calibration_samples": args.calibration_samples,
        "calibration_prompt_count": len(prompts),
        "captured_calls": len(captures),
        "pooled_hidden_states": int(pooled.shape[0]),
        "shared_gate_storage": shared_gate_storage,
        "shared_down_storage": shared_down_storage,
        "load_seconds": load_seconds,
        "calibration_generation_seconds": calibration_seconds,
        "quantize_pack_seconds": quantize_seconds,
        "baseline_seconds": baseline_seconds,
        "q4_seconds": q4_seconds,
        "baseline_text": baseline_text,
        "q4_text": q4_text,
        "token_agreement": agreement,
        "loaded_memory_bytes": loaded_memory,
        "bf16_plus_packed_memory_bytes": packed_memory,
        "converted_memory_bytes": converted_memory,
        "memory_saved_bytes": memory_saved,
        "packed_layer_bytes": packed_bytes,
        "q4_gate_calls": q4_experts.q4_gate_calls,
        "q4_down_calls": q4_experts.q4_down_calls,
        "q4_tokens": q4_experts.q4_tokens,
        "expert_results": expert_results,
        "summary": {
            "minimum_routed_samples": min(
                value["routed_samples"] for value in expert_results
            ),
            "experts_using_fallback": sum(
                value["fallback_samples"] > 0 for value in expert_results
            ),
            "mean_gate_runtime_nrmse": mean(
                value["gate_runtime_nrmse"] for value in expert_results
            ),
            "maximum_gate_runtime_nrmse": max(
                value["gate_runtime_nrmse"] for value in expert_results
            ),
            "mean_down_runtime_nrmse": mean(
                value["down_runtime_nrmse"] for value in expert_results
            ),
            "maximum_down_runtime_nrmse": max(
                value["down_runtime_nrmse"] for value in expert_results
            ),
        },
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"JSON results: {args.json_output}")
    print(f"Peak memory: {torch.cuda.max_memory_allocated(device) / GIB:.3f} GiB")


if __name__ == "__main__":
    main()

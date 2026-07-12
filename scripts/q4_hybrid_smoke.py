from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from q4_activation_probe import capture_hook, routed_samples
from q4_expert_probe import synchronise, tensor_metrics
from q4_gptq_probe import quantize_linear_gptq
from q4_inventory import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    find_local_snapshot,
)
from q4_kernel_probe import make_runtime

GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a baseline DiffusionGemma generation, calibrate one routed "
            "expert with GPTQ, replace that expert's projections with packed "
            "Triton Q4 modules, and run the same generation again."
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
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--calibration-samples", type=int, default=512)
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
        default=Path("var/q4-hybrid-smoke.json"),
    )
    return parser.parse_args()


class HybridQ4Experts(nn.Module):
    def __init__(
        self,
        original: nn.Module,
        *,
        expert: int,
        gate_up: nn.Module,
        down: nn.Module,
    ) -> None:
        super().__init__()
        self.original = original
        self.expert = expert
        self.gate_up = gate_up
        self.down = down
        self.num_experts = original.num_experts
        self.act_fn = original.act_fn
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
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)),
                0,
            ).nonzero()

        for expert_tensor in expert_hit:
            expert_index = int(expert_tensor[0].item())
            if expert_index == self.num_experts:
                continue

            top_k_position, token_index = torch.where(
                expert_mask[expert_index]
            )
            current_state = hidden_states[token_index]

            if expert_index == self.expert:
                gate, up = self.gate_up(current_state).chunk(2, dim=-1)
                self.q4_gate_calls += 1
                self.q4_tokens += int(current_state.shape[0])
            else:
                gate, up = F.linear(
                    current_state,
                    self.original.gate_up_proj[expert_index],
                ).chunk(2, dim=-1)

            intermediate = self.act_fn(gate) * up

            if expert_index == self.expert:
                current_hidden_states = self.down(intermediate)
                self.q4_down_calls += 1
            else:
                current_hidden_states = F.linear(
                    intermediate,
                    self.original.down_proj[expert_index],
                )

            current_hidden_states = (
                current_hidden_states
                * top_k_weights[token_index, top_k_position, None]
            )
            final_hidden_states.index_add_(
                0,
                token_index,
                current_hidden_states.to(final_hidden_states.dtype),
            )

        return final_hidden_states


def make_inputs(
    processor: Any,
    prompt: str,
    device: torch.device,
) -> Any:
    return processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)


def generate(
    *,
    model: nn.Module,
    inputs: Any,
    seed: int,
    max_new_tokens: int,
    denoising_steps: int,
    temperature: float,
    device: torch.device,
) -> tuple[Any, float]:
    torch.manual_seed(seed)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_denoising_steps=denoising_steps,
            t_max=temperature,
            t_min=min(0.4, temperature),
            disable_compile=True,
        )
    synchronise(device)
    return output, time.perf_counter() - started


def output_sequences(output: Any) -> torch.Tensor:
    return output.sequences if hasattr(output, "sequences") else output


def decode_generated(
    processor: Any,
    inputs: Any,
    output: Any,
) -> tuple[torch.Tensor, str]:
    sequences = output_sequences(output)
    prompt_length = int(inputs["input_ids"].shape[-1])
    generated_tokens = sequences[0, prompt_length:].detach().to("cpu")
    text = processor.decode(
        generated_tokens,
        skip_special_tokens=True,
    )
    return generated_tokens, text


def token_agreement(
    baseline: torch.Tensor,
    hybrid: torch.Tensor,
) -> dict[str, float | int]:
    shared = min(baseline.numel(), hybrid.numel())
    matching = int((baseline[:shared] == hybrid[:shared]).sum().item())
    return {
        "baseline_tokens": int(baseline.numel()),
        "hybrid_tokens": int(hybrid.numel()),
        "shared_tokens": shared,
        "matching_tokens": matching,
        "agreement": matching / shared if shared else 1.0,
        "exact_match": bool(torch.equal(baseline, hybrid)),
    }


def main() -> None:
    args = parse_args()
    if args.layer < 0 or args.expert < 0:
        raise SystemExit("--layer and --expert must be non-negative")
    if args.group_size != 32:
        raise SystemExit("This first hybrid smoke is pinned to group size 32")
    if args.calibration_samples < 1:
        raise SystemExit("--calibration-samples must be at least 1")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This smoke test requires ROCm through torch.cuda")

    snapshot = find_local_snapshot(
        args.cache_root,
        args.model_id,
        args.revision,
    )
    print(f"Snapshot: {snapshot}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Torch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
    print(f"Layer: {args.layer}")
    print(f"Expert: {args.expert}")

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
    if args.layer >= len(decoder_layers):
        raise SystemExit(
            f"Layer {args.layer} is outside 0..{len(decoder_layers) - 1}"
        )
    original_experts = decoder_layers[args.layer].experts
    if args.expert >= original_experts.num_experts:
        raise SystemExit(
            f"Expert {args.expert} is outside "
            f"0..{original_experts.num_experts - 1}"
        )

    captures: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(list)
    handle = original_experts.register_forward_pre_hook(
        capture_hook(args.layer, captures)
    )
    try:
        baseline_inputs = make_inputs(processor, args.prompt, device)
        baseline_output, baseline_seconds = generate(
            model=model,
            inputs=baseline_inputs,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            denoising_steps=args.denoising_steps,
            temperature=args.temperature,
            device=device,
        )
    finally:
        handle.remove()

    baseline_tokens, baseline_text = decode_generated(
        processor,
        baseline_inputs,
        baseline_output,
    )
    print(f"Baseline generation: {baseline_seconds:.3f} s")
    print(f"Baseline text: {baseline_text!r}")
    print(f"Captured expert calls: {len(captures[args.layer])}")

    hidden_states_cpu, route_weights_cpu = routed_samples(
        captures[args.layer],
        args.expert,
        args.calibration_samples,
    )
    hidden_states = hidden_states_cpu.to(
        device=device,
        dtype=torch.bfloat16,
    )
    print(f"Calibration samples: {hidden_states.shape[0]}")

    gate_up_weight = original_experts.gate_up_proj[args.expert]
    down_weight = original_experts.down_proj[args.expert]

    print("Calibrating gate/up...")
    gate_result = quantize_linear_gptq(
        weight=gate_up_weight,
        calibration_inputs=hidden_states,
        group_size=args.group_size,
        calibration_batch_size=128,
        block_size=128,
        damp_percent=0.01,
    )
    gate_values, up_values = F.linear(
        hidden_states,
        gate_result.weight,
    ).chunk(2, dim=-1)
    down_calibration = (
        original_experts.act_fn(gate_values) * up_values
    )

    print("Calibrating down...")
    down_result = quantize_linear_gptq(
        weight=down_weight,
        calibration_inputs=down_calibration,
        group_size=args.group_size,
        calibration_batch_size=128,
        block_size=128,
        damp_percent=0.01,
    )

    print("Building packed Triton Q4 modules...")
    q4_gate = make_runtime(TritonV2Linear, gate_result, device)
    q4_down = make_runtime(TritonV2Linear, down_result, device)

    with torch.inference_mode():
        gate_reference = F.linear(hidden_states, gate_result.weight)
        gate_runtime = q4_gate(hidden_states)
        gate_runtime_metrics = tensor_metrics(
            gate_reference,
            gate_runtime,
        )
        down_reference = F.linear(down_calibration, down_result.weight)
        down_runtime = q4_down(down_calibration)
        down_runtime_metrics = tensor_metrics(
            down_reference,
            down_runtime,
        )

    hybrid_experts = HybridQ4Experts(
        original_experts,
        expert=args.expert,
        gate_up=q4_gate,
        down=q4_down,
    )
    decoder_layers[args.layer].experts = hybrid_experts

    hybrid_inputs = make_inputs(processor, args.prompt, device)
    hybrid_output, hybrid_seconds = generate(
        model=model,
        inputs=hybrid_inputs,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        device=device,
    )
    hybrid_tokens, hybrid_text = decode_generated(
        processor,
        hybrid_inputs,
        hybrid_output,
    )
    agreement = token_agreement(
        baseline_tokens,
        hybrid_tokens,
    )

    print()
    print(f"Hybrid generation: {hybrid_seconds:.3f} s")
    print(f"Hybrid text: {hybrid_text!r}")
    print(
        "Q4 invocation: "
        f"gate_calls={hybrid_experts.q4_gate_calls}, "
        f"down_calls={hybrid_experts.q4_down_calls}, "
        f"tokens={hybrid_experts.q4_tokens}"
    )
    print(
        "Token agreement: "
        f"{agreement['matching_tokens']}/{agreement['shared_tokens']} "
        f"({agreement['agreement']:.2%}), "
        f"exact={agreement['exact_match']}"
    )
    print(
        "Packed gate runtime drift: "
        f"cosine={gate_runtime_metrics['cosine_similarity']:.8f}, "
        f"NRMSE={gate_runtime_metrics['normalized_rmse']:.6f}"
    )
    print(
        "Packed down runtime drift: "
        f"cosine={down_runtime_metrics['cosine_similarity']:.8f}, "
        f"NRMSE={down_runtime_metrics['normalized_rmse']:.6f}"
    )

    payload = {
        "model_id": args.model_id,
        "revision": args.revision,
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name(device),
        "layer": args.layer,
        "expert": args.expert,
        "group_size": args.group_size,
        "calibration_samples": int(hidden_states.shape[0]),
        "load_seconds": load_seconds,
        "baseline_seconds": baseline_seconds,
        "hybrid_seconds": hybrid_seconds,
        "baseline_text": baseline_text,
        "hybrid_text": hybrid_text,
        "token_agreement": agreement,
        "q4_gate_calls": hybrid_experts.q4_gate_calls,
        "q4_down_calls": hybrid_experts.q4_down_calls,
        "q4_tokens": hybrid_experts.q4_tokens,
        "gate_gptq_metadata": gate_result.metadata,
        "down_gptq_metadata": down_result.metadata,
        "gate_runtime_metrics": {
            "cosine": gate_runtime_metrics["cosine_similarity"],
            "nrmse": gate_runtime_metrics["normalized_rmse"],
            "maximum_absolute_error": gate_runtime_metrics[
                "maximum_absolute_error"
            ],
        },
        "down_runtime_metrics": {
            "cosine": down_runtime_metrics["cosine_similarity"],
            "nrmse": down_runtime_metrics["normalized_rmse"],
            "maximum_absolute_error": down_runtime_metrics[
                "maximum_absolute_error"
            ],
        },
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

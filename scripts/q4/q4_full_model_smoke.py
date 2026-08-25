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
import torch.nn.functional as F
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from safetensors.torch import save_file
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from q4_calibration_coverage import load_prompts
from q4_expert_probe import synchronise, tensor_metrics
from q4_full_layer_smoke import (
    FullQ4Experts,
    activation_capture,
    combined_calibration,
    memory_bytes,
)
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
from q4_runtime_roundtrip import checkpoint_tensors

GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one BF16 calibration suite, stream every shared "
            "DiffusionGemma expert layer through GPTQ, release each BF16 layer "
            "immediately, and run an end-to-end all-Q4 smoke test."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("MODELDECK_HF_HUB_CACHE", DEFAULT_CACHE_ROOT)),
    )
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--calibration-denoising-steps", type=int, default=8)
    parser.add_argument("--smoke-denoising-steps", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument(
        "--prompt-limit",
        type=int,
        help="Use only the first N calibration prompts for a shorter diagnostic run.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("var/q4-full-model-smoke.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Optionally write one packed expert safetensors shard per layer "
            "plus q4-manifest.json. The artifact references the pinned base "
            "snapshot for non-expert weights."
        ),
    )
    return parser.parse_args()


def compact_metric(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "maximum": max(values),
    }


def write_progress(
    path: Path,
    *,
    base: dict[str, Any],
    layers: list[dict[str, Any]],
    state: str,
) -> None:
    payload = dict(base)
    payload.update(
        {
            "state": state,
            "completed_layers": len(layers),
            "layers": layers,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_q4_layer(
    checkpoint_dir: Path,
    *,
    layer: int,
    module: FullQ4Experts,
) -> dict[str, Any]:
    tensors: dict[str, torch.Tensor] = {}
    for projection, runtimes in (
        ("gate_up", module.gate_up),
        ("down", module.down),
    ):
        for expert, runtime in enumerate(runtimes):
            for name, tensor in checkpoint_tensors(runtime).items():
                tensors[f"{projection}.{expert}.{name}"] = tensor

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filename = f"experts-layer-{layer:02d}.safetensors"
    path = checkpoint_dir / filename
    save_file(
        tensors,
        str(path),
        metadata={
            "format": "modeldeck-diffusiongemma-expert-gptq-v1",
            "layer": str(layer),
            "bits": "4",
            "group_size": "32",
            "sym": "true",
            "desc_act": "false",
            "qzero_format": "2",
        },
    )
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    file_bytes = path.stat().st_size
    del tensors
    return {
        "layer": layer,
        "file": filename,
        "tensor_bytes": tensor_bytes,
        "file_bytes": file_bytes,
    }


def write_manifest(
    checkpoint_dir: Path,
    *,
    model_id: str,
    revision: str,
    layers: list[dict[str, Any]],
    state: str,
) -> None:
    manifest = {
        "format": "modeldeck-diffusiongemma-expert-gptq",
        "format_version": 1,
        "state": state,
        "base_model_id": model_id,
        "base_model_revision": revision,
        "generation_family": "text-diffusion",
        "dtype": "bfloat16",
        "quantization": {
            "method": "gptq",
            "bits": 4,
            "group_size": 32,
            "symmetric": True,
            "desc_act": False,
            "qzero_format": 2,
            "runtime": "gptqmodel-triton-v2",
        },
        "experts": {
            "layer_count": 30,
            "experts_per_layer": 128,
            "encoder_decoder_storage": "shared",
            "gate_up_shape": [1408, 2816],
            "down_shape": [2816, 704],
            "state_tensors": ["qweight", "qzeros", "scales", "g_idx"],
            "layers": layers,
        },
        "non_expert_weights": {
            "source": "base_model",
            "excluded_suffixes": [
                ".experts.gate_up_proj",
                ".experts.down_proj",
            ],
        },
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "q4-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def quantize_layer(
    *,
    layer: int,
    captures: list[dict[str, torch.Tensor]],
    encoder_layers: Any,
    decoder_layers: Any,
    calibration_samples: int,
    group_size: int,
    device: torch.device,
) -> tuple[FullQ4Experts, dict[str, Any]]:
    encoder_experts = encoder_layers[layer].experts
    decoder_experts = decoder_layers[layer].experts
    if encoder_experts.num_experts != decoder_experts.num_experts:
        raise RuntimeError(f"Layer {layer} encoder/decoder expert counts differ")

    num_experts = int(decoder_experts.num_experts)
    shared_gate = (
        encoder_experts.gate_up_proj.data_ptr()
        == decoder_experts.gate_up_proj.data_ptr()
    )
    shared_down = (
        encoder_experts.down_proj.data_ptr()
        == decoder_experts.down_proj.data_ptr()
    )
    if not shared_gate or not shared_down:
        raise RuntimeError(
            f"Layer {layer} expert storage is not shared between encoder and decoder"
        )

    pooled = torch.cat(
        [capture["hidden_states"] for capture in captures],
        dim=0,
    )
    q4_gate: list[torch.nn.Module] = []
    q4_down: list[torch.nn.Module] = []
    expert_results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for expert in range(num_experts):
        calibration_cpu, routed_count, fallback_count = combined_calibration(
            captures,
            pooled,
            expert,
            calibration_samples,
        )
        calibration = calibration_cpu.to(device=device, dtype=torch.bfloat16)
        gate_weight = decoder_experts.gate_up_proj[expert]
        down_weight = decoder_experts.down_proj[expert]

        gate_result = quantize_linear_gptq(
            weight=gate_weight,
            calibration_inputs=calibration,
            group_size=group_size,
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
            group_size=group_size,
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
        if (expert + 1) % 32 == 0 or expert + 1 == num_experts:
            print(
                f"  Layer {layer}: packed {expert + 1}/{num_experts}; "
                f"last routed={routed_count}, fallback={fallback_count}"
            )

    q4_experts = FullQ4Experts(
        gate_up=q4_gate,
        down=q4_down,
        act_fn=decoder_experts.act_fn,
    )
    packed_memory = memory_bytes(device)
    encoder_layers[layer].experts = q4_experts
    decoder_layers[layer].experts = q4_experts
    del q4_gate, q4_down, encoder_experts, decoder_experts, pooled
    gc.collect()
    torch.cuda.empty_cache()
    converted_memory = memory_bytes(device)

    gate_nrmse = [value["gate_runtime_nrmse"] for value in expert_results]
    down_nrmse = [value["down_runtime_nrmse"] for value in expert_results]
    result = {
        "layer": layer,
        "captured_calls": len(captures),
        "minimum_routed_samples": min(
            value["routed_samples"] for value in expert_results
        ),
        "experts_using_fallback": sum(
            value["fallback_samples"] > 0 for value in expert_results
        ),
        "experts_fully_pooled": sum(
            value["routed_samples"] == 0 for value in expert_results
        ),
        "packed_bytes": sum(value["packed_bytes"] for value in expert_results),
        "packed_plus_bf16_memory_bytes": packed_memory,
        "converted_memory_bytes": converted_memory,
        "wall_seconds": time.perf_counter() - started,
        "gate_runtime_nrmse": compact_metric(gate_nrmse),
        "down_runtime_nrmse": compact_metric(down_nrmse),
        "experts": expert_results,
    }
    return q4_experts, result


def main() -> None:
    args = parse_args()
    if args.group_size != 32:
        raise SystemExit("This full-model smoke is pinned to --group-size 32")
    if args.calibration_samples < 32:
        raise SystemExit("--calibration-samples must be at least 32")
    if args.prompt_limit is not None and args.prompt_limit < 1:
        raise SystemExit("--prompt-limit must be at least 1")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This smoke test requires ROCm through torch.cuda")

    prompts = load_prompts(args.prompts_file)
    if args.prompt_limit is not None:
        prompts = prompts[: args.prompt_limit]
    snapshot = find_local_snapshot(args.cache_root, args.model_id, args.revision)
    print(f"Snapshot: {snapshot}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Torch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
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
    if len(encoder_layers) != len(decoder_layers):
        raise RuntimeError("Encoder and decoder layer counts differ")
    num_layers = len(decoder_layers)
    captures: dict[int, list[dict[str, torch.Tensor]]] = {
        layer: [] for layer in range(num_layers)
    }
    handles = []
    for layer in range(num_layers):
        handles.extend(
            [
                encoder_layers[layer].experts.register_forward_pre_hook(
                    activation_capture(captures[layer])
                ),
                decoder_layers[layer].experts.register_forward_pre_hook(
                    activation_capture(captures[layer])
                ),
            ]
        )

    calibration_generation_seconds: list[float] = []
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
            calibration_generation_seconds.append(elapsed)
            print(
                f"Calibration run {index + 1}/{len(prompts)}: "
                f"{elapsed:.3f} s"
            )
    finally:
        for handle in handles:
            handle.remove()

    total_captured_calls = sum(len(value) for value in captures.values())
    total_captured_rows = sum(
        int(capture["hidden_states"].shape[0])
        for layer_captures in captures.values()
        for capture in layer_captures
    )
    print(f"Captured expert calls: {total_captured_calls}")
    print(f"Captured hidden-state rows: {total_captured_rows}")

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

    base_result = {
        "model_id": args.model_id,
        "revision": args.revision,
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name(device),
        "group_size": args.group_size,
        "calibration_samples": args.calibration_samples,
        "calibration_prompt_count": len(prompts),
        "num_layers": num_layers,
        "load_seconds": load_seconds,
        "loaded_memory_bytes": loaded_memory,
        "calibration_generation_seconds": calibration_generation_seconds,
        "total_captured_calls": total_captured_calls,
        "total_captured_rows": total_captured_rows,
        "baseline_seconds": baseline_seconds,
        "baseline_text": baseline_text,
        "checkpoint_dir": (
            str(args.checkpoint_dir) if args.checkpoint_dir is not None else None
        ),
    }
    layer_results: list[dict[str, Any]] = []
    q4_layers: list[FullQ4Experts] = []
    checkpoint_layers: list[dict[str, Any]] = []
    if args.checkpoint_dir is not None:
        write_manifest(
            args.checkpoint_dir,
            model_id=args.model_id,
            revision=args.revision,
            layers=checkpoint_layers,
            state="converting",
        )
    write_progress(
        args.json_output,
        base=base_result,
        layers=layer_results,
        state="converting",
    )

    conversion_started = time.perf_counter()
    for layer in range(num_layers):
        print()
        print(f"Converting layer {layer + 1}/{num_layers} (index {layer})...")
        q4_experts, result = quantize_layer(
            layer=layer,
            captures=captures[layer],
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            calibration_samples=args.calibration_samples,
            group_size=args.group_size,
            device=device,
        )
        if args.checkpoint_dir is not None:
            checkpoint_result = save_q4_layer(
                args.checkpoint_dir,
                layer=layer,
                module=q4_experts,
            )
            checkpoint_layers.append(checkpoint_result)
            result["checkpoint"] = checkpoint_result
            write_manifest(
                args.checkpoint_dir,
                model_id=args.model_id,
                revision=args.revision,
                layers=checkpoint_layers,
                state="converting",
            )
        q4_layers.append(q4_experts)
        layer_results.append(result)
        del captures[layer]
        gc.collect()
        print(
            f"Layer {layer}: {result['wall_seconds']:.2f} s, "
            f"memory={result['converted_memory_bytes'] / GIB:.3f} GiB, "
            f"fallback={result['experts_using_fallback']}, "
            f"fully pooled={result['experts_fully_pooled']}, "
            f"max gate/down NRMSE="
            f"{result['gate_runtime_nrmse']['maximum']:.6f}/"
            f"{result['down_runtime_nrmse']['maximum']:.6f}"
        )
        write_progress(
            args.json_output,
            base=base_result,
            layers=layer_results,
            state="converting",
        )

    conversion_seconds = time.perf_counter() - conversion_started
    converted_memory = memory_bytes(device)
    memory_saved = loaded_memory - converted_memory
    packed_bytes = sum(value["packed_bytes"] for value in layer_results)
    print()
    print(f"All-layer conversion: {conversion_seconds:.3f} s")
    print(f"Packed expert storage: {packed_bytes / GIB:.3f} GiB")
    print(f"Allocated after conversion: {converted_memory / GIB:.3f} GiB")
    print(f"Net allocated saving: {memory_saved / GIB:.3f} GiB")

    q4_inputs = make_inputs(processor, prompts[0], device)
    q4_output, q4_seconds = generate(
        model=model,
        inputs=q4_inputs,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        denoising_steps=args.smoke_denoising_steps,
        temperature=args.temperature,
        device=device,
    )
    q4_tokens, q4_text = decode_generated(
        processor,
        q4_inputs,
        q4_output,
    )
    agreement = token_agreement(baseline_tokens, q4_tokens)
    gate_calls = sum(layer.q4_gate_calls for layer in q4_layers)
    down_calls = sum(layer.q4_down_calls for layer in q4_layers)
    q4_routed_tokens = sum(layer.q4_tokens for layer in q4_layers)
    print(f"Full Q4 smoke: {q4_seconds:.3f} s")
    print(f"Full Q4 text: {q4_text!r}")
    print(
        "Token agreement: "
        f"{agreement['matching_tokens']}/{agreement['shared_tokens']} "
        f"({agreement['agreement']:.2%}), exact={agreement['exact_match']}"
    )
    print(
        "Q4 invocation: "
        f"gate_calls={gate_calls}, down_calls={down_calls}, "
        f"tokens={q4_routed_tokens}"
    )

    final_result = dict(base_result)
    final_result.update(
        {
            "state": "complete",
            "completed_layers": len(layer_results),
            "layers": layer_results,
            "conversion_seconds": conversion_seconds,
            "packed_expert_bytes": packed_bytes,
            "converted_memory_bytes": converted_memory,
            "memory_saved_bytes": memory_saved,
            "q4_seconds": q4_seconds,
            "q4_text": q4_text,
            "token_agreement": agreement,
            "q4_gate_calls": gate_calls,
            "q4_down_calls": down_calls,
            "q4_tokens": q4_routed_tokens,
            "summary": {
                "experts_using_fallback": sum(
                    value["experts_using_fallback"] for value in layer_results
                ),
                "experts_fully_pooled": sum(
                    value["experts_fully_pooled"] for value in layer_results
                ),
                "minimum_routed_samples": min(
                    value["minimum_routed_samples"] for value in layer_results
                ),
                "maximum_gate_runtime_nrmse": max(
                    value["gate_runtime_nrmse"]["maximum"]
                    for value in layer_results
                ),
                "maximum_down_runtime_nrmse": max(
                    value["down_runtime_nrmse"]["maximum"]
                    for value in layer_results
                ),
            },
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "checkpoint_layers": checkpoint_layers,
        }
    )
    if args.checkpoint_dir is not None:
        write_manifest(
            args.checkpoint_dir,
            model_id=args.model_id,
            revision=args.revision,
            layers=checkpoint_layers,
            state="complete",
        )
    args.json_output.write_text(
        json.dumps(final_result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"JSON results: {args.json_output}")
    print(f"Peak memory: {torch.cuda.max_memory_allocated(device) / GIB:.3f} GiB")


if __name__ == "__main__":
    main()

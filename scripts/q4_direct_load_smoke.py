from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoProcessor,
    DiffusionGemmaForBlockDiffusion,
)

from q4_expert_probe import synchronise
from q4_full_layer_smoke import FullQ4Experts
from q4_hybrid_smoke import decode_generated, generate, make_inputs
from q4_inventory import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    find_local_snapshot,
)
from q4_runtime_roundtrip import restore_runtime

GIB = 1024**3
EXPERT_SUFFIXES = (
    ".experts.gate_up_proj",
    ".experts.down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a ModelDeck DiffusionGemma Q4 expert delta directly onto "
            "ROCm from a meta model plus the pinned base model's non-expert "
            "weights, without materialising BF16 expert weights."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("MODELDECK_HF_HUB_CACHE", DEFAULT_CACHE_ROOT)),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("var/diffusiongemma-26b-a4b-it-gptq-q4-g32"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--denoising-steps", type=int, default=48)
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
        default=Path("var/q4-direct-load-smoke.json"),
    )
    return parser.parse_args()


def load_manifest(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Q4 manifest was not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "modeldeck-diffusiongemma-expert-gptq":
        raise SystemExit("Unsupported Q4 manifest format")
    if manifest.get("format_version") != 1:
        raise SystemExit("Unsupported Q4 manifest version")
    if manifest.get("state") != "complete":
        raise SystemExit("Q4 manifest is not complete")
    if manifest.get("base_model_id") != args.model_id:
        raise SystemExit("Manifest base model does not match --model-id")
    if manifest.get("base_model_revision") != args.revision:
        raise SystemExit("Manifest base revision does not match --revision")
    quantization = manifest.get("quantization", {})
    expected = {
        "method": "gptq",
        "bits": 4,
        "group_size": 32,
        "symmetric": True,
        "desc_act": False,
        "qzero_format": 2,
        "runtime": "gptqmodel-triton-v2",
    }
    if any(quantization.get(key) != value for key, value in expected.items()):
        raise SystemExit("Manifest quantization settings are unsupported")
    return manifest


def get_object(root: Any, name: str) -> Any:
    value = root
    for component in name.split("."):
        value = getattr(value, component)
    return value


def set_object(root: Any, name: str, value: Any) -> None:
    components = name.split(".")
    parent = get_object(root, ".".join(components[:-1]))
    setattr(parent, components[-1], value)


def is_expert_weight(name: str) -> bool:
    return name.endswith(EXPERT_SUFFIXES)


def load_q4_layers(
    *,
    model: DiffusionGemmaForBlockDiffusion,
    checkpoint_dir: Path,
    manifest: dict[str, Any],
    device: torch.device,
) -> tuple[list[FullQ4Experts], int]:
    encoder_layers = model.model.encoder.language_model.layers
    decoder_layers = model.model.decoder.layers
    expert_manifest = manifest["experts"]
    layer_entries = expert_manifest["layers"]
    if len(layer_entries) != len(decoder_layers):
        raise RuntimeError("Manifest does not contain every model layer")

    gate_shape = expert_manifest["gate_up_shape"]
    down_shape = expert_manifest["down_shape"]
    state_names = expert_manifest["state_tensors"]
    experts_per_layer = int(expert_manifest["experts_per_layer"])
    q4_layers: list[FullQ4Experts] = []
    loaded_bytes = 0

    for expected_layer, entry in enumerate(layer_entries):
        layer = int(entry["layer"])
        if layer != expected_layer:
            raise RuntimeError("Manifest layer order is not contiguous")
        path = checkpoint_dir / entry["file"]
        if not path.is_file():
            raise RuntimeError(f"Missing Q4 layer shard: {path}")
        tensors = load_file(str(path), device="cpu")
        q4_gate = []
        q4_down = []
        for expert in range(experts_per_layer):
            gate_tensors = {
                name: tensors[f"gate_up.{expert}.{name}"]
                for name in state_names
            }
            down_tensors = {
                name: tensors[f"down.{expert}.{name}"]
                for name in state_names
            }
            q4_gate.append(
                restore_runtime(
                    gate_tensors,
                    in_features=int(gate_shape[1]),
                    out_features=int(gate_shape[0]),
                    device=device,
                )
            )
            q4_down.append(
                restore_runtime(
                    down_tensors,
                    in_features=int(down_shape[1]),
                    out_features=int(down_shape[0]),
                    device=device,
                )
            )

        act_fn = decoder_layers[layer].experts.act_fn
        q4_experts = FullQ4Experts(
            gate_up=q4_gate,
            down=q4_down,
            act_fn=act_fn,
        )
        encoder_layers[layer].experts = q4_experts
        decoder_layers[layer].experts = q4_experts
        q4_layers.append(q4_experts)
        loaded_bytes += sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors.values()
        )
        del tensors, gate_tensors, down_tensors, q4_gate, q4_down
        gc.collect()
        print(
            f"Loaded Q4 layer {layer + 1}/{len(layer_entries)}; "
            f"allocated={torch.cuda.memory_allocated(device) / GIB:.3f} GiB"
        )

    return q4_layers, loaded_bytes


def base_weight_map(snapshot: Path) -> dict[str, str]:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return {str(key): str(value) for key, value in payload["weight_map"].items()}
    single = snapshot / "model.safetensors"
    if single.is_file():
        with safe_open(single, framework="pt", device="cpu") as handle:
            return {name: single.name for name in handle.keys()}
    raise RuntimeError("The base snapshot has no safetensors checkpoint")


def load_base_non_experts(
    *,
    model: DiffusionGemmaForBlockDiffusion,
    snapshot: Path,
    device: torch.device,
) -> tuple[int, int]:
    weight_map = base_weight_map(snapshot)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        if not is_expert_weight(name):
            by_shard[shard].append(name)

    loaded_tensors = 0
    loaded_bytes = 0
    for index, (shard, names) in enumerate(sorted(by_shard.items())):
        path = snapshot / shard
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in sorted(names):
                tensor = handle.get_tensor(name)
                set_module_tensor_to_device(
                    model,
                    name,
                    device,
                    value=tensor,
                )
                loaded_tensors += 1
                loaded_bytes += tensor.numel() * tensor.element_size()
                del tensor
        gc.collect()
        print(
            f"Loaded base shard {index + 1}/{len(by_shard)}; "
            f"allocated={torch.cuda.memory_allocated(device) / GIB:.3f} GiB"
        )
    return loaded_tensors, loaded_bytes


def tie_encoder_parameters(model: DiffusionGemmaForBlockDiffusion) -> int:
    tied = 0
    prefix = "model.encoder.language_model."
    for name, parameter in list(model.named_parameters(remove_duplicate=False)):
        if not parameter.is_meta or not name.startswith(prefix):
            continue
        decoder_name = "model.decoder." + name[len(prefix) :]
        try:
            source = get_object(model, decoder_name)
        except AttributeError:
            continue
        if isinstance(source, torch.Tensor) and not source.is_meta:
            set_object(model, name, source)
            tied += 1
    return tied


def tie_output_heads(model: DiffusionGemmaForBlockDiffusion) -> int:
    tied = 0
    loaded_embeddings = [
        (name, parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if name.endswith("embed_tokens.weight") and not parameter.is_meta
    ]
    for name, parameter in list(model.named_parameters(remove_duplicate=False)):
        if not parameter.is_meta or not name.endswith("lm_head.weight"):
            continue
        matches = [
            value
            for _, value in loaded_embeddings
            if value.shape == parameter.shape
        ]
        if len(matches) == 1:
            set_object(model, name, matches[0])
            tied += 1
    return tied


def remaining_meta(model: DiffusionGemmaForBlockDiffusion) -> list[str]:
    names = [
        name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.is_meta
    ]
    names.extend(
        name
        for name, buffer in model.named_buffers(remove_duplicate=False)
        if buffer.is_meta
    )
    return sorted(set(names))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This smoke test requires ROCm through torch.cuda")

    snapshot = find_local_snapshot(args.cache_root, args.model_id, args.revision)
    manifest = load_manifest(args.checkpoint_dir / "q4-manifest.json", args)
    print(f"Base snapshot: {snapshot}")
    print(f"Q4 checkpoint: {args.checkpoint_dir}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Torch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    config = AutoConfig.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    processor = AutoProcessor.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    with init_empty_weights(include_buffers=False):
        model = DiffusionGemmaForBlockDiffusion(config)
    model.eval()
    print("Created meta model skeleton")

    q4_layers, q4_bytes = load_q4_layers(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
        manifest=manifest,
        device=device,
    )
    print(f"Loaded packed Q4 experts: {q4_bytes / GIB:.3f} GiB")
    non_expert_tensors, non_expert_bytes = load_base_non_experts(
        model=model,
        snapshot=snapshot,
        device=device,
    )
    tied_encoder = tie_encoder_parameters(model)
    tied_heads = tie_output_heads(model)
    meta_names = remaining_meta(model)
    if meta_names:
        preview = "\n  ".join(meta_names[:40])
        raise RuntimeError(
            f"Direct loader left {len(meta_names)} meta tensors:\n  {preview}"
        )

    model.to(device)
    model.eval()
    synchronise(device)
    load_seconds = time.perf_counter() - started
    loaded_memory = int(torch.cuda.memory_allocated(device))
    print(f"Tied encoder parameters: {tied_encoder}")
    print(f"Tied output heads: {tied_heads}")
    print(f"Loaded non-expert tensors: {non_expert_tensors}")
    print(f"Loaded non-expert storage: {non_expert_bytes / GIB:.3f} GiB")
    print(f"Direct load: {load_seconds:.3f} s")
    print(f"Allocated after direct load: {loaded_memory / GIB:.3f} GiB")

    inputs = make_inputs(processor, args.prompt, device)
    output, generation_seconds = generate(
        model=model,
        inputs=inputs,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        device=device,
    )
    _, text = decode_generated(processor, inputs, output)
    gate_calls = sum(layer.q4_gate_calls for layer in q4_layers)
    down_calls = sum(layer.q4_down_calls for layer in q4_layers)
    routed_tokens = sum(layer.q4_tokens for layer in q4_layers)
    print(f"Generation: {generation_seconds:.3f} s")
    print(f"Generated text: {text!r}")
    print(
        "Q4 invocation: "
        f"gate_calls={gate_calls}, down_calls={down_calls}, "
        f"tokens={routed_tokens}"
    )

    payload = {
        "model_id": args.model_id,
        "revision": args.revision,
        "checkpoint_dir": str(args.checkpoint_dir),
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name(device),
        "q4_bytes": q4_bytes,
        "non_expert_tensors": non_expert_tensors,
        "non_expert_bytes": non_expert_bytes,
        "tied_encoder_parameters": tied_encoder,
        "tied_output_heads": tied_heads,
        "remaining_meta_tensors": meta_names,
        "load_seconds": load_seconds,
        "loaded_memory_bytes": loaded_memory,
        "generation_seconds": generation_seconds,
        "text": text,
        "q4_gate_calls": gate_calls,
        "q4_down_calls": down_calls,
        "q4_tokens": routed_tokens,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"JSON results: {args.json_output}")
    print(f"Peak memory: {payload['peak_memory_bytes'] / GIB:.3f} GiB")


if __name__ == "__main__":
    main()

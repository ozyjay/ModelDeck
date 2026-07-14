from __future__ import annotations

import gc
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoProcessor, DiffusionGemmaForBlockDiffusion

GIB = 1024**3
EXPERT_SUFFIXES = (
    ".experts.gate_up_proj",
    ".experts.down_proj",
)
EXPECTED_QUANTIZATION = {
    "method": "gptq",
    "bits": 4,
    "group_size": 32,
    "symmetric": True,
    "desc_act": False,
    "qzero_format": 2,
    "runtime": "gptqmodel-triton-v2",
}


@dataclass(frozen=True)
class Q4LoadResult:
    processor: Any
    model: DiffusionGemmaForBlockDiffusion
    q4_layers: list[FullQ4Experts]
    details: dict[str, Any]


class FullQ4Experts(nn.Module):
    def __init__(
        self,
        *,
        gate_up: list[nn.Module],
        down: list[nn.Module],
        act_fn: nn.Module,
    ) -> None:
        super().__init__()
        if len(gate_up) != len(down):
            raise ValueError("Q4 gate/up and down expert counts differ")
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
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_tensor in expert_hit:
            expert = int(expert_tensor[0].item())
            top_k_position, token_index = torch.where(expert_mask[expert])
            current_state = hidden_states[token_index]
            gate, up = self.gate_up[expert](current_state).chunk(2, dim=-1)
            intermediate = self.act_fn(gate) * up
            current_hidden_states = self.down[expert](intermediate)
            current_hidden_states = current_hidden_states * top_k_weights[token_index, top_k_position, None]
            final_hidden_states.index_add_(
                0,
                token_index,
                current_hidden_states.to(final_hidden_states.dtype),
            )
            self.q4_gate_calls += 1
            self.q4_down_calls += 1
            self.q4_tokens += int(current_state.shape[0])

        return final_hidden_states


def find_local_snapshot(cache_root: Path, model_id: str, revision: str) -> Path:
    repository = cache_root / f"models--{model_id.replace('/', '--')}"
    snapshot = repository / "snapshots" / revision
    if snapshot.is_dir():
        return snapshot
    raise RuntimeError(
        f"Pinned local snapshot was not found beneath {repository / 'snapshots'} for revision {revision}"
    )


def load_manifest(
    path: Path,
    *,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Q4 manifest was not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "modeldeck-diffusiongemma-expert-gptq":
        raise RuntimeError("Unsupported Q4 manifest format")
    format_version = manifest.get("format_version")
    if format_version not in {1, 2}:
        raise RuntimeError("Unsupported Q4 manifest version")
    if manifest.get("state") != "complete":
        raise RuntimeError("Q4 manifest is not complete")
    if manifest.get("base_model_id") != model_id:
        raise RuntimeError("Manifest base model does not match the worker model")
    if manifest.get("base_model_revision") != revision:
        raise RuntimeError("Manifest base revision does not match the worker revision")
    quantization = manifest.get("quantization", {})
    if any(quantization.get(key) != value for key, value in EXPECTED_QUANTIZATION.items()):
        raise RuntimeError("Manifest quantization settings are unsupported")
    if format_version == 2:
        if manifest.get("artifact_type") != "self-contained":
            raise RuntimeError("Q4 manifest version 2 must describe a self-contained artifact")
        non_experts = manifest.get("non_expert_weights")
        if not isinstance(non_experts, dict) or non_experts.get("source") != "checkpoint":
            raise RuntimeError("Self-contained Q4 manifest has no packaged non-expert weights")
        _safe_flat_path(non_experts.get("index_file"), label="non-expert index")
    return manifest


def _safe_flat_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str):
        raise RuntimeError(f"Manifest {label} path is missing")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise RuntimeError(f"Manifest {label} path is unsafe: {value!r}")
    return Path(path.name)


def _runtime_constructor(*, in_features: int, out_features: int) -> TritonV2Linear:
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


def _restore_runtime(
    tensors: dict[str, torch.Tensor],
    *,
    in_features: int,
    out_features: int,
    device: torch.device,
) -> TritonV2Linear:
    runtime = _runtime_constructor(in_features=in_features, out_features=out_features)
    for name, tensor in tensors.items():
        if "." in name:
            raise RuntimeError(f"Unexpected nested packed state key: {name}")
        if hasattr(runtime, name):
            delattr(runtime, name)
        runtime.register_buffer(name, tensor.to(device))
    runtime.qzero_format(format=2)
    runtime.eval()
    return runtime


def _is_expert_weight(name: str) -> bool:
    return name.endswith(EXPERT_SUFFIXES)


def _is_encoder_mirror(name: str) -> bool:
    return name.startswith("model.encoder.language_model.")


def _base_weight_map(snapshot: Path, *, index_file: str = "model.safetensors.index.json") -> dict[str, str]:
    index_path = snapshot / _safe_flat_path(index_file, label="weight index")
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"Safetensors index has no weight map: {index_path}")
        result: dict[str, str] = {}
        for key, value in weight_map.items():
            relative = _safe_flat_path(value, label="weight shard")
            result[str(key)] = relative.as_posix()
        return result
    single_name = "model.safetensors" if index_file == "model.safetensors.index.json" else ""
    single = snapshot / single_name
    if single.is_file():
        with safe_open(single, framework="pt", device="cpu") as handle:
            return {name: single.name for name in handle.keys()}
    raise RuntimeError("The base snapshot has no safetensors checkpoint")


def _load_base_non_experts(
    *,
    model: DiffusionGemmaForBlockDiffusion,
    snapshot: Path,
    device: torch.device,
    index_file: str = "model.safetensors.index.json",
) -> tuple[int, int]:
    weight_map = _base_weight_map(snapshot, index_file=index_file)
    buffer_names = {name for name, _ in model.named_buffers(remove_duplicate=False)}
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        skip_tied_encoder_parameter = _is_encoder_mirror(name) and name not in buffer_names
        if not _is_expert_weight(name) and not skip_tied_encoder_parameter:
            by_shard[shard].append(name)

    loaded_tensors = 0
    loaded_bytes = 0
    for shard, names in sorted(by_shard.items()):
        path = snapshot / _safe_flat_path(shard, label="weight shard")
        if not path.is_file():
            raise RuntimeError(f"Non-expert weight shard is missing: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in sorted(names):
                tensor = handle.get_tensor(name)
                set_module_tensor_to_device(
                    model,
                    name,
                    device,
                    value=tensor,
                    dtype=tensor.dtype,
                )
                loaded_tensors += 1
                loaded_bytes += tensor.numel() * tensor.element_size()
                del tensor
        gc.collect()
    return loaded_tensors, loaded_bytes


def _load_q4_layers(
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
    if set(state_names) != {"g_idx", "qweight", "qzeros", "scales"}:
        raise RuntimeError("Manifest packed state tensor inventory is unsupported")
    experts_per_layer = int(expert_manifest["experts_per_layer"])
    q4_layers: list[FullQ4Experts] = []
    loaded_bytes = 0

    for expected_layer, entry in enumerate(layer_entries):
        layer = int(entry["layer"])
        if layer != expected_layer:
            raise RuntimeError("Manifest layer order is not contiguous")
        path = checkpoint_dir / _safe_flat_path(entry.get("file"), label="expert shard")
        if not path.is_file():
            raise RuntimeError(f"Missing Q4 layer shard: {path}")
        tensors = load_file(str(path), device="cpu")
        q4_gate = []
        q4_down = []
        for expert in range(experts_per_layer):
            gate_tensors = {name: tensors[f"gate_up.{expert}.{name}"] for name in state_names}
            down_tensors = {name: tensors[f"down.{expert}.{name}"] for name in state_names}
            q4_gate.append(
                _restore_runtime(
                    gate_tensors,
                    in_features=int(gate_shape[1]),
                    out_features=int(gate_shape[0]),
                    device=device,
                )
            )
            q4_down.append(
                _restore_runtime(
                    down_tensors,
                    in_features=int(down_shape[1]),
                    out_features=int(down_shape[0]),
                    device=device,
                )
            )

        q4_experts = FullQ4Experts(
            gate_up=q4_gate,
            down=q4_down,
            act_fn=decoder_layers[layer].experts.act_fn,
        )
        encoder_layers[layer].experts = q4_experts
        decoder_layers[layer].experts = q4_experts
        q4_layers.append(q4_experts)
        loaded_bytes += sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        del tensors, gate_tensors, down_tensors, q4_gate, q4_down
        gc.collect()

    return q4_layers, loaded_bytes


def _remaining_meta(model: DiffusionGemmaForBlockDiffusion) -> list[str]:
    names = [name for name, parameter in model.named_parameters(remove_duplicate=False) if parameter.is_meta]
    names.extend(name for name, buffer in model.named_buffers(remove_duplicate=False) if buffer.is_meta)
    return sorted(set(names))


def _parameter_dtype_summary(model: DiffusionGemmaForBlockDiffusion) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for parameter in model.parameters():
        counts[str(parameter.dtype)] += parameter.numel()
    return dict(sorted(counts.items()))


def load_diffusiongemma_q4(
    *,
    model_id: str,
    revision: str,
    cache_root: Path | None,
    checkpoint_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Q4LoadResult:
    if dtype != torch.bfloat16:
        raise RuntimeError("The DiffusionGemma Q4 runtime requires bfloat16 non-expert weights")
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    manifest = load_manifest(
        checkpoint_dir / "q4-manifest.json",
        model_id=model_id,
        revision=revision,
    )
    self_contained = manifest["format_version"] == 2
    if self_contained:
        snapshot = checkpoint_dir
        non_expert_index = manifest["non_expert_weights"]["index_file"]
        weight_source = "checkpoint"
    else:
        if cache_root is None:
            raise RuntimeError("The expert-delta Q4 checkpoint requires the pinned base cache")
        snapshot = find_local_snapshot(cache_root, model_id, revision)
        non_expert_index = "model.safetensors.index.json"
        weight_source = "pinned-base-cache"
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
    original_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with init_empty_weights(include_buffers=False):
            model = DiffusionGemmaForBlockDiffusion(config)
    finally:
        torch.set_default_dtype(original_default_dtype)
    model.eval()

    generation_config_path = snapshot / "generation_config.json"
    if generation_config_path.is_file():
        model.generation_config = model.generation_config_class.from_pretrained(
            snapshot,
            local_files_only=True,
        )

    non_expert_tensors, non_expert_bytes = _load_base_non_experts(
        model=model,
        snapshot=snapshot,
        device=device,
        index_file=non_expert_index,
    )
    meta_before_tie = set(_remaining_meta(model))
    model.model.tie_weights()
    model.tie_weights()
    native_tied_tensors = len(meta_before_tie - set(_remaining_meta(model)))
    q4_layers, q4_bytes = _load_q4_layers(
        model=model,
        checkpoint_dir=checkpoint_dir,
        manifest=manifest,
        device=device,
    )

    meta_names = _remaining_meta(model)
    if meta_names:
        preview = "\n  ".join(meta_names[:40])
        raise RuntimeError(f"Direct Q4 loader left {len(meta_names)} meta tensors:\n  {preview}")
    unexpected_parameters = [
        name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16
    ]
    if unexpected_parameters:
        preview = "\n  ".join(unexpected_parameters[:40])
        raise RuntimeError("Direct Q4 loader produced non-BF16 parameters:\n  " + preview)

    model.to(device)
    model.eval()
    return Q4LoadResult(
        processor=processor,
        model=model,
        q4_layers=q4_layers,
        details={
            "runtime": "text-diffusion-gptq-rocm",
            "quantization": "gptq-q4-g32-expert-only",
            "artifact_type": (
                "self-contained-q4-bf16-hybrid" if self_contained else "expert-only-quantized-weight-delta"
            ),
            "weight_source": weight_source,
            "base_model_runtime_dependency": not self_contained,
            "q4_checkpoint_dir": str(checkpoint_dir),
            "q4_bytes": q4_bytes,
            "non_expert_tensors": non_expert_tensors,
            "non_expert_bytes": non_expert_bytes,
            "native_tied_tensors": native_tied_tensors,
            "parameter_dtypes": _parameter_dtype_summary(model),
        },
    )


def q4_invocation_metrics(layers: list[FullQ4Experts]) -> dict[str, int]:
    return {
        "q4_gate_calls": sum(layer.q4_gate_calls for layer in layers),
        "q4_down_calls": sum(layer.q4_down_calls for layer in layers),
        "q4_tokens": sum(layer.q4_tokens for layer in layers),
    }

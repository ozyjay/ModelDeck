from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from accelerate import init_empty_weights
from q4_inventory import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    find_local_snapshot,
)
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoProcessor,
    DiffusionGemmaForBlockDiffusion,
    GenerationConfig,
)

DEFAULT_CHECKPOINT_DIR = Path("var/diffusiongemma-26b-a4b-it-gptq-q4-g32")
EXPERT_SUFFIXES = (
    ".experts.gate_up_proj",
    ".experts.down_proj",
)
NON_EXPERT_INDEX = "non-expert-model.safetensors.index.json"


class MaterializationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialise a self-contained DiffusionGemma Q4/BF16 hybrid from an "
            "existing ModelDeck expert-only Q4 checkpoint and the pinned base snapshot."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("MODELDECK_HF_HUB_CACHE", DEFAULT_CACHE_ROOT)),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory; defaults to upgrading --checkpoint-dir in place.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaterializationError(f"Could not read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise MaterializationError(f"Expected a JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_flat_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str):
        raise MaterializationError(f"{label} path is missing")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise MaterializationError(f"Unsafe {label} path: {value!r}")
    return Path(path.name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def base_weight_map(snapshot: Path) -> dict[str, str]:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        payload = read_json(index_path)
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise MaterializationError("The base Safetensors index has no weight map")
        return {str(name): str(shard) for name, shard in weight_map.items()}
    single = snapshot / "model.safetensors"
    if single.is_file():
        with safe_open(single, framework="pt", device="cpu") as handle:
            return {name: single.name for name in handle.keys()}
    raise MaterializationError("The pinned base snapshot has no Safetensors checkpoint")


def validate_source_manifest(
    manifest: dict[str, Any],
    *,
    model_id: str,
    revision: str,
) -> None:
    if manifest.get("format") != "modeldeck-diffusiongemma-expert-gptq":
        raise MaterializationError("Unsupported Q4 checkpoint format")
    if manifest.get("format_version") not in {1, 2}:
        raise MaterializationError("Unsupported Q4 checkpoint version")
    if manifest.get("state") != "complete":
        raise MaterializationError("The Q4 checkpoint is not complete")
    if manifest.get("base_model_id") != model_id:
        raise MaterializationError("The Q4 checkpoint references a different base model")
    if manifest.get("base_model_revision") != revision:
        raise MaterializationError("The Q4 checkpoint references a different base revision")
    layers = manifest.get("experts", {}).get("layers")
    if not isinstance(layers, list) or len(layers) != 30:
        raise MaterializationError("The Q4 checkpoint does not contain all 30 expert layers")


def copy_expert_shards(
    source: Path,
    destination: Path,
    manifest: dict[str, Any],
) -> None:
    if source == destination:
        return
    for entry in manifest["experts"]["layers"]:
        relative = safe_flat_path(entry.get("file"), label="expert shard")
        source_path = source / relative
        destination_path = destination / relative
        if not source_path.is_file():
            raise MaterializationError(f"Expert shard is missing: {source_path}")
        if destination_path.exists():
            if destination_path.stat().st_size != source_path.stat().st_size:
                raise MaterializationError(f"Existing destination shard differs: {destination_path}")
            continue
        try:
            os.link(source_path, destination_path)
        except OSError:
            shutil.copy2(source_path, destination_path)


def model_buffer_names(snapshot: Path) -> set[str]:
    config = AutoConfig.from_pretrained(
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
    names = {name for name, _ in model.named_buffers(remove_duplicate=False)}
    del model
    gc.collect()
    return names


def is_expert_weight(name: str) -> bool:
    return name.endswith(EXPERT_SUFFIXES)


def selected_non_experts(snapshot: Path) -> dict[str, list[str]]:
    buffer_names = model_buffer_names(snapshot)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in base_weight_map(snapshot).items():
        encoder_mirror = name.startswith("model.encoder.language_model.")
        skip_tied_encoder_parameter = encoder_mirror and name not in buffer_names
        if not is_expert_weight(name) and not skip_tied_encoder_parameter:
            relative = safe_flat_path(shard, label="base weight shard")
            by_shard[relative.as_posix()].append(name)
    if not by_shard:
        raise MaterializationError("No non-expert weights were selected from the base model")
    return dict(sorted(by_shard.items()))


def write_non_expert_shards(
    *,
    snapshot: Path,
    staging: Path,
    selected: dict[str, list[str]],
    model_id: str,
    revision: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    weight_map: dict[str, str] = {}
    shard_entries: list[dict[str, Any]] = []
    total_tensor_bytes = 0
    total_tensors = 0
    shard_count = len(selected)

    for index, (source_shard, names) in enumerate(selected.items(), 1):
        filename = f"non-expert-model-{index:05d}-of-{shard_count:05d}.safetensors"
        output = staging / filename
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(snapshot / source_shard, framework="pt", device="cpu") as handle:
            for name in sorted(names):
                tensors[name] = handle.get_tensor(name)
        save_file(
            tensors,
            str(output),
            metadata={
                "format": "modeldeck-diffusiongemma-non-expert-bf16-v1",
                "base_model": model_id,
                "base_revision": revision,
            },
        )
        tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        for name in tensors:
            weight_map[name] = filename
        entry = {
            "file": filename,
            "tensor_count": len(tensors),
            "tensor_bytes": tensor_bytes,
            "file_bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        }
        shard_entries.append(entry)
        total_tensor_bytes += tensor_bytes
        total_tensors += len(tensors)
        print(
            f"Wrote non-expert shard {index}/{shard_count}: {filename}, "
            f"tensors={len(tensors)}, bytes={output.stat().st_size}"
        )
        del tensors
        gc.collect()

    index_payload = {
        "metadata": {
            "total_size": total_tensor_bytes,
            "tensor_count": total_tensors,
            "format": "modeldeck-diffusiongemma-non-expert-bf16-v1",
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    write_json_atomic(staging / NON_EXPERT_INDEX, index_payload)
    summary = {
        "source": "checkpoint",
        "format": "safetensors",
        "index_file": NON_EXPERT_INDEX,
        "tensor_count": total_tensors,
        "tensor_bytes": total_tensor_bytes,
        "shard_count": shard_count,
        "shards": shard_entries,
        "excluded_suffixes": list(EXPERT_SUFFIXES),
    }
    return summary, shard_entries


def write_runtime_files(snapshot: Path, staging: Path) -> list[dict[str, Any]]:
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    generation_config = GenerationConfig.from_pretrained(snapshot, local_files_only=True)
    config.save_pretrained(staging)
    processor.save_pretrained(staging)
    generation_config.save_pretrained(staging)
    model_index = snapshot / "model_index.json"
    if model_index.is_file():
        shutil.copy2(model_index, staging / model_index.name)

    paths = sorted(
        path
        for path in staging.iterdir()
        if path.is_file() and path.name != NON_EXPERT_INDEX and not path.name.startswith("non-expert-model-")
    )
    names = {path.name for path in paths}
    required = {"config.json", "generation_config.json", "tokenizer_config.json"}
    missing = sorted(required - names)
    if missing:
        raise MaterializationError(f"Processor export omitted required runtime files: {missing}")
    return [
        {
            "file": path.name,
            "file_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def move_staged_files(staging: Path, output_dir: Path) -> None:
    for source in sorted(staging.iterdir()):
        if source.is_file():
            source.replace(output_dir / source.name)


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output_dir = (args.output_dir or checkpoint_dir).expanduser().resolve()
    snapshot = find_local_snapshot(args.cache_root, args.model_id, args.revision)
    manifest_path = checkpoint_dir / "q4-manifest.json"
    source_manifest = read_json(manifest_path)
    validate_source_manifest(
        source_manifest,
        model_id=args.model_id,
        revision=args.revision,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_expert_shards(checkpoint_dir, output_dir, source_manifest)
    selected = selected_non_experts(snapshot)
    print(f"Base snapshot: {snapshot}")
    print(f"Source Q4 checkpoint: {checkpoint_dir}")
    print(f"Self-contained output: {output_dir}")
    print(f"Selected source shards: {len(selected)}")

    with tempfile.TemporaryDirectory(prefix=".q4-self-contained-", dir=output_dir) as temporary:
        staging = Path(temporary)
        non_experts, _ = write_non_expert_shards(
            snapshot=snapshot,
            staging=staging,
            selected=selected,
            model_id=args.model_id,
            revision=args.revision,
        )
        runtime_files = write_runtime_files(snapshot, staging)
        move_staged_files(staging, output_dir)

    manifest = dict(source_manifest)
    manifest.update(
        {
            "format_version": 2,
            "artifact_type": "self-contained",
            "state": "complete",
            "non_expert_weights": non_experts,
            "runtime_files": runtime_files,
        }
    )
    write_json_atomic(output_dir / "q4-manifest.json", manifest)

    referenced_non_expert_files = {entry["file"] for entry in non_experts["shards"]}
    for stale in output_dir.glob("non-expert-model-*.safetensors"):
        if stale.name not in referenced_non_expert_files:
            stale.unlink()

    summary = {
        "format_version": 2,
        "artifact_type": "self-contained",
        "checkpoint_dir": str(output_dir),
        "base_model": {"id": args.model_id, "revision": args.revision},
        "expert_shards": len(manifest["experts"]["layers"]),
        "non_expert_shards": non_experts["shard_count"],
        "non_expert_tensors": non_experts["tensor_count"],
        "non_expert_bytes": non_experts["tensor_bytes"],
        "runtime_files": len(runtime_files),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except MaterializationError as error:
        raise SystemExit(str(error)) from error

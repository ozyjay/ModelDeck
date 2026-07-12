from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig

DEFAULT_MODEL_ID = "google/diffusiongemma-26B-A4B-it"
DEFAULT_REVISION = "52de6b914ee1749a7d4933202505ddf5b414ec43"
DEFAULT_CACHE_ROOT = "/mnt/work/models/huggingface/hub"
GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a locally cached DiffusionGemma checkpoint and estimate the "
            "storage impact of group-wise Q4 packed-expert quantisation. Weight "
            "tensors are not materialised."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("HF_HUB_CACHE", DEFAULT_CACHE_ROOT)),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Q4 scale group size along the input dimension.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=12,
        help="Maximum number of packed expert tensors to print.",
    )
    return parser.parse_args()


def find_local_snapshot(cache_root: Path, model_id: str, revision: str) -> Path:
    """Locate a pinned Hub snapshot without requiring every repository file."""
    repository = cache_root / f"models--{model_id.replace('/', '--')}"
    snapshot = repository / "snapshots" / revision
    if snapshot.is_dir():
        return snapshot.resolve()

    reference = repository / "refs" / revision
    if reference.is_file():
        resolved_revision = reference.read_text(encoding="utf-8").strip()
        referenced_snapshot = repository / "snapshots" / resolved_revision
        if referenced_snapshot.is_dir():
            return referenced_snapshot.resolve()

    raise SystemExit(
        "Local model snapshot was not found. Expected it beneath "
        f"{repository / 'snapshots'} for revision {revision}."
    )


def q4_storage_bytes(shape: tuple[int, ...], group_size: int) -> int:
    """Estimate symmetric INT4 payload plus one BF16 scale per input group."""
    parameters = math.prod(shape)
    packed_weights = (parameters + 1) // 2
    rows = math.prod(shape[:-1])
    scales_per_row = math.ceil(shape[-1] / group_size)
    bf16_scales = rows * scales_per_row * 2
    return packed_weights + bf16_scales


def main() -> None:
    args = parse_args()
    if args.group_size < 1:
        raise SystemExit("--group-size must be at least 1")
    if args.show < 0:
        raise SystemExit("--show must be non-negative")

    snapshot = find_local_snapshot(args.cache_root, args.model_id, args.revision)
    config = AutoConfig.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )

    shards = sorted(snapshot.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No safetensors shards found in {snapshot}")

    total_parameters = 0
    packed_expert_parameters = 0
    packed_expert_q4_bytes = 0
    packed_tensors: list[tuple[str, tuple[int, ...], int]] = []

    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as weights:
            for name in weights.keys():
                shape = tuple(weights.get_slice(name).get_shape())
                parameters = math.prod(shape)
                total_parameters += parameters

                is_packed_expert = len(shape) == 3 and (
                    "gate_up_proj" in name or "down_proj" in name
                )
                if not is_packed_expert:
                    continue

                packed_expert_parameters += parameters
                packed_expert_q4_bytes += q4_storage_bytes(shape, args.group_size)
                packed_tensors.append((name, shape, parameters))

    total_bf16_bytes = total_parameters * 2
    packed_expert_bf16_bytes = packed_expert_parameters * 2
    expert_fraction = (
        packed_expert_parameters / total_parameters if total_parameters else 0.0
    )

    print(f"Snapshot: {snapshot}")
    print(f"Config class: {type(config).__name__}")
    print(f"Model type: {config.model_type}")
    print(f"Safetensors shards: {len(shards)}")
    print()
    print(f"Total parameters: {total_parameters:,}")
    print(f"Estimated complete BF16 weights: {total_bf16_bytes / GIB:.2f} GiB")
    print(f"Packed expert parameters: {packed_expert_parameters:,}")
    print(f"Packed expert proportion: {expert_fraction:.2%}")
    print(f"Packed experts as BF16: {packed_expert_bf16_bytes / GIB:.2f} GiB")
    print(
        f"Packed experts as Q4 g{args.group_size}: "
        f"{packed_expert_q4_bytes / GIB:.2f} GiB"
    )
    print(
        "Estimated expert-only saving: "
        f"{(packed_expert_bf16_bytes - packed_expert_q4_bytes) / GIB:.2f} GiB"
    )
    print()
    print(f"Packed expert tensor count: {len(packed_tensors)}")

    for name, shape, parameters in packed_tensors[: args.show]:
        print(f"{name}: {shape} = {parameters:,}")


if __name__ == "__main__":
    main()

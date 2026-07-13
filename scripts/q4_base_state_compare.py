from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, DiffusionGemmaForBlockDiffusion

from q4_direct_load_smoke import (
    is_expert_weight,
    load_base_non_experts,
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
            "Compare sampled non-expert parameters and buffers from the Q4 "
            "meta reconstruction against a normal BF16 from_pretrained load."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("MODELDECK_HF_HUB_CACHE", DEFAULT_CACHE_ROOT)),
    )
    parser.add_argument("--samples", type=int, default=257)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("var/q4-base-state-compare.json"),
    )
    return parser.parse_args()


def tensor_signature(tensor: torch.Tensor, samples: int) -> dict[str, Any]:
    flat = tensor.detach().reshape(-1)
    count = min(samples, flat.numel())
    if count == 0:
        values: list[Any] = []
    elif flat.numel() == count:
        values = flat.to(device="cpu", dtype=torch.float64).tolist()
    else:
        indices = torch.linspace(
            0,
            flat.numel() - 1,
            steps=count,
            device=flat.device,
            dtype=torch.float64,
        ).round().to(torch.long)
        values = flat[indices].to(device="cpu", dtype=torch.float64).tolist()
    digest = hashlib.sha256(
        json.dumps(values, allow_nan=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "sample_count": count,
        "sample_sha256": digest,
        "sample_min": min(values) if values else None,
        "sample_max": max(values) if values else None,
    }


def collect_state(
    model: DiffusionGemmaForBlockDiffusion,
    samples: int,
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for kind, entries in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in entries:
            if is_expert_weight(name) or ".experts." in name or tensor.is_meta:
                continue
            values[f"{kind}:{name}"] = tensor_signature(tensor, samples)
    return values


def compare_states(
    direct: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    direct_keys = set(direct)
    baseline_keys = set(baseline)
    shared = sorted(direct_keys & baseline_keys)
    mismatches = [
        {
            "name": name,
            "direct": direct[name],
            "baseline": baseline[name],
        }
        for name in shared
        if direct[name] != baseline[name]
    ]
    return {
        "direct_items": len(direct),
        "baseline_items": len(baseline),
        "shared_items": len(shared),
        "missing_from_direct": sorted(baseline_keys - direct_keys),
        "missing_from_baseline": sorted(direct_keys - baseline_keys),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("This comparison requires ROCm through torch.cuda")

    snapshot = find_local_snapshot(args.cache_root, args.model_id, args.revision)
    print(f"Snapshot: {snapshot}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Samples per tensor: {args.samples}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    config = AutoConfig.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    original_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with init_empty_weights(include_buffers=False):
            direct_model = DiffusionGemmaForBlockDiffusion(config)
    finally:
        torch.set_default_dtype(original_default_dtype)
    direct_model.eval()
    direct_tensors, direct_bytes = load_base_non_experts(
        model=direct_model,
        snapshot=snapshot,
        device=device,
    )
    direct_model.model.tie_weights()
    direct_model.tie_weights()
    direct_state = collect_state(direct_model, args.samples)
    direct_memory = int(torch.cuda.memory_allocated(device))
    print(f"Direct items: {len(direct_state)}")
    print(f"Direct allocation: {direct_memory / GIB:.3f} GiB")

    del direct_model, config
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.memory_allocated(device) != 0:
        print(
            "Warning: allocation after direct release: "
            f"{torch.cuda.memory_allocated(device) / GIB:.3f} GiB"
        )

    baseline_model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
    )
    baseline_model.to(device)
    baseline_model.eval()
    baseline_state = collect_state(baseline_model, args.samples)
    baseline_memory = int(torch.cuda.memory_allocated(device))
    print(f"Baseline items: {len(baseline_state)}")
    print(f"Baseline allocation: {baseline_memory / GIB:.3f} GiB")

    comparison = compare_states(direct_state, baseline_state)
    print(f"State mismatches: {comparison['mismatch_count']}")
    for mismatch in comparison["mismatches"][:40]:
        print(
            f"  {mismatch['name']}: "
            f"direct={mismatch['direct']['sample_sha256'][:12]} "
            f"baseline={mismatch['baseline']['sample_sha256'][:12]}"
        )
    if comparison["missing_from_direct"]:
        print(f"Missing from direct: {len(comparison['missing_from_direct'])}")
        for name in comparison["missing_from_direct"][:40]:
            print(f"  {name}")

    payload = {
        "model_id": args.model_id,
        "revision": args.revision,
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name(device),
        "samples_per_tensor": args.samples,
        "direct_loaded_tensors": direct_tensors,
        "direct_loaded_bytes": direct_bytes,
        "direct_memory_bytes": direct_memory,
        "baseline_memory_bytes": baseline_memory,
        "comparison": comparison,
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

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from modeldeck.fp8_kernel import (
    DEFAULT_CONFIG,
    KERNEL_COMMIT,
    KERNEL_REPO_ID,
    KERNEL_VERSION,
    QWEN38_FP8_WEIGHT_SHAPES,
    TUNING_MANIFEST_VERSION,
    runtime_fingerprint,
    triton_cache_path,
    tuning_manifest_path,
    validate_fp8_kernel_snapshot,
    write_tuning_manifest,
)

M_BUCKET_INPUTS = {16: 1, 32: 24, 64: 48, 128: 96}
CANDIDATES = tuple(
    {"num_warps": num_warps, "num_stages": num_stages}
    for num_warps in (2, 4, 8, 16)
    for num_stages in (2, 3, 4)
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune the reviewed Qwen3.8 ROCm FP8 kernel")
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("MODELDECK_DATA_DIR", ".modeldeck")),
    )
    parser.add_argument("--stage", choices=("decode", "full"), default="full")
    parser.add_argument("--candidate-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--secondary-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--candidate-json", help=argparse.SUPPRESS)
    return parser


def _candidate_process(arguments: argparse.Namespace) -> int:
    request = json.loads(arguments.candidate_json)
    import torch
    import triton

    validated = validate_fp8_kernel_snapshot(arguments.cache_root)
    os.environ["LOCAL_KERNELS"] = f"{KERNEL_REPO_ID}={validated.snapshot_path}"
    os.environ["KERNELS_CACHE"] = str(arguments.cache_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from kernels import get_kernel

    kernel = get_kernel(KERNEL_REPO_ID, version=KERNEL_VERSION)
    matmul_module = __import__(f"{kernel.__package__}.matmul", fromlist=["matmul"])
    autotuner = matmul_module.w8a8_block_dynamic_fp8_matmul_kernel
    autotuner.configs = [
        triton.Config(
            {},
            num_warps=int(request["num_warps"]),
            num_stages=int(request["num_stages"]),
        )
    ]

    torch.manual_seed(7)
    m, n, k = int(request["m"]), int(request["n"]), int(request["k"])
    activation = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).contiguous()
    weight = (torch.randn((n, k), device="cuda") * 0.2).to(torch.float8_e4m3fn).contiguous()
    scales = (torch.rand((math.ceil(n / 128), math.ceil(k / 128)), device="cuda") * 0.05 + 0.01)
    scales = scales.contiguous()

    for _ in range(arguments.warmups):
        output = kernel.matmul_2d(activation, weight, scales, [128, 128], torch.bfloat16)
        torch.cuda.synchronize()
    timings = []
    for _ in range(arguments.repetitions):
        started = time.perf_counter()
        output = kernel.matmul_2d(activation, weight, scales, [128, 128], torch.bfloat16)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1_000)

    if not torch.isfinite(output).all():
        raise RuntimeError("FP8 candidate produced non-finite output")
    expanded_scales = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
    reference = activation.float() @ (weight.float() * expanded_scales).T
    mean_absolute_error = float((output.float() - reference).abs().mean().item())
    reference_magnitude = max(float(reference.abs().mean().item()), 1e-6)
    relative_error = mean_absolute_error / reference_magnitude
    if relative_error > 0.12:
        raise RuntimeError(f"FP8 candidate relative error {relative_error:.4f} exceeded 0.12")
    print(
        json.dumps(
            {
                "median_ms": statistics.median(timings),
                "minimum_ms": min(timings),
                "relative_error": relative_error,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_candidate(
    arguments: argparse.Namespace,
    request: dict[str, int],
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cache-root",
        str(arguments.cache_root),
        "--data-dir",
        str(arguments.data_dir),
        "--warmups",
        str(arguments.warmups),
        "--repetitions",
        str(arguments.repetitions),
        "--candidate-json",
        json.dumps(request, separators=(",", ":")),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TRITON_CACHE_AUTOTUNING": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    primary = (request["num_warps"], request["num_stages"]) in {(4, 2), (8, 2), (8, 3)}
    timeout = (
        arguments.candidate_timeout_seconds
        if primary
        else min(arguments.candidate_timeout_seconds, arguments.secondary_timeout_seconds)
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return {**request, "status": "rejected", "reason": "timeout"}
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip().splitlines()
        return {
            **request,
            "status": "rejected",
            "reason": message[-1][:500] if message else f"exit-{completed.returncode}",
        }
    try:
        metrics = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {**request, "status": "rejected", "reason": "invalid-result"}
    return {**request, "status": "accepted", **metrics}


def _candidate_key(value: dict[str, Any]) -> tuple[int, int, int, str, int, int]:
    return (
        int(value["n"]),
        int(value["k"]),
        int(value["block_size_m"]),
        str(value["dtype"]),
        int(value["num_warps"]),
        int(value["num_stages"]),
    )


def _load_known_bad(
    path: Path,
    *,
    fingerprint: dict[str, str],
    stage: str,
) -> set[tuple[int, int, int, str, int, int]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if (
        document.get("format") != "modeldeck-fp8-tuning"
        or document.get("version") != TUNING_MANIFEST_VERSION
        or document.get("fingerprint") != fingerprint
        or document.get("stage") != stage
    ):
        return set()
    known_bad = set()
    for entry in document.get("rejected", []):
        try:
            known_bad.add(_candidate_key(entry))
        except (KeyError, TypeError, ValueError):
            continue
    return known_bad


def tune(arguments: argparse.Namespace) -> Path:
    import torch

    validated = validate_fp8_kernel_snapshot(arguments.cache_root)
    fingerprint = runtime_fingerprint(torch, kernel_manifest_digest=validated.manifest_digest)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache_path(arguments.data_dir, fingerprint))
    target = tuning_manifest_path(arguments.data_dir)
    buckets = (16,) if arguments.stage == "decode" else tuple(M_BUCKET_INPUTS)
    winners = []
    rejected = []
    known_bad = _load_known_bad(target, fingerprint=fingerprint, stage=arguments.stage)
    for n, k in QWEN38_FP8_WEIGHT_SHAPES:
        for block_size_m in buckets:
            results = []
            for candidate in CANDIDATES:
                request = {
                    "m": M_BUCKET_INPUTS[block_size_m],
                    "n": n,
                    "k": k,
                    "block_size_m": block_size_m,
                    "dtype": "bfloat16",
                    **candidate,
                }
                rejection_key = _candidate_key(request)
                if rejection_key in known_bad:
                    result = {**request, "status": "rejected", "reason": "known-bad-for-bucket"}
                else:
                    result = _run_candidate(arguments, request)
                    if result["status"] == "rejected":
                        known_bad.add(rejection_key)
                print(
                    json.dumps(
                        {
                            "event": "candidate-complete",
                            "n": n,
                            "k": k,
                            "block_size_m": block_size_m,
                            "num_warps": candidate["num_warps"],
                            "num_stages": candidate["num_stages"],
                            "status": result["status"],
                            "median_ms": result.get("median_ms"),
                            "reason": result.get("reason"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if result["status"] == "accepted":
                    results.append(result)
                else:
                    rejected.append(result)
            if results:
                winner = min(results, key=lambda item: item["median_ms"])
            else:
                winner = {
                    "n": n,
                    "k": k,
                    "block_size_m": block_size_m,
                    "dtype": "bfloat16",
                    **DEFAULT_CONFIG,
                    "status": "fallback",
                }
            winners.append(winner)
    document = {
        "format": "modeldeck-fp8-tuning",
        "version": TUNING_MANIFEST_VERSION,
        "model_id": "Qwen/Qwen3.8-27B-FP8",
        "kernel_commit": KERNEL_COMMIT,
        "stage": arguments.stage,
        "fingerprint": fingerprint,
        "winners": winners,
        "rejected": rejected,
    }
    write_tuning_manifest(target, document)
    return target


def main() -> int:
    arguments = _parser().parse_args()
    if (
        arguments.candidate_timeout_seconds <= 0
        or arguments.secondary_timeout_seconds <= 0
        or arguments.warmups < 1
        or arguments.repetitions < 1
    ):
        raise SystemExit("Timeout, warm-ups and repetitions must be positive")
    if arguments.candidate_json:
        return _candidate_process(arguments)
    target = tune(arguments)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

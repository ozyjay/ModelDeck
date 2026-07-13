from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Any


GIB = 1024**3
DEFAULT_MODEL_ID = "google/diffusiongemma-26B-A4B-it"
DEFAULT_REVISION = "52de6b914ee1749a7d4933202505ddf5b414ec43"
DEFAULT_CACHE_ROOT = "/mnt/work/models/huggingface/hub"
Q4_WORKER = "diffusiongemma-q4-rocm"
BF16_WORKER = "diffusiongemma-rocm"
Q4_ALIAS = "text-diffusion-q4"
BF16_ALIAS = "text-diffusion"


@dataclass(frozen=True)
class PromptSpec:
    id: str
    prompt: str
    required_groups: tuple[tuple[str, ...], ...] = ()
    minimum_words: int = 12
    minimum_sentences: int = 0


DEFAULT_PROMPTS = (
    PromptSpec(
        id="factual-sky",
        prompt="Explain why the sky appears blue in three concise sentences.",
        minimum_words=30,
        minimum_sentences=3,
    ),
    PromptSpec(
        id="python-primes",
        prompt="Write a short Python function that returns the prime numbers below n.",
        required_groups=(("def ",), ("return",)),
        minimum_words=12,
    ),
    PromptSpec(
        id="science-comparison",
        prompt="Compare photosynthesis and cellular respiration for a high-school student.",
        required_groups=(("photosynthesis",), ("respiration",)),
        minimum_words=35,
    ),
    PromptSpec(
        id="travel-plan",
        prompt="Plan a three-day visit to Cairns with indoor and outdoor alternatives.",
        required_groups=(("cairns",), ("day",), ("indoor",), ("outdoor",)),
        minimum_words=35,
    ),
    PromptSpec(
        id="arithmetic-reasoning",
        prompt=(
            "A train travels 180 kilometres in 2.5 hours. "
            "State the numerical average speed first, then explain the calculation briefly."
        ),
        required_groups=(("72",),),
        minimum_words=20,
    ),
    PromptSpec(
        id="creative-scene",
        prompt="Write a brief imaginative scene about a robot discovering rain for the first time.",
        required_groups=(
            ("robot",),
            ("rain", "raindrop", "downpour", "deluge"),
        ),
        minimum_words=35,
    ),
    PromptSpec(
        id="balanced-analysis",
        prompt="Explain one benefit and one risk of using artificial intelligence in education.",
        required_groups=(("benefit", "advantage"), ("risk", "concern", "drawback")),
        minimum_words=30,
    ),
    PromptSpec(
        id="translation",
        prompt=(
            "Translate 'The library opens at nine tomorrow morning' into French "
            "and explain one grammar choice."
        ),
        required_groups=(("bibliothèque", "bibliotheque"), ("neuf", "9")),
        minimum_words=15,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the packaged DiffusionGemma Q4 worker with its pinned BF16 "
            "baseline and apply local quality, determinism, memory, and latency gates."
        )
    )
    parser.add_argument("--management-url", default="http://127.0.0.1:3600")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8600")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("MODELDECK_HF_HUB_CACHE", DEFAULT_CACHE_ROOT)),
    )
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--seed-repeats", type=int, default=1)
    parser.add_argument("--stability-runs", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--denoising-steps", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--job-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--leave-worker",
        choices=("q4", "bf16", "none"),
        default="q4",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("var/q4-quality-evaluation.json"),
    )
    return parser.parse_args()


def load_prompts(path: Path | None) -> list[PromptSpec]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise SystemExit("--prompts-file must contain a non-empty JSON list")
    prompts: list[PromptSpec] = []
    for index, item in enumerate(payload):
        if isinstance(item, str):
            prompt = item.strip()
            prompt_id = f"prompt-{index + 1}"
        elif isinstance(item, dict):
            prompt = str(item.get("prompt", "")).strip()
            prompt_id = str(item.get("id") or f"prompt-{index + 1}")
        else:
            raise SystemExit("prompt entries must be strings or objects")
        if not prompt:
            raise SystemExit("evaluation prompts must be non-empty")
        prompts.append(PromptSpec(id=prompt_id, prompt=prompt))
    return prompts


def find_snapshot(cache_root: Path, model_id: str, revision: str) -> Path:
    snapshot = (
        cache_root
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    if not snapshot.is_dir():
        raise SystemExit(f"Pinned model snapshot not found: {snapshot}")
    return snapshot


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = client.request(method, url, json=payload)
    if not response.is_success:
        raise RuntimeError(f"{method} {url} failed ({response.status_code}): {response.text}")
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError(f"{method} {url} returned a non-object response")
    return value


def start_worker(client: httpx.Client, management_url: str, worker_id: str) -> dict[str, Any]:
    worker = request_json(
        client,
        "POST",
        f"{management_url}/api/workers/{worker_id}/start",
    )
    if worker.get("state") != "ready":
        raise RuntimeError(f"Worker {worker_id} did not become ready: {worker.get('state')}")
    return worker


def stop_worker(client: httpx.Client, management_url: str, worker_id: str) -> None:
    response = client.post(f"{management_url}/api/workers/{worker_id}/stop")
    if response.status_code not in {200, 404}:
        raise RuntimeError(
            f"Could not stop {worker_id} ({response.status_code}): {response.text}"
        )


def metrics(client: httpx.Client, endpoint: str) -> dict[str, Any]:
    return request_json(client, "GET", f"{endpoint}/metrics")


def run_diffusion(
    client: httpx.Client,
    *,
    gateway_url: str,
    alias: str,
    spec: PromptSpec,
    seed: int,
    max_length: int,
    denoising_steps: int,
    temperature: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    queued = request_json(
        client,
        "POST",
        f"{gateway_url}/v1/diffuse",
        payload={
            "model": alias,
            "prompt": spec.prompt,
            "max_length": max_length,
            "denoising_steps": denoising_steps,
            "block_length": max_length,
            "temperature": temperature,
            "seed": seed,
            "stream_intermediate_frames": False,
        },
    )
    job_id = str(queued.get("job_id", ""))
    if not job_id:
        raise RuntimeError("Gateway did not return a diffusion job id")
    deadline = time.monotonic() + timeout_seconds
    while True:
        job = request_json(client, "GET", f"{gateway_url}/v1/jobs/{job_id}")
        if job.get("state") in {"complete", "failed", "cancelled"}:
            break
        if time.monotonic() >= deadline:
            client.post(f"{gateway_url}/v1/jobs/{job_id}/cancel")
            raise TimeoutError(f"Diffusion job {job_id} exceeded {timeout_seconds:.0f} seconds")
        time.sleep(0.25)
    wall_seconds = time.perf_counter() - started
    frames = job.get("frames") or []
    terminal = frames[-1] if frames else {}
    text = str(job.get("text", ""))
    contract_checks = {
        "complete_state": job.get("state") == "complete",
        "correct_alias": job.get("model") == alias,
        "has_frames": bool(frames),
        "terminal_complete": terminal.get("complete") is True,
        "terminal_unmasked": terminal.get("masked_tokens") == 0,
        "not_cancelled": terminal.get("cancelled") is False,
        "nonempty_text": len(text.strip()) >= 20,
        "no_replacement_characters": "�" not in text,
    }
    return {
        "prompt_id": spec.id,
        "seed": seed,
        "job_id": job_id,
        "text": text,
        "frame_count": len(frames),
        "stable_tokens": terminal.get("stable_tokens"),
        "worker_seconds": job.get("metrics", {}).get("total_seconds"),
        "wall_seconds": wall_seconds,
        "contract_checks": contract_checks,
        "contract_passed": all(contract_checks.values()),
        "constraint": evaluate_constraints(spec, text),
    }


def normalise_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def evaluate_constraints(spec: PromptSpec, text: str) -> dict[str, Any]:
    normalised = normalise_text(text)
    words = re.findall(r"\b[\w'-]+\b", normalised)
    sentences = [item for item in re.split(r"[.!?]+", text) if item.strip()]
    group_results = [
        any(normalise_text(candidate) in normalised for candidate in group)
        for group in spec.required_groups
    ]
    checks = {
        "minimum_words": len(words) >= spec.minimum_words,
        "minimum_sentences": len(sentences) >= spec.minimum_sentences,
        "required_groups": all(group_results),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "word_count": len(words),
        "sentence_count": len(sentences),
        "required_group_results": group_results,
    }


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def compare_outputs(tokenizer: Any, q4: dict[str, Any], bf16: dict[str, Any]) -> dict[str, Any]:
    q4_tokens = token_ids(tokenizer, q4["text"])
    bf16_tokens = token_ids(tokenizer, bf16["text"])
    shared = min(len(q4_tokens), len(bf16_tokens))
    positional_matches = sum(
        left == right for left, right in zip(q4_tokens[:shared], bf16_tokens[:shared], strict=True)
    )
    return {
        "prompt_id": q4["prompt_id"],
        "seed": q4["seed"],
        "q4_tokens": len(q4_tokens),
        "bf16_tokens": len(bf16_tokens),
        "positional_token_agreement": positional_matches / shared if shared else 1.0,
        "token_edit_similarity": SequenceMatcher(
            None,
            q4_tokens,
            bf16_tokens,
            autojunk=False,
        ).ratio(),
        "word_edit_similarity": SequenceMatcher(
            None,
            q4["text"].split(),
            bf16["text"].split(),
            autojunk=False,
        ).ratio(),
        "exact_text": q4["text"] == bf16["text"],
    }


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round(0.95 * (len(ordered) - 1))
    return ordered[index]


def run_suite(
    client: httpx.Client,
    *,
    gateway_url: str,
    alias: str,
    prompts: list[PromptSpec],
    seed: int,
    seed_repeats: int,
    max_length: int,
    denoising_steps: int,
    temperature: float,
    timeout_seconds: float,
    endpoint: str,
) -> tuple[list[dict[str, Any]], list[int]]:
    results: list[dict[str, Any]] = []
    memory_samples: list[int] = []
    total = len(prompts) * seed_repeats
    run_index = 0
    for repeat in range(seed_repeats):
        for prompt_index, spec in enumerate(prompts):
            run_index += 1
            run_seed = seed + repeat * 10_000 + prompt_index
            result = run_diffusion(
                client,
                gateway_url=gateway_url,
                alias=alias,
                spec=spec,
                seed=run_seed,
                max_length=max_length,
                denoising_steps=denoising_steps,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )
            results.append(result)
            memory_samples.append(int(metrics(client, endpoint)["memory_allocated_bytes"]))
            print(
                f"  {alias} {run_index}/{total}: {spec.id}, seed={run_seed}, "
                f"{result['wall_seconds']:.2f}s, contract={result['contract_passed']}, "
                f"constraint={result['constraint']['passed']}"
            )
    return results, memory_samples


def phase_summary(results: list[dict[str, Any]], memory_samples: list[int]) -> dict[str, Any]:
    latencies = [float(result["wall_seconds"]) for result in results]
    allocated_range = max(memory_samples) - min(memory_samples) if memory_samples else 0
    return {
        "runs": len(results),
        "contract_passes": sum(result["contract_passed"] for result in results),
        "constraint_passes": sum(result["constraint"]["passed"] for result in results),
        "constraint_pass_rate": (
            sum(result["constraint"]["passed"] for result in results) / len(results)
            if results
            else 0.0
        ),
        "median_wall_seconds": median(latencies) if latencies else 0.0,
        "p95_wall_seconds": percentile_95(latencies),
        "memory_allocated_min_bytes": min(memory_samples) if memory_samples else 0,
        "memory_allocated_max_bytes": max(memory_samples) if memory_samples else 0,
        "memory_allocated_range_bytes": allocated_range,
    }


def restore_worker(
    client: httpx.Client,
    *,
    management_url: str,
    leave_worker: str,
) -> None:
    if leave_worker == "q4":
        start_worker(client, management_url, Q4_WORKER)
    elif leave_worker == "bf16":
        start_worker(client, management_url, BF16_WORKER)
    else:
        stop_worker(client, management_url, Q4_WORKER)
        stop_worker(client, management_url, BF16_WORKER)


def validate_args(args: argparse.Namespace) -> None:
    if args.seed_repeats < 1:
        raise SystemExit("--seed-repeats must be at least 1")
    if args.stability_runs < 0:
        raise SystemExit("--stability-runs cannot be negative")
    if not 8 <= args.max_length <= 256:
        raise SystemExit("--max-length must be between 8 and 256")
    if not 1 <= args.denoising_steps <= 48:
        raise SystemExit("--denoising-steps must be between 1 and 48")
    if not 0 < args.temperature <= 2:
        raise SystemExit("--temperature must be greater than zero and at most 2")


def main() -> None:
    import httpx
    from transformers import AutoProcessor

    args = parse_args()
    validate_args(args)
    prompts = load_prompts(args.prompts_file)
    snapshot = find_snapshot(args.cache_root, args.model_id, args.revision)
    processor = AutoProcessor.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer = processor.tokenizer
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    timeout = httpx.Timeout(900.0, connect=3.0)

    with httpx.Client(timeout=timeout) as client:
        profile_response = client.get(f"{args.management_url}/api/profiles")
        profile_response.raise_for_status()
        profile_payload = profile_response.json()
        if not isinstance(profile_payload, list):
            raise RuntimeError("Management profiles endpoint returned a non-list response")
        profile_ids = {str(profile.get("id")) for profile in profile_payload}
        missing_profiles = {Q4_WORKER, BF16_WORKER} - profile_ids
        if missing_profiles:
            raise RuntimeError(
                "The running management service is missing profiles: "
                + ", ".join(sorted(missing_profiles))
            )

        print("Phase 1/3: Q4 quality, determinism, and stability")
        q4_worker = start_worker(client, args.management_url, Q4_WORKER)
        q4_endpoint = str(q4_worker["endpoint"])
        q4_metrics_before = metrics(client, q4_endpoint)
        if q4_metrics_before.get("quantization") != "gptq-q4-g32-expert-only":
            raise RuntimeError("Q4 worker did not report the expected quantization")
        q4_results, q4_memory = run_suite(
            client,
            gateway_url=args.gateway_url,
            alias=Q4_ALIAS,
            prompts=prompts,
            seed=args.seed,
            seed_repeats=args.seed_repeats,
            max_length=args.max_length,
            denoising_steps=args.denoising_steps,
            temperature=args.temperature,
            timeout_seconds=args.job_timeout_seconds,
            endpoint=q4_endpoint,
        )

        replay = run_diffusion(
            client,
            gateway_url=args.gateway_url,
            alias=Q4_ALIAS,
            spec=prompts[0],
            seed=args.seed,
            max_length=args.max_length,
            denoising_steps=args.denoising_steps,
            temperature=args.temperature,
            timeout_seconds=args.job_timeout_seconds,
        )
        q4_memory.append(int(metrics(client, q4_endpoint)["memory_allocated_bytes"]))
        deterministic_reference = next(
            result
            for result in q4_results
            if result["prompt_id"] == prompts[0].id and result["seed"] == args.seed
        )
        deterministic = {
            "prompt_id": prompts[0].id,
            "seed": args.seed,
            "exact_text": replay["text"] == deterministic_reference["text"],
            "reference_job_id": deterministic_reference["job_id"],
            "replay_job_id": replay["job_id"],
        }
        print(f"  deterministic replay: exact={deterministic['exact_text']}")

        stability_results: list[dict[str, Any]] = []
        stability_memory: list[int] = []
        for index in range(args.stability_runs):
            spec = prompts[index % len(prompts)]
            result = run_diffusion(
                client,
                gateway_url=args.gateway_url,
                alias=Q4_ALIAS,
                spec=spec,
                seed=args.seed + 50_000 + index,
                max_length=args.max_length,
                denoising_steps=args.denoising_steps,
                temperature=args.temperature,
                timeout_seconds=args.job_timeout_seconds,
            )
            stability_results.append(result)
            allocated = int(metrics(client, q4_endpoint)["memory_allocated_bytes"])
            q4_memory.append(allocated)
            stability_memory.append(allocated)
            print(
                f"  stability {index + 1}/{args.stability_runs}: {spec.id}, "
                f"{result['wall_seconds']:.2f}s, contract={result['contract_passed']}"
            )
        q4_metrics_after = metrics(client, q4_endpoint)

        print("Phase 2/3: pinned BF16 comparison suite")
        bf16_worker = start_worker(client, args.management_url, BF16_WORKER)
        bf16_endpoint = str(bf16_worker["endpoint"])
        bf16_metrics_before = metrics(client, bf16_endpoint)
        bf16_results, bf16_memory = run_suite(
            client,
            gateway_url=args.gateway_url,
            alias=BF16_ALIAS,
            prompts=prompts,
            seed=args.seed,
            seed_repeats=args.seed_repeats,
            max_length=args.max_length,
            denoising_steps=args.denoising_steps,
            temperature=args.temperature,
            timeout_seconds=args.job_timeout_seconds,
            endpoint=bf16_endpoint,
        )
        bf16_metrics_after = metrics(client, bf16_endpoint)

        print("Phase 3/3: comparison and release gates")
        comparisons = [
            compare_outputs(tokenizer, q4, bf16)
            for q4, bf16 in zip(q4_results, bf16_results, strict=True)
        ]
        q4_summary = phase_summary(q4_results, q4_memory)
        bf16_summary = phase_summary(bf16_results, bf16_memory)
        stability_summary = phase_summary(stability_results, stability_memory)
        token_similarity = [item["token_edit_similarity"] for item in comparisons]
        word_similarity = [item["word_edit_similarity"] for item in comparisons]
        positional_agreement = [item["positional_token_agreement"] for item in comparisons]
        comparison_summary = {
            "pairs": len(comparisons),
            "mean_token_edit_similarity": mean(token_similarity),
            "minimum_token_edit_similarity": min(token_similarity),
            "mean_word_edit_similarity": mean(word_similarity),
            "minimum_word_edit_similarity": min(word_similarity),
            "mean_positional_token_agreement": mean(positional_agreement),
            "exact_text_pairs": sum(item["exact_text"] for item in comparisons),
        }
        q4_gate_delta = int(q4_metrics_after.get("q4_gate_calls", 0)) - int(
            q4_metrics_before.get("q4_gate_calls", 0)
        )
        q4_down_delta = int(q4_metrics_after.get("q4_down_calls", 0)) - int(
            q4_metrics_before.get("q4_down_calls", 0)
        )
        latency_ratio = (
            q4_summary["median_wall_seconds"] / bf16_summary["median_wall_seconds"]
            if bf16_summary["median_wall_seconds"]
            else float("inf")
        )
        checks = {
            "q4_contracts": q4_summary["contract_passes"] == q4_summary["runs"],
            "bf16_contracts": bf16_summary["contract_passes"] == bf16_summary["runs"],
            "q4_stability_contracts": (
                stability_summary["contract_passes"] == stability_summary["runs"]
            ),
            "deterministic_replay": deterministic["exact_text"],
            "q4_kernel_invoked": q4_gate_delta > 0 and q4_gate_delta == q4_down_delta,
            "q4_peak_memory_under_24_gib": (
                int(q4_metrics_after["peak_memory_allocated_bytes"]) <= 24 * GIB
            ),
            "q4_memory_range_under_1_gib": (
                q4_summary["memory_allocated_range_bytes"] <= GIB
            ),
            "q4_latency_under_3x_bf16": latency_ratio <= 3.0,
            "mean_token_edit_similarity_at_least_0_35": (
                comparison_summary["mean_token_edit_similarity"] >= 0.35
            ),
            "q4_constraints_not_materially_worse": (
                q4_summary["constraint_pass_rate"]
                >= max(0.5, bf16_summary["constraint_pass_rate"] - 0.25)
            ),
        }
        passed = all(checks.values())
        report = {
            "format": "modeldeck-diffusiongemma-q4-evaluation",
            "format_version": 1,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "total_seconds": time.perf_counter() - started,
            "passed": passed,
            "checks": checks,
            "configuration": {
                "model_id": args.model_id,
                "revision": args.revision,
                "prompt_count": len(prompts),
                "seed": args.seed,
                "seed_repeats": args.seed_repeats,
                "stability_runs": args.stability_runs,
                "max_length": args.max_length,
                "denoising_steps": args.denoising_steps,
                "temperature": args.temperature,
            },
            "q4": {
                "worker": q4_worker,
                "metrics_before": q4_metrics_before,
                "metrics_after": q4_metrics_after,
                "summary": q4_summary,
                "results": q4_results,
                "deterministic_replay": {**deterministic, "result": replay},
                "stability_summary": stability_summary,
                "stability_results": stability_results,
                "q4_gate_call_delta": q4_gate_delta,
                "q4_down_call_delta": q4_down_delta,
            },
            "bf16": {
                "worker": bf16_worker,
                "metrics_before": bf16_metrics_before,
                "metrics_after": bf16_metrics_after,
                "summary": bf16_summary,
                "results": bf16_results,
            },
            "comparison": {
                "latency_ratio_q4_to_bf16": latency_ratio,
                "summary": comparison_summary,
                "pairs": comparisons,
            },
        }

        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        restore_worker(
            client,
            management_url=args.management_url,
            leave_worker=args.leave_worker,
        )

    display = {
        "passed": passed,
        "checks": checks,
        "q4_median_seconds": round(q4_summary["median_wall_seconds"], 4),
        "bf16_median_seconds": round(bf16_summary["median_wall_seconds"], 4),
        "latency_ratio": round(latency_ratio, 4),
        "q4_peak_gib": round(q4_metrics_after["peak_memory_allocated_bytes"] / GIB, 3),
        "mean_token_edit_similarity": round(
            comparison_summary["mean_token_edit_similarity"],
            4,
        ),
        "q4_constraint_pass_rate": round(q4_summary["constraint_pass_rate"], 4),
        "bf16_constraint_pass_rate": round(bf16_summary["constraint_pass_rate"], 4),
        "json_output": str(args.json_output),
    }
    print(json.dumps(display, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

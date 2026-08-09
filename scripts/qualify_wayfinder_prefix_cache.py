from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from statistics import median
from typing import Any

import httpx

ALLOWED_MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
}
MEMORY_KEYS = (
    "memory_allocated_bytes",
    "memory_reserved_bytes",
    "system_gtt_used_bytes",
    "system_vram_used_bytes",
    "host_memory_used_bytes",
)


class QualificationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qualify application-managed WayFinder prefix caching on physical ROCm workers."
    )
    parser.add_argument("--management-url", default="http://127.0.0.1:3600")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8600")
    parser.add_argument("--workers", nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, default=5, choices=range(3, 11))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = client.request(method, url, json=payload)
    if not response.is_success:
        raise QualificationError(f"{method} {url} failed with HTTP {response.status_code}")
    document = response.json()
    if not isinstance(document, dict):
        raise QualificationError(f"{method} {url} returned a non-object response")
    return document


def ensure_thermal_safe(client: httpx.Client, gateway_url: str) -> None:
    thermal = request_json(client, "GET", gateway_url + "/v1/thermal")
    if thermal.get("state") == "critical":
        raise QualificationError(
            "ModelDeck's active thermal policy entered its critical state; qualification stopped"
        )


def output_digest(trace: dict[str, Any]) -> str:
    events = trace.get("events") or []
    text = events[-1].get("text_so_far", "") if events else ""
    return hashlib.sha256(str(text).encode()).hexdigest()


def assert_trace_equivalent(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    reference_events = reference.get("events") or []
    candidate_events = candidate.get("events") or []
    if output_digest(reference) != output_digest(candidate):
        raise QualificationError("Cache and bypass output text differ")
    if len(reference_events) != len(candidate_events):
        raise QualificationError("Cache and bypass output token counts differ")
    for expected, actual in zip(reference_events, candidate_events, strict=True):
        expected_selected = expected.get("selected") or {}
        actual_selected = actual.get("selected") or {}
        if expected_selected.get("token_id") != actual_selected.get("token_id"):
            raise QualificationError("Cache and bypass selected token IDs differ")
        selected_delta = abs(
            float(expected_selected.get("probability", 0))
            - float(actual_selected.get("probability", 0))
        )
        if selected_delta > 1e-5:
            raise QualificationError("Selected-token probabilities exceed the 1e-5 tolerance")
        expected_top = expected.get("alternatives") or []
        actual_top = actual.get("alternatives") or []
        if [item.get("token_id") for item in expected_top] != [item.get("token_id") for item in actual_top]:
            raise QualificationError("Cache and bypass top-k token identities differ")
        for expected_item, actual_item in zip(expected_top, actual_top, strict=True):
            top_delta = abs(
                float(expected_item.get("probability", 0))
                - float(actual_item.get("probability", 0))
            )
            if top_delta > 1e-5:
                raise QualificationError("Top-k probabilities exceed the 1e-5 tolerance")
        for name in ("mean", "l2_norm"):
            expected_hidden = expected.get("hidden_state_summary") or {}
            actual_hidden = actual.get("hidden_state_summary") or {}
            if name in expected_hidden or name in actual_hidden:
                if abs(float(expected_hidden.get(name, 0)) - float(actual_hidden.get(name, 0))) > 1e-4:
                    raise QualificationError("Hidden-state summaries exceed the 1e-4 tolerance")


def trace_request(
    client: httpx.Client,
    trace_url: str,
    route: str,
    *,
    cache_hint: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": route,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the local WayFinder coding agent. Follow the supplied tool contracts, "
                    "keep private inputs private, and answer the user's current request precisely. "
                )
                * 64,
            },
            {"role": "user", "content": "Reply with a concise explanation of prefix caching."},
        ],
        "seed": 7,
        "max_tokens": 64,
        "min_tokens": 64,
        "temperature": 0,
        "top_k": 5,
        "include_hidden_state_summary": True,
        "stream": False,
    }
    if cache_hint:
        payload["modeldeck"] = {
            "prefix_cache": {
                "stable_message_count": 1,
                "profile_version": "wayfinder-agent-v1",
            }
        }
    return request_json(client, "POST", trace_url.rstrip("/") + "/native/autoregressive/trace", payload)


def memory_snapshot(client: httpx.Client, endpoint: str, management_url: str) -> dict[str, int]:
    metrics = request_json(client, "GET", endpoint.rstrip("/") + "/metrics")
    snapshot = {
        key: int(metrics[key])
        for key in MEMORY_KEYS
        if isinstance(metrics.get(key), int) and not isinstance(metrics.get(key), bool)
    }
    telemetry = request_json(client, "GET", management_url + "/api/telemetry")
    memory = telemetry.get("memory") or {}
    total = memory.get("total_bytes")
    available = memory.get("available_bytes")
    if isinstance(total, int) and isinstance(available, int):
        snapshot["host_memory_used_bytes"] = max(total - available, 0)
    return snapshot


def assert_no_monotonic_growth(samples: list[dict[str, int]]) -> None:
    for key in MEMORY_KEYS:
        values = [sample[key] for sample in samples if key in sample]
        if len(values) >= 3 and all(
            after > before for before, after in zip(values, values[1:], strict=False)
        ):
            raise QualificationError(f"Repeated cases show monotonic growth in {key}")


def cancellation_case(client: httpx.Client, endpoint: str, route: str, index: int) -> None:
    request_id = f"prefix-cache-qualification-cancel-{index}"
    payload = {
        "request_id": request_id,
        "model": route,
        "messages": [
            {"role": "system", "content": "Stable WayFinder instructions. " * 256},
            {"role": "user", "content": "Generate a long deterministic response."},
        ],
        "modeldeck": {
            "prefix_cache": {
                "stable_message_count": 1,
                "profile_version": "wayfinder-agent-v1",
            }
        },
        "seed": 7,
        "max_tokens": 512,
        "min_tokens": 512,
        "temperature": 0,
        "top_k": 1,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            request_json,
            client,
            "POST",
            endpoint.rstrip("/") + "/native/autoregressive/trace",
            payload,
        )
        cancelled = False
        for _ in range(50):
            if running.done():
                break
            response = request_json(
                client,
                "POST",
                endpoint.rstrip("/") + "/cancel",
                {"request_id": request_id},
            )
            if response.get("ok") is True:
                cancelled = True
                break
            time.sleep(0.02)
        result = running.result(timeout=30)
    if not cancelled or not (result.get("metrics") or {}).get("cancelled"):
        raise QualificationError("Worker cancellation did not stop the active qualification request")


def resolve_workers(workers: list[dict[str, Any]], selectors: list[str]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for selector in dict.fromkeys(selectors):
        matches = [item for item in workers if item.get("id") == selector or item.get("name") == selector]
        if len(matches) != 1:
            raise QualificationError(f"Worker selector must resolve uniquely: {selector}")
        worker = dict(matches[0])
        if worker.get("model_id") not in ALLOWED_MODELS:
            raise QualificationError(f"Worker is not an allowlisted WayFinder Qwen2.5 model: {selector}")
        capabilities = worker.get("capabilities") or {}
        if capabilities.get("prefix_caching") != "application-managed" or not capabilities.get(
            "prefix_cache_enabled"
        ):
            raise QualificationError(f"Prefix caching is not enabled for Worker: {selector}")
        worker["route"] = f"worker-{worker['id'][:8]}"
        resolved.append(worker)
    return resolved


def qualify_worker(
    client: httpx.Client,
    management_url: str,
    gateway_url: str,
    worker: dict[str, Any],
    repetitions: int,
) -> dict[str, Any]:
    started_here = worker.get("state") == "stopped"
    if worker.get("state") not in {"ready", "stopped"}:
        raise QualificationError(f"Worker {worker['name']} must be ready or stopped")
    if started_here:
        worker = request_json(client, "POST", f"{management_url}/api/workers/{worker['id']}/start")
    endpoint = worker.get("endpoint")
    if worker.get("state") != "ready" or not isinstance(endpoint, str):
        raise QualificationError(f"Worker {worker['name']} did not become ready")
    bypass_ttft: list[float] = []
    hit_ttft: list[float] = []
    memory_samples: list[dict[str, int]] = []
    try:
        for _ in range(repetitions):
            ensure_thermal_safe(client, gateway_url)
            request_json(
                client,
                "POST",
                f"{management_url}/api/workers/{worker['id']}/prefix-cache/clear",
            )
            cold = trace_request(client, endpoint, worker["route"], cache_hint=True)
            ensure_thermal_safe(client, gateway_url)
            warm = trace_request(client, endpoint, worker["route"], cache_hint=True)
            ensure_thermal_safe(client, gateway_url)
            bypass = trace_request(client, endpoint, worker["route"], cache_hint=False)
            assert_trace_equivalent(bypass, cold)
            assert_trace_equivalent(bypass, warm)
            cold_cache = ((cold.get("metrics") or {}).get("prefix_cache") or {})
            warm_cache = ((warm.get("metrics") or {}).get("prefix_cache") or {})
            bypass_cache = ((bypass.get("metrics") or {}).get("prefix_cache") or {})
            if cold_cache.get("status") != "miss" or warm_cache.get("status") != "hit":
                raise QualificationError("Cold-miss and warm-hit cache status was not observed")
            if bypass_cache.get("status") != "bypass":
                raise QualificationError("Deliberate bypass did not use full prefill")
            if int(warm_cache.get("prefix_tokens", 0)) > 8192:
                raise QualificationError("Retained prefix exceeds 8,192 tokens")
            if int(warm_cache.get("cache_bytes", 0)) > 512 * 1024 * 1024:
                raise QualificationError("Retained prefix cache exceeds 512 MiB")
            hit_ttft.append(float((warm.get("metrics") or {})["first_token_seconds"]))
            bypass_ttft.append(float((bypass.get("metrics") or {})["first_token_seconds"]))
            memory_samples.append(memory_snapshot(client, endpoint, management_url))
        for index in range(3):
            ensure_thermal_safe(client, gateway_url)
            cancellation_case(client, endpoint, worker["route"], index)
            memory_samples.append(memory_snapshot(client, endpoint, management_url))
        assert_no_monotonic_growth(memory_samples)
        median_hit = median(hit_ttft)
        median_bypass = median(bypass_ttft)
        improvement = 1 - (median_hit / median_bypass)
        if improvement < 0.20:
            raise QualificationError(
                f"Median warm-hit TTFT improvement was {improvement:.1%}; at least 20% is required"
            )
        metrics = request_json(client, "GET", endpoint.rstrip("/") + "/metrics")
        return {
            "worker_id": worker["id"],
            "model_id": worker["model_id"],
            "revision": worker["revision"],
            "qualified": True,
            "repetitions": repetitions,
            "median_warm_hit_ttft_seconds": median_hit,
            "median_bypass_ttft_seconds": median_bypass,
            "warm_hit_improvement": improvement,
            "cache_bytes": metrics.get("prefix_cache_bytes"),
            "cache_tokens": metrics.get("prefix_cache_tokens"),
            "memory_samples": memory_samples,
        }
    finally:
        request_json(
            client,
            "POST",
            f"{management_url}/api/workers/{worker['id']}/prefix-cache/clear",
        )
        if started_here:
            request_json(client, "POST", f"{management_url}/api/workers/{worker['id']}/stop")


def main() -> int:
    args = parse_args()
    try:
        with httpx.Client(timeout=httpx.Timeout(900, connect=2)) as client:
            workers = client.get(args.management_url + "/api/workers").json()
            if not isinstance(workers, list):
                raise QualificationError("Management workers endpoint returned a non-list response")
            selected = resolve_workers(workers, args.workers)
            results = [
                qualify_worker(
                    client,
                    args.management_url,
                    args.gateway_url,
                    worker,
                    args.repetitions,
                )
                for worker in selected
            ]
        report = {
            "format": "modeldeck-wayfinder-prefix-cache-qualification",
            "version": 1,
            "results": results,
        }
        rendered = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (httpx.HTTPError, ValueError, QualificationError) as error:
        print(f"WayFinder prefix-cache qualification failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

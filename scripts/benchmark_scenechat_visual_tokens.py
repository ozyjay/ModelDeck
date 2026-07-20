from __future__ import annotations

import argparse
import base64
import json
import math
import socket
import sys
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import uvicorn
from modeldeck.config import Settings
from modeldeck.contracts.scenechat import external_prompt
from modeldeck.domain import WorkerDefinition
from modeldeck.gateway import create_gateway_app
from PIL import Image

MANAGEMENT_URL = "http://127.0.0.1:3600"
ROUTE_NAME = "scenechat-vision"
IMAGE_PATH = Path("/mnt/work/GitHubProjects/SceneChat/replay_assets/demo_booth.png")
REQUEST_DEADLINE_SECONDS = 120
EXPECTED_KEYS = {"summary", "objects", "relationships", "uncertainties", "safety_notes"}


def _json_request(url: str, *, payload: dict[str, Any] | None = None, timeout: float = 10) -> Any:
    encoded = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        url,
        data=encoded,
        method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json"} if encoded is not None else {},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _post(url: str, *, timeout: float = 900) -> Any:
    request = Request(url, data=b"", method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _free_loopback_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def _wait_for(url: str, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _json_request(url, timeout=2)
            return
        except (HTTPError, URLError, TimeoutError, OSError):
            time.sleep(0.25)
    raise RuntimeError("The benchmark-only gateway did not become available")


def _worker_by_id(workers: list[dict[str, Any]], worker_id: str) -> dict[str, Any]:
    matches = [worker for worker in workers if worker["id"] == worker_id or worker["name"] == worker_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one configured Worker matching {worker_id!r}")
    return matches[0]


def _validate_workers(worker_280: dict[str, Any], worker_140: dict[str, Any]) -> None:
    for worker, budget in ((worker_280, 280), (worker_140, 140)):
        if worker["generation_family"] != "vision-language":
            raise RuntimeError(f"Worker {worker['name']!r} is not a vision-language Worker")
        if worker["settings"].get("visual_token_budget") != budget:
            raise RuntimeError(f"Worker {worker['name']!r} does not have visual token budget {budget}")
    identities = {(worker["model_id"], worker["revision"]) for worker in (worker_280, worker_140)}
    if len(identities) != 1:
        raise RuntimeError("The benchmark Workers must use the same pinned model and revision")


def _validate_route() -> None:
    live = _json_request(f"{MANAGEMENT_URL}/api/live")
    routes = [route for route in live.get("routes", []) if route.get("public_name") == ROUTE_NAME]
    if len(routes) != 1:
        raise RuntimeError(f"The published Event must contain exactly one {ROUTE_NAME!r} Route")


def _profile(worker: dict[str, Any]):
    definition = WorkerDefinition.model_validate(
        {name: worker[name] for name in WorkerDefinition.model_fields if name in worker}
    )
    return definition.to_profile()


def _image_payload() -> tuple[str, dict[str, int]]:
    if not IMAGE_PATH.is_file():
        raise RuntimeError(f"Prepared benchmark image is missing: {IMAGE_PATH}")
    image_bytes = IMAGE_PATH.read_bytes()
    with Image.open(IMAGE_PATH) as image:
        image.load()
        metadata = {"width": image.width, "height": image.height, "bytes": len(image_bytes)}
        image_format = image.format
    mime_type = {"PNG": "image/png", "JPEG": "image/jpeg"}.get(image_format)
    if mime_type is None:
        raise RuntimeError("The prepared benchmark image must be PNG or JPEG")
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}", metadata


def _payload(data_url: str) -> dict[str, Any]:
    return {
        "model": ROUTE_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": external_prompt("Describe the scene.")},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
        "stream": False,
    }


def _schema_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != EXPECTED_KEYS:
        return False
    if not isinstance(value["summary"], str) or not value["summary"].strip():
        return False
    if not all(isinstance(value[name], list) for name in EXPECTED_KEYS - {"summary"}):
        return False
    required_object_keys = {"label", "description", "approximate_location"}
    return all(isinstance(item, dict) and set(item) == required_object_keys for item in value["objects"])


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 4)
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(interpolated, 4)


def _summary(samples: list[dict[str, Any]], failures: Counter[str]) -> dict[str, Any]:
    latencies = [sample["latency_seconds"] for sample in samples]
    completions = [sample["completion_tokens"] for sample in samples]
    return {
        "measured_requests": len(samples) + sum(failures.values()),
        "valid_responses": len(samples),
        "failure_categories": dict(sorted(failures.items())),
        "schema_valid_responses": sum(sample["schema_valid"] for sample in samples),
        "latency_seconds": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
        },
        "completion_tokens": {
            "minimum": min(completions) if completions else None,
            "maximum": max(completions) if completions else None,
            "p50": _percentile([float(value) for value in completions], 0.50),
            "at_limit": sum(value >= 512 for value in completions),
        },
    }


def _run_request(gateway_url: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    started = time.perf_counter()
    try:
        response = _json_request(
            f"{gateway_url}/v1/chat/completions",
            payload=payload,
            timeout=REQUEST_DEADLINE_SECONDS,
        )
        analysis = json.loads(response["choices"][0]["message"]["content"])
        usage = response.get("usage", {})
        return (
            {
                "latency_seconds": time.perf_counter() - started,
                "completion_tokens": int(usage.get("completion_tokens", -1)),
                "schema_valid": _schema_valid(analysis),
                "analysis": analysis,
            },
            None,
        )
    except HTTPError as error:
        try:
            code = json.load(error).get("error", {}).get("code")
        except (AttributeError, ValueError, TypeError):
            code = None
        return None, str(code or f"http_{error.code}")
    except (TimeoutError, URLError):
        return None, "benchmark_request_timeout"
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None, "invalid_response"


def _review(samples: list[dict[str, Any]], budget: int) -> dict[str, int] | None:
    if not samples:
        return None
    reviewed = samples[: min(3, len(samples))]
    accepted = 0
    for index, sample in enumerate(reviewed, start=1):
        print(f"\nBudget {budget}, review sample {index}:\n", file=sys.stderr)
        print(json.dumps(sample["analysis"], indent=2), file=sys.stderr)
        answer = input("Does this accurately cover the important visible objects? [y/N] ")
        accepted += answer.strip().casefold() == "y"
    return {"reviewed": len(reviewed), "accepted": accepted}


def _benchmark_arm(
    gateway_url: str,
    worker: dict[str, Any],
    other: dict[str, Any],
    payload: dict[str, Any],
    warmups: int,
    runs: int,
    human_review: bool,
) -> dict[str, Any]:
    _post(f"{MANAGEMENT_URL}/api/workers/{other['id']}/stop")
    _post(f"{MANAGEMENT_URL}/api/workers/{worker['id']}/start")
    warmup_failures: Counter[str] = Counter()
    valid_warmups = 0
    for _ in range(warmups):
        warmup_sample, warmup_failure = _run_request(gateway_url, payload)
        if warmup_sample is None:
            warmup_failures[warmup_failure or "unknown"] += 1
        else:
            valid_warmups += 1
    budget = int(worker["settings"]["visual_token_budget"])
    if valid_warmups == 0:
        return {
            "visual_token_budget": budget,
            "benchmark_status": "not_run_no_valid_warmups",
            "warmup_valid_responses": 0,
            "warmup_failure_categories": dict(sorted(warmup_failures.items())),
            **_summary([], Counter()),
            "human_review": None,
        }
    samples: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for _ in range(runs):
        sample, failure = _run_request(gateway_url, payload)
        if sample is None:
            failures[failure or "unknown"] += 1
        else:
            samples.append(sample)
    result = _summary(samples, failures)
    result["visual_token_budget"] = budget
    result["benchmark_status"] = "completed"
    result["warmup_valid_responses"] = valid_warmups
    result["warmup_failure_categories"] = dict(sorted(warmup_failures.items()))
    result["human_review"] = _review(samples, budget) if human_review else None
    for sample in samples:
        sample.pop("analysis", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SceneChat Gemma 4 visual-token budgets")
    parser.add_argument("--worker-280", required=True)
    parser.add_argument("--worker-140", required=True)
    parser.add_argument("--warmups", type=int, choices=range(3, 6), default=4)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--human-review", action="store_true")
    arguments = parser.parse_args()
    if arguments.runs < 50:
        parser.error("--runs must be at least 50")

    workers = _json_request(f"{MANAGEMENT_URL}/api/workers")
    worker_280 = _worker_by_id(workers, arguments.worker_280)
    worker_140 = _worker_by_id(workers, arguments.worker_140)
    _validate_workers(worker_280, worker_140)
    _validate_route()
    originally_ready = [worker for worker in (worker_280, worker_140) if worker["state"] == "ready"]
    data_url, image_metadata = _image_payload()
    payload = _payload(data_url)
    gateway_port = _free_loopback_port()
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    gateway_app = create_gateway_app(
        alias_routes={ROUTE_NAME: [_profile(worker_280), _profile(worker_140)]},
        settings=Settings(
            gateway_port=gateway_port,
            scenechat_timeout_seconds=REQUEST_DEADLINE_SECONDS,
        ),
    )
    server = uvicorn.Server(
        uvicorn.Config(
            gateway_app,
            host="127.0.0.1",
            port=gateway_port,
            access_log=False,
            log_level="warning",
        )
    )
    gateway_thread = threading.Thread(target=server.run, name="scenechat-benchmark-gateway")
    gateway_thread.start()
    try:
        _wait_for(f"{gateway_url}/v1/health", timeout=15)
        results = [
            _benchmark_arm(
                gateway_url,
                worker_140,
                worker_280,
                payload,
                arguments.warmups,
                arguments.runs,
                arguments.human_review,
            ),
            _benchmark_arm(
                gateway_url,
                worker_280,
                worker_140,
                payload,
                arguments.warmups,
                arguments.runs,
                arguments.human_review,
            ),
        ]
    finally:
        server.should_exit = True
        gateway_thread.join(timeout=10)
        for worker in (worker_280, worker_140):
            try:
                _post(f"{MANAGEMENT_URL}/api/workers/{worker['id']}/stop")
            except (HTTPError, URLError, TimeoutError):
                pass
        for worker in originally_ready:
            _post(f"{MANAGEMENT_URL}/api/workers/{worker['id']}/start")

    document = {
        "format": "modeldeck-scenechat-visual-token-benchmark",
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "route": ROUTE_NAME,
        "model_id": worker_280["model_id"],
        "revision": worker_280["revision"],
        "image": image_metadata,
        "warmups_per_arm": arguments.warmups,
        "request_deadline_seconds": REQUEST_DEADLINE_SECONDS,
        "arms": results,
        "privacy": "No image data, prompt text, model descriptions or credentials are retained.",
    }
    output_dir = Path("var/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"scenechat_visual_tokens_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

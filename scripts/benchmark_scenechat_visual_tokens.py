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


def _temperatures() -> list[dict[str, Any]]:
    payload = _json_request(f"{MANAGEMENT_URL}/api/telemetry", timeout=5)
    if not isinstance(payload, dict):
        raise RuntimeError("The telemetry endpoint returned an invalid response")
    readings = []
    for reading in payload.get("temperatures", []):
        value = reading.get("celsius") if isinstance(reading, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        readings.append(reading)
    return readings


def _maximum_temperature(readings: list[dict[str, Any]]) -> float | None:
    values = [float(reading["celsius"]) for reading in readings]
    return max(values) if values else None


def _gpu_temperature(readings: list[dict[str, Any]]) -> float | None:
    values = [float(reading["celsius"]) for reading in readings if reading.get("source") == "amdgpu"]
    return max(values) if values else None


def _wait_for_cooldown(target: float, maximum_wait_seconds: float = 600) -> None:
    deadline = time.monotonic() + maximum_wait_seconds
    while time.monotonic() < deadline:
        readings = _temperatures()
        maximum = _maximum_temperature(readings)
        if maximum is None:
            raise RuntimeError("No temperature sensors are available for the benchmark")
        if maximum <= target:
            return
        time.sleep(2)
    raise RuntimeError(f"Hardware did not cool below {target:g}°C in time")


class ThermalGuard:
    def __init__(self, maximum_celsius: float) -> None:
        self.maximum_celsius = maximum_celsius
        self.maximum_observed_celsius: float | None = None
        self.maximum_gpu_celsius: float | None = None
        self.maximum_observed_sensor: dict[str, Any] | None = None
        self.maximum_gpu_sensor: dict[str, Any] | None = None
        self.trigger_sensor: dict[str, Any] | None = None
        self.current_temperature_celsius: float | None = None
        self.sample_count = 0
        self.abort_reason: str | None = None
        self.triggered = threading.Event()
        self.finished = threading.Event()
        self.worker_id: str | None = None
        self.thread = threading.Thread(target=self._monitor, name="scenechat-benchmark-thermal-guard")

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.finished.set()
        self.thread.join(timeout=35)

    def _abort(self, reason: str, sensor: dict[str, Any] | None = None) -> None:
        worker_id = self.worker_id
        self.abort_reason = reason
        self.trigger_sensor = sensor
        self.triggered.set()
        if worker_id:
            try:
                _post(f"{MANAGEMENT_URL}/api/workers/{worker_id}/stop", timeout=30)
            except (HTTPError, URLError, TimeoutError, OSError):
                pass

    def wait_for_cooldown(self, target: float, maximum_wait_seconds: float = 600) -> float:
        started = time.monotonic()
        initial_sample_count = self.sample_count
        while time.monotonic() - started < maximum_wait_seconds:
            if self.triggered.is_set():
                return time.monotonic() - started
            if (
                self.sample_count > initial_sample_count
                and self.current_temperature_celsius is not None
                and self.current_temperature_celsius <= target
            ):
                return time.monotonic() - started
            self.finished.wait(0.25)
        self._abort("cooldown_timeout")
        return time.monotonic() - started

    def _monitor(self) -> None:
        while not self.finished.wait(0.5):
            try:
                readings = _temperatures()
            except (HTTPError, URLError, TimeoutError, OSError, RuntimeError, TypeError, ValueError):
                self._abort("telemetry_unavailable")
                return
            maximum = _maximum_temperature(readings)
            if maximum is None:
                self._abort("temperature_sensors_unavailable")
                return
            hottest = max(readings, key=lambda reading: float(reading["celsius"]))
            hottest_sensor = {
                "source": str(hottest.get("source") or "unknown"),
                "label": str(hottest.get("label") or hottest.get("source") or "unknown"),
                "celsius": float(hottest["celsius"]),
            }
            gpu_readings = [reading for reading in readings if reading.get("source") == "amdgpu"]
            hottest_gpu = (
                max(gpu_readings, key=lambda reading: float(reading["celsius"])) if gpu_readings else None
            )
            gpu = _gpu_temperature(readings)
            self.sample_count += 1
            self.current_temperature_celsius = maximum
            self.maximum_observed_celsius = max(
                maximum,
                self.maximum_observed_celsius or maximum,
            )
            if self.maximum_observed_sensor is None or maximum > float(
                self.maximum_observed_sensor["celsius"]
            ):
                self.maximum_observed_sensor = hottest_sensor
            if gpu is not None:
                self.maximum_gpu_celsius = max(gpu, self.maximum_gpu_celsius or gpu)
                if self.maximum_gpu_sensor is None or gpu > float(self.maximum_gpu_sensor["celsius"]):
                    self.maximum_gpu_sensor = {
                        "source": str(hottest_gpu.get("source") or "amdgpu"),
                        "label": str(hottest_gpu.get("label") or "amdgpu"),
                        "celsius": gpu,
                    }
            if maximum >= self.maximum_celsius:
                self._abort("temperature_limit", hottest_sensor)
                return


def _thermal_summary(
    guard: ThermalGuard,
    maximum_temperature_celsius: float,
    cooldown_temperature_celsius: float,
) -> dict[str, Any]:
    return {
        "maximum_allowed_celsius": maximum_temperature_celsius,
        "cooldown_target_celsius": cooldown_temperature_celsius,
        "maximum_observed_celsius": guard.maximum_observed_celsius,
        "maximum_gpu_celsius": guard.maximum_gpu_celsius,
        "maximum_observed_sensor": guard.maximum_observed_sensor,
        "maximum_gpu_sensor": guard.maximum_gpu_sensor,
        "sample_count": guard.sample_count,
        "abort": guard.triggered.is_set(),
        "abort_reason": guard.abort_reason,
        "trigger_sensor": guard.trigger_sensor,
    }


def _finish_guarded_arm(worker: dict[str, Any], guard: ThermalGuard) -> None:
    if not guard.triggered.is_set():
        try:
            _post(f"{MANAGEMENT_URL}/api/workers/{worker['id']}/stop", timeout=30)
        except (HTTPError, URLError, TimeoutError, OSError):
            pass
    guard.worker_id = None
    guard.close()


def _pace_requests(
    guard: ThermalGuard,
    cooldown_temperature_celsius: float,
    pacing: dict[str, Any],
) -> bool:
    elapsed = guard.wait_for_cooldown(cooldown_temperature_celsius)
    pacing["cooldown_events"] += 1
    pacing["total_cooldown_seconds"] = round(pacing["total_cooldown_seconds"] + elapsed, 4)
    return not guard.triggered.is_set()


def _worker_by_id(workers: list[dict[str, Any]], worker_id: str) -> dict[str, Any]:
    matches = [worker for worker in workers if worker["id"] == worker_id or worker["name"] == worker_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one configured Worker matching {worker_id!r}")
    return matches[0]


def _validate_workers(workers: list[dict[str, Any]]) -> None:
    if not workers:
        raise RuntimeError("Supply at least one benchmark Worker")
    budgets: set[int] = set()
    for worker in workers:
        budget = worker["settings"].get("visual_token_budget")
        if worker["generation_family"] != "vision-language":
            raise RuntimeError(f"Worker {worker['name']!r} is not a vision-language Worker")
        if budget not in {70, 140, 280}:
            raise RuntimeError(
                f"Worker {worker['name']!r} does not have an allowlisted visual token budget"
            )
        if budget in budgets:
            raise RuntimeError(f"More than one Worker has visual token budget {budget}")
        budgets.add(budget)
    identities = {(worker["model_id"], worker["revision"]) for worker in workers}
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
    other_workers: list[dict[str, Any]],
    payload: dict[str, Any],
    warmups: int,
    runs: int,
    human_review: bool,
    maximum_temperature_celsius: float,
    cooldown_temperature_celsius: float,
    requests_per_thermal_batch: int,
) -> dict[str, Any]:
    for other in other_workers:
        _post(f"{MANAGEMENT_URL}/api/workers/{other['id']}/stop")
    _wait_for_cooldown(cooldown_temperature_celsius)
    guard = ThermalGuard(maximum_temperature_celsius)
    guard.worker_id = worker["id"]
    guard.start()
    pacing = {
        "requests_per_batch": requests_per_thermal_batch,
        "cooldown_events": 0,
        "total_cooldown_seconds": 0.0,
    }
    requests_since_cooldown = 0
    try:
        _post(f"{MANAGEMENT_URL}/api/workers/{worker['id']}/start")
        warmup_failures: Counter[str] = Counter()
        valid_warmups = 0
        for _ in range(warmups):
            if requests_since_cooldown >= requests_per_thermal_batch:
                if not _pace_requests(guard, cooldown_temperature_celsius, pacing):
                    warmup_failures[guard.abort_reason or "thermal_abort"] += 1
                    break
                requests_since_cooldown = 0
            warmup_sample, warmup_failure = _run_request(gateway_url, payload)
            requests_since_cooldown += 1
            if guard.triggered.is_set():
                warmup_failures[guard.abort_reason or "thermal_abort"] += 1
                break
            if warmup_sample is None:
                warmup_failures[warmup_failure or "unknown"] += 1
            else:
                valid_warmups += 1
        budget = int(worker["settings"]["visual_token_budget"])
        if guard.triggered.is_set() or valid_warmups == 0:
            _finish_guarded_arm(worker, guard)
            return {
                "visual_token_budget": budget,
                "benchmark_status": (
                    "thermal_abort" if guard.triggered.is_set() else "not_run_no_valid_warmups"
                ),
                "warmup_valid_responses": valid_warmups,
                "warmup_failure_categories": dict(sorted(warmup_failures.items())),
                **_summary([], Counter()),
                "human_review": None,
                "pacing": pacing,
                "thermal": _thermal_summary(
                    guard,
                    maximum_temperature_celsius,
                    cooldown_temperature_celsius,
                ),
            }
        samples: list[dict[str, Any]] = []
        failures: Counter[str] = Counter()
        for _ in range(runs):
            if requests_since_cooldown >= requests_per_thermal_batch:
                if not _pace_requests(guard, cooldown_temperature_celsius, pacing):
                    failures[guard.abort_reason or "thermal_abort"] += 1
                    break
                requests_since_cooldown = 0
            sample, failure = _run_request(gateway_url, payload)
            requests_since_cooldown += 1
            if guard.triggered.is_set():
                failures[guard.abort_reason or "thermal_abort"] += 1
                break
            if sample is None:
                failures[failure or "unknown"] += 1
            else:
                samples.append(sample)
        _finish_guarded_arm(worker, guard)
        result = _summary(samples, failures)
        result["visual_token_budget"] = budget
        result["benchmark_status"] = "thermal_abort" if guard.triggered.is_set() else "completed"
        result["warmup_valid_responses"] = valid_warmups
        result["warmup_failure_categories"] = dict(sorted(warmup_failures.items()))
        result["human_review"] = (
            _review(samples, budget) if human_review and not guard.triggered.is_set() else None
        )
        result["pacing"] = pacing
        result["thermal"] = _thermal_summary(
            guard,
            maximum_temperature_celsius,
            cooldown_temperature_celsius,
        )
        for sample in samples:
            sample.pop("analysis", None)
        return result
    finally:
        guard.worker_id = None
        guard.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SceneChat visual-token budgets")
    parser.add_argument("--worker-70")
    parser.add_argument("--worker-140")
    parser.add_argument("--worker-280")
    parser.add_argument("--warmups", type=int, choices=range(3, 6), default=4)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--human-review", action="store_true")
    parser.add_argument("--maximum-temperature-celsius", type=float, default=80)
    parser.add_argument("--cooldown-temperature-celsius", type=float, default=65)
    parser.add_argument("--requests-per-thermal-batch", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.runs < 50:
        parser.error("--runs must be at least 50")
    if not 65 <= arguments.maximum_temperature_celsius <= 90:
        parser.error("--maximum-temperature-celsius must be between 65 and 90")
    if not 45 <= arguments.cooldown_temperature_celsius <= 75:
        parser.error("--cooldown-temperature-celsius must be between 45 and 75")
    if arguments.cooldown_temperature_celsius >= arguments.maximum_temperature_celsius:
        parser.error("cooldown temperature must be below maximum temperature")
    if not 1 <= arguments.requests_per_thermal_batch <= 10:
        parser.error("--requests-per-thermal-batch must be between 1 and 10")

    worker_arguments = [arguments.worker_70, arguments.worker_140, arguments.worker_280]
    worker_ids = [worker_id for worker_id in worker_arguments if worker_id]
    if not worker_ids:
        parser.error("supply at least one of --worker-70, --worker-140 or --worker-280")

    configured_workers = _json_request(f"{MANAGEMENT_URL}/api/workers")
    workers = [_worker_by_id(configured_workers, worker_id) for worker_id in worker_ids]
    _validate_workers(workers)
    _validate_route()
    originally_ready = [worker for worker in workers if worker["state"] == "ready"]
    data_url, image_metadata = _image_payload()
    payload = _payload(data_url)
    gateway_port = _free_loopback_port()
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    gateway_app = create_gateway_app(
        alias_routes={ROUTE_NAME: [_profile(worker) for worker in workers]},
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
    results: list[dict[str, Any]] = []
    thermal_abort = False
    try:
        _wait_for(f"{gateway_url}/v1/health", timeout=15)
        for worker in workers:
            result = _benchmark_arm(
                gateway_url,
                worker,
                [other for other in workers if other["id"] != worker["id"]],
                payload,
                arguments.warmups,
                arguments.runs,
                arguments.human_review,
                arguments.maximum_temperature_celsius,
                arguments.cooldown_temperature_celsius,
                arguments.requests_per_thermal_batch,
            )
            results.append(result)
            if result["thermal"]["abort"]:
                thermal_abort = True
                break
    finally:
        server.should_exit = True
        gateway_thread.join(timeout=10)
        for worker in workers:
            try:
                _post(f"{MANAGEMENT_URL}/api/workers/{worker['id']}/stop")
            except (HTTPError, URLError, TimeoutError):
                pass
        if not thermal_abort:
            for worker in originally_ready:
                _post(f"{MANAGEMENT_URL}/api/workers/{worker['id']}/start")

    document = {
        "format": "modeldeck-scenechat-visual-token-benchmark",
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "route": ROUTE_NAME,
        "model_id": workers[0]["model_id"],
        "revision": workers[0]["revision"],
        "image": image_metadata,
        "warmups_per_arm": arguments.warmups,
        "request_deadline_seconds": REQUEST_DEADLINE_SECONDS,
        "maximum_temperature_celsius": arguments.maximum_temperature_celsius,
        "cooldown_temperature_celsius": arguments.cooldown_temperature_celsius,
        "requests_per_thermal_batch": arguments.requests_per_thermal_batch,
        "thermal_abort": thermal_abort,
        "arms": results,
        "privacy": "No image data, prompt text, model descriptions or credentials are retained.",
    }
    output_dir = Path("var/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"scenechat_visual_tokens_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(output)
    if thermal_abort:
        raise SystemExit("The benchmark stopped because its thermal safety guard triggered")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import base64
import hashlib
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
from modeldeck.contracts.scenechat import (
    CONTRACT_VERSION,
    CURATED_QUESTIONS,
    SYSTEM_PROMPT,
    SceneAnalysis,
    external_prompt,
)
from modeldeck.domain import WorkerDefinition
from modeldeck.gateway import create_gateway_app
from PIL import Image
from pydantic import ValidationError

MANAGEMENT_URL = "http://127.0.0.1:3600"
ROUTE_NAME = "scenechat-vision"
IMAGE_PATH = Path("/mnt/work/GitHubProjects/SceneChat/replay_assets/demo_booth.png")
EXPECTED_IMAGE_WIDTH = 1280
EXPECTED_IMAGE_HEIGHT = 720
EXPECTED_IMAGE_BYTES = 59_214
REQUEST_DEADLINE_SECONDS = 120
SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "backend"
    / "modeldeck"
    / "contracts"
    / "scenechat"
    / "scene_analysis.schema.json"
)


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
            raise RuntimeError(f"Worker {worker['name']!r} does not have an allowlisted visual token budget")
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
    expected = {
        "width": EXPECTED_IMAGE_WIDTH,
        "height": EXPECTED_IMAGE_HEIGHT,
        "bytes": EXPECTED_IMAGE_BYTES,
    }
    if metadata != expected:
        raise RuntimeError(
            "The prepared benchmark image does not match the committed 1280×720, "
            "59,214-byte SceneChat fixture"
        )
    mime_type = {"PNG": "image/png", "JPEG": "image/jpeg"}.get(image_format)
    if mime_type is None:
        raise RuntimeError("The prepared benchmark image must be PNG or JPEG")
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}", metadata


def _payload(data_url: str, question: str, maximum_tokens: int) -> dict[str, Any]:
    return {
        "automatic": True,
        "model": ROUTE_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": external_prompt(question)},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": maximum_tokens,
        "response_format": {"type": "json_object"},
        "stream": False,
    }


def _schema_valid(value: Any) -> bool:
    try:
        SceneAnalysis.model_validate(value)
    except ValidationError:
        return False
    return True


def _question_id(index: int) -> str:
    return f"question-{index + 1:02d}"


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


def _distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "minimum": round(min(values), 4) if values else None,
        "maximum": round(max(values), 4) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _summary(
    samples: list[dict[str, Any]],
    failures: Counter[str],
    *,
    token_limit: int,
) -> dict[str, Any]:
    latencies = [sample["latency_seconds"] for sample in samples]
    completions = [sample["completion_tokens"] for sample in samples]
    prompt_tokens = [sample["prompt_tokens"] for sample in samples]
    visual_tokens = [sample["visual_tokens"] for sample in samples if sample["visual_tokens"] is not None]
    throughput = [
        sample["tokens_per_second"] for sample in samples if sample["tokens_per_second"] is not None
    ]
    stage_names = (
        "preprocessing_seconds",
        "inference_seconds",
        "validation_seconds",
        "total_worker_seconds",
        "gateway_overhead_seconds",
    )
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
            "p95": _percentile([float(value) for value in completions], 0.95),
            "limit": token_limit,
            "at_limit": sum(value >= token_limit for value in completions),
        },
        "prompt_tokens": _distribution([float(value) for value in prompt_tokens]),
        "visual_tokens": _distribution([float(value) for value in visual_tokens]),
        "tokens_per_second": _distribution(throughput),
        "finish_reasons": dict(sorted(Counter(sample["finish_reason"] for sample in samples).items())),
        "stage_seconds": {
            name: _distribution([sample[name] for sample in samples if sample.get(name) is not None])
            for name in stage_names
        },
    }


def _worker_diagnostics(worker: dict[str, Any]) -> dict[str, Any] | None:
    try:
        metrics = _json_request(f"{worker['endpoint']}/metrics", timeout=5)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, TypeError, KeyError):
        return None
    diagnostic = metrics.get("last_request") if isinstance(metrics, dict) else None
    return diagnostic if isinstance(diagnostic, dict) else None


def _failure_category(code: str, diagnostic: dict[str, Any] | None) -> str:
    if diagnostic:
        category = diagnostic.get("output_failure_category")
        if category == "token_limit_reached":
            return "token_limit"
        if category in {"invalid_json", "schema_violation"}:
            return str(category)
    if code in {"generation_timeout", "benchmark_request_timeout"}:
        return "timeout"
    if code in {
        "local_route_unavailable",
        "model_not_ready",
        "worker_busy",
        "worker_unavailable",
    } or code.startswith("http_"):
        return "route_failure"
    return code


def _run_request(
    gateway_url: str,
    worker: dict[str, Any],
    payload: dict[str, Any],
    *,
    question_id: str,
    token_limit: int,
) -> tuple[dict[str, Any] | None, str | None]:
    started = time.perf_counter()
    try:
        response = _json_request(
            f"{gateway_url}/v1/chat/completions",
            payload=payload,
            timeout=REQUEST_DEADLINE_SECONDS,
        )
        latency_seconds = time.perf_counter() - started
        diagnostic = _worker_diagnostics(worker)
        if diagnostic is None:
            return None, "metrics_unavailable"
        analysis = json.loads(response["choices"][0]["message"]["content"])
        if not _schema_valid(analysis):
            return None, "schema_violation"
        usage = response.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens", -1))
        prompt_tokens = int(usage.get("prompt_tokens", -1))
        finish_reason = str(response["choices"][0].get("finish_reason") or "unknown")
        if finish_reason == "length" or completion_tokens >= token_limit:
            return None, "token_limit"
        total_worker_seconds = diagnostic.get("total_worker_seconds")
        return (
            {
                "question_id": question_id,
                "latency_seconds": latency_seconds,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_per_second": diagnostic.get("tokens_per_second"),
                "visual_tokens": diagnostic.get("visual_tokens"),
                "preprocessing_seconds": diagnostic.get("preprocessing_seconds"),
                "inference_seconds": diagnostic.get("inference_seconds"),
                "validation_seconds": diagnostic.get("validation_seconds"),
                "total_worker_seconds": total_worker_seconds,
                "gateway_overhead_seconds": (
                    max(0.0, latency_seconds - float(total_worker_seconds))
                    if isinstance(total_worker_seconds, (int, float))
                    else None
                ),
                "finish_reason": finish_reason,
                "schema_valid": True,
                "analysis": analysis,
            },
            None,
        )
    except HTTPError as error:
        try:
            code = json.load(error).get("error", {}).get("code")
        except (AttributeError, ValueError, TypeError):
            code = None
        safe_code = str(code or f"http_{error.code}")
        return None, _failure_category(safe_code, _worker_diagnostics(worker))
    except (TimeoutError, URLError):
        return None, "timeout"
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None, "invalid_response"


def _review(samples: list[dict[str, Any]], budget: int) -> dict[str, int] | None:
    if not samples:
        return None
    reviewed = []
    seen_questions: set[str] = set()
    for sample in samples:
        if sample["question_id"] in seen_questions:
            continue
        reviewed.append(sample)
        seen_questions.add(sample["question_id"])
    accepted = 0
    for sample in reviewed:
        print(
            f"\nBudget {budget}, {sample['question_id']} review sample:\n",
            file=sys.stderr,
        )
        print(json.dumps(sample["analysis"], indent=2), file=sys.stderr)
        answer = input("Does this accurately cover important visible objects and uncertainty? [y/N] ")
        accepted += answer.strip().casefold() == "y"
    return {"reviewed": len(reviewed), "accepted": accepted}


def _validate_human_review_mode(enabled: bool, input_stream: Any) -> None:
    if enabled and not input_stream.isatty():
        raise ValueError("--human-review requires an interactive terminal")


def _worker_configuration(worker: dict[str, Any]) -> dict[str, Any]:
    settings = worker["settings"]
    return {
        "worker_id": worker["id"],
        "runtime": worker["runtime"],
        "runtime_template_id": worker.get("runtime_template_id"),
        "runtime_template_version": worker.get("runtime_template_version"),
        "dtype": worker["dtype"],
        "context_length": settings.get("context_length"),
        "maximum_new_tokens": settings.get("maximum_new_tokens"),
        "visual_token_budget": settings.get("visual_token_budget"),
        "generation_timeout_seconds": settings.get("generation_timeout_seconds"),
    }


def _benchmark_arm(
    gateway_url: str,
    worker: dict[str, Any],
    other_workers: list[dict[str, Any]],
    data_url: str,
    warmups: int,
    runs_per_question: int,
    human_review: bool,
    maximum_temperature_celsius: float,
    cooldown_temperature_celsius: float,
    requests_per_thermal_batch: int,
    minimum_duration_seconds: float,
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
    token_limit = int(worker["settings"]["maximum_new_tokens"])

    def run_one(
        question: str,
        question_id: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        nonlocal requests_since_cooldown
        if requests_since_cooldown >= requests_per_thermal_batch:
            if not _pace_requests(guard, cooldown_temperature_celsius, pacing):
                return None, guard.abort_reason or "thermal_abort"
            requests_since_cooldown = 0
        sample, failure = _run_request(
            gateway_url,
            worker,
            _payload(data_url, question, token_limit),
            question_id=question_id,
            token_limit=token_limit,
        )
        requests_since_cooldown += 1
        if guard.triggered.is_set():
            return None, guard.abort_reason or "thermal_abort"
        return sample, failure

    try:
        _post(f"{MANAGEMENT_URL}/api/workers/{worker['id']}/start")
        warmup_failures: Counter[str] = Counter()
        valid_warmups = 0
        for index in range(warmups):
            question = CURATED_QUESTIONS[index % len(CURATED_QUESTIONS)]
            warmup_sample, warmup_failure = run_one(question, _question_id(index))
            if warmup_sample is None:
                warmup_failures[warmup_failure or "unknown"] += 1
                if guard.triggered.is_set():
                    break
            else:
                valid_warmups += 1
        budget = int(worker["settings"]["visual_token_budget"])
        if guard.triggered.is_set() or valid_warmups == 0:
            _finish_guarded_arm(worker, guard)
            return {
                "configuration": _worker_configuration(worker),
                "visual_token_budget": budget,
                "maximum_new_tokens": token_limit,
                "benchmark_status": (
                    "thermal_abort" if guard.triggered.is_set() else "not_run_no_valid_warmups"
                ),
                "warmup_valid_responses": valid_warmups,
                "warmup_failure_categories": dict(sorted(warmup_failures.items())),
                **_summary([], Counter(), token_limit=token_limit),
                "questions": [],
                "burn_in": None,
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
        samples_by_question: dict[str, list[dict[str, Any]]] = {}
        failures_by_question: dict[str, Counter[str]] = {}
        measured_started = time.monotonic()
        for index, question in enumerate(CURATED_QUESTIONS):
            question_id = _question_id(index)
            question_samples = samples_by_question.setdefault(question_id, [])
            question_failures = failures_by_question.setdefault(question_id, Counter())
            for _ in range(runs_per_question):
                sample, failure = run_one(question, question_id)
                if sample is None:
                    category = failure or "unknown"
                    failures[category] += 1
                    question_failures[category] += 1
                    if guard.triggered.is_set():
                        break
                else:
                    samples.append(sample)
                    question_samples.append(sample)
            if guard.triggered.is_set():
                break

        burn_in_samples: list[dict[str, Any]] = []
        burn_in_failures: Counter[str] = Counter()
        burn_in_index = 0
        while (
            not guard.triggered.is_set()
            and minimum_duration_seconds > 0
            and time.monotonic() - measured_started < minimum_duration_seconds
        ):
            question_index = burn_in_index % len(CURATED_QUESTIONS)
            sample, failure = run_one(
                CURATED_QUESTIONS[question_index],
                _question_id(question_index),
            )
            burn_in_index += 1
            if sample is None:
                burn_in_failures[failure or "unknown"] += 1
            else:
                burn_in_samples.append(sample)

        _finish_guarded_arm(worker, guard)
        result = _summary(samples, failures, token_limit=token_limit)
        question_results = []
        for index, _question in enumerate(CURATED_QUESTIONS):
            question_id = _question_id(index)
            question_results.append(
                {
                    "question_id": question_id,
                    **_summary(
                        samples_by_question.get(question_id, []),
                        failures_by_question.get(question_id, Counter()),
                        token_limit=token_limit,
                    ),
                }
            )
        result["questions"] = question_results
        result["configuration"] = _worker_configuration(worker)
        result["visual_token_budget"] = budget
        result["maximum_new_tokens"] = token_limit
        result["benchmark_status"] = "thermal_abort" if guard.triggered.is_set() else "completed"
        result["warmup_valid_responses"] = valid_warmups
        result["warmup_failure_categories"] = dict(sorted(warmup_failures.items()))
        result["burn_in"] = (
            {
                "minimum_duration_seconds": minimum_duration_seconds,
                "actual_duration_seconds": round(time.monotonic() - measured_started, 4),
                **_summary(
                    burn_in_samples,
                    burn_in_failures,
                    token_limit=token_limit,
                ),
            }
            if minimum_duration_seconds > 0
            else None
        )
        result["human_review"] = (
            _review(samples, budget) if human_review and not guard.triggered.is_set() else None
        )
        result["acceptance"] = {
            "all_questions_complete": all(
                question["measured_requests"] == runs_per_question for question in question_results
            ),
            "zero_failures": not failures,
            "zero_token_limit_hits": result["completion_tokens"]["at_limit"] == 0
            and failures["token_limit"] == 0,
            "median_at_most_8_seconds": (
                result["latency_seconds"]["p50"] is not None and result["latency_seconds"]["p50"] <= 8
            ),
            "p95_at_most_12_seconds": (
                result["latency_seconds"]["p95"] is not None and result["latency_seconds"]["p95"] <= 12
            ),
            "per_question_completion_p95_below_limit": all(
                question["completion_tokens"]["p95"] is not None
                and question["completion_tokens"]["p95"] < token_limit
                for question in question_results
            ),
            "preferred_per_question_completion_p95_at_most_260": all(
                question["completion_tokens"]["p95"] is not None
                and question["completion_tokens"]["p95"] <= 260
                for question in question_results
            ),
        }
        result["pacing"] = pacing
        result["thermal"] = _thermal_summary(
            guard,
            maximum_temperature_celsius,
            cooldown_temperature_celsius,
        )
        for sample in [*samples, *burn_in_samples]:
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
    parser.add_argument("--warmups", type=int, choices=range(2, 6), default=2)
    parser.add_argument(
        "--runs-per-question",
        "--runs",
        dest="runs_per_question",
        type=int,
        default=10,
    )
    parser.add_argument("--load-mode", choices=("isolated", "combined"), default="isolated")
    parser.add_argument("--minimum-duration-seconds", type=float, default=0)
    parser.add_argument("--human-review", action="store_true")
    parser.add_argument("--maximum-temperature-celsius", type=float, default=80)
    parser.add_argument("--cooldown-temperature-celsius", type=float, default=65)
    parser.add_argument("--requests-per-thermal-batch", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.runs_per_question < 10:
        parser.error("--runs-per-question must be at least 10")
    if arguments.minimum_duration_seconds < 0:
        parser.error("--minimum-duration-seconds cannot be negative")
    if arguments.minimum_duration_seconds and arguments.load_mode != "combined":
        parser.error("--minimum-duration-seconds requires --load-mode combined")
    if not 65 <= arguments.maximum_temperature_celsius <= 90:
        parser.error("--maximum-temperature-celsius must be between 65 and 90")
    if not 45 <= arguments.cooldown_temperature_celsius <= 75:
        parser.error("--cooldown-temperature-celsius must be between 45 and 75")
    if arguments.cooldown_temperature_celsius >= arguments.maximum_temperature_celsius:
        parser.error("cooldown temperature must be below maximum temperature")
    if not 1 <= arguments.requests_per_thermal_batch <= 10:
        parser.error("--requests-per-thermal-batch must be between 1 and 10")
    try:
        _validate_human_review_mode(arguments.human_review, sys.stdin)
    except ValueError as error:
        parser.error(str(error))

    worker_arguments = [arguments.worker_70, arguments.worker_140, arguments.worker_280]
    worker_ids = [worker_id for worker_id in worker_arguments if worker_id]
    if not worker_ids:
        parser.error("supply at least one of --worker-70, --worker-140 or --worker-280")

    configured_workers = _json_request(f"{MANAGEMENT_URL}/api/workers")
    workers = [_worker_by_id(configured_workers, worker_id) for worker_id in worker_ids]
    _validate_workers(workers)
    _validate_route()
    originally_ready = [worker for worker in configured_workers if worker["state"] == "ready"]
    data_url, image_metadata = _image_payload()
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
            workers_to_stop = [
                other
                for other in configured_workers
                if other["id"] != worker["id"]
                and (
                    other["id"] in {candidate["id"] for candidate in workers}
                    or (arguments.load_mode == "isolated" and other["state"] == "ready")
                )
            ]
            result = _benchmark_arm(
                gateway_url,
                worker,
                workers_to_stop,
                data_url,
                arguments.warmups,
                arguments.runs_per_question,
                arguments.human_review,
                arguments.maximum_temperature_celsius,
                arguments.cooldown_temperature_celsius,
                arguments.requests_per_thermal_batch,
                arguments.minimum_duration_seconds,
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
        "version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "route": ROUTE_NAME,
        "model_id": workers[0]["model_id"],
        "revision": workers[0]["revision"],
        "contract_version": CONTRACT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "schema_sha256": hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
        "image": image_metadata,
        "warmups_per_arm": arguments.warmups,
        "curated_question_count": len(CURATED_QUESTIONS),
        "runs_per_question": arguments.runs_per_question,
        "load_mode": arguments.load_mode,
        "minimum_duration_seconds": arguments.minimum_duration_seconds,
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

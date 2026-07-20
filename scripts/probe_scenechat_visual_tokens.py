from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_scenechat_visual_tokens as benchmark  # noqa: E402

VISUAL_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


def _temperatures() -> list[dict[str, Any]]:
    payload = benchmark._json_request(f"{benchmark.MANAGEMENT_URL}/api/telemetry", timeout=5)
    return [
        reading
        for reading in payload.get("temperatures", [])
        if isinstance(reading, dict) and isinstance(reading.get("celsius"), (int, float))
    ]


def _maximum_temperature(readings: list[dict[str, Any]]) -> float | None:
    values = [float(reading["celsius"]) for reading in readings]
    return max(values) if values else None


def _gpu_temperature(readings: list[dict[str, Any]]) -> float | None:
    values = [float(reading["celsius"]) for reading in readings if reading.get("source") == "amdgpu"]
    return max(values) if values else None


def _wait_for_cooldown(target: float, maximum_wait_seconds: float = 600) -> None:
    deadline = time.monotonic() + maximum_wait_seconds
    while time.monotonic() < deadline:
        maximum = _maximum_temperature(_temperatures())
        if maximum is None or maximum <= target:
            return
        time.sleep(2)
    raise RuntimeError(f"Hardware did not cool below {target:g}°C in time")


class ThermalGuard:
    def __init__(self, maximum_celsius: float) -> None:
        self.maximum_celsius = maximum_celsius
        self.maximum_observed_celsius: float | None = None
        self.maximum_gpu_celsius: float | None = None
        self.triggered = threading.Event()
        self.finished = threading.Event()
        self.worker_id: str | None = None
        self.thread = threading.Thread(target=self._monitor, name="scenechat-thermal-guard")

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.finished.set()
        self.thread.join(timeout=5)

    def _monitor(self) -> None:
        while not self.finished.wait(0.5):
            try:
                readings = _temperatures()
            except (HTTPError, URLError, TimeoutError, OSError):
                continue
            maximum = _maximum_temperature(readings)
            gpu = _gpu_temperature(readings)
            if maximum is not None:
                self.maximum_observed_celsius = max(
                    maximum,
                    self.maximum_observed_celsius or maximum,
                )
            if gpu is not None:
                self.maximum_gpu_celsius = max(gpu, self.maximum_gpu_celsius or gpu)
            if maximum is None or maximum < self.maximum_celsius:
                continue
            self.triggered.set()
            worker_id = self.worker_id
            if worker_id:
                try:
                    benchmark._post(
                        f"{benchmark.MANAGEMENT_URL}/api/workers/{worker_id}/stop",
                        timeout=30,
                    )
                except (HTTPError, URLError, TimeoutError, OSError):
                    pass
            return


def _workers_for_sweep(model_id: str) -> list[dict[str, Any]]:
    workers = benchmark._json_request(f"{benchmark.MANAGEMENT_URL}/api/workers")
    selected = []
    for budget in VISUAL_TOKEN_BUDGETS:
        matches = [
            worker
            for worker in workers
            if worker.get("model_id") == model_id
            and worker.get("settings", {}).get("visual_token_budget") == budget
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one {model_id} Worker with visual token budget {budget}")
        selected.append(matches[0])
    identities = {(worker["model_id"], worker["revision"]) for worker in selected}
    if len(identities) != 1:
        raise RuntimeError("All probe Workers must use the same pinned model revision")
    return selected


def _safe_result(
    budget: int,
    sample: dict[str, Any] | None,
    failure: str | None,
    diagnostics: dict[str, Any] | None,
    guard: ThermalGuard,
) -> dict[str, Any]:
    return {
        "visual_token_budget": budget,
        "outcome": "valid_response" if sample and sample.get("schema_valid") else "failure",
        "failure_category": failure,
        "latency_seconds": round(float(sample["latency_seconds"]), 4) if sample else None,
        "completion_tokens": sample.get("completion_tokens") if sample else None,
        "schema_valid": bool(sample and sample.get("schema_valid")),
        "worker_diagnostics": diagnostics,
        "maximum_temperature_celsius": guard.maximum_observed_celsius,
        "maximum_gpu_temperature_celsius": guard.maximum_gpu_celsius,
        "thermal_abort": guard.triggered.is_set(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one guarded SceneChat request per visual budget")
    parser.add_argument("--model-id", default="google/gemma-4-12B-it")
    parser.add_argument("--maximum-temperature-celsius", type=float, default=80)
    parser.add_argument("--cooldown-temperature-celsius", type=float, default=65)
    arguments = parser.parse_args()
    if arguments.cooldown_temperature_celsius >= arguments.maximum_temperature_celsius:
        parser.error("cooldown temperature must be below maximum temperature")

    workers = _workers_for_sweep(arguments.model_id)
    benchmark._validate_route()
    data_url, image_metadata = benchmark._image_payload()
    payload = benchmark._payload(data_url)
    gateway_port = benchmark._free_loopback_port()
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    gateway_app = benchmark.create_gateway_app(
        alias_routes={benchmark.ROUTE_NAME: [benchmark._profile(worker) for worker in workers]},
        settings=benchmark.Settings(
            gateway_port=gateway_port,
            scenechat_timeout_seconds=benchmark.REQUEST_DEADLINE_SECONDS,
        ),
    )
    server = benchmark.uvicorn.Server(
        benchmark.uvicorn.Config(
            gateway_app,
            host="127.0.0.1",
            port=gateway_port,
            access_log=False,
            log_level="warning",
        )
    )
    gateway_thread = threading.Thread(target=server.run, name="scenechat-probe-gateway")
    gateway_thread.start()
    guard = ThermalGuard(arguments.maximum_temperature_celsius)
    guard.start()
    results: list[dict[str, Any]] = []
    try:
        benchmark._wait_for(f"{gateway_url}/v1/health", timeout=15)
        for worker in workers:
            _wait_for_cooldown(arguments.cooldown_temperature_celsius)
            guard.maximum_observed_celsius = None
            guard.maximum_gpu_celsius = None
            budget = int(worker["settings"]["visual_token_budget"])
            benchmark._post(
                f"{benchmark.MANAGEMENT_URL}/api/workers/{worker['id']}/start",
                timeout=900,
            )
            guard.worker_id = worker["id"]
            sample, failure = benchmark._run_request(gateway_url, payload)
            metrics = None
            try:
                metrics_payload = benchmark._json_request(f"http://127.0.0.1:{worker['port']}/metrics")
                metrics = metrics_payload.get("last_request")
            except (HTTPError, URLError, TimeoutError, OSError):
                pass
            if sample is not None:
                sample.pop("analysis", None)
            results.append(_safe_result(budget, sample, failure, metrics, guard))
            guard.worker_id = None
            try:
                benchmark._post(
                    f"{benchmark.MANAGEMENT_URL}/api/workers/{worker['id']}/stop",
                    timeout=30,
                )
            except (HTTPError, URLError, TimeoutError, OSError):
                pass
            if guard.triggered.is_set():
                break
    finally:
        guard.worker_id = None
        guard.close()
        server.should_exit = True
        gateway_thread.join(timeout=10)
        for worker in workers:
            try:
                benchmark._post(
                    f"{benchmark.MANAGEMENT_URL}/api/workers/{worker['id']}/stop",
                    timeout=30,
                )
            except (HTTPError, URLError, TimeoutError, OSError):
                pass

    document = {
        "format": "modeldeck-scenechat-visual-token-probe",
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "route": benchmark.ROUTE_NAME,
        "model_id": workers[0]["model_id"],
        "revision": workers[0]["revision"],
        "image": image_metadata,
        "request_deadline_seconds": benchmark.REQUEST_DEADLINE_SECONDS,
        "maximum_temperature_celsius": arguments.maximum_temperature_celsius,
        "cooldown_temperature_celsius": arguments.cooldown_temperature_celsius,
        "results": results,
        "sweep_complete": len(results) == len(VISUAL_TOKEN_BUDGETS),
        "privacy": "No image data, prompt text, model descriptions or credentials are retained.",
    }
    output_dir = Path("var/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"scenechat_visual_token_probe_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import httpx
from modeldeck.compatibility import evidence_fingerprint
from PIL import Image

REPORT_FORMAT = "modeldeck-benchmark"
REPORT_VERSION = 1
ALLOWED_INITIAL_STATES = {"ready", "stopped"}
SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "backend/modeldeck/contracts/scenechat/scene_analysis_system.txt"
)


@dataclass(frozen=True)
class BenchmarkPreset:
    repetitions: int
    autoregressive_tokens: int = 64
    diffusion_tokens: int = 128
    diffusion_steps: int = 24
    vision_tokens: int = 256
    llama_tokens: int = 256


PRESETS = {
    "quick": BenchmarkPreset(repetitions=2),
    "standard": BenchmarkPreset(repetitions=5),
}


class BenchmarkError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark allowlisted ModelDeck ROCm workers.")
    parser.add_argument("--management-url", default="http://127.0.0.1:3600")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8600")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="standard")
    parser.add_argument(
        "--workers",
        nargs="+",
        required=True,
        help="Editable Worker names or immutable Worker UUIDs.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def validate_workers(workers: list[str]) -> list[str]:
    selected = list(dict.fromkeys(workers))
    if not selected:
        raise BenchmarkError("At least one physical Worker must be selected")
    return selected


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "minimum": min(values) if values else None,
        "median": median(values) if values else None,
        "p95": percentile(values, 0.95),
        "maximum": max(values) if values else None,
    }


def output_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sanitise_error(error: BaseException) -> dict[str, str]:
    message = str(error)
    message = re.sub(r"(?i)bearer\s+[^\s]+", "Bearer [redacted]", message)
    message = re.sub(r"(?i)(?:sk|hf)_[A-Za-z0-9_-]+", "[redacted-token]", message)
    message = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[redacted-image]", message)
    message = re.sub(r"/(?:mnt|home|tmp)/[^\s\"']+", "[redacted-path]", message)
    return {"category": type(error).__name__, "message": message[:500]}


def safe_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in ("memory", "swap", "temperatures", "fans")
        if payload.get(key) is not None
    }


def safe_runtime_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "runtime",
        "device",
        "device_name",
        "torch_version",
        "hip_version",
        "rocm_version",
        "transformers_version",
        "load_seconds",
        "memory_allocated_bytes",
        "memory_reserved_bytes",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
        "quantization",
        "q4_gate_calls",
        "q4_down_calls",
        "system_gtt_used_bytes",
        "system_gtt_total_bytes",
        "system_gtt_peak_used_bytes",
        "system_vram_used_bytes",
        "system_vram_total_bytes",
    }
    return {key: payload[key] for key in allowed if key in payload}


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], httpx.Headers]:
    response = client.request(method, url, json=payload)
    if not response.is_success:
        raise BenchmarkError(f"{method} {url} failed with HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError as error:
        raise BenchmarkError(f"{method} {url} returned invalid JSON") from error
    if not isinstance(body, dict):
        raise BenchmarkError(f"{method} {url} returned a non-object response")
    return body, response.headers


class BenchmarkRunner:
    def __init__(
        self,
        client: httpx.Client,
        *,
        management_url: str,
        gateway_url: str,
        preset: BenchmarkPreset,
    ) -> None:
        self.client = client
        self.management_url = management_url.rstrip("/")
        self.gateway_url = gateway_url.rstrip("/")
        self.preset = preset

    def get(self, path: str) -> dict[str, Any]:
        return request_json(self.client, "GET", self.management_url + path)[0]

    def post(self, path: str) -> dict[str, Any]:
        return request_json(self.client, "POST", self.management_url + path)[0]

    def workers(self) -> list[dict[str, Any]]:
        response = self.client.get(self.management_url + "/api/workers")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise BenchmarkError("Management workers endpoint returned a non-list response")
        return payload

    def telemetry(self) -> dict[str, Any]:
        return safe_telemetry(self.get("/api/telemetry"))

    def preflight(self, selected: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        workers = self.workers()
        resolved = []
        for selector in selected:
            matches = [
                worker
                for worker in workers
                if worker.get("id") == selector or worker.get("name") == selector
            ]
            if not matches:
                raise BenchmarkError(f"Running ModelDeck has no Worker named or identified by: {selector}")
            if len(matches) > 1:
                raise BenchmarkError(f"Worker name is ambiguous; use its UUID: {selector}")
            if matches[0].get("runtime") == "mock":
                raise BenchmarkError(f"Mock Workers are not physical benchmark targets: {selector}")
            resolved.append(dict(matches[0]))
        for worker in workers:
            if worker.get("state") not in ALLOWED_INITIAL_STATES:
                raise BenchmarkError(
                    f"Worker {worker.get('id')} is {worker.get('state')}; "
                    "benchmarks require ready or stopped workers"
                )
        live = self.get("/api/live")
        for worker in resolved:
            routes = [
                route
                for route in live.get("routes", [])
                if worker["id"] in route.get("worker_ids", [])
            ]
            if not routes:
                raise BenchmarkError(
                    f"Worker {worker['name']} is not assigned to a Route in the published Event"
                )
            if len(routes) > 1:
                raise BenchmarkError(
                    f"Worker {worker['name']} serves several published Routes; benchmark them separately"
                )
            worker["gateway_model"] = routes[0]["public_name"]
        return resolved, self.get("/api/hardware")

    def stop_all(self) -> None:
        self.post("/api/workers/stop-all")

    def start_worker(self, worker_id: str) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        worker = self.post(f"/api/workers/{worker_id}/start")
        elapsed = time.perf_counter() - started
        if worker.get("state") != "ready":
            raise BenchmarkError(f"Worker {worker_id} did not become ready")
        return worker, elapsed

    def stop_worker(self, worker_id: str) -> None:
        worker = self.post(f"/api/workers/{worker_id}/stop")
        if worker.get("state") != "stopped" or worker.get("pid") is not None:
            raise BenchmarkError(f"Worker {worker_id} did not stop cleanly")

    def worker_payload(self, endpoint: str, path: str) -> dict[str, Any]:
        return request_json(self.client, "GET", endpoint.rstrip("/") + path)[0]

    def run_autoregressive(self, profile: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": profile["gateway_model"],
            "prompt": "Summarise the role of local inference in one concise paragraph.",
            "seed": 7,
            "max_tokens": self.preset.autoregressive_tokens,
            "min_tokens": self.preset.autoregressive_tokens,
            "temperature": 0,
            "top_k": 1,
            "stream": False,
        }
        started = time.perf_counter()
        body, _ = request_json(
            self.client, "POST", self.gateway_url + "/native/autoregressive/trace", payload=payload
        )
        wall = time.perf_counter() - started
        events = body.get("events") or []
        metrics = body.get("metrics") or {}
        text = events[-1].get("text_so_far", "") if events else ""
        if not text or not metrics.get("generated_tokens"):
            raise BenchmarkError("Autoregressive benchmark returned no generated output")
        return {
            "wall_seconds": wall,
            "worker_seconds": metrics.get("total_seconds"),
            "first_output_seconds": metrics.get("first_token_seconds"),
            "throughput_tokens_per_second": metrics.get("tokens_per_second"),
            "prompt_tokens": len(body.get("prompt_token_ids") or []),
            "generated_tokens": metrics.get("generated_tokens"),
            "output_sha256": output_hash(text),
        }

    def run_llama_vulkan(self, profile: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": profile["gateway_model"],
            "messages": [
                {
                    "role": "user",
                    "content": "Summarise the role of local inference in one concise paragraph.",
                }
            ],
            "seed": 7,
            "max_tokens": self.preset.llama_tokens,
            "temperature": 0,
            "stream": False,
        }
        started = time.perf_counter()
        body, _ = request_json(
            self.client, "POST", self.gateway_url + "/v1/chat/completions", payload=payload
        )
        wall = time.perf_counter() - started
        choices = body.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        if not str(content or "").strip():
            raise BenchmarkError("GPT-OSS benchmark returned no visible generated output")
        timings = body.get("timings") or {}
        usage = body.get("usage") or {}
        worker_ms = timings.get("predicted_ms")
        return {
            "wall_seconds": wall,
            "worker_seconds": float(worker_ms) / 1000 if worker_ms is not None else None,
            "first_output_seconds": None,
            "throughput_tokens_per_second": timings.get("predicted_per_second"),
            "prompt_tokens": usage.get("prompt_tokens") or timings.get("prompt_n"),
            "generated_tokens": usage.get("completion_tokens") or timings.get("predicted_n"),
            "output_sha256": output_hash(content),
        }

    def run_diffusion(self, profile: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": profile["gateway_model"],
            "prompt": "Explain why reproducible local benchmarks are useful in concise prose.",
            "max_length": self.preset.diffusion_tokens,
            "block_length": self.preset.diffusion_tokens,
            "denoising_steps": self.preset.diffusion_steps,
            "temperature": 0.8,
            "seed": 11,
            "stream_intermediate_frames": False,
        }
        started = time.perf_counter()
        queued, _ = request_json(
            self.client, "POST", self.gateway_url + "/v1/diffuse", payload=payload
        )
        job_id = str(queued.get("job_id") or "")
        if not job_id:
            raise BenchmarkError("Diffusion benchmark returned no job identifier")
        deadline = time.monotonic() + 900
        while True:
            job, _ = request_json(self.client, "GET", f"{self.gateway_url}/v1/jobs/{job_id}")
            if job.get("state") in {"complete", "failed", "cancelled"}:
                break
            if time.monotonic() >= deadline:
                self.client.post(f"{self.gateway_url}/v1/jobs/{job_id}/cancel")
                raise BenchmarkError("Diffusion benchmark exceeded 900 seconds")
            time.sleep(0.25)
        if job.get("state") != "complete" or not str(job.get("text") or "").strip():
            raise BenchmarkError(f"Diffusion benchmark ended in state {job.get('state')}")
        return {
            "wall_seconds": time.perf_counter() - started,
            "worker_seconds": (job.get("metrics") or {}).get("total_seconds"),
            "output_sha256": output_hash(job["text"]),
        }

    def run_vision(self, profile: dict[str, Any]) -> dict[str, Any]:
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), color=(70, 100, 130)).save(buffer, "PNG")
        image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        prompt += "\n\nSelected curated question:\nDescribe the scene."
        payload = {
            "model": profile["gateway_model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": self.preset.vision_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        started = time.perf_counter()
        body, _ = request_json(
            self.client, "POST", self.gateway_url + "/v1/vision/analyse", payload=payload
        )
        wall = time.perf_counter() - started
        choices = body.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        if not content:
            raise BenchmarkError("Vision benchmark returned no structured output")
        parsed = json.loads(content)
        usage = body.get("usage") or {}
        return {
            "wall_seconds": wall,
            "prompt_tokens": usage.get("prompt_tokens"),
            "generated_tokens": usage.get("completion_tokens"),
            "output_sha256": output_hash(parsed),
        }

    def run_workload(self, profile: dict[str, Any]) -> dict[str, Any]:
        if profile.get("runtime") == "llama-vulkan":
            return self.run_llama_vulkan(profile)
        family = profile["generation_family"]
        if family == "autoregressive":
            return self.run_autoregressive(profile)
        if family == "text-diffusion":
            return self.run_diffusion(profile)
        if family == "vision-language":
            return self.run_vision(profile)
        raise BenchmarkError(f"Unsupported generation family: {family}")

    def benchmark_profile(self, profile: dict[str, Any], hardware: dict[str, Any]) -> dict[str, Any]:
        before_telemetry = self.telemetry()
        started_at = datetime.now(UTC).isoformat()
        try:
            worker, cold_wall = self.start_worker(profile["id"])
            endpoint = str(worker["endpoint"])
            model = self.worker_payload(endpoint, "/model")
            metrics_before = safe_runtime_metrics(self.worker_payload(endpoint, "/metrics"))
            self.run_workload(profile)  # excluded benchmark warm-up
            samples: list[dict[str, Any]] = []
            failures: list[dict[str, str]] = []
            for index in range(self.preset.repetitions):
                try:
                    samples.append({"iteration": index + 1, **self.run_workload(profile)})
                except Exception as error:
                    failures.append({"iteration": str(index + 1), **sanitise_error(error)})
            metrics_after = safe_runtime_metrics(self.worker_payload(endpoint, "/metrics"))
            if not samples:
                raise BenchmarkError("Every measured request failed")
            fingerprint_fields = build_fingerprint_fields(hardware, profile, model, metrics_after)
            result = {
                "worker_id": profile["id"],
                "worker_name": profile["name"],
                "route_name": profile["gateway_model"],
                "model_id": profile["model_id"],
                "model_revision": profile["revision"],
                "generation_family": profile["generation_family"],
                "runtime": profile["runtime"],
                "dtype": profile["dtype"],
                "status": "success" if not failures else "partial-failure",
                "started_at": started_at,
                "cold_start_wall_seconds": cold_wall,
                "model_load_seconds": metrics_after.get("load_seconds"),
                "fingerprint": evidence_fingerprint(fingerprint_fields),
                "fingerprint_fields": fingerprint_fields,
                "telemetry_before": before_telemetry,
                "telemetry_after": self.telemetry(),
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
                "samples": samples,
                "failures": failures,
                "summary": summarise_samples(samples, failures, self.preset.repetitions),
            }
            return result
        finally:
            self.stop_worker(profile["id"])

    def restore(self, initially_ready: list[str]) -> dict[str, Any]:
        outcomes = []
        for worker_id in initially_ready:
            try:
                worker, _ = self.start_worker(worker_id)
                outcomes.append({"worker_id": worker_id, "status": worker.get("state")})
            except Exception as error:
                outcomes.append({"worker_id": worker_id, "status": "failed", "error": sanitise_error(error)})
        return {
            "requested_ready_workers": initially_ready,
            "outcomes": outcomes,
            "passed": all(item["status"] == "ready" for item in outcomes),
        }


def build_fingerprint_fields(
    hardware: dict[str, Any],
    profile: dict[str, Any],
    model: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    configured = hardware.get("configured") or {}
    detected = hardware.get("detected") or {}
    return {
        "hardware_profile": configured.get("profile_id"),
        "fedora_version": detected.get("fedora_release"),
        "kernel": detected.get("kernel"),
        "gpu": metrics.get("device_name"),
        "gpu_architecture": configured.get("gpu_architecture"),
        "rocm_version": metrics.get("rocm_version") or metrics.get("hip_version"),
        "torch_version": metrics.get("torch_version"),
        "transformers_version": metrics.get("transformers_version"),
        "vllm_version": None,
        "model_id": model.get("model_id", profile.get("model_id")),
        "model_revision": model.get("revision", profile.get("revision")),
        "quantisation": model.get("quantization", "none"),
        "dtype": model.get("dtype", profile.get("dtype")),
        "runtime": profile.get("runtime"),
        "environment_overrides": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LD_PRELOAD": None,
        },
    }


def summarise_samples(
    samples: list[dict[str, Any]], failures: list[dict[str, str]], requested: int
) -> dict[str, Any]:
    def numbers(key: str) -> list[float]:
        return [float(sample[key]) for sample in samples if sample.get(key) is not None]

    hashes = [str(sample["output_sha256"]) for sample in samples]
    return {
        "requested_requests": requested,
        "successful_requests": len(samples),
        "failed_requests": len(failures),
        "wall_seconds": numeric_summary(numbers("wall_seconds")),
        "worker_seconds": numeric_summary(numbers("worker_seconds")),
        "first_output_seconds": numeric_summary(numbers("first_output_seconds")),
        "throughput_tokens_per_second": numeric_summary(numbers("throughput_tokens_per_second")),
        "deterministic_outputs": len(set(hashes)) <= 1,
    }


def default_output_paths() -> tuple[Path, Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = Path("var/benchmarks") / f"modeldeck-benchmark-{stamp}"
    return stem.with_suffix(".json"), stem.with_suffix(".md")


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# ModelDeck benchmark",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Preset: `{report['configuration']['preset']}`",
        f"- Status: `{report['status']}`",
        "",
    ]
    families = ("autoregressive", "text-diffusion", "vision-language")
    for family in families:
        rows = [result for result in report["results"] if result.get("generation_family") == family]
        if not rows:
            continue
        lines.extend(
            [
                f"## {family}",
                "",
                "| Worker | Status | Cold start (s) | Median request (s) | p95 request (s) "
                "| Median throughput (tok/s) | Peak device/GTT (GiB) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in rows:
            summary = item.get("summary") or {}
            wall = summary.get("wall_seconds") or {}
            throughput = summary.get("throughput_tokens_per_second") or {}
            metrics = item.get("metrics_after") or {}
            peak = metrics.get("peak_memory_allocated_bytes")
            if peak is None:
                peak = metrics.get("system_gtt_peak_used_bytes")
            lines.append(
                "| {profile} | {status} | {cold} | {wall} | {p95} | {throughput} | {peak} |".format(
                    profile=item.get("worker_name") or item.get("worker_id", "unknown"),
                    status=item.get("status", "failed"),
                    cold=format_number(item.get("cold_start_wall_seconds")),
                    wall=format_number(wall.get("median")),
                    p95=format_number(wall.get("p95")),
                    throughput=format_number(throughput.get("median")),
                    peak=format_number(peak / 1024**3 if isinstance(peak, int | float) else None),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Lifecycle restoration",
            "",
            "Passed."
            if report["restoration"]["passed"]
            else "One or more initial worker states were not restored.",
            "",
            "Performance values are observational and are not release or compatibility gates.",
            "",
        ]
    )
    return "\n".join(lines)


def format_number(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def run_benchmark(
    runner: BenchmarkRunner,
    *,
    selected: list[str],
    preset_name: str,
) -> dict[str, Any]:
    profiles, hardware = runner.preflight(selected)
    initial_workers = runner.workers()
    initially_ready = [worker["id"] for worker in initial_workers if worker.get("state") == "ready"]
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    results: list[dict[str, Any]] = []
    restoration: dict[str, Any] = {
        "requested_ready_workers": initially_ready,
        "outcomes": [],
        "passed": False,
    }
    runner.stop_all()
    try:
        for profile in profiles:
            worker_id = profile["id"]
            print(f"Benchmarking {worker_id}…", flush=True)
            try:
                results.append(runner.benchmark_profile(profile, hardware))
            except BaseException as error:
                results.append(
                    {
                        "worker_id": worker_id,
                        "worker_name": profile["name"],
                        "model_id": profile["model_id"],
                        "generation_family": profile["generation_family"],
                        "status": "failed",
                        "error": sanitise_error(error),
                    }
                )
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
    finally:
        try:
            runner.stop_all()
        finally:
            restoration = runner.restore(initially_ready)
    failed = any(result["status"] != "success" for result in results)
    status = "completed" if not failed and restoration["passed"] else "completed-with-failures"
    return {
        "format": REPORT_FORMAT,
        "format_version": REPORT_VERSION,
        "run_id": datetime.now(UTC).strftime("benchmark-%Y%m%dT%H%M%SZ"),
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "total_seconds": time.perf_counter() - started,
        "status": status,
        "configuration": {
            "preset": preset_name,
            "selected_workers": [profile["id"] for profile in profiles],
            "warmup_requests": 1,
            "measured_requests_per_worker": runner.preset.repetitions,
            "autoregressive_tokens": runner.preset.autoregressive_tokens,
            "diffusion_tokens": runner.preset.diffusion_tokens,
            "diffusion_steps": runner.preset.diffusion_steps,
            "vision_tokens": runner.preset.vision_tokens,
            "llama_tokens": runner.preset.llama_tokens,
        },
        "results": results,
        "restoration": restoration,
    }


def write_reports(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")


def report_exit_code(report: dict[str, Any]) -> int:
    return 0 if report.get("status") == "completed" else 1


def main() -> int:
    args = parse_args()
    try:
        selected = validate_workers(args.workers)
        default_json, default_markdown = default_output_paths()
        json_path = args.json_output or default_json
        markdown_path = args.markdown_output or (
            args.json_output.with_suffix(".md") if args.json_output else default_markdown
        )
        if json_path.resolve() == markdown_path.resolve():
            raise BenchmarkError("JSON and Markdown outputs must use different paths")
        timeout = httpx.Timeout(920.0, connect=3.0)
        with httpx.Client(timeout=timeout) as client:
            runner = BenchmarkRunner(
                client,
                management_url=args.management_url,
                gateway_url=args.gateway_url,
                preset=PRESETS[args.preset],
            )
            report = run_benchmark(runner, selected=selected, preset_name=args.preset)
        write_reports(report, json_path, markdown_path)
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {markdown_path}")
        return report_exit_code(report)
    except BenchmarkError as error:
        print(f"Benchmark setup failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

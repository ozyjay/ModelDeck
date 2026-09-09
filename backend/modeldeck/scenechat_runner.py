"""Staged execution using the existing benchmark's isolated gateway and payload builder."""

from __future__ import annotations

import base64
import json
import math
import socket
import subprocess
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import psutil
import uvicorn

from modeldeck.config import Settings
from modeldeck.domain import WorkerDefinition
from modeldeck.gateway import create_gateway_app
from modeldeck.scenechat_evidence import registration, verify_bundle
from modeldeck.scenechat_experiment import (
    CURATED_QUESTIONS,
    Corpus,
    Journal,
    Matrix,
    digest,
    file_digest,
    freeze,
)


def sensor_sample() -> dict[str, Any]:
    readings = psutil.sensors_temperatures()
    cpu = [
        r.current for name, rows in readings.items() if name == "k10temp" for r in rows if r.label == "Tctl"
    ]
    gpu = [r.current for name, rows in readings.items() if name == "amdgpu" for r in rows]
    if not cpu or not gpu or any(not math.isfinite(v) for v in cpu + gpu):
        raise RuntimeError("Fresh Tctl and AMD GPU thermal readings are required")
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gtt_files = sorted(Path("/sys/class/drm").glob("card[0-9]*/device/mem_info_gtt_used"))
    if not gtt_files:
        raise RuntimeError("Whole-device GTT telemetry is unavailable")
    return {
        "sampled_at": datetime.now(UTC).isoformat(),
        "monotonic": time.monotonic(),
        "tctl_celsius": max(cpu),
        "gpu_celsius": max(gpu),
        "available_bytes": memory.available,
        "swap_in_bytes": swap.sin,
        "swap_out_bytes": swap.sout,
        "gtt_bytes": sum(int(p.read_text()) for p in gtt_files),
    }


class SafetyMonitor:
    def __init__(self, path: Path, stop: Any) -> None:
        self.journal = Journal(path)
        self.stop_worker = stop
        self.finished = threading.Event()
        self.aborted = threading.Event()
        self.reason: str | None = None
        self.latest: dict[str, Any] | None = None
        self.worker_pid: int | None = None
        self.worker_metrics: Any = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="scenechat-experiment-safety")
        self.watchdog = threading.Thread(target=self._watch, daemon=True, name="scenechat-telemetry-watchdog")
        self.started_at = time.monotonic()

    def _abort(self, reason: str) -> None:
        if self.aborted.is_set():
            return
        self.reason = reason
        self.aborted.set()
        try:
            self.stop_worker()
        except Exception as cleanup:
            self.reason += f"; stop_failed:{type(cleanup).__name__}"

    def _watch(self) -> None:
        while not self.finished.wait(0.1) and not self.aborted.is_set():
            last_sample = self.latest["monotonic"] if self.latest else self.started_at
            if time.monotonic() - last_sample > 0.6:
                self._abort("telemetry_watchdog_expired")
                return

    def _run(self) -> None:
        try:
            while not self.finished.is_set() and not self.aborted.is_set():
                start = time.monotonic()
                sample = sensor_sample()
                self.latest = sample
                sample["worker_rss_bytes"] = (
                    psutil.Process(self.worker_pid).memory_info().rss if self.worker_pid else None
                )
                sample["worker_memory"] = self.worker_metrics() if self.worker_metrics else None
                self.journal.append({"kind": "telemetry", **sample})
                if max(sample["tctl_celsius"], sample["gpu_celsius"]) >= 80:
                    raise RuntimeError("temperature_limit")
                if sample["available_bytes"] < 16 * 1024**3:
                    raise RuntimeError("insufficient_host_memory")
                if time.monotonic() - start > 0.5:
                    raise RuntimeError("telemetry_deadline")
                self.finished.wait(max(0, 0.45 - (time.monotonic() - start)))
        except Exception as error:
            self._abort(str(error) if isinstance(error, RuntimeError) else type(error).__name__)
        finally:
            self.journal.close()

    def start(self) -> None:
        self.started_at = time.monotonic()
        self.thread.start()
        self.watchdog.start()

    def check(self) -> None:
        if self.aborted.is_set():
            raise RuntimeError(f"Safety abort: {self.reason}")
        if (
            not self.thread.is_alive()
            or self.latest is None
            or time.monotonic() - self.latest["monotonic"] > 1
        ):
            raise RuntimeError("Live safety monitor is unavailable or stale")

    def cooldown(self) -> float:
        started = time.monotonic()
        while time.monotonic() - started < 600:
            self.check()
            if max(self.latest["tctl_celsius"], self.latest["gpu_celsius"]) <= 65:
                return time.monotonic() - started
            self.finished.wait(0.25)
        raise RuntimeError("Cooldown exceeded ten minutes")

    def close(self) -> None:
        self.finished.set()
        self.thread.join(timeout=3)
        self.watchdog.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError("Safety monitor did not terminate")


def validate_worker(worker: dict[str, Any], configuration: Any) -> None:
    definition = WorkerDefinition.model_validate(
        {k: v for k, v in worker.items() if k in WorkerDefinition.model_fields}
    )
    if digest(definition.model_dump(mode="json")) != configuration.worker_definition_sha256:
        raise ValueError("Worker definition differs from the frozen matrix")
    if (definition.model_id, definition.revision, definition.runtime) != (
        configuration.model_id,
        configuration.revision,
        configuration.runtime,
    ):
        raise ValueError("Worker model/runtime identity differs from the frozen matrix")
    if definition.dtype != "bfloat16" or definition.generation_family != "vision-language":
        raise ValueError("Evaluation requires a dedicated BF16 SceneChat Worker")
    for key, expected in {
        "context_length": 8192,
        "maximum_new_tokens": 1024,
        "generation_timeout_seconds": 60,
        "visual_token_budget": 140,
    }.items():
        if definition.settings.get(key) != expected:
            raise ValueError(f"Evaluation setting mismatch: {key}")
    if (
        definition.settings.get("prefix_cache_enabled", False)
        or definition.settings.get("thinking_mode", "disabled") != "disabled"
    ):
        raise ValueError("Evaluation requires thinking and prefix caching disabled")


def validate_window(value: dict[str, Any], worker_ids: set[str]) -> None:
    now = datetime.now(UTC)
    if value.get("approved") is not True or not value.get("approved_by"):
        raise ValueError("An operator-approved maintenance receipt is required")
    start, end = (datetime.fromisoformat(value[key]) for key in ("starts_at", "ends_at"))
    if start.tzinfo is None or end.tzinfo is None or not start <= now < end:
        raise ValueError("The approved maintenance window is not active")
    if not worker_ids <= set(value.get("allowed_worker_ids", [])):
        raise ValueError("Maintenance approval does not cover the evaluation Workers")


def execute(
    harness: Any,
    *,
    corpus: Corpus,
    corpus_root: Path,
    matrix: Matrix,
    frozen: dict[str, Any],
    output: Path,
    port: int,
    maintenance: dict[str, Any],
) -> None:
    worker_ids = {c.worker_id for c in matrix.configurations}
    validate_window(maintenance, worker_ids)
    if not 1024 <= port <= 65535 or port in {3600, 8600, 8000} or 8610 <= port <= 8624:
        raise ValueError("Choose an explicitly reserved evaluation gateway port")
    if port not in maintenance.get("reserved_evaluation_ports", []):
        raise ValueError("Evaluation port must be recorded in the maintenance inventory")
    configured = harness._json_request(f"{harness.MANAGEMENT_URL}/api/workers")
    if any(w["port"] == port for w in configured):
        raise ValueError("Evaluation gateway port is allocated to a Worker")
    # Bind before starting Workers and retain the socket throughout the run.
    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", port))
        listener.listen(128)
        _execute_bound(
            harness, corpus, corpus_root, matrix, frozen, output, port, maintenance, configured, listener
        )
    finally:
        listener.close()


def _execute_bound(
    harness: Any,
    corpus: Corpus,
    corpus_root: Path,
    matrix: Matrix,
    frozen: dict[str, Any],
    output: Path,
    port: int,
    maintenance: dict[str, Any],
    configured: list[dict[str, Any]],
    listener: socket.socket,
) -> None:
    workers = {c.id: harness._worker_by_id(configured, c.worker_id) for c in matrix.configurations}
    stop_ids = set(maintenance.get("stop_worker_ids", []))
    if not stop_ids <= set(maintenance.get("allowed_worker_ids", [])):
        raise ValueError("Maintenance approval does not authorise every requested Worker stop")
    active_ids = {w["id"] for w in configured if w["state"] not in {"stopped", "failed"}}
    load_mode = maintenance.get("load_mode", "isolated")
    if load_mode not in {"isolated", "combined"}:
        raise ValueError("Unknown experiment load mode")
    if load_mode == "isolated" and active_ids - stop_ids:
        raise ValueError("Isolated trials require explicit permission to stop every active Worker")
    if frozen["stage"] == "sustained" and (
        load_mode != "combined" or not maintenance.get("resident_workloads")
    ):
        raise ValueError("Sustained qualification requires the recorded detector and resident workloads")
    corpus.verify_files(corpus_root, {row["image_id"] for row in frozen["requests"]})
    for configuration in matrix.configurations:
        worker = workers[configuration.id]
        validate_worker(worker, configuration)
        value = registration(configuration.worker_id)
        stage_hashes = {r["image_sha256"] for r in frozen["requests"]}
        if (
            value is None
            or value["bundle_sha256"] != configuration.bundle_sha256
            or set(value["image_hashes"]) != stage_hashes
        ):
            raise ValueError("Benchmark registration must match the bundle and exact stage image hashes")
        verify_bundle(configuration.bundle_sha256)
        if value["diagnostic_timing"] != configuration.diagnostic_timing:
            raise ValueError("Diagnostic timing changed from the frozen configuration")
        if not configuration.artifact_files:
            raise ValueError("An artifact hash inventory is required before physical trials")
        snapshot = (
            Path(worker["settings"]["cache_root"])
            / ("models--" + configuration.model_id.replace("/", "--"))
            / "snapshots"
            / configuration.revision
        )
        inventory = {str(p.relative_to(snapshot)): file_digest(p) for p in snapshot.rglob("*") if p.is_file()}
        if inventory != configuration.artifact_files:
            raise ValueError("Model/processor artifact inventory changed")
        if worker["state"] not in {"stopped", "failed"}:
            raise ValueError("Evaluation Workers must be stopped before the run")
    journal = Journal(output / "attempts.jsonl")
    journal.append(
        {
            "kind": "run_receipt",
            "schedule_sha256": frozen["sha256"],
            "matrix": matrix.model_dump(),
            "corpus_sha256": frozen["corpus_sha256"],
            "maintenance_receipt_sha256": digest(maintenance),
            "stage": frozen["stage"],
            "started_at": datetime.now(UTC).isoformat(),
        }
    )
    current: dict[str, Any] | None = None
    server = None
    gateway_thread = None

    def stop() -> None:
        monitor.worker_pid = None
        monitor.worker_metrics = None
        if current:
            harness._post(f"{harness.MANAGEMENT_URL}/api/workers/{current['id']}/stop", timeout=2)

    monitor = SafetyMonitor(output / "telemetry.jsonl", stop)
    monitor.start()
    started = time.monotonic()
    sustained_started = None
    measured_count = 0
    try:
        deadline = time.monotonic() + 2
        while monitor.latest is None and not monitor.aborted.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        monitor.check()
        for worker_id in sorted(stop_ids):
            harness._worker_by_id(configured, worker_id)
            harness._post(f"{harness.MANAGEMENT_URL}/api/workers/{worker_id}/stop", timeout=30)
            journal.append({"kind": "maintenance_stop", "worker_id": worker_id})
        image_by_id = {i.id: i for i in corpus.images}
        last_block = None
        for row in frozen["requests"]:
            validate_window(maintenance, {c.worker_id for c in matrix.configurations})
            monitor.check()
            if frozen["stage"] == "sustained" and not row["warmup"]:
                if sustained_started is None:
                    sustained_started = time.monotonic()
                target = sustained_started + measured_count * 7200 / 209
                while time.monotonic() < target:
                    monitor.check()
                    validate_window(maintenance, {c.worker_id for c in matrix.configurations})
                    time.sleep(min(0.25, max(0, target - time.monotonic())))
                measured_count += 1
            block = (row["block"], row["configuration_id"])
            if block != last_block:
                if server:
                    server.should_exit = True
                    gateway_thread.join(timeout=3)
                    if gateway_thread.is_alive():
                        raise RuntimeError("Evaluation gateway did not stop")
                stop()
                current = workers[row["configuration_id"]]
                cooldown = monitor.cooldown()
                baseline = sensor_sample()
                load_started = time.monotonic()
                harness._post(f"{harness.MANAGEMENT_URL}/api/workers/{current['id']}/start")
                health = harness._json_request(f"{current['endpoint']}/health")
                configuration = next(c for c in matrix.configurations if c.id == row["configuration_id"])
                if (
                    not health.get("ready")
                    or health.get("worker_id") != current["id"]
                    or health.get("model_revision") != current["revision"]
                    or health.get("benchmark_bundle_sha256") != configuration.bundle_sha256
                ):
                    raise RuntimeError("Started Worker did not attest its expected identity and readiness")
                refreshed = harness._worker_by_id(
                    harness._json_request(f"{harness.MANAGEMENT_URL}/api/workers"), current["id"]
                )
                monitor.worker_pid = refreshed.get("pid")
                endpoint = current["endpoint"]

                def memory_metrics(worker_endpoint=endpoint):
                    values = harness._json_request(f"{worker_endpoint}/metrics", timeout=0.2)
                    return {key: value for key, value in values.items() if "memory" in key}

                monitor.worker_metrics = memory_metrics
                journal.append(
                    {
                        "kind": "load",
                        "configuration_id": row["configuration_id"],
                        "block": row["block"],
                        "cold_load_seconds": time.monotonic() - load_started,
                        "cooldown_seconds": cooldown,
                        "baseline": baseline,
                        "health": health,
                        "model": harness._json_request(f"{current['endpoint']}/model"),
                        "metrics": harness._json_request(f"{current['endpoint']}/metrics"),
                    }
                )
                app = create_gateway_app(
                    alias_routes={harness.ROUTE_NAME: [harness._profile(current)]},
                    settings=Settings(gateway_port=port, scenechat_timeout_seconds=60),
                )
                server = uvicorn.Server(
                    uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False, log_level="warning")
                )
                # uvicorn owns its socket; use a duplicate so the reservation survives blocks.
                gateway_thread = threading.Thread(
                    target=server.run, kwargs={"sockets": [listener.dup()]}, daemon=True
                )
                gateway_thread.start()
                harness._wait_for(f"http://127.0.0.1:{port}/v1/health", timeout=15)
                last_block = block
            image = image_by_id[row["image_id"]]
            path = corpus_root / image.path
            if file_digest(path) != row["image_sha256"]:
                raise ValueError("Corpus image changed after freezing")
            request_id = str(uuid.uuid4())
            payload = harness._payload(
                "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode(),
                CURATED_QUESTIONS[int(row["question_id"][1:]) - 1],
                1024,
            )
            payload["request_id"] = request_id
            attempt_started = time.monotonic()
            journal.append(
                {
                    "kind": "started",
                    "attempt_id": row["id"],
                    "request_id": request_id,
                    "started_at": datetime.now(UTC).isoformat(),
                }
            )
            status = "invalid_response"
            try:
                request_url = (
                    f"{current['endpoint']}/v1/chat/completions"
                    if frozen["execution_path"] == "worker"
                    else f"http://127.0.0.1:{port}/v1/chat/completions"
                )
                if frozen["execution_path"] == "worker":
                    # Direct Worker requests use the same body and caller-generated identity.
                    import os
                    from urllib.request import Request, urlopen

                    upstream = Request(
                        request_url,
                        data=json.dumps({k: v for k, v in payload.items() if k != "request_id"}).encode(),
                        headers={
                            "Content-Type": "application/json",
                            "X-Request-ID": request_id,
                            "Authorization": "Bearer "
                            + os.environ.get("MODELDECK_SCENECHAT_API_KEY", "local"),
                        },
                    )
                    with urlopen(upstream, timeout=60) as response_stream:
                        response = json.load(response_stream)
                else:
                    response = harness._json_request(request_url, payload=payload, timeout=60)
                content = json.loads(response["choices"][0]["message"]["content"])
                status = "success" if harness._schema_valid(content) else "schema_violation"
            except HTTPError as error:
                status = f"http_{error.code}"
            except (OSError, TimeoutError):
                status = "timeout_or_transport_error"
            except (KeyError, ValueError, TypeError, IndexError):
                status = "invalid_response"
            duration = time.monotonic() - attempt_started
            diagnostics = harness._worker_diagnostics(current)
            if not diagnostics or diagnostics.get("request_id") != request_id:
                diagnostics = {}
                if status == "success":
                    status = "diagnostics_unavailable_or_stale"
            if diagnostics.get("token_limit_reached"):
                status = "token_limit"
            if monitor.aborted.is_set():
                status = "thermal_or_safety_abort"
            journal.append(
                {
                    **diagnostics,
                    "kind": "finished",
                    "attempt_id": row["id"],
                    "request_id": request_id,
                    "status": status,
                    "duration_seconds": duration,
                }
            )
            monitor.check()
        if frozen["stage"] == "sustained" and time.monotonic() - started < 7200:
            # Do not turn an abbreviated fixed schedule into a two-hour qualification.
            raise RuntimeError("Sustained schedule ended before two hours; qualification remains incomplete")
    finally:
        failures = []
        if server:
            server.should_exit = True
            gateway_thread.join(timeout=3)
            if gateway_thread.is_alive():
                failures.append("gateway_stop_failed")
        try:
            stop()
        except Exception as error:
            failures.append(f"worker_stop_failed:{type(error).__name__}")
        try:
            monitor.close()
        except Exception as error:
            failures.append(f"monitor_stop_failed:{type(error).__name__}")
        journal.append({"kind": "cleanup", "failures": failures, "safety_abort": monitor.reason})
        journal.close()
        if failures:
            raise RuntimeError("; ".join(failures))


def main(harness: Any, arguments: Any) -> None:
    corpus_path = Path(arguments.corpus)
    corpus = Corpus.model_validate_json(corpus_path.read_text())
    matrix = Matrix.model_validate_json(Path(arguments.configuration_matrix).read_text())
    frozen = freeze(corpus, matrix, arguments.stage, arguments.execution_path)
    corpus.verify_files(corpus_path.parent, {row["image_id"] for row in frozen["requests"]})
    schedule_path = Path(arguments.schedule)
    if arguments.execute:
        if json.loads(schedule_path.read_text()) != frozen:
            raise ValueError("Schedule, corpus, matrix, contract or gates changed after freezing")
        if arguments.execution_path == "direct":
            if len(matrix.configurations) != 1 or not arguments.maintenance_receipt:
                raise ValueError("Direct diagnostics require one configuration and a maintenance receipt")
            from modeldeck.scenechat_evidence import launch_environment

            configuration = matrix.configurations[0]
            configured = harness._json_request(f"{harness.MANAGEMENT_URL}/api/workers")
            worker = harness._worker_by_id(configured, configuration.worker_id)
            validate_worker(worker, configuration)
            if worker["state"] != "stopped":
                raise ValueError("Stop the evaluation Worker before direct-runtime diagnostics")
            maintenance = json.loads(Path(arguments.maintenance_receipt).read_text())
            validate_window(maintenance, {configuration.worker_id})
            output = schedule_path.parent / f"run-{uuid.uuid4()}"
            output.mkdir()
            worker_path = output / "worker.json"
            worker_path.write_text(json.dumps(worker))
            python = Path(".venv-rocm72/bin/python").absolute()
            import os

            environment = dict(os.environ)
            launch_environment(configuration.worker_id, python, environment)
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "modeldeck.scenechat_direct",
                    "--worker",
                    str(worker_path.absolute()),
                    "--corpus",
                    str(corpus_path.absolute()),
                    "--matrix",
                    str(Path(arguments.configuration_matrix).absolute()),
                    "--schedule",
                    str(schedule_path.absolute()),
                    "--maintenance",
                    str(Path(arguments.maintenance_receipt).absolute()),
                ],
                env=environment,
                check=True,
            )
            return
        if arguments.evaluation_port is None or not arguments.maintenance_receipt:
            raise ValueError("Execution requires an explicit port and maintenance receipt")
        execute(
            harness,
            corpus=corpus,
            corpus_root=corpus_path.parent,
            matrix=matrix,
            frozen=frozen,
            output=schedule_path.parent / f"run-{uuid.uuid4()}",
            port=arguments.evaluation_port,
            maintenance=json.loads(Path(arguments.maintenance_receipt).read_text()),
        )
    else:
        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        with schedule_path.open("x") as stream:
            json.dump(frozen, stream, indent=2)
            stream.write("\n")
        print(
            f"Frozen {arguments.stage} schedule: {schedule_path}; "
            "no inference or lifecycle operations performed."
        )

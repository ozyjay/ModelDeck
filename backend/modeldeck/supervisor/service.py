from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from modeldeck.profiles import ModelProfile
from modeldeck.protocol import WorkerEvent, WorkerState


@dataclass
class ManagedWorker:
    profile: ModelProfile
    state: WorkerState = WorkerState.STOPPED
    process: asyncio.subprocess.Process | None = None
    started_at: datetime | None = None
    last_error: str | None = None
    log_session_id: str | None = None
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.profile.id,
            "state": self.state,
            "model_id": self.profile.model_id,
            "generation_family": self.profile.generation_family,
            "runtime": self.profile.preferred_runtime,
            "lifecycle": self.profile.lifecycle,
            "alias": self.profile.alias,
            "endpoint": f"http://127.0.0.1:{self.profile.port}",
            "port": self.profile.port,
            "pid": self.process.pid if self.process and self.process.returncode is None else None,
            "started_at": self.started_at,
            "last_error": self.last_error,
            "log_session_id": self.log_session_id,
            "capabilities": self.profile.capabilities.model_dump(),
        }


class WorkerSupervisor:
    _MAX_LOG_RECORDS = 500

    def __init__(
        self,
        profiles: list[ModelProfile],
        *,
        startup_timeout: float = 10.0,
        stop_timeout: float = 4.0,
        log_dir: Path | None = None,
    ) -> None:
        self.workers = {profile.id: ManagedWorker(profile=profile) for profile in profiles}
        self.startup_timeout = startup_timeout
        self.stop_timeout = stop_timeout
        self._load_lock = asyncio.Lock()
        self._worker_locks = defaultdict(asyncio.Lock)
        self._events: asyncio.Queue[WorkerEvent] = asyncio.Queue(maxsize=256)
        self._event_history: deque[WorkerEvent] = deque(maxlen=256)
        self._logs: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self._MAX_LOG_RECORDS)
        )
        self.log_dir = log_dir
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted_logs()

    def list_workers(self) -> list[dict[str, Any]]:
        self._refresh_exits()
        return [worker.snapshot() for worker in self.workers.values()]

    def get_worker(self, worker_id: str) -> dict[str, Any]:
        self._refresh_exits()
        return self._require(worker_id).snapshot()

    def register_profile(self, profile: ModelProfile) -> None:
        if profile.id in self.workers:
            raise ValueError(f"Worker profile already exists: {profile.id}")
        if any(worker.profile.port == profile.port for worker in self.workers.values()):
            raise ValueError(f"Worker port is already assigned: {profile.port}")
        self.workers[profile.id] = ManagedWorker(profile=profile)

    async def remove_profile(self, worker_id: str) -> None:
        worker = self._require(worker_id)
        async with self._worker_locks[worker_id]:
            self._refresh_exits()
            if worker.process and worker.process.returncode is None:
                raise RuntimeError("Stop the worker before removing its runtime configuration")
            if worker.state not in {WorkerState.STOPPED, WorkerState.FAILED}:
                raise RuntimeError("Wait for the worker lifecycle transition to finish")
            del self.workers[worker_id]
            self._worker_locks.pop(worker_id, None)
            self._logs.pop(worker_id, None)

    def logs(self, worker_id: str) -> list[dict[str, Any]]:
        worker = self._require(worker_id)
        records = list(self._logs[worker_id])
        if worker.log_session_id is None:
            return records
        return [record for record in records if record.get("session_id") == worker.log_session_id]

    async def start(self, worker_id: str) -> dict[str, Any]:
        worker = self._require(worker_id)
        async with self._load_lock, self._worker_locks[worker_id]:
            if worker.process and worker.process.returncode is None:
                return worker.snapshot()
            if worker.profile.lifecycle.value == "exclusive":
                for other_id, other in self.workers.items():
                    if (
                        other_id != worker_id
                        and other.profile.lifecycle.value == "exclusive"
                        and other.process
                        and other.process.returncode is None
                    ):
                        await self.stop(other_id)
            await self._transition(worker, WorkerState.VALIDATING, "Validating allowlisted worker manifest")
            if not port_available(worker.profile.port):
                worker.last_error = f"Port {worker.profile.port} is already in use"
                await self._transition(worker, WorkerState.FAILED, worker.last_error)
                raise RuntimeError(worker.last_error)

            try:
                launch = build_worker_launch(worker.profile)
            except ValueError as error:
                worker.last_error = str(error)
                await self._transition(worker, WorkerState.FAILED, worker.last_error)
                raise RuntimeError(worker.last_error) from error
            await self._transition(worker, WorkerState.STARTING, "Starting isolated worker process")
            worker.log_session_id = uuid.uuid4().hex
            try:
                worker.process = await asyncio.create_subprocess_exec(
                    *launch.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=launch.environment,
                )
            except OSError as error:
                worker.last_error = str(error)
                await self._transition(worker, WorkerState.FAILED, f"Worker launch failed: {error}")
                raise RuntimeError(worker.last_error) from error
            worker.started_at = datetime.now(UTC)
            worker.last_error = None
            worker.tasks = {
                asyncio.create_task(self._capture(worker, worker.process.stdout, "stdout")),
                asyncio.create_task(self._capture(worker, worker.process.stderr, "stderr")),
                asyncio.create_task(self._watch_exit(worker)),
            }
            await self._transition(worker, WorkerState.LOADING, "Waiting for worker health and model load")
            try:
                startup_timeout = float(
                    worker.profile.settings.get("startup_timeout_seconds", self.startup_timeout)
                )
                await asyncio.wait_for(self._wait_until_loaded(worker), timeout=startup_timeout)
                await self._transition(worker, WorkerState.WARMING, "Running configured warmup")
                warmup_timeout = float(worker.profile.settings.get("warmup_timeout_seconds", 10.0))
                async with httpx.AsyncClient(timeout=warmup_timeout) as client:
                    response = await client.post(f"http://127.0.0.1:{worker.profile.port}/warmup")
                    response.raise_for_status()
                    if response.json().get("ready") is not True:
                        raise RuntimeError("worker warmup did not report readiness")
                await self._transition(worker, WorkerState.READY, "Worker passed health and warmup checks")
            except Exception as error:
                worker.last_error = f"Startup failed: {type(error).__name__}: {error}"
                await self._transition(worker, WorkerState.FAILED, worker.last_error)
                await self._terminate(worker)
                raise RuntimeError(worker.last_error) from error
            return worker.snapshot()

    async def stop(self, worker_id: str) -> dict[str, Any]:
        worker = self._require(worker_id)
        async with self._worker_locks[worker_id]:
            if not worker.process or worker.process.returncode is not None:
                worker.process = None
                await self._transition(worker, WorkerState.STOPPED, "Worker is stopped")
                return worker.snapshot()
            await self._transition(worker, WorkerState.STOPPING, "Requesting graceful worker shutdown")
            try:
                async with httpx.AsyncClient(timeout=1.5) as client:
                    await client.post(f"http://127.0.0.1:{worker.profile.port}/shutdown")
            except httpx.HTTPError:
                pass
            try:
                await asyncio.wait_for(worker.process.wait(), timeout=self.stop_timeout)
            except TimeoutError:
                await self._terminate(worker)
            worker.process = None
            worker.started_at = None
            await self._transition(worker, WorkerState.STOPPED, "Worker stopped and process exited")
            return worker.snapshot()

    async def restart(self, worker_id: str) -> dict[str, Any]:
        await self.stop(worker_id)
        return await self.start(worker_id)

    async def stop_all(self) -> None:
        for worker_id in self.workers:
            await self.stop(worker_id)

    async def next_event(self) -> WorkerEvent:
        return await self._events.get()

    def event_history(self) -> list[dict[str, Any]]:
        return [event.model_dump(mode="json") for event in self._event_history]

    async def _wait_until_loaded(self, worker: ManagedWorker) -> None:
        url = f"http://127.0.0.1:{worker.profile.port}/health"
        async with httpx.AsyncClient(timeout=0.5) as client:
            while True:
                if worker.process is None or worker.process.returncode is not None:
                    raise RuntimeError("worker exited before readiness")
                try:
                    response = await client.get(url)
                    payload = response.json()
                    if payload.get("state") == WorkerState.FAILED:
                        raise RuntimeError(payload.get("error") or "worker model load failed")
                    if response.is_success and (
                        payload.get("ready") is True or payload.get("state") == WorkerState.WARMING
                    ):
                        return
                except (httpx.HTTPError, ValueError):
                    pass
                await asyncio.sleep(0.5)

    async def _capture(
        self,
        worker: ManagedWorker,
        stream: asyncio.StreamReader | None,
        source: str,
    ) -> None:
        if stream is None:
            return
        while line := await stream.readline():
            message = redact_log(line.decode(errors="replace").rstrip())
            if message:
                self._append_log(worker.profile.id, source, message)

    def _append_log(self, worker_id: str, source: str, message: str) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "source": source,
            "level": classify_log_level(message),
            "message": redact_log(message),
        }
        session_id = self._require(worker_id).log_session_id
        if session_id is not None:
            record["session_id"] = session_id
        logs = self._logs[worker_id]
        was_full = len(logs) == self._MAX_LOG_RECORDS
        logs.append(record)
        if self.log_dir is None:
            return
        path = self.log_dir / f"{worker_id}.jsonl"
        if was_full:
            self._write_log_file(path, logs)
        else:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _load_persisted_logs(self) -> None:
        if self.log_dir is None:
            return
        for worker_id in self.workers:
            path = self.log_dir / f"{worker_id}.jsonl"
            if not path.is_file():
                continue
            logs = self._logs[worker_id]
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(record, dict) and {"timestamp", "source", "message"} <= record.keys():
                    logs.append(record)
            self._write_log_file(path, logs)

    @staticmethod
    def _write_log_file(path: Path, logs: deque[dict[str, Any]]) -> None:
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in logs:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        temporary.replace(path)

    async def _watch_exit(self, worker: ManagedWorker) -> None:
        process = worker.process
        if process is None:
            return
        return_code = await process.wait()
        if worker.state not in {WorkerState.STOPPING, WorkerState.STOPPED, WorkerState.FAILED}:
            worker.last_error = f"Worker process exited unexpectedly with code {return_code}"
            await self._transition(worker, WorkerState.FAILED, worker.last_error)

    async def _terminate(self, worker: ManagedWorker) -> None:
        process = worker.process
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except TimeoutError:
            process.kill()
            await process.wait()

    async def _transition(self, worker: ManagedWorker, state: WorkerState, message: str) -> None:
        worker.state = state
        event = WorkerEvent(worker_id=worker.profile.id, state=state, message=message)
        self._event_history.append(event)
        if self._events.full():
            self._events.get_nowait()
        self._events.put_nowait(event)

    def _refresh_exits(self) -> None:
        for worker in self.workers.values():
            if (
                worker.process
                and worker.process.returncode is not None
                and worker.state
                not in {
                    WorkerState.STOPPED,
                    WorkerState.FAILED,
                }
            ):
                worker.state = WorkerState.FAILED
                worker.last_error = f"Worker process exited with code {worker.process.returncode}"

    def _require(self, worker_id: str) -> ManagedWorker:
        try:
            return self.workers[worker_id]
        except KeyError as error:
            raise KeyError(f"Unknown worker: {worker_id}") from error


@dataclass(frozen=True)
class WorkerLaunch:
    command: list[str]
    environment: dict[str, str]


def build_worker_launch(profile: ModelProfile) -> WorkerLaunch:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    common = [
        "--worker-id",
        profile.id,
        "--model-id",
        profile.model_id,
        "--revision",
        profile.revision,
        "--port",
        str(profile.port),
    ]
    builder = TRUSTED_LAUNCH_BUILDERS.get(profile.preferred_runtime)
    if builder is None:
        raise ValueError(f"Runtime launch is not implemented: {profile.preferred_runtime}")
    return builder(profile, environment, common)


LaunchBuilder = Callable[[ModelProfile, dict[str, str], list[str]], WorkerLaunch]


def _mock_launch(profile: ModelProfile, environment: dict[str, str], common: list[str]) -> WorkerLaunch:
    return WorkerLaunch(
        command=[
            sys.executable,
            "-m",
            "modeldeck.workers.mock_worker",
            *common,
            "--family",
            profile.generation_family.value,
        ],
        environment=environment,
    )


def _rocm_python() -> Path:
    python = Path(os.environ.get("MODELDECK_ROCM72_PYTHON", ".venv-rocm72/bin/python")).expanduser()
    if not python.is_file():
        raise ValueError("ROCm 7.2 runtime is missing; run pwsh -NoProfile -File scripts/setup.ps1")
    return python


def _autoregressive_launch(
    profile: ModelProfile, environment: dict[str, str], common: list[str]
) -> WorkerLaunch:
    python = _rocm_python()
    if cache_root := profile.settings.get("cache_root"):
        environment["HF_HUB_CACHE"] = str(cache_root)
    return WorkerLaunch(
        command=[
            str(python.absolute()),
            "-m",
            "modeldeck.workers.autoregressive_worker",
            *common,
            "--dtype",
            profile.dtype,
            "--context-length",
            str(profile.settings.get("context_length", 2048)),
            "--maximum-new-tokens",
            str(profile.settings.get("maximum_new_tokens", 128)),
        ],
        environment=environment,
    )


def _vision_language_launch(
    profile: ModelProfile, environment: dict[str, str], common: list[str]
) -> WorkerLaunch:
    python = _rocm_python()
    cache_root = profile.settings.get("cache_root")
    if not cache_root:
        raise ValueError("SceneChat worker requires an allowlisted Hugging Face cache root")
    environment["HF_HUB_CACHE"] = str(cache_root)
    environment["MODELDECK_SCENECHAT_API_KEY"] = os.environ.get("MODELDECK_SCENECHAT_API_KEY", "local")
    return WorkerLaunch(
        command=[
            str(python.absolute()),
            "-m",
            "modeldeck.workers.scenechat_worker",
            *common,
            "--cache-root",
            str(cache_root),
            "--dtype",
            profile.dtype,
            "--context-length",
            str(profile.settings.get("context_length", 8192)),
            "--maximum-new-tokens",
            str(profile.settings.get("maximum_new_tokens", 512)),
            "--generation-timeout-seconds",
            str(profile.settings.get("generation_timeout_seconds", 60)),
        ],
        environment=environment,
    )


def _llama_vulkan_launch(
    profile: ModelProfile, environment: dict[str, str], common: list[str]
) -> WorkerLaunch:
    artifact_path = profile.settings.get("artifact_path")
    if not artifact_path:
        raise ValueError("GPT-OSS Vulkan worker requires an allowlisted GGUF artefact")
    return WorkerLaunch(
        command=[
            sys.executable,
            "-m",
            "modeldeck.workers.llama_vulkan_worker",
            *common,
            "--artifact-path",
            str(artifact_path),
            "--context-length",
            str(profile.settings.get("context_length", 8192)),
            "--maximum-new-tokens",
            str(profile.settings.get("maximum_new_tokens", 256)),
            "--execution-preset",
            str(profile.settings.get("execution_preset", "vulkan-full")),
        ],
        environment=environment,
    )


def _moshiko_launch(profile: ModelProfile, environment: dict[str, str], common: list[str]) -> WorkerLaunch:
    python = Path(os.environ.get("MODELDECK_MOSHIKO_PYTHON", ".venv-moshi-rocm72/bin/python")).expanduser()
    if not python.is_file():
        raise ValueError(
            "Moshiko ROCm runtime is missing; run pwsh -NoProfile -File scripts/setup_moshiko_rocm72.ps1"
        )
    cache_root = profile.settings.get("cache_root")
    if not cache_root:
        raise ValueError("Moshiko worker requires an allowlisted Hugging Face cache root")
    environment["HF_HUB_CACHE"] = str(cache_root)
    return WorkerLaunch(
        command=[
            str(python.absolute()),
            "-m",
            "modeldeck.workers.moshiko_worker",
            *common,
            "--alias",
            profile.alias,
            "--cache-root",
            str(cache_root),
        ],
        environment=environment,
    )


def _text_diffusion_launch(
    profile: ModelProfile, environment: dict[str, str], common: list[str]
) -> WorkerLaunch:
    is_q4 = profile.preferred_runtime == "text-diffusion-gptq-rocm"
    if is_q4:
        configured_python = os.environ.get("MODELDECK_ROCM72_Q4_PYTHON")
        default_q4_python = Path(".venv-rocm72-q4/bin/python")
        if configured_python:
            python = Path(configured_python).expanduser()
        elif default_q4_python.is_file():
            python = default_q4_python
        else:
            python = Path(os.environ.get("MODELDECK_ROCM72_PYTHON", default_q4_python)).expanduser()
    else:
        python = Path(os.environ.get("MODELDECK_ROCM72_PYTHON", ".venv-rocm72/bin/python")).expanduser()
    if not python.is_file():
        if is_q4:
            raise ValueError(
                "Q4 ROCm runtime is missing; create .venv-rocm72-q4 and install "
                "requirements-rocm72-q4-gptqmodel.txt"
            )
        raise ValueError("ROCm 7.2 runtime is missing; run pwsh -NoProfile -File scripts/setup.ps1")
    environment.pop("LD_PRELOAD", None)
    if cache_root := profile.settings.get("cache_root"):
        environment["HF_HUB_CACHE"] = str(cache_root)
    elif is_q4:
        environment.pop("HF_HUB_CACHE", None)
    if profile.settings.get("hsa_preload_evidence"):
        hsa_runtime = Path("/usr/lib64/libhsa-runtime64.so.1")
        if not hsa_runtime.is_file():
            raise ValueError("Evidence-gated HSA runtime preload library is missing")
        environment["LD_PRELOAD"] = str(hsa_runtime)
    command = [
        str(python.absolute()),
        "-m",
        "modeldeck.workers.text_diffusion_worker",
        *common,
        "--dtype",
        profile.dtype,
        "--maximum-new-tokens",
        str(profile.settings.get("maximum_new_tokens", 256)),
        "--maximum-denoising-steps",
        str(profile.settings.get("maximum_denoising_steps", 48)),
    ]
    if is_q4:
        if cache_root := profile.settings.get("cache_root"):
            command.extend(["--cache-root", str(cache_root)])
        command.extend(["--q4-checkpoint-dir", str(profile.settings["q4_checkpoint_dir"])])
    return WorkerLaunch(command=command, environment=environment)


TRUSTED_LAUNCH_BUILDERS: dict[str, LaunchBuilder] = {
    "mock": _mock_launch,
    "transformers-rocm": _autoregressive_launch,
    "vision-language-transformers-rocm": _vision_language_launch,
    "text-diffusion-transformers-rocm": _text_diffusion_launch,
    "text-diffusion-gptq-rocm": _text_diffusion_launch,
    "llama-vulkan": _llama_vulkan_launch,
    "moshiko-rocm": _moshiko_launch,
}


def build_mock_worker_command(profile: ModelProfile) -> list[str]:
    if profile.preferred_runtime != "mock":
        raise ValueError("Profile is not a mock runtime")
    return build_worker_launch(profile).command


def classify_log_level(message: str) -> str:
    lowered = message.casefold()
    if "{{- raise_exception(" in lowered:
        return "info"
    if any(
        marker in lowered
        for marker in ("error", "exception", "traceback", "critical", "out of memory", "oom")
    ):
        return "error"
    if "warning" in lowered or lowered.lstrip().startswith("warn") or "skipping import" in lowered:
        return "warning"
    return "info"


def port_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def redact_log(message: str) -> str:
    lowered = message.lower()
    for marker in (
        "authorization:",
        "hf_token=",
        "api_key=",
        "prompt=",
        "generated_text=",
        "image_url=",
        ";base64,",
    ):
        index = lowered.find(marker)
        if index >= 0:
            return f"{message[:index]}{marker}[redacted]"
    if message.startswith("{"):
        try:
            payload = json.loads(message)
            for key in (
                "prompt",
                "messages",
                "generated_text",
                "image_url",
                "content",
                "token",
                "api_key",
                "authorization",
            ):
                if key in payload:
                    payload[key] = "[redacted]"
            return json.dumps(payload, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError):
            pass
    return message

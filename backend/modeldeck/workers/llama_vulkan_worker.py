from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import socket
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from modeldeck.llama_runtime import (
    GPT_OSS_LLAMA_REQUIRED_FLAGS,
    ValidatedLlamaInstallation,
    ValidatedQwenRuntime,
    configuration_fingerprint,
    validate_llama_installation,
    validate_qwen_runtime,
)
from modeldeck.protocol import GenerationFamily

REASONING_MARKERS = re.compile(
    r"<\|(?:analysis|reasoning|channel)[^>]*\>.*?<\|(?:end|final)[^>]*\>", re.DOTALL
)
AMD_VENDOR_ID = "0x1002"
QWEN_OPENAI_REQUEST_FIELDS = frozenset(
    {
        "frequency_penalty",
        "logit_bias",
        "max_completion_tokens",
        "max_tokens",
        "messages",
        "model",
        "n",
        "presence_penalty",
        "prompt",
        "reasoning_effort",
        "response_format",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_p",
        "user",
    }
)
QWEN_REASONING_EFFORTS = frozenset({"default", "none", "minimal", "low", "medium", "high", "xhigh", "max"})
_QWEN_TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^\s>]+)>\s*(?P<body>.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_QWEN_TOOL_PARAMETER = re.compile(r"<parameter=(?P<name>[^\s>]+)>\s*(?P<value>.*?)\s*</parameter>", re.DOTALL)


def fixed_llama_server() -> Path:
    return Path(".runtime-tools/llama.cpp/bin/llama-server").resolve()


def gpt_oss_configuration_fingerprint(
    args: argparse.Namespace, installation: ValidatedLlamaInstallation
) -> str:
    payload = {
        "model_id": args.model_id,
        "model_revision": args.revision,
        "context_length": args.context_length,
        "execution_preset": args.execution_preset,
        "llama_cpp_commit": installation.receipt.commit,
        "llama_server_sha256": installation.executable_sha256,
        "llama_build_receipt_sha256": installation.receipt_sha256,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def llama_command(
    *, model: Path, port: int, context_length: int, preset: str, executable: Path | None = None
) -> list[str]:
    if preset != "vulkan-full":
        raise ValueError("Unknown allowlisted GPT-OSS execution preset")
    executable = executable or fixed_llama_server()
    if not executable.is_file():
        raise ValueError(
            "Pinned llama.cpp Vulkan runtime is missing; run "
            "pwsh -NoProfile -File scripts/setup_llama_vulkan.ps1"
        )
    allowed_names = {
        "gpt-oss-120b-MXFP4.gguf",
        "gpt-oss-120b-mxfp4-00001-of-00003.gguf",
    }
    if not model.is_file() or model.name not in allowed_names:
        raise ValueError("The allowlisted GPT-OSS MXFP4 GGUF artefact is missing")
    command = [
        str(executable),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        str(model),
        "--ctx-size",
        str(context_length),
        "--parallel",
        "1",
        "--n-gpu-layers",
        "999",
        "--flash-attn",
        "on",
        "--jinja",
    ]
    return command


def qwen_llama_command(
    *, runtime: ValidatedQwenRuntime, port: int, thinking_mode: str | None = None
) -> list[str]:
    manifest = runtime.manifest
    effective_thinking_mode = thinking_mode or (
        "disabled" if manifest.id == "qwen35-4b-q8-vulkan" else "adaptive"
    )
    if effective_thinking_mode not in {"adaptive", "disabled"}:
        raise ValueError("The Qwen llama.cpp command requires a trusted thinking mode")
    command = [
        str(runtime.executable),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        str(runtime.model),
        "--ctx-size",
        str(manifest.context_length),
        "--parallel",
        "1",
        "--device",
        "Vulkan0",
        "--gpu-layers",
        "all",
        "--fit",
        "off",
        "--flash-attn",
        "on",
        "--cache-type-k",
        manifest.cache_type_k,
        "--cache-type-v",
        manifest.cache_type_v,
        "--jinja",
        "--reasoning-format",
        "deepseek",
        "--metrics",
        "--slots",
        "--offline",
        "--log-colors",
        "off",
        "-lv",
        "4",
    ]
    if runtime.projector is not None:
        command.extend(["--mmproj", str(runtime.projector)])
    else:
        command.append("--no-mmproj")
    if effective_thinking_mode == "disabled":
        command.extend(["--reasoning-effort", "none"])
    if runtime.mtp_model is not None and manifest.mtp_draft_tokens is not None:
        command.extend(
            [
                "--spec-type",
                "draft-mtp",
                "--spec-draft-model",
                str(runtime.mtp_model),
                "--spec-draft-device",
                "Vulkan0",
                "--spec-draft-ngl",
                "all",
                "--spec-draft-n-max",
                str(manifest.mtp_draft_tokens),
            ]
        )
    return command


def allocate_private_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class LlamaEvidence:
    """Bounded startup evidence and speculative-decoding metrics from llama-server."""

    _DRAFT = re.compile(
        r"draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+) accepted\s*/\s*(\d+) generated",
        re.IGNORECASE,
    )
    _PROMPT = re.compile(r"prompt eval time.+?([0-9.]+) tokens per second", re.IGNORECASE)
    _GENERATION = re.compile(r"eval time.+?([0-9.]+) tokens per second", re.IGNORECASE)

    def __init__(self) -> None:
        self.lines: deque[str] = deque(maxlen=500)
        self.backend_vulkan = False
        self.device_expected = False
        self.full_offload = False
        self.architecture_qwen = False
        self.quantisation_loaded = False
        self.projector_loaded = False
        self.mtp_enabled = False
        self.draft_proposed = 0
        self.draft_accepted = 0
        self.acceptance_ratio: float | None = None
        self.prompt_tokens_per_second: float | None = None
        self.generated_tokens_per_second: float | None = None

    def feed(self, line: str, *, quantisation: str) -> None:
        self.lines.append(line)
        lowered = line.casefold()
        self.backend_vulkan |= "vulkan" in lowered
        self.device_expected |= "vulkan0" in lowered and any(
            marker in lowered for marker in ("radeon", "amd", "gfx1151")
        )
        self.full_offload |= bool(
            re.search(r"offload(?:ed|ing).*(?:all|\d+\s*/\s*\d+).*layer", lowered)
            or "all layers to gpu" in lowered
        )
        self.architecture_qwen |= "qwen35" in lowered or "qwen3.8" in lowered
        self.quantisation_loaded |= quantisation.casefold() in lowered
        self.projector_loaded |= "mmproj" in lowered and any(
            marker in lowered for marker in ("load", "vision", "projector")
        )
        self.mtp_enabled |= "draft-mtp" in lowered or "spec_type" in lowered and "mtp" in lowered
        if match := self._DRAFT.search(line):
            self.acceptance_ratio = float(match.group(1))
            self.draft_accepted = int(match.group(2))
            self.draft_proposed = int(match.group(3))
        if match := self._PROMPT.search(line):
            self.prompt_tokens_per_second = float(match.group(1))
        elif match := self._GENERATION.search(line):
            self.generated_tokens_per_second = float(match.group(1))

    def startup_checks(
        self, *, projector_required: bool = True, mtp_required: bool = True
    ) -> dict[str, bool]:
        checks = {
            "Vulkan backend": self.backend_vulkan,
            "expected AMD Vulkan device": self.device_expected,
            "complete GPU layer offload": self.full_offload,
            "Qwen architecture": self.architecture_qwen,
            "expected quantisation": self.quantisation_loaded,
        }
        if projector_required:
            checks["BF16 vision projector"] = self.projector_loaded
        if mtp_required:
            checks["MTP speculative decoder"] = self.mtp_enabled
        return checks

    def startup_errors(self, *, projector_required: bool = True, mtp_required: bool = True) -> list[str]:
        checks = self.startup_checks(projector_required=projector_required, mtp_required=mtp_required)
        return [name for name, passed in checks.items() if not passed]

    def record_generation_timings(self, payload: Any) -> None:
        if not isinstance(payload, dict) or not isinstance(payload.get("timings"), dict):
            return
        timings = payload["timings"]
        proposed = timings.get("draft_n")
        accepted = timings.get("draft_n_accepted")
        if (
            isinstance(proposed, int)
            and not isinstance(proposed, bool)
            and isinstance(accepted, int)
            and not isinstance(accepted, bool)
            and proposed >= accepted >= 0
        ):
            self.draft_proposed = proposed
            self.draft_accepted = accepted
            self.acceptance_ratio = accepted / proposed if proposed else None


def remove_reasoning(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: remove_reasoning(item)
            for key, item in value.items()
            if key not in {"reasoning", "reasoning_content", "analysis"}
        }
    if isinstance(value, list):
        return [remove_reasoning(item) for item in value]
    if isinstance(value, str):
        return REASONING_MARKERS.sub("", value)
    return value


def qwen_request(
    body: Any,
    *,
    model_id: str,
    maximum_new_tokens: int,
    thinking_mode: str = "adaptive",
) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    if thinking_mode not in {"adaptive", "disabled"}:
        raise ValueError("The Qwen llama.cpp Worker requires a trusted thinking mode")
    result = {key: value for key, value in body.items() if key in QWEN_OPENAI_REQUEST_FIELDS}
    result["model"] = model_id
    reasoning_effort = result.get("reasoning_effort")
    if thinking_mode == "disabled":
        if reasoning_effort not in {None, "none"}:
            raise ValueError("reasoning_effort must be none when thinking is disabled")
        result["reasoning_effort"] = "none"
    elif reasoning_effort is not None and (
        not isinstance(reasoning_effort, str) or reasoning_effort not in QWEN_REASONING_EFFORTS
    ):
        raise ValueError("reasoning_effort is not supported by the trusted llama.cpp runtime")
    requested = result.get("max_completion_tokens", result.get("max_tokens"))
    if isinstance(requested, bool) or (
        requested is not None and (not isinstance(requested, int) or requested < 1)
    ):
        raise ValueError("max_tokens must be a positive integer")
    if isinstance(requested, int) and requested > maximum_new_tokens:
        raise ValueError("max_tokens exceeds the configured generation limit")
    return result


def normalise_qwen_chat_completion(payload: Any, *, tools: Any) -> Any:
    """Canonicalise known llama.cpp Qwen tool-call variants at the OpenAI boundary.

    The local llama.cpp server owns prompt rendering and generation, but its Qwen
    templates may omit a call identifier, use an empty value for an empty-object
    argument list, or leave a complete XML call in a reasoning field.  None of
    those variants is valid on ModelDeck's public OpenAI-compatible surface.
    """

    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        return payload
    parameter_schemas = _qwen_tool_parameter_schemas(tools)
    for choice in payload["choices"]:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            continue
        message = choice["message"]
        calls = message.get("tool_calls")
        if isinstance(calls, list) and calls:
            message["tool_calls"] = [_normalise_qwen_tool_call(call, parameter_schemas) for call in calls]
            continue
        recovered, clean_content = _recover_qwen_tool_calls(message, parameter_schemas)
        if recovered:
            message["tool_calls"] = recovered
            if clean_content is not None:
                message["content"] = clean_content
            choice["finish_reason"] = "tool_calls"
    return payload


def _qwen_tool_parameter_schemas(tools: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(tools, list):
        return {}
    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        parameters = function.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        schemas[function["name"]] = properties if isinstance(properties, dict) else {}
    return schemas


def _normalise_qwen_tool_call(call: Any, parameter_schemas: dict[str, dict[str, Any]]) -> Any:
    if not isinstance(call, dict):
        return call
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        return call
    normalised = {**call, "type": "function", "function": dict(function)}
    arguments = normalised["function"].get("arguments")
    if isinstance(arguments, dict):
        normalised["function"]["arguments"] = json.dumps(arguments)
    elif arguments == "" and not parameter_schemas.get(function["name"]):
        normalised["function"]["arguments"] = "{}"
    if not isinstance(normalised.get("id"), str) or not normalised["id"]:
        normalised["id"] = f"call_{uuid.uuid4().hex}"
    return normalised


def _recover_qwen_tool_calls(
    message: dict[str, Any], parameter_schemas: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], str | None]:
    recovered: list[dict[str, Any]] = []
    clean_content: str | None = None
    for field in ("content", "reasoning_content"):
        value = message.get(field)
        if not isinstance(value, str):
            continue
        matches = list(_QWEN_TOOL_CALL.finditer(value))
        for match in matches:
            arguments = _qwen_xml_arguments(
                match.group("body"), parameter_schemas.get(match.group("name"), {})
            )
            if arguments is None:
                continue
            recovered.append(
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": match.group("name"), "arguments": json.dumps(arguments)},
                }
            )
        if field == "content" and recovered:
            remaining = _QWEN_TOOL_CALL.sub("", value).strip()
            clean_content = remaining or None
    return recovered, clean_content


def _qwen_xml_arguments(body: str, parameter_schemas: dict[str, Any]) -> dict[str, Any] | None:
    arguments: dict[str, Any] = {}
    for parameter in _QWEN_TOOL_PARAMETER.finditer(body):
        name = parameter.group("name")
        value = parameter.group("value").strip()
        schema = parameter_schemas.get(name)
        if isinstance(schema, dict) and schema.get("type") not in {None, "string"}:
            try:
                arguments[name] = json.loads(value)
            except json.JSONDecodeError:
                return None
        else:
            arguments[name] = value
    if _QWEN_TOOL_PARAMETER.sub("", body).strip():
        return None
    return arguments


def amd_gpu_memory_metrics() -> dict[str, int]:
    """Read whole-device AMD memory counters from the fixed Linux DRM sysfs interface."""
    for device in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        try:
            if (device / "vendor").read_text(encoding="utf-8").strip().lower() != AMD_VENDOR_ID:
                continue
            values = {}
            for source, key in (
                ("mem_info_gtt_used", "system_gtt_used_bytes"),
                ("mem_info_gtt_total", "system_gtt_total_bytes"),
                ("mem_info_vram_used", "system_vram_used_bytes"),
                ("mem_info_vram_total", "system_vram_total_bytes"),
            ):
                values[key] = int((device / source).read_text(encoding="utf-8").strip())
            return values
        except (OSError, ValueError):
            continue
    return {}


class LlamaProcess:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Allocated immediately before launch so constructing an app never reserves a port.
        self.internal_port = 0
        # Keep the catalogue-approved snapshot filename for the strict GGUF allowlist.
        # Hugging Face snapshots are symlinks whose resolved blob names are opaque hashes.
        self.artifact_path = Path(args.artifact_path).absolute()
        self.process: asyncio.subprocess.Process | None = None
        self.log_tasks: list[asyncio.Task[None]] = []
        self.restart_lock = asyncio.Lock()
        self.evidence = LlamaEvidence()
        self.qwen_runtime: ValidatedQwenRuntime | None = None
        self.llama_installation: ValidatedLlamaInstallation | None = None
        self.memory_task: asyncio.Task[None] | None = None
        self.peak_gtt_used_bytes: int | None = None
        self.started = time.monotonic()
        self.load_seconds: float | None = None
        self.last_time_to_first_token_seconds: float | None = None
        self.startup_failure_category: str | None = None

    async def start(self) -> None:
        self.started = time.monotonic()
        self.internal_port = allocate_private_port()
        self.evidence = LlamaEvidence()
        self.startup_failure_category = None
        if getattr(self.args, "runtime_profile", None):
            self.qwen_runtime = validate_qwen_runtime(
                self.args.runtime_profile,
                self.artifact_path.parent,
                data_dir=Path(self.args.data_dir) if self.args.data_dir else None,
                candidate_id=self.args.candidate_manifest_id,
            )
            if self.args.context_length != self.qwen_runtime.manifest.context_length:
                raise ValueError("Configured context length does not match the trusted Qwen manifest")
            command = qwen_llama_command(
                runtime=self.qwen_runtime,
                port=self.internal_port,
                thinking_mode=self.args.thinking_mode,
            )
        else:
            self.llama_installation = validate_llama_installation(required_flags=GPT_OSS_LLAMA_REQUIRED_FLAGS)
            command = llama_command(
                model=self.artifact_path,
                port=self.internal_port,
                context_length=self.args.context_length,
                preset=self.args.execution_preset,
                executable=self.llama_installation.executable,
            )
        environment = dict(os.environ)
        environment["GGML_VK_VISIBLE_DEVICES"] = "0"
        self.process = await asyncio.create_subprocess_exec(
            *command,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.log_tasks = [
            asyncio.create_task(self._capture(self.process.stdout)),
            asyncio.create_task(self._capture(self.process.stderr)),
        ]
        self.memory_task = asyncio.create_task(self._sample_gpu_memory())

    async def stop(self) -> None:
        try:
            if self.process is None or self.process.returncode is not None:
                return
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=8)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        finally:
            if self.memory_task is not None:
                self.memory_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.memory_task
                self.memory_task = None
            for task in self.log_tasks:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            self.log_tasks.clear()

    async def restart(self) -> None:
        async with self.restart_lock:
            await self.stop()
            self.process = None
            self.load_seconds = None
            await self.start()

    async def _capture(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        quantisation = self.qwen_runtime.manifest.quantisation if self.qwen_runtime else "mxfp4"
        while line := await stream.readline():
            message = line.decode(errors="replace").rstrip()
            self.evidence.feed(message, quantisation=quantisation)
            self.startup_failure_category = (
                classify_llama_startup_failure(message) or self.startup_failure_category
            )

    def child_failure(self) -> dict[str, Any] | None:
        if self.process is None or self.process.returncode is None:
            return None
        category = self.startup_failure_category or "llama_child_exited"
        descriptions = {
            "accelerator_memory_allocation_failed": "accelerator memory allocation failed",
            "model_load_failed": "model loading failed",
            "vulkan_initialisation_failed": "Vulkan initialisation failed",
            "llama_child_exited": "the llama.cpp process exited",
        }
        return {
            "failure_category": category,
            "child_exit_code": self.process.returncode,
            "error": (
                f"llama.cpp child exited during model loading with code "
                f"{self.process.returncode}: {descriptions[category]}"
            ),
        }

    def memory_metrics(self) -> dict[str, int]:
        metrics = amd_gpu_memory_metrics()
        current = metrics.get("system_gtt_used_bytes")
        if current is not None:
            self.peak_gtt_used_bytes = max(self.peak_gtt_used_bytes or current, current)
        if self.peak_gtt_used_bytes is not None:
            metrics["system_gtt_peak_used_bytes"] = self.peak_gtt_used_bytes
        return metrics

    async def _sample_gpu_memory(self) -> None:
        while True:
            self.memory_metrics()
            await asyncio.sleep(0.1)

    async def ready(self) -> bool:
        if self.process is None or self.process.returncode is not None:
            return False
        try:
            async with httpx.AsyncClient(timeout=0.5) as client:
                response = await client.get(f"http://127.0.0.1:{self.internal_port}/health")
            verified = not self.qwen_runtime or not self.evidence.startup_errors(
                projector_required=self.qwen_runtime.projector is not None,
                mtp_required=self.qwen_runtime.mtp_model is not None,
            )
            if response.is_success and verified and self.load_seconds is None:
                self.load_seconds = round(time.monotonic() - self.started, 4)
            return response.is_success and verified
        except httpx.HTTPError:
            return False


def classify_llama_startup_failure(message: str) -> str | None:
    """Return a safe failure category without retaining child-process log content."""
    lowered = message.casefold()
    if any(
        marker in lowered
        for marker in (
            "out of memory",
            "failed to allocate",
            "allocation failed",
            "cannot allocate memory",
        )
    ):
        return "accelerator_memory_allocation_failed"
    if any(marker in lowered for marker in ("failed to load model", "model load failed")):
        return "model_load_failed"
    if "vulkan" in lowered and any(
        marker in lowered for marker in ("error", "failed", "failure", "initialization")
    ):
        return "vulkan_initialisation_failed"
    return None


def create_app(args: argparse.Namespace) -> FastAPI:
    runtime = LlamaProcess(args)
    is_qwen = bool(getattr(args, "runtime_profile", None))
    thinking_mode = getattr(args, "thinking_mode", None)
    if is_qwen and thinking_mode not in {"adaptive", "disabled"}:
        raise ValueError("The selected Qwen llama.cpp Worker requires a trusted thinking mode")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await runtime.start()
        yield
        await runtime.stop()

    app = FastAPI(title="ModelDeck llama.cpp Vulkan worker", lifespan=lifespan)
    app.state.shutdown_callback = None

    @app.get("/health")
    async def health():
        ready = await runtime.ready()
        failure = runtime.child_failure()
        manifest = runtime.qwen_runtime.manifest if runtime.qwen_runtime else None
        installation_identity = (
            {
                "llama_cpp_commit": runtime.qwen_runtime.source_revision,
                "llama_server_sha256": runtime.qwen_runtime.executable_sha256,
                "llama_build_receipt_sha256": runtime.qwen_runtime.receipt_sha256,
            }
            if runtime.qwen_runtime
            else {
                "llama_cpp_commit": runtime.llama_installation.receipt.commit,
                "llama_server_sha256": runtime.llama_installation.executable_sha256,
                "llama_build_receipt_sha256": runtime.llama_installation.receipt_sha256,
            }
            if getattr(runtime, "llama_installation", None)
            else {}
        )
        return {
            "protocol_version": "1",
            "worker_id": args.worker_id,
            "runtime": "llama-vulkan",
            "generation_family": GenerationFamily.AUTOREGRESSIVE,
            "state": "failed" if failure else "ready" if ready else "loading",
            "model_id": args.model_id,
            "model_revision": args.revision,
            "device": "vulkan:0",
            "device_name": "AMD Radeon 8060S (Vulkan)" if manifest else "AMD Vulkan",
            "rocm_version": None,
            "ready": ready,
            **installation_identity,
            **(failure or {}),
            **(
                {
                    "runtime_profile": manifest.id,
                    "thinking_mode": thinking_mode,
                    "configuration_fingerprint": configuration_fingerprint(
                        runtime.qwen_runtime, thinking_mode=thinking_mode
                    ),
                    "verified_capabilities": (
                        ["chat", "completions", "streaming", "cancellation"]
                        + (
                            ["image_input", "structured_output", "tool_calling"]
                            if runtime.qwen_runtime.projector
                            else []
                        )
                        + (["reasoning"] if thinking_mode == "adaptive" else [])
                        + (["mtp"] if runtime.qwen_runtime.mtp_model else [])
                        if ready
                        else []
                    ),
                }
                if manifest
                else {
                    "configuration_fingerprint": gpt_oss_configuration_fingerprint(
                        args, runtime.llama_installation
                    )
                }
                if getattr(runtime, "llama_installation", None)
                else {}
            ),
        }

    @app.post("/warmup")
    async def warmup():
        if not await runtime.ready():
            return JSONResponse({"ready": False}, status_code=503)
        payload = {
            "prompt": "Count from 1 to 20, separated by spaces:\n1 2 3 4 5",
            "n_predict": 48 if is_qwen else 1,
            "temperature": 0,
            **({"ignore_eos": True} if is_qwen else {}),
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"http://127.0.0.1:{runtime.internal_port}/completion", json=payload)
        if response.is_success and is_qwen:
            runtime.evidence.record_generation_timings(response.json())
        mtp_required = bool(runtime.qwen_runtime and runtime.qwen_runtime.mtp_model)
        mtp_verified = not mtp_required or runtime.evidence.draft_accepted > 0
        ready = response.is_success and mtp_verified
        return JSONResponse({"ready": ready, "mtp_verified": mtp_verified}, status_code=200 if ready else 503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": args.model_id, "object": "model"}]}

    @app.get("/model")
    async def model():
        manifest = runtime.qwen_runtime.manifest if runtime.qwen_runtime else None
        return {
            "model_id": args.model_id,
            "revision": args.revision,
            "generation_family": GenerationFamily.AUTOREGRESSIVE,
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": manifest.quantisation if manifest else "mxfp4",
            "quantization": manifest.quantisation if manifest else "mxfp4",
            **(
                {
                    "llama_cpp_commit": runtime.llama_installation.receipt.commit,
                    "llama_server_sha256": runtime.llama_installation.executable_sha256,
                    "llama_build_receipt_sha256": runtime.llama_installation.receipt_sha256,
                    "configuration_fingerprint": gpt_oss_configuration_fingerprint(
                        args, runtime.llama_installation
                    ),
                }
                if runtime.llama_installation
                else {}
            ),
            **(
                {
                    "original_model_id": manifest.original_model_id,
                    "original_model_revision": manifest.original_model_revision,
                    "artefact_model_id": manifest.artefact_model_id,
                    "artefact_revision": manifest.artefact_revision,
                    "gguf_sha256": manifest.model.sha256,
                    **({"projector_sha256": manifest.projector.sha256} if manifest.projector else {}),
                    **({"mtp_model_sha256": manifest.mtp_model.sha256} if manifest.mtp_model else {}),
                    "llama_cpp_commit": manifest.llama_cpp_commit,
                    "llama_server_sha256": runtime.qwen_runtime.executable_sha256,
                    "llama_build_receipt_sha256": runtime.qwen_runtime.receipt_sha256,
                    "backend": manifest.backend,
                    "context_length": manifest.context_length,
                    "cache_type_k": manifest.cache_type_k,
                    "cache_type_v": manifest.cache_type_v,
                    "mtp_enabled": runtime.qwen_runtime.mtp_model is not None,
                    "thinking_mode": thinking_mode,
                    "mtp_draft_tokens": manifest.mtp_draft_tokens or 0,
                    "configuration_fingerprint": configuration_fingerprint(
                        runtime.qwen_runtime, thinking_mode=thinking_mode
                    ),
                }
                if manifest
                else {}
            ),
        }

    @app.get("/metrics")
    async def metrics():
        startup_checks = (
            runtime.evidence.startup_checks(
                projector_required=runtime.qwen_runtime.projector is not None,
                mtp_required=runtime.qwen_runtime.mtp_model is not None,
            )
            if runtime.qwen_runtime
            else {}
        )
        return {
            "runtime": "llama-vulkan",
            "execution_preset": args.execution_preset,
            "load_seconds": runtime.load_seconds,
            "mtp_enabled": bool(runtime.qwen_runtime and runtime.qwen_runtime.mtp_model),
            **({"thinking_mode": thinking_mode} if is_qwen else {}),
            "mtp_draft_tokens": (
                runtime.qwen_runtime.manifest.mtp_draft_tokens
                if runtime.qwen_runtime and runtime.qwen_runtime.manifest.mtp_draft_tokens
                else 0
            ),
            "draft_proposed_tokens": runtime.evidence.draft_proposed,
            "draft_accepted_tokens": runtime.evidence.draft_accepted,
            "draft_rejected_tokens": max(
                0, runtime.evidence.draft_proposed - runtime.evidence.draft_accepted
            ),
            "mtp_acceptance_ratio": runtime.evidence.acceptance_ratio,
            "prompt_tokens_per_second": runtime.evidence.prompt_tokens_per_second,
            "generated_tokens_per_second": runtime.evidence.generated_tokens_per_second,
            "time_to_first_token_seconds": runtime.last_time_to_first_token_seconds,
            "throughput_basis": "effective_mtp" if runtime.qwen_runtime else "backend_reported",
            "startup_checks": startup_checks,
            "startup_errors": [name for name, passed in startup_checks.items() if not passed],
            **(runtime.child_failure() or {}),
            **runtime.memory_metrics(),
        }

    async def proxy(request: Request, path: str):
        raw_body = await request.json()
        try:
            body = (
                qwen_request(
                    raw_body,
                    model_id=args.model_id,
                    maximum_new_tokens=args.maximum_new_tokens,
                    thinking_mode=thinking_mode,
                )
                if is_qwen
                else raw_body
            )
        except ValueError as error:
            return JSONResponse(
                {"error": {"code": "invalid_request", "message": str(error)}}, status_code=400
            )
        body["model"] = args.model_id
        filter_reasoning = not is_qwen or thinking_mode == "disabled"
        request_started = time.monotonic()
        client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=1))
        try:
            response = await client.send(
                client.build_request("POST", f"http://127.0.0.1:{runtime.internal_port}{path}", json=body),
                stream=bool(body.get("stream")),
            )
        except httpx.HTTPError:
            await client.aclose()
            if runtime.process is None or runtime.process.returncode is not None:
                await runtime.restart()
            return JSONResponse({"error": {"code": "llama_runtime_unavailable"}}, status_code=503)
        if body.get("stream"):

            async def filtered_stream():
                try:
                    async for line in response.aiter_lines():
                        if runtime.last_time_to_first_token_seconds is None and line.startswith("data: "):
                            runtime.last_time_to_first_token_seconds = round(
                                time.monotonic() - request_started, 6
                            )
                        if filter_reasoning and line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                line = "data: " + json.dumps(remove_reasoning(json.loads(line[6:])))
                            except json.JSONDecodeError:
                                continue
                        yield (line + "\n").encode()
                except asyncio.CancelledError:
                    await runtime.restart()
                    raise
                finally:
                    await response.aclose()
                    await client.aclose()

            return StreamingResponse(filtered_stream(), media_type="text/event-stream")
        try:
            payload = response.json()
            if is_qwen and path == "/v1/chat/completions":
                payload = normalise_qwen_chat_completion(payload, tools=body.get("tools"))
            return JSONResponse(
                remove_reasoning(payload) if filter_reasoning else payload,
                status_code=response.status_code,
            )
        finally:
            await response.aclose()
            await client.aclose()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await proxy(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await proxy(request, "/v1/completions")

    @app.post("/cancel")
    async def cancel():
        await runtime.restart()
        return {"ok": True, "runtime_recreated": True}

    @app.post("/shutdown")
    async def shutdown():
        if app.state.shutdown_callback:
            asyncio.get_running_loop().call_later(0.05, app.state.shutdown_callback)
        return {"ok": True}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument(
        "--runtime-profile",
        choices=(
            "qwen35-4b-q8-vulkan",
            "qwen35-approved-q8-vulkan",
            "qwen38-q8-mtp-vulkan",
            "qwen38-q4-mtp-vulkan",
        ),
    )
    parser.add_argument("--candidate-manifest-id")
    parser.add_argument("--data-dir")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument("--thinking-mode", choices=("adaptive", "disabled"), default="adaptive")
    parser.add_argument(
        "--execution-preset",
        choices=("vulkan-full",),
        default="vulkan-full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)
    )
    app.state.shutdown_callback = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()

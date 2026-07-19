from __future__ import annotations

import grp
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil

PACKAGE_NAMES = (
    "torch",
    "transformers",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "vllm",
)
MODEL_PORTS = (8000, 8019, 8100, 8300, 8600, *range(8610, 8700), 8700, 8800, 8900)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _fedora_release() -> str | None:
    for path in (Path("/etc/fedora-release"), Path("/etc/os-release")):
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return None


def _command_version(command: str, *args: str) -> dict[str, Any]:
    executable = shutil.which(command)
    if not executable:
        return {"available": False, "path": None, "version": None}
    try:
        result = subprocess.run([executable, *args], capture_output=True, text=True, timeout=3, check=False)
        output = (result.stdout or result.stderr).strip().splitlines()
        version = output[0] if output else None
    except (OSError, subprocess.TimeoutExpired) as error:
        version = f"probe failed: {type(error).__name__}"
    return {"available": True, "path": executable, "version": version}


def _rocm_package_versions() -> list[str]:
    rpm = shutil.which("rpm")
    if not rpm:
        return []
    try:
        result = subprocess.run(
            [rpm, "-qa", "rocm-core", "rocm-runtime", "rocminfo", "hipcc"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        return sorted(line for line in result.stdout.splitlines() if line)
    except (OSError, subprocess.TimeoutExpired):
        return []


def _torch_details(allocation_test: bool) -> dict[str, Any]:
    if package_version("torch") is None:
        return {"version": None, "hip": None, "cuda_available": False, "allocation_test": "not-run"}
    try:
        import torch

        details: dict[str, Any] = {
            "version": str(torch.__version__),
            "hip": torch.version.hip,
            "cuda_available": bool(torch.cuda.is_available()),
            "allocation_test": "not-requested",
        }
        if allocation_test:
            try:
                tensor = torch.ones((2, 2), device="cuda")
                details["allocation_test"] = "passed"
                del tensor
            except Exception as error:  # hardware-specific diagnostic
                details["allocation_test"] = f"failed: {type(error).__name__}: {error}"
        return details
    except Exception as error:  # broken binary stack is a useful result
        return {
            "version": package_version("torch"),
            "hip": None,
            "cuda_available": False,
            "allocation_test": f"import-failed: {type(error).__name__}: {error}",
        }


def _filesystem(path: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {"path": path, "available": False}
    usage = psutil.disk_usage(path)
    return {
        "path": path,
        "available": True,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "percent": usage.percent,
    }


def _listening_ports() -> list[int]:
    listening: set[int] = set()
    for port in MODEL_PORTS:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.01)
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    listening.add(port)
        except OSError:
            # Hardened and sandboxed environments may deny socket creation entirely.
            return sorted(listening)
    return sorted(listening)


def _active_model_processes() -> list[dict[str, Any]]:
    terms = ("vllm", "ollama", "llama", "transformers", "diffusion", "modeldeck")
    matches = []
    for process in psutil.process_iter(("pid", "name", "cmdline")):
        try:
            arguments = process.info.get("cmdline") or ()
            command = " ".join(arguments)
            if process.pid != os.getpid() and any(term in command.lower() for term in terms):
                matches.append(
                    {
                        "pid": process.pid,
                        "name": process.info.get("name"),
                        "command": _safe_process_command(arguments)[:240],
                    }
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return matches


def _safe_process_command(arguments: list[str] | tuple[str, ...]) -> str:
    secret_flags = {"--api-key", "--token", "--hf-token", "--authorization"}
    safe: list[str] = []
    redact_next = False
    for argument in arguments:
        lowered = argument.lower()
        if redact_next:
            safe.append("[redacted]")
            redact_next = False
        elif lowered in secret_flags:
            safe.append(argument)
            redact_next = True
        elif any(lowered.startswith(f"{flag}=") for flag in secret_flags):
            safe.append(f"{argument.split('=', 1)[0]}=[redacted]")
        else:
            safe.append(argument)
    return " ".join(safe)


def _temperatures() -> list[dict[str, Any]]:
    readings = []
    try:
        for source, sensors in psutil.sensors_temperatures().items():
            for sensor in sensors:
                readings.append(
                    {"source": source, "label": sensor.label or source, "celsius": sensor.current}
                )
    except (AttributeError, OSError):
        pass
    return readings


def _fans() -> list[dict[str, Any]]:
    readings = []
    try:
        for source, fans in psutil.sensors_fans().items():
            for fan in fans:
                readings.append({"source": source, "label": fan.label or source, "rpm": fan.current})
    except (AttributeError, OSError):
        pass
    return readings


def cache_candidates() -> list[str]:
    candidates = []
    if os.getenv("HF_HUB_CACHE"):
        candidates.append(os.environ["HF_HUB_CACHE"])
    if os.getenv("HF_HOME"):
        candidates.append(str(Path(os.environ["HF_HOME"]) / "hub"))
    candidates.extend((str(Path.home() / ".cache/huggingface/hub"), "/mnt/work/models/huggingface/hub"))
    return list(dict.fromkeys(candidates))


def probe_environment(*, allocation_test: bool = False) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    detected = {
        "fedora_release": _fedora_release(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "packages": {name: package_version(name) for name in PACKAGE_NAMES},
        "torch": _torch_details(allocation_test),
        "rocm_packages": _rocm_package_versions(),
        "gpu_device_nodes": {path: Path(path).exists() for path in ("/dev/kfd", "/dev/dri")},
        "hsa_runtime_candidates": {
            "/usr/lib64/libhsa-runtime64.so.1": Path("/usr/lib64/libhsa-runtime64.so.1").exists()
        },
        "groups": sorted(
            group.gr_name
            for group in grp.getgrall()
            if os.getgid() == group.gr_gid or __import__("getpass").getuser() in group.gr_mem
        ),
        "tools": {
            "rocminfo": _command_version("rocminfo", "--version"),
            "rocm-smi": _command_version("rocm-smi", "--version"),
            "amd-smi": _command_version("amd-smi", "version"),
            "podman": _command_version("podman", "--version"),
            "docker": _command_version("docker", "--version"),
        },
        "memory": {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "percent": memory.percent,
        },
        "swap": {"total_bytes": swap.total, "used_bytes": swap.used, "percent": swap.percent},
        "filesystems": [_filesystem("/"), _filesystem("/mnt/work")],
        "temperatures": _temperatures(),
        "fans": _fans(),
        "huggingface_cache_candidates": [
            {"path": path, "exists": Path(path).is_dir()} for path in cache_candidates()
        ],
        "active_model_processes": _active_model_processes(),
        "listening_model_ports": _listening_ports(),
    }
    return {
        "configured": {
            "profile_id": "framework-desktop-rocm72",
            "os": "Fedora 44",
            "gpu": "AMD Radeon 8060S Graphics",
            "gpu_architecture": "gfx1151",
            "rocm_family": "7.2.x",
            "work_mount": "/mnt/work",
        },
        "detected": detected,
        "last_tested": None,
        "diagnostic_note": (
            "ROCm PyTorch uses the 'cuda' device API for AMD accelerators; this does not imply an NVIDIA GPU."
        ),
    }


def main() -> None:
    print(json.dumps(probe_environment(allocation_test="--allocation-test" in sys.argv), indent=2))


if __name__ == "__main__":
    main()

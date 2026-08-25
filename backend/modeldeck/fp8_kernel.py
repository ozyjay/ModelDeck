from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Literal

KERNEL_REPO_ID = "kernels-community/finegrained-fp8"
KERNEL_REVISION = "v3"
KERNEL_COMMIT = "fcf89a79d85eab78182c62fb986ed01f2cbf7422"
KERNEL_VERSION = 3
KERNEL_PACKAGE_VERSION = "0.15.2"
KERNELS_DATA_VERSION = "0.16.1"
TOMLKIT_VERSION = "0.15.1"
TRANSFORMERS_VERSION = "5.13.0"
TRITON_VERSION = "3.5.1+rocm7.2.1.gita272dfa8"
TORCH_VERSION = "2.9.1+rocm7.2.1.lw.gitff65f5bc"
SUPPORTED_GPU_ARCHITECTURE = "gfx1151"
TRUST_MANIFEST_RESOURCE = "finegrained_fp8_rocm_v3.json"
TUNING_MANIFEST_VERSION = 1
DEFAULT_CONFIG = {"num_warps": 4, "num_stages": 2}
QWEN38_FP8_WEIGHT_SHAPES = (
    (10_240, 5_120),
    (6_144, 5_120),
    (5_120, 6_144),
    (5_120, 17_408),
    (17_408, 5_120),
    (1_024, 5_120),
    (12_288, 5_120),
)
QWEN38_FP8_BLOCK_SIZE_M_BUCKETS = (16, 32, 64, 128)
TUNING_DTYPE = "bfloat16"
REQUIRED_TUNING_KEYS = frozenset(
    (n, k, block_size_m, TUNING_DTYPE)
    for n, k in QWEN38_FP8_WEIGHT_SHAPES
    for block_size_m in QWEN38_FP8_BLOCK_SIZE_M_BUCKETS
)


class FP8KernelValidationError(RuntimeError):
    """Raised before untrusted or incompatible kernel code can be imported."""


@dataclass(frozen=True)
class ValidatedFP8Kernel:
    snapshot_path: Path
    variant_path: Path
    manifest_digest: str
    tuning_manifest_path: Path | None
    tuning_manifest_digest: str | None
    tuning_status: Literal["validated", "missing", "stale"]

    def runtime_details(self) -> dict[str, Any]:
        return {
            "execution_mode": "native_fp8",
            "kernels_version": KERNEL_PACKAGE_VERSION,
            "kernel_repo_id": KERNEL_REPO_ID,
            "kernel_revision": KERNEL_REVISION,
            "kernel_commit": KERNEL_COMMIT,
            "kernel_manifest_sha256": self.manifest_digest,
            "triton_version": TRITON_VERSION,
            "tuning_profile_sha256": self.tuning_manifest_digest,
            "tuning_status": self.tuning_status,
            "kernel_snapshot_path": str(self.snapshot_path),
        }


def _trust_manifest_bytes() -> bytes:
    from importlib.resources import files

    return files("modeldeck").joinpath("registry_data", TRUST_MANIFEST_RESOURCE).read_bytes()


def trust_manifest() -> dict[str, Any]:
    try:
        value = json.loads(_trust_manifest_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise FP8KernelValidationError("The packaged FP8 kernel trust manifest is unreadable") from error
    if not isinstance(value, dict):
        raise FP8KernelValidationError("The packaged FP8 kernel trust manifest must be an object")
    return value


def kernel_snapshot_path(cache_root: Path) -> Path:
    return Path(cache_root) / "kernels--kernels-community--finegrained-fp8" / "snapshots" / KERNEL_COMMIT


def validate_fp8_kernel_snapshot(cache_root: Path) -> ValidatedFP8Kernel:
    manifest_bytes = _trust_manifest_bytes()
    manifest = trust_manifest()
    expected_identity = {
        "repo_id": KERNEL_REPO_ID,
        "revision": KERNEL_REVISION,
        "commit": KERNEL_COMMIT,
        "kernel_id": "_finegrained_fp8_rocm_846165b",
        "kernel_version": KERNEL_VERSION,
        "backend": "rocm",
    }
    for name, expected in expected_identity.items():
        if manifest.get(name) != expected:
            raise FP8KernelValidationError(f"The packaged FP8 kernel manifest has an invalid {name}")

    repo_root = kernel_snapshot_path(cache_root).parents[1]
    ref_path = repo_root / "refs" / KERNEL_REVISION
    try:
        resolved_ref = ref_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise FP8KernelValidationError(f"The pinned FP8 kernel ref is missing: {ref_path}") from error
    if resolved_ref != KERNEL_COMMIT:
        raise FP8KernelValidationError("The cached FP8 kernel v3 ref does not match the reviewed commit")

    snapshot = kernel_snapshot_path(cache_root)
    variant = snapshot / "build" / "torch-rocm"
    try:
        resolved_snapshot = snapshot.resolve(strict=True)
        resolved_variant = variant.resolve(strict=True)
        resolved_variant.relative_to(resolved_snapshot)
    except (OSError, ValueError) as error:
        raise FP8KernelValidationError(
            "The reviewed ROCm FP8 kernel snapshot is missing or unsafe"
        ) from error

    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise FP8KernelValidationError("The FP8 kernel trust manifest has no file hashes")
    actual_files = {
        path.relative_to(resolved_variant).as_posix()
        for path in resolved_variant.rglob("*")
        if path.is_file()
    }
    expected_names = set(expected_files)
    if actual_files != expected_names:
        missing = sorted(expected_names - actual_files)
        unexpected = sorted(actual_files - expected_names)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unexpected:
            detail.append("unexpected " + ", ".join(unexpected))
        raise FP8KernelValidationError("The FP8 kernel file set changed: " + "; ".join(detail))
    for relative_name, expected_digest in expected_files.items():
        path = resolved_variant / relative_name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise FP8KernelValidationError(f"The FP8 kernel digest changed: {relative_name}")

    try:
        metadata = json.loads((resolved_variant / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FP8KernelValidationError("The FP8 kernel metadata is unreadable") from error
    if (
        metadata.get("name") != "finegrained-fp8"
        or metadata.get("id") != expected_identity["kernel_id"]
        or metadata.get("version") != KERNEL_VERSION
        or metadata.get("backend", {}).get("type") != "rocm"
    ):
        raise FP8KernelValidationError("The FP8 kernel metadata does not match the reviewed ROCm build")

    return ValidatedFP8Kernel(
        snapshot_path=resolved_snapshot,
        variant_path=resolved_variant,
        manifest_digest=hashlib.sha256(manifest_bytes).hexdigest(),
        tuning_manifest_path=None,
        tuning_manifest_digest=None,
        tuning_status="missing",
    )


def _require_package_version(distribution: str, expected: str) -> None:
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise FP8KernelValidationError(f"Required FP8 runtime package is missing: {distribution}") from error
    if actual != expected:
        raise FP8KernelValidationError(
            f"FP8 runtime package {distribution} must be {expected}; detected {actual}"
        )


def _gpu_architecture(torch: Any) -> str:
    properties = torch.cuda.get_device_properties(0)
    value = getattr(properties, "gcnArchName", "") or getattr(properties, "gcn_arch_name", "")
    return str(value).split(":", 1)[0]


def runtime_fingerprint(torch: Any, *, kernel_manifest_digest: str) -> dict[str, str]:
    return {
        "gpu_architecture": _gpu_architecture(torch),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "hip_version": str(torch.version.hip),
        "torch_version": str(torch.__version__),
        "triton_version": importlib.metadata.version("triton"),
        "transformers_version": importlib.metadata.version("transformers"),
        "kernels_version": importlib.metadata.version("kernels"),
        "kernel_commit": KERNEL_COMMIT,
        "kernel_manifest_sha256": kernel_manifest_digest,
    }


def tuning_manifest_path(data_dir: Path) -> Path:
    return Path(data_dir) / "runtime" / "fp8-tuning" / f"qwen38-{SUPPORTED_GPU_ARCHITECTURE}.json"


def load_tuning_manifest(
    path: Path,
    *,
    expected_fingerprint: dict[str, str],
    required_keys: frozenset[tuple[int, int, int, str]] | None = None,
) -> tuple[dict[tuple[int, int, int, str], dict[str, int]], str] | None:
    if not path.is_file():
        return None
    content = path.read_bytes()
    try:
        document = json.loads(content)
    except json.JSONDecodeError:
        return None
    if document.get("format") != "modeldeck-fp8-tuning" or document.get("version") != 1:
        return None
    if document.get("fingerprint") != expected_fingerprint:
        return None
    winners: dict[tuple[int, int, int, str], dict[str, int]] = {}
    physically_validated: set[tuple[int, int, int, str]] = set()
    for entry in document.get("winners", []):
        try:
            key = (
                int(entry["n"]),
                int(entry["k"]),
                int(entry["block_size_m"]),
                str(entry["dtype"]),
            )
            config = {
                "num_warps": int(entry["num_warps"]),
                "num_stages": int(entry["num_stages"]),
            }
        except (KeyError, TypeError, ValueError):
            return None
        if config["num_warps"] not in {2, 4, 8, 16} or config["num_stages"] not in {2, 3, 4}:
            return None
        winners[key] = config
        if entry.get("status") == "accepted":
            physically_validated.add(key)
    if required_keys is not None and not required_keys.issubset(physically_validated):
        return None
    return winners, hashlib.sha256(content).hexdigest()


def write_tuning_manifest(path: Path, document: dict[str, Any]) -> str:
    if document.get("format") != "modeldeck-fp8-tuning" or document.get("version") != 1:
        raise ValueError("Invalid FP8 tuning manifest")
    content = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return hashlib.sha256(content).hexdigest()


def _triton_config(config: dict[str, int]) -> Any:
    import triton

    return triton.Config({}, num_warps=config["num_warps"], num_stages=config["num_stages"])


def _validate_runtime_interfaces(get_kernel: Any) -> None:
    import triton
    from transformers import FineGrainedFP8Config
    from transformers.integrations import finegrained_fp8

    expected_parameters = (
        (get_kernel, {"repo_id", "version", "backend", "trust_remote_code"}, "kernels.get_kernel"),
        (triton.Config, {"kwargs", "num_warps", "num_stages"}, "triton.Config"),
        (
            FineGrainedFP8Config,
            {"activation_scheme", "weight_block_size", "dequantize", "scale_fmt"},
            "transformers.FineGrainedFP8Config",
        ),
    )
    for callable_value, required, name in expected_parameters:
        try:
            available = set(inspect.signature(callable_value).parameters)
        except (TypeError, ValueError) as error:
            raise FP8KernelValidationError(f"Unable to inspect the pinned {name} interface") from error
        if not required.issubset(available):
            raise FP8KernelValidationError(f"The pinned {name} interface changed")
    matcher_parameters = set(inspect.signature(finegrained_fp8.should_convert_module).parameters)
    if matcher_parameters != {"full_name", "patterns"}:
        raise FP8KernelValidationError("The pinned Transformers FP8 module matcher interface changed")
    # Transformers 5.13 treats every skip-list item as an unescaped regular-expression
    # prefix. Qwen3.8's exact `.mlp.gate` router names consequently also match
    # `.mlp.gate_proj`, dropping the latter's scales and corrupting native-FP8 output.
    # This exact/hierarchical matcher retains intentional child and short-name skips
    # without allowing a reviewed module name to match a merely similar sibling.
    finegrained_fp8.should_convert_module = _should_convert_exact_module


def _should_convert_exact_module(full_name: str, patterns: list[str] | None = None) -> bool:
    if patterns is None:
        return True
    return not any(
        full_name == pattern or full_name.startswith(pattern + ".") or full_name.endswith("." + pattern)
        for pattern in patterns
    )


def _install_config_selector(
    autotuner: Any,
    winners: dict[tuple[int, int, int, str], dict[str, int]],
) -> None:
    if not all(hasattr(autotuner, name) for name in ("arg_names", "configs", "run")):
        raise FP8KernelValidationError("The pinned Triton autotuner interface changed")
    original_run = autotuner.run
    lock = threading.Lock()

    def selected_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        arguments = {**dict(zip(self.arg_names, args, strict=False)), **kwargs}
        try:
            key = (
                int(arguments["N"]),
                int(arguments["K"]),
                int(arguments["BLOCK_SIZE_M"]),
                TUNING_DTYPE,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FP8KernelValidationError(
                "The FP8 kernel invocation no longer exposes its tuning key"
            ) from error
        selected = _triton_config(winners.get(key, DEFAULT_CONFIG))
        with lock:
            previous = self.configs
            self.configs = [selected]
            try:
                return original_run(*args, **kwargs)
            finally:
                self.configs = previous

    autotuner.run = MethodType(selected_run, autotuner)


def triton_cache_path(data_dir: Path, fingerprint: dict[str, str]) -> Path:
    encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return Path(data_dir) / "runtime" / "triton-cache" / digest


def prepare_native_fp8(
    *,
    cache_root: Path,
    data_dir: Path,
    torch: Any,
) -> ValidatedFP8Kernel:
    validated = validate_fp8_kernel_snapshot(cache_root)
    for distribution, expected in (
        ("torch", TORCH_VERSION),
        ("triton", TRITON_VERSION),
        ("transformers", TRANSFORMERS_VERSION),
        ("kernels", KERNEL_PACKAGE_VERSION),
        ("kernels-data", KERNELS_DATA_VERSION),
        ("tomlkit", TOMLKIT_VERSION),
    ):
        _require_package_version(distribution, expected)
    if not torch.cuda.is_available() or _gpu_architecture(torch) != SUPPORTED_GPU_ARCHITECTURE:
        raise FP8KernelValidationError("Native FP8 is reviewed only for the detected gfx1151 ROCm device")

    fingerprint = runtime_fingerprint(torch, kernel_manifest_digest=validated.manifest_digest)
    manifest_path = tuning_manifest_path(data_dir)
    loaded = load_tuning_manifest(
        manifest_path,
        expected_fingerprint=fingerprint,
        required_keys=REQUIRED_TUNING_KEYS,
    )
    if loaded is None:
        winners, tuning_digest = {}, None
        status = "stale" if manifest_path.exists() else "missing"
    else:
        winners, tuning_digest = loaded
        status = "validated"

    os.environ["LOCAL_KERNELS"] = f"{KERNEL_REPO_ID}={validated.snapshot_path}"
    os.environ["KERNELS_CACHE"] = str(Path(cache_root))
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache_path(data_dir, fingerprint))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True

    from kernels import get_kernel

    _validate_runtime_interfaces(get_kernel)
    kernel = get_kernel(KERNEL_REPO_ID, version=KERNEL_VERSION)
    matmul_module = importlib.import_module(f"{kernel.__package__}.matmul")
    autotuner = getattr(matmul_module, "w8a8_block_dynamic_fp8_matmul_kernel", None)
    if autotuner is None:
        raise FP8KernelValidationError("The reviewed FP8 block matmul kernel is missing")
    _install_config_selector(autotuner, winners)

    return ValidatedFP8Kernel(
        snapshot_path=validated.snapshot_path,
        variant_path=validated.variant_path,
        manifest_digest=validated.manifest_digest,
        tuning_manifest_path=manifest_path if loaded is not None else None,
        tuning_manifest_digest=tuning_digest,
        tuning_status=status,
    )

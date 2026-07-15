from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

GIB = 1024**3
DEFAULT_CHECKPOINT_DIR = Path("/mnt/work/models/modeldeck/diffusiongemma-26b-a4b-it-gptq-q4-g32")
DEFAULT_EVALUATION_REPORT = Path("var/q4-quality-evaluation.json")
DEFAULT_LICENSE = Path("docs/licenses/APACHE-2.0.txt")
EXPECTED_MODEL_ID = "google/diffusiongemma-26B-A4B-it"
EXPECTED_REVISION = "52de6b914ee1749a7d4933202505ddf5b414ec43"
EXPECTED_QUANTIZATION = {
    "method": "gptq",
    "bits": 4,
    "group_size": 32,
    "symmetric": True,
    "desc_act": False,
    "qzero_format": 2,
    "runtime": "gptqmodel-triton-v2",
}
PUBLIC_EVALUATION_REDACTED_KEYS = {
    "endpoint",
    "pid",
    "q4_checkpoint_dir",
}
PUBLIC_EVALUATION_REDACTION_CATEGORIES = [
    "local worker endpoints",
    "local checkpoint paths",
    "process identifiers",
    "request and job identifiers",
]


class ReleaseError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package and verify the ModelDeck DiffusionGemma GPTQ Q4 g32 "
            "checkpoint as a reproducible release bundle."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=DEFAULT_EVALUATION_REPORT,
    )
    parser.add_argument("--license-file", type=Path, default=DEFAULT_LICENSE)
    parser.add_argument(
        "--source-commit",
        help="ModelDeck commit to record; defaults to the current Git HEAD.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing bundle without rewriting any files.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReleaseError(f"Required JSON file was not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"Could not read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseError(f"Expected a JSON object: {path}")
    return value


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ReleaseError(f"Unsafe release path: {value!r}")
    return Path(*pure.parts)


def validate_checkpoint(
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], list[Path], list[Path], list[Path]]:
    manifest_path = checkpoint_dir / "q4-manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("format") != "modeldeck-diffusiongemma-expert-gptq":
        raise ReleaseError("Unsupported Q4 checkpoint format")
    format_version = manifest.get("format_version")
    if format_version not in {1, 2}:
        raise ReleaseError("Unsupported Q4 checkpoint format version")
    if manifest.get("state") != "complete":
        raise ReleaseError("Q4 checkpoint conversion is not complete")
    if manifest.get("base_model_id") != EXPECTED_MODEL_ID:
        raise ReleaseError("Q4 checkpoint does not reference the pinned base model")
    if manifest.get("base_model_revision") != EXPECTED_REVISION:
        raise ReleaseError("Q4 checkpoint does not reference the pinned base revision")
    quantization = manifest.get("quantization")
    if not isinstance(quantization, dict) or any(
        quantization.get(key) != value for key, value in EXPECTED_QUANTIZATION.items()
    ):
        raise ReleaseError("Q4 checkpoint quantization settings do not match the release format")

    experts = manifest.get("experts")
    if not isinstance(experts, dict):
        raise ReleaseError("Q4 manifest has no expert metadata")
    layer_count = experts.get("layer_count")
    layers = experts.get("layers")
    if layer_count != 30:
        raise ReleaseError("Q4 manifest must contain exactly 30 layers")
    if not isinstance(layers, list) or len(layers) != layer_count:
        raise ReleaseError("Q4 manifest does not contain every expert layer")
    expected_expert_metadata = {
        "experts_per_layer": 128,
        "encoder_decoder_storage": "shared",
        "gate_up_shape": [1408, 2816],
        "down_shape": [2816, 704],
        "state_tensors": ["qweight", "qzeros", "scales", "g_idx"],
    }
    if any(experts.get(key) != value for key, value in expected_expert_metadata.items()):
        raise ReleaseError("Q4 manifest expert topology is incompatible")

    shard_paths: list[Path] = []
    seen_files: set[str] = set()
    for expected_layer, entry in enumerate(layers):
        if not isinstance(entry, dict) or entry.get("layer") != expected_layer:
            raise ReleaseError("Q4 manifest layer entries are not contiguous")
        filename = entry.get("file")
        if not isinstance(filename, str):
            raise ReleaseError(f"Layer {expected_layer} has no shard filename")
        relative = safe_relative_path(filename)
        if len(relative.parts) != 1 or relative.suffix != ".safetensors":
            raise ReleaseError(f"Layer {expected_layer} has an invalid shard filename")
        if filename in seen_files:
            raise ReleaseError(f"Duplicate Q4 shard in manifest: {filename}")
        seen_files.add(filename)
        path = checkpoint_dir / relative
        if not path.is_file():
            raise ReleaseError(f"Q4 shard is missing: {path}")
        expected_bytes = entry.get("file_bytes")
        if isinstance(expected_bytes, int) and path.stat().st_size != expected_bytes:
            raise ReleaseError(
                f"Q4 shard size mismatch for {filename}: "
                f"expected {expected_bytes}, found {path.stat().st_size}"
            )
        shard_paths.append(path)
    non_expert_paths: list[Path] = []
    runtime_paths: list[Path] = []
    if format_version == 2:
        if manifest.get("artifact_type") != "self-contained":
            raise ReleaseError("Q4 checkpoint version 2 must be self-contained")
        non_experts = manifest.get("non_expert_weights")
        if not isinstance(non_experts, dict) or non_experts.get("source") != "checkpoint":
            raise ReleaseError("Self-contained checkpoint has no packaged non-expert weights")
        index_name = non_experts.get("index_file")
        if not isinstance(index_name, str):
            raise ReleaseError("Self-contained checkpoint has no non-expert index")
        index_relative = safe_relative_path(index_name)
        if len(index_relative.parts) != 1 or index_relative.suffix != ".json":
            raise ReleaseError("Self-contained checkpoint has an invalid non-expert index")
        index_path = checkpoint_dir / index_relative
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ReleaseError("Non-expert Safetensors index has no weight map")
        if any(_is_expert_weight(str(name)) for name in weight_map):
            raise ReleaseError("Non-expert Safetensors index contains BF16 expert tensors")
        if non_experts.get("tensor_count") != len(weight_map):
            raise ReleaseError("Non-expert tensor count does not match its index")

        shard_entries = non_experts.get("shards")
        if not isinstance(shard_entries, list) or not shard_entries:
            raise ReleaseError("Self-contained checkpoint has no non-expert shards")
        if non_experts.get("shard_count") != len(shard_entries):
            raise ReleaseError("Non-expert shard count does not match the manifest")
        indexed_shards = {str(value) for value in weight_map.values()}
        manifest_shards: set[str] = set()
        for entry in shard_entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                raise ReleaseError("Non-expert shard metadata is invalid")
            filename = entry["file"]
            relative = safe_relative_path(filename)
            if len(relative.parts) != 1 or relative.suffix != ".safetensors":
                raise ReleaseError(f"Invalid non-expert shard filename: {filename}")
            if filename in manifest_shards:
                raise ReleaseError(f"Duplicate non-expert shard: {filename}")
            manifest_shards.add(filename)
            path = checkpoint_dir / relative
            if not path.is_file():
                raise ReleaseError(f"Non-expert shard is missing: {path}")
            if path.stat().st_size != entry.get("file_bytes"):
                raise ReleaseError(f"Non-expert shard size mismatch: {filename}")
            expected_hash = entry.get("sha256")
            if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
                raise ReleaseError(f"Non-expert shard checksum mismatch: {filename}")
            non_expert_paths.append(path)
        if manifest_shards != indexed_shards:
            raise ReleaseError("Non-expert index and manifest reference different shards")
        non_expert_paths.append(index_path)

        runtime_files = manifest.get("runtime_files")
        if not isinstance(runtime_files, list) or not runtime_files:
            raise ReleaseError("Self-contained checkpoint has no runtime metadata files")
        runtime_names: set[str] = set()
        for entry in runtime_files:
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                raise ReleaseError("Runtime file metadata is invalid")
            filename = entry["file"]
            relative = safe_relative_path(filename)
            if len(relative.parts) != 1 or filename in runtime_names:
                raise ReleaseError(f"Invalid or duplicate runtime file: {filename}")
            runtime_names.add(filename)
            path = checkpoint_dir / relative
            if not path.is_file():
                raise ReleaseError(f"Runtime file is missing: {path}")
            if path.stat().st_size != entry.get("file_bytes"):
                raise ReleaseError(f"Runtime file size mismatch: {filename}")
            expected_hash = entry.get("sha256")
            if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
                raise ReleaseError(f"Runtime file checksum mismatch: {filename}")
            runtime_paths.append(path)
        required_runtime = {"config.json", "generation_config.json", "tokenizer_config.json"}
        if not required_runtime.issubset(runtime_names):
            missing = sorted(required_runtime - runtime_names)
            raise ReleaseError(f"Self-contained checkpoint is missing runtime files: {missing}")
    return manifest, shard_paths, non_expert_paths, runtime_paths


def _is_expert_weight(name: str) -> bool:
    return name.endswith((".experts.gate_up_proj", ".experts.down_proj"))


def require_complete_phase(report: dict[str, Any], key: str) -> None:
    phase = report.get(key)
    if not isinstance(phase, dict):
        raise ReleaseError(f"Evaluation report is missing the {key} phase")
    summary = phase.get("summary")
    if not isinstance(summary, dict):
        raise ReleaseError(f"Evaluation report is missing the {key} summary")
    runs = summary.get("runs")
    if not isinstance(runs, int) or runs < 1:
        raise ReleaseError(f"Evaluation report has no {key} runs")
    if summary.get("contract_passes") != runs:
        raise ReleaseError(f"Not every {key} contract passed")
    if summary.get("constraint_passes") != runs:
        raise ReleaseError(f"Not every {key} constraint passed")
    if summary.get("constraint_pass_rate") != 1.0:
        raise ReleaseError(f"The {key} constraint pass rate is not 1.0")


def validate_evaluation(
    report: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
) -> None:
    if report.get("format") != "modeldeck-diffusiongemma-q4-evaluation":
        raise ReleaseError("Unsupported Q4 evaluation report format")
    if report.get("format_version") != 1:
        raise ReleaseError("Unsupported Q4 evaluation report version")
    if report.get("passed") is not True:
        raise ReleaseError("The Q4 evaluation release gate did not pass")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise ReleaseError("One or more Q4 evaluation checks did not pass")

    configuration = report.get("configuration")
    if not isinstance(configuration, dict):
        raise ReleaseError("Evaluation configuration is missing")
    if configuration.get("model_id") != checkpoint_manifest.get("base_model_id"):
        raise ReleaseError("Evaluation base model does not match the Q4 checkpoint")
    if configuration.get("revision") != checkpoint_manifest.get("base_model_revision"):
        raise ReleaseError("Evaluation base revision does not match the Q4 checkpoint")
    if configuration.get("prompt_count", 0) < 8:
        raise ReleaseError("The release evaluation must contain at least eight prompts")

    require_complete_phase(report, "q4")
    require_complete_phase(report, "bf16")
    q4 = report["q4"]
    stability = q4.get("stability_summary")
    if not isinstance(stability, dict) or stability.get("runs", 0) < 4:
        raise ReleaseError("The release evaluation must contain at least four stability runs")
    if stability.get("contract_passes") != stability.get("runs"):
        raise ReleaseError("Not every Q4 stability contract passed")
    deterministic = q4.get("deterministic_replay")
    if not isinstance(deterministic, dict) or deterministic.get("exact_text") is not True:
        raise ReleaseError("The Q4 deterministic replay was not exact")
    metrics = q4.get("metrics_after")
    if not isinstance(metrics, dict):
        raise ReleaseError("The Q4 runtime metrics are missing")
    if metrics.get("runtime") != "text-diffusion-gptq-rocm":
        raise ReleaseError("The evaluation did not use the packaged Q4 runtime")
    if metrics.get("quantization") != "gptq-q4-g32-expert-only":
        raise ReleaseError("The evaluation did not use the expected Q4 quantization")
    if metrics.get("q4_gate_calls", 0) <= 0:
        raise ReleaseError("The Q4 gate kernel was not invoked")
    if metrics.get("q4_gate_calls") != metrics.get("q4_down_calls"):
        raise ReleaseError("Q4 gate/down invocation counts differ")


def is_private_evaluation_field(key: str, value: Any) -> bool:
    normalized = key.lower()
    if normalized in PUBLIC_EVALUATION_REDACTED_KEYS:
        return True
    if normalized == "job_id" or normalized.endswith("_job_id"):
        return True
    if isinstance(value, str):
        if re.match(r"https?://(?:127\.0\.0\.1|localhost|0\.0\.0\.0)(?::\d+)?", value):
            return True
        if value.startswith(("/home/", "/mnt/", "/tmp/", "/workspace/")) and any(
            token in normalized for token in ("path", "dir", "checkpoint")
        ):
            return True
    return False


def public_evaluation_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return an evidence-preserving report with host-local identifiers removed."""

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: sanitize(item)
                for key, item in value.items()
                if not is_private_evaluation_field(key, item)
            }
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value

    sanitized = sanitize(report)
    if not isinstance(sanitized, dict):
        raise ReleaseError("Expected the public evaluation report to remain an object")
    sanitized["publication"] = {
        "sanitized": True,
        "removed": PUBLIC_EVALUATION_REDACTION_CATEGORIES,
    }
    validate_public_evaluation(sanitized)
    return sanitized


def validate_public_evaluation(report: dict[str, Any]) -> None:
    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                item_path = f"{path}.{key}" if path else key
                if is_private_evaluation_field(key, item):
                    raise ReleaseError(f"Public evaluation report contains private field: {item_path}")
                walk(item, item_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(report, "")
    publication = report.get("publication")
    if not isinstance(publication, dict) or publication.get("sanitized") is not True:
        raise ReleaseError("Public evaluation report has no sanitization marker")


def current_source_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ReleaseError("Tracked files must be clean before packaging a release bundle")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ReleaseError("Could not determine the ModelDeck Git commit; pass --source-commit explicitly")
    return value


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "gptqmodel",
        "torch",
        "transformers",
        "triton",
        "accelerate",
        "safetensors",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def gibibytes(value: int | float) -> float:
    return float(value) / GIB


def render_model_card(
    checkpoint_manifest: dict[str, Any],
    report: dict[str, Any],
    source_commit: str,
) -> str:
    model_id = checkpoint_manifest["base_model_id"]
    revision = checkpoint_manifest["base_model_revision"]
    q4_summary = report["q4"]["summary"]
    bf16_summary = report["bf16"]["summary"]
    q4_metrics = report["q4"]["metrics_after"]
    comparison = report["comparison"]
    comparison_summary = comparison["summary"]
    reduction = 100 * (
        1 - q4_metrics["memory_allocated_bytes"] / report["bf16"]["metrics_after"]["memory_allocated_bytes"]
    )
    q4_contracts = f"{q4_summary['contract_passes']}/{q4_summary['runs']}"
    bf16_contracts = f"{bf16_summary['contract_passes']}/{bf16_summary['runs']}"
    q4_constraints = f"{q4_summary['constraint_passes']}/{q4_summary['runs']}"
    bf16_constraints = f"{bf16_summary['constraint_passes']}/{bf16_summary['runs']}"
    q4_median = q4_summary["median_wall_seconds"]
    bf16_median = bf16_summary["median_wall_seconds"]
    q4_allocated = gibibytes(q4_metrics["memory_allocated_bytes"])
    bf16_allocated = gibibytes(report["bf16"]["metrics_after"]["memory_allocated_bytes"])
    stability_summary = report["q4"]["stability_summary"]
    stability_contracts = f"{stability_summary['contract_passes']}/{stability_summary['runs']}"
    self_contained = checkpoint_manifest.get("format_version") == 2
    if self_contained:
        package_description = f"""This is the self-contained ModelDeck Q4/BF16 hybrid for `{model_id}`. It
quantizes all 30 Mixture-of-Experts layers to symmetric GPTQ 4-bit weights with group
size 32 and packages the remaining model weights in BF16.

The pinned upstream model and revision remain recorded for provenance, but this release
does **not** download or load the upstream checkpoint at runtime. All model, processor,
tokenizer, and generation files needed by the ModelDeck loader are included here."""
        compatibility_description = (
            "This package is **not a standard Transformers or GPTQ checkpoint**. Use "
            "the\ncustom ModelDeck direct Q4 loader; Ollama, llama.cpp, generic "
            "`from_pretrained()`, and\nvLLM do not directly understand this hybrid "
            "checkpoint layout."
        )
        packaged_non_experts = checkpoint_manifest["non_expert_weights"]
        non_expert_line = (
            f"- Packaged non-expert BF16 tensors: {gibibytes(packaged_non_experts['tensor_bytes']):.3f} GiB"
        )
    else:
        package_description = (
            f"This is a ModelDeck expert-weight delta for `{model_id}`. It quantizes "
            "all 30\nMixture-of-Experts layers to symmetric GPTQ 4-bit weights with "
            "group size 32 while\nretaining the non-expert weights from the pinned "
            "base model in BF16."
        )
        compatibility_description = (
            "This package is **not a standard standalone GPTQ model** and is not "
            "directly\nloadable with `transformers.AutoModel`. It requires the base "
            f"model at revision\n`{revision}` and the custom ModelDeck direct Q4 "
            "loader. The package does not include the\nbase model's non-expert weights."
        )
        non_expert_line = "- Non-expert BF16 tensors: loaded from the pinned base model"
    return f"""---
license: apache-2.0
base_model: {model_id}
base_model_relation: quantized
tags:
  - diffusiongemma
  - text-diffusion
  - gptq
  - modeldeck
  - rocm
  - amd
---

# DiffusionGemma 26B A4B IT — ModelDeck self-contained GPTQ Q4 g32

[ModelDeck source and runtime](https://github.com/ozyjay/ModelDeck)

{package_description}

{compatibility_description}

## Quantization

- Method: GPTQ, 4 bits, symmetric
- Group size: 32
- Activation ordering: disabled
- Runtime: GPTQModel Triton V2 on ROCm
- Quantized tensors: expert `gate_up_proj` and `down_proj`
- Packed expert tensors: {gibibytes(q4_metrics["q4_bytes"]):.3f} GiB
{non_expert_line}
- ModelDeck source commit: `{source_commit}`

## Validated configuration

- Base model: `{model_id}`
- Base revision: `{revision}`
- Device: {q4_metrics["device_name"]}
- Torch: `{q4_metrics["torch_version"]}`
- HIP: `{q4_metrics["hip_version"]}`
- Transformers: `{q4_metrics["transformers_version"]}`
- Maximum output length: {report["configuration"]["max_length"]} tokens
- Maximum denoising steps: {report["configuration"]["denoising_steps"]}
- Temperature: {report["configuration"]["temperature"]}

## Release evaluation

| Measure | Q4 | BF16 |
|---|---:|---:|
| Contract passes | {q4_contracts} | {bf16_contracts} |
| Constraint passes | {q4_constraints} | {bf16_constraints} |
| Median wall time | {q4_median:.3f} s | {bf16_median:.3f} s |
| Steady allocated memory | {q4_allocated:.3f} GiB | {bf16_allocated:.3f} GiB |

- Q4/BF16 median latency ratio: {comparison["latency_ratio_q4_to_bf16"]:.4f}×
- Peak Q4 allocated memory: {gibibytes(q4_metrics["peak_memory_allocated_bytes"]):.3f} GiB
- Steady allocation reduction versus BF16: {reduction:.2f}%
- Mean token edit similarity versus BF16: {comparison_summary["mean_token_edit_similarity"]:.4f}
- Exact same-seed deterministic replay: passed
- Additional Q4 stability contracts: {stability_contracts}

The complete prompt-level results and release gates are included in
`q4-quality-evaluation.json`.

## ModelDeck usage

Download the immutable release into ModelDeck's expected checkpoint directory:

```powershell
hf download ozyjay/diffusiongemma-26b-a4b-it-modeldeck-gptq-q4-g32 `
    --revision v1.1.0 `
    --local-dir /mnt/work/models/modeldeck/diffusiongemma-26b-a4b-it-gptq-q4-g32
```

Then start the isolated worker from the ModelDeck repository:

```powershell
./scripts/start_diffusiongemma_q4.ps1 -Smoke
```

Verify every packaged file before use:

```powershell
./scripts/package_diffusiongemma_q4_release.ps1 -VerifyOnly
```

## Scope and limitations

- Validated for text-diffusion generation on one AMD Radeon 8060S (`gfx1151`) using
  the pinned ROCm stack above.
- The Q4 experts reduce memory substantially but are slower than BF16 on the tested
  hardware.
- The package has not been validated for multimodal generation, other GPUs, other base
  revisions, or other GPTQ runtimes.
- Compatibility is tied to the ModelDeck source commit above; treat a different loader
  revision as unvalidated until the release gate passes again.
- Generated output can be inaccurate, biased, or unsafe. Apply task-appropriate safety
  and factuality checks.

## Licence and provenance

The base DiffusionGemma model is published by Google DeepMind under Apache License 2.0.
This package contains transformed expert weights derived from the pinned base revision.
See `LICENSE` and `THIRD_PARTY_NOTICES.md` for redistribution information.
"""


def render_notices(checkpoint_manifest: dict[str, Any]) -> str:
    self_contained = checkpoint_manifest.get("format_version") == 2
    weight_notice = (
        "The expert weights in this release were modified by GPTQ quantization. The "
        "remaining model weights are included in BF16, making the ModelDeck artifact "
        "self-contained at runtime."
        if self_contained
        else "The expert weights in this release were modified by GPTQ quantization. "
        "The base model's non-expert weights are not included and must be obtained "
        "separately from the upstream model repository."
    )
    return f"""# Third-party notices

## DiffusionGemma

- Upstream model: `{checkpoint_manifest["base_model_id"]}`
- Pinned revision: `{checkpoint_manifest["base_model_revision"]}`
- Author: Google DeepMind
- Model page: https://huggingface.co/{checkpoint_manifest["base_model_id"]}
- Licence: Apache License 2.0
- Licence page: https://ai.google.dev/gemma/apache_2

{weight_notice}

The pinned upstream snapshot does not contain a separate `NOTICE` file. If a future
base revision adds one, it must be included in any derivative release built from that
revision.

GPTQModel, PyTorch, Transformers, Triton, Accelerate, and Safetensors are runtime tools
and dependencies; their source or binary distributions are not included in this model
package.
"""


def file_entry(checkpoint_dir: Path, path: Path, role: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(checkpoint_dir).as_posix(),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_release(
    *,
    checkpoint_dir: Path,
    evaluation_report: Path,
    license_file: Path,
    source_commit: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ReleaseError("--source-commit must be a full 40-character Git SHA")
    checkpoint_dir = checkpoint_dir.resolve()
    checkpoint_manifest, shard_paths, non_expert_paths, runtime_paths = validate_checkpoint(checkpoint_dir)
    report = read_json(evaluation_report)
    validate_evaluation(report, checkpoint_manifest)
    if not license_file.is_file():
        raise ReleaseError(f"Apache 2.0 licence file was not found: {license_file}")

    packaged_report = checkpoint_dir / "q4-quality-evaluation.json"
    write_json(packaged_report, public_evaluation_report(report))
    shutil.copy2(license_file, checkpoint_dir / "LICENSE")
    write_text(
        checkpoint_dir / "README.md",
        render_model_card(checkpoint_manifest, report, source_commit),
    )
    write_text(
        checkpoint_dir / "THIRD_PARTY_NOTICES.md",
        render_notices(checkpoint_manifest),
    )

    entries = [
        file_entry(
            checkpoint_dir,
            checkpoint_dir / "q4-manifest.json",
            "checkpoint-manifest",
        )
    ]
    entries.extend(file_entry(checkpoint_dir, path, "expert-shard") for path in shard_paths)
    entries.extend(
        file_entry(
            checkpoint_dir,
            path,
            "non-expert-index" if path.suffix == ".json" else "non-expert-shard",
        )
        for path in non_expert_paths
    )
    entries.extend(file_entry(checkpoint_dir, path, "runtime-metadata") for path in runtime_paths)
    entries.extend(
        file_entry(checkpoint_dir, checkpoint_dir / name, role)
        for name, role in (
            ("q4-quality-evaluation.json", "evaluation-report"),
            ("README.md", "model-card"),
            ("LICENSE", "license"),
            ("THIRD_PARTY_NOTICES.md", "third-party-notices"),
        )
    )
    q4_metrics = report["q4"]["metrics_after"]
    release_manifest = {
        "format": "modeldeck-diffusiongemma-q4-release",
        "format_version": 2 if checkpoint_manifest.get("format_version") == 2 else 1,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "release_name": "diffusiongemma-26b-a4b-it-gptq-q4-g32",
        "artifact_type": (
            "self-contained-q4-bf16-hybrid"
            if checkpoint_manifest.get("format_version") == 2
            else "expert-only-quantized-weight-delta"
        ),
        "modeldeck_source_commit": source_commit,
        "base_model": {
            "id": checkpoint_manifest["base_model_id"],
            "revision": checkpoint_manifest["base_model_revision"],
            "license": "apache-2.0",
        },
        "quantization": checkpoint_manifest["quantization"],
        "runtime": {
            "python": platform.python_version(),
            "packages": package_versions(),
            "torch": q4_metrics["torch_version"],
            "hip": q4_metrics["hip_version"],
            "transformers": q4_metrics["transformers_version"],
        },
        "validated_hardware": {
            "device_name": q4_metrics["device_name"],
            "device": q4_metrics["device"],
        },
        "evaluation": {
            "report": "q4-quality-evaluation.json",
            "completed_at": report["completed_at"],
            "checks": report["checks"],
            "q4_contract_passes": report["q4"]["summary"]["contract_passes"],
            "q4_constraint_pass_rate": report["q4"]["summary"]["constraint_pass_rate"],
            "bf16_constraint_pass_rate": report["bf16"]["summary"]["constraint_pass_rate"],
            "deterministic_replay": report["q4"]["deterministic_replay"]["exact_text"],
            "q4_peak_memory_bytes": q4_metrics["peak_memory_allocated_bytes"],
            "q4_median_seconds": report["q4"]["summary"]["median_wall_seconds"],
            "bf16_median_seconds": report["bf16"]["summary"]["median_wall_seconds"],
            "latency_ratio_q4_to_bf16": report["comparison"]["latency_ratio_q4_to_bf16"],
            "mean_token_edit_similarity": report["comparison"]["summary"]["mean_token_edit_similarity"],
        },
        "files": sorted(entries, key=lambda item: item["path"]),
    }
    release_manifest_path = checkpoint_dir / "release-manifest.json"
    write_json(release_manifest_path, release_manifest)

    checksums = {entry["path"]: entry["sha256"] for entry in release_manifest["files"]}
    checksums[release_manifest_path.name] = sha256_file(release_manifest_path)
    write_text(
        checkpoint_dir / "SHA256SUMS",
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
    )
    return release_manifest


def parse_sha256sums(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ReleaseError(f"Checksum file was not found: {path}")
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ReleaseError(f"Invalid SHA256SUMS line {line_number}")
        digest, name = match.groups()
        safe_relative_path(name)
        if name in result:
            raise ReleaseError(f"Duplicate checksum entry: {name}")
        result[name] = digest
    return result


def verify_release(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.resolve()
    release_manifest_path = checkpoint_dir / "release-manifest.json"
    manifest = read_json(release_manifest_path)
    if manifest.get("format") != "modeldeck-diffusiongemma-q4-release":
        raise ReleaseError("Unsupported release manifest format")
    if manifest.get("format_version") not in {1, 2}:
        raise ReleaseError("Unsupported release manifest version")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ReleaseError("Release manifest contains no files")

    expected_checksums: dict[str, str] = {}
    total_bytes = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise ReleaseError("Release manifest contains an invalid file entry")
        name = entry.get("path")
        if not isinstance(name, str):
            raise ReleaseError("Release manifest file path is missing")
        path = checkpoint_dir / safe_relative_path(name)
        if not path.is_file():
            raise ReleaseError(f"Release file is missing: {name}")
        size = path.stat().st_size
        if size != entry.get("size_bytes"):
            raise ReleaseError(f"Release file size mismatch: {name}")
        digest = sha256_file(path)
        if digest != entry.get("sha256"):
            raise ReleaseError(f"Release file checksum mismatch: {name}")
        expected_checksums[name] = digest
        total_bytes += size

    expected_checksums[release_manifest_path.name] = sha256_file(release_manifest_path)
    checksum_file = parse_sha256sums(checkpoint_dir / "SHA256SUMS")
    if checksum_file != expected_checksums:
        missing = sorted(set(expected_checksums) - set(checksum_file))
        extra = sorted(set(checksum_file) - set(expected_checksums))
        raise ReleaseError(
            f"SHA256SUMS does not match the release manifest (missing={missing}, extra={extra})"
        )
    checkpoint_manifest, _, _, _ = validate_checkpoint(checkpoint_dir)
    packaged_report = read_json(checkpoint_dir / "q4-quality-evaluation.json")
    validate_public_evaluation(packaged_report)
    validate_evaluation(packaged_report, checkpoint_manifest)
    return {
        "release_name": manifest.get("release_name"),
        "files_verified": len(files),
        "payload_bytes": total_bytes,
        "payload_gib": round(gibibytes(total_bytes), 3),
        "source_commit": manifest.get("modeldeck_source_commit"),
        "base_model": manifest.get("base_model"),
        "artifact_type": manifest.get("artifact_type"),
        "self_contained": manifest.get("artifact_type") == "self-contained-q4-bf16-hybrid",
        "evaluation_passed": packaged_report.get("passed") is True,
    }


def main() -> None:
    args = parse_args()
    try:
        if args.verify_only:
            summary = verify_release(args.checkpoint_dir)
        else:
            source_commit = args.source_commit or current_source_commit()
            build_release(
                checkpoint_dir=args.checkpoint_dir,
                evaluation_report=args.evaluation_report,
                license_file=args.license_file,
                source_commit=source_commit,
            )
            summary = verify_release(args.checkpoint_dir)
    except ReleaseError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

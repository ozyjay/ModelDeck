from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from modeldeck.hardware.probe import cache_candidates
from modeldeck.q4_release import Q4ReleaseError, inspect_modeldeck_q4_release
from modeldeck.speechshift import SPEECHSHIFT_MODEL_SPECS, validate_speechshift_snapshot


def resolve_cache_paths(env: Mapping[str, str] | None = None) -> list[Path]:
    if env is None:
        return [Path(path).expanduser() for path in cache_candidates()]
    candidates = []
    if env.get("HF_HUB_CACHE"):
        candidates.append(Path(env["HF_HUB_CACHE"]))
    if env.get("HF_HOME"):
        candidates.append(Path(env["HF_HOME"]) / "hub")
    home = Path(env.get("HOME", str(Path.home())))
    candidates.extend((home / ".cache/huggingface/hub", Path("/mnt/work/models/huggingface/hub")))
    return list(dict.fromkeys(path.expanduser() for path in candidates))


def _revision(model_dir: Path, snapshot: Path) -> str:
    del model_dir
    return snapshot.name


def _snapshot_complete(snapshot: Path) -> bool:
    safetensors = list(snapshot.glob("*.safetensors"))
    pytorch_weights = list(snapshot.glob("pytorch_model*.bin"))
    return (
        _weight_files_complete(safetensors)
        or _weight_files_complete(pytorch_weights)
        or any(snapshot.glob("*.gguf"))
    )


def _weight_files_complete(paths: list[Path]) -> bool:
    if not paths:
        return False
    shard_pattern = re.compile(r"-(\d+)-of-(\d+)\.(?:safetensors|bin)$")
    shard_matches = [shard_pattern.search(path.name) for path in paths]
    if not any(shard_matches):
        return True
    if any(match is None for match in shard_matches):
        return True
    expected_counts = {int(match.group(2)) for match in shard_matches if match is not None}
    shard_numbers = {int(match.group(1)) for match in shard_matches if match is not None}
    if len(expected_counts) != 1:
        return False
    expected = expected_counts.pop()
    return shard_numbers == set(range(1, expected + 1))


def _artifacts(snapshot: Path, repo_id: str) -> list[dict[str, Any]]:
    if repo_id != "ggml-org/gpt-oss-120b-GGUF":
        return []
    consolidated = snapshot / "gpt-oss-120b-MXFP4.gguf"
    if consolidated.is_file():
        return [
            {
                "artifact_id": "gpt-oss-120b-mxfp4",
                "kind": "gguf",
                "format": "mxfp4",
                "filenames": [consolidated.name],
            }
        ]
    shards = sorted(snapshot.glob("gpt-oss-120b-mxfp4-*-of-*.gguf"))
    if len(shards) != 3:
        return []
    return [
        {
            "artifact_id": "gpt-oss-120b-mxfp4",
            "kind": "gguf",
            "format": "mxfp4",
            "filenames": [path.name for path in shards],
        }
    ]


def _physical_size(paths: Iterable[Path]) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    for root in paths:
        for path in root.rglob("*"):
            try:
                stat = path.stat()
            except OSError:
                continue
            key = (stat.st_dev, stat.st_ino)
            if path.is_file() and key not in seen:
                total += stat.st_size
                seen.add(key)
    return total


def _generation_family(snapshot: Path, repo_id: str = "") -> str | None:
    if spec := SPEECHSHIFT_MODEL_SPECS.get(repo_id):
        return spec.generation_family
    if repo_id == "kyutai/moshiko-pytorch-bf16":
        return "speech-conversation"
    if repo_id == "ggml-org/gpt-oss-120b-GGUF":
        return "autoregressive"
    try:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    architectures = " ".join(config.get("architectures") or ()).lower()
    model_type = str(config.get("model_type", "")).lower()
    if "diffusion" in architectures or "diffusion" in model_type:
        return "text-diffusion"
    if (
        "multimodal" in architectures
        or model_type in {"gemma4", "gemma4_unified"}
        or (config.get("vision_config") and config.get("text_config"))
    ):
        return "vision-language"
    if "causallm" in architectures or config.get("is_decoder"):
        return "autoregressive"
    return None


def _capability_hints(generation_family: str | None) -> list[str]:
    return {
        "autoregressive": ["text-generation", "chat"],
        "vision-language": ["text-generation", "chat", "image-input", "structured-output"],
        "text-diffusion": [
            "text-generation",
            "iterative-refinement",
            "intermediate-frames",
            "seeded-generation",
        ],
        "speech-conversation": ["audio-input", "audio-output", "full-duplex"],
        "text-translation": ["translation"],
        "speech-synthesis": ["audio-output", "speech-synthesis"],
        "speech-recognition": ["audio-input", "speech-recognition"],
    }.get(generation_family, [])


def _configuration_support(snapshot: Path, repo_id: str = "", revision: str = "") -> tuple[str | None, str]:
    if spec := SPEECHSHIFT_MODEL_SPECS.get(repo_id):
        if error := validate_speechshift_snapshot(snapshot, repo_id, revision):
            return None, error
        if spec.generation_family == "text-translation":
            return spec.configuration_support, (
                f"Supported by the pinned OPUS {spec.source_language}→{spec.target_language} CPU worker."
            )
        return spec.configuration_support, "Supported by the isolated Qwen3-TTS ROCm worker."
    if repo_id == "kyutai/moshiko-pytorch-bf16":
        required = {
            "model.safetensors",
            "tokenizer_spm_32k_3.model",
            "tokenizer-e351c8d8-checkpoint125.safetensors",
        }
        missing = sorted(name for name in required if not (snapshot / name).is_file())
        if missing:
            return None, f"The Moshiko snapshot is incomplete: missing {', '.join(missing)}."
        return "moshiko-speech", "Supported by the isolated Repartee Moshiko ROCm worker."
    if repo_id == "ggml-org/gpt-oss-120b-GGUF":
        if _artifacts(snapshot, repo_id):
            return "gpt-oss-llama-vulkan", (
                "Supported by the pinned Repartee llama.cpp Vulkan runtime; "
                "hardware verification is required."
            )
        return None, (
            "The GPT-OSS MXFP4 GGUF snapshot must contain the official consolidated "
            "artefact or all three legacy shards."
        )
    try:
        q4_release = inspect_modeldeck_q4_release(snapshot)
    except Q4ReleaseError as error:
        return None, f"ModelDeck Q4 release validation failed: {error}"
    if q4_release is not None:
        return (
            "diffusiongemma-modeldeck-q4",
            "Supported by the dedicated ModelDeck DiffusionGemma Q4 runtime.",
        )
    try:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "The snapshot has no readable Transformers configuration."
    architectures = {str(value) for value in config.get("architectures") or ()}
    model_type = str(config.get("model_type", "")).lower()
    qwen35_scenechat_models = {
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
    }
    if repo_id == "openai/gpt-oss-120b" or model_type == "gpt_oss":
        return None, (
            "This is the GPT-OSS source snapshot. Configure the pinned "
            "ggml-org/gpt-oss-120b-GGUF MXFP4 companion snapshot for AMD Vulkan."
        )
    if config.get("quantization_config"):
        return None, "Quantised snapshots require a dedicated, compatibility-tested runtime."
    if (
        repo_id in qwen35_scenechat_models
        and model_type == "qwen3_5"
        and "Qwen3_5ForConditionalGeneration" in architectures
    ):
        return "scenechat-qwen35", "Supported by the dedicated SceneChat Qwen3.5 worker."
    if any(architecture.endswith("ForCausalLM") for architecture in architectures) or config.get(
        "is_decoder"
    ):
        return "autoregressive-transformers", "Supported by the local Transformers ROCm worker."
    if (model_type == "gemma4" and "Gemma4ForConditionalGeneration" in architectures) or (
        model_type == "gemma4_unified" and "Gemma4UnifiedForConditionalGeneration" in architectures
    ):
        return "scenechat-gemma4", "Supported by the dedicated SceneChat Gemma 4 worker."
    if model_type == "diffusiongemma" or "DiffusionGemmaForBlockDiffusion" in architectures:
        return "diffusiongemma-transformers", (
            "Supported by the dedicated DiffusionGemma Transformers worker."
        )
    return None, "No allowlisted ModelDeck worker supports this architecture yet."


def discover_huggingface_models(paths: Iterable[Path] | None = None) -> list[dict[str, Any]]:
    models = []
    for cache_root in paths or resolve_cache_paths():
        if not cache_root.is_dir():
            continue
        for model_dir in sorted(cache_root.glob("models--*")):
            snapshots = [path for path in (model_dir / "snapshots").glob("*") if path.is_dir()]
            complete = [path for path in snapshots if _snapshot_complete(path)]
            partial = any(model_dir.rglob("*.incomplete")) or bool(snapshots and not complete)
            if not snapshots and not partial:
                continue
            repo_id = model_dir.name.removeprefix("models--").replace("--", "/")
            pinned_spec = SPEECHSHIFT_MODEL_SPECS.get(repo_id)
            pinned_snapshot = model_dir / "snapshots" / pinned_spec.revision if pinned_spec else None
            chosen = (
                pinned_snapshot
                if pinned_snapshot is not None and pinned_snapshot in complete
                else complete[-1]
                if complete
                else snapshots[-1]
                if snapshots
                else None
            )
            state = "partial" if partial and not complete else "installed-untested" if complete else "partial"
            revision = _revision(model_dir, chosen) if chosen else None
            support, support_reason = (
                _configuration_support(chosen, repo_id, revision or "")
                if chosen and complete
                else (None, "Finish the local snapshot before configuring a runtime.")
            )
            try:
                q4_release = inspect_modeldeck_q4_release(chosen) if chosen and complete else None
            except Q4ReleaseError:
                q4_release = None
            generation_family = _generation_family(chosen, repo_id) if chosen else None
            models.append(
                {
                    "model_id": repo_id,
                    "revision": revision,
                    "cache_location": str(model_dir),
                    "snapshot_location": str(chosen) if chosen else None,
                    "physical_size_bytes": _physical_size((model_dir,)),
                    "download_state": state,
                    "generation_family_hint": generation_family,
                    "capability_hints": _capability_hints(generation_family),
                    "configuration_support": support,
                    "configuration_support_reason": support_reason,
                    "base_model_id": q4_release.get("base_model_id") if q4_release else None,
                    "base_model_revision": (q4_release.get("base_model_revision") if q4_release else None),
                    "runnable": False,
                    "runnable_reason": "Compatibility has not been tested for the current stack.",
                    "artifacts": _artifacts(chosen, repo_id) if chosen and complete else [],
                }
            )
    return models

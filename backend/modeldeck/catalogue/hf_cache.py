from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from modeldeck.hardware.probe import cache_candidates
from modeldeck.q4_release import Q4ReleaseError, inspect_modeldeck_q4_release


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
    refs_main = model_dir / "refs/main"
    try:
        return refs_main.read_text(encoding="utf-8").strip() or snapshot.name
    except OSError:
        return snapshot.name


def _snapshot_complete(snapshot: Path) -> bool:
    weights = any(snapshot.glob("*.safetensors")) or any(snapshot.glob("pytorch_model*.bin"))
    return weights


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


def _generation_family(snapshot: Path) -> str | None:
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
        or model_type == "gemma4"
        or (config.get("vision_config") and config.get("text_config"))
    ):
        return "vision-language"
    if "causallm" in architectures or config.get("is_decoder"):
        return "autoregressive"
    return None


def _configuration_support(snapshot: Path) -> tuple[str | None, str]:
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
    if config.get("quantization_config"):
        return None, "Quantised snapshots require a dedicated, compatibility-tested runtime."
    if any(architecture.endswith("ForCausalLM") for architecture in architectures) or config.get(
        "is_decoder"
    ):
        return "autoregressive-transformers", "Supported by the local Transformers ROCm worker."
    if model_type == "gemma4" and "Gemma4ForConditionalGeneration" in architectures:
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
            chosen = complete[-1] if complete else (snapshots[-1] if snapshots else None)
            repo_id = model_dir.name.removeprefix("models--").replace("--", "/")
            state = "partial" if partial and not complete else "installed-untested" if complete else "partial"
            support, support_reason = (
                _configuration_support(chosen)
                if chosen and complete
                else (None, "Finish the local snapshot before configuring a runtime.")
            )
            try:
                q4_release = inspect_modeldeck_q4_release(chosen) if chosen and complete else None
            except Q4ReleaseError:
                q4_release = None
            models.append(
                {
                    "model_id": repo_id,
                    "revision": _revision(model_dir, chosen) if chosen else None,
                    "cache_location": str(model_dir),
                    "snapshot_location": str(chosen) if chosen else None,
                    "physical_size_bytes": _physical_size((model_dir,)),
                    "download_state": state,
                    "generation_family_hint": _generation_family(chosen) if chosen else None,
                    "configuration_support": support,
                    "configuration_support_reason": support_reason,
                    "base_model_id": q4_release.get("base_model_id") if q4_release else None,
                    "base_model_revision": (q4_release.get("base_model_revision") if q4_release else None),
                    "runnable": False,
                    "runnable_reason": "Compatibility has not been tested for the current stack.",
                }
            )
    return models

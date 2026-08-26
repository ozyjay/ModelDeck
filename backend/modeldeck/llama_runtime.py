from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LLAMA_CPP_COMMIT = "9d77fa17254e1dee4b9e92504c91611a60b1359f"
LLAMA_RUNTIME_ROOT = Path(".runtime-tools/llama.cpp")
LLAMA_SERVER_RELATIVE_PATH = Path("bin/llama-server")
LLAMA_BUILD_RECEIPT_RELATIVE_PATH = Path("bin/modeldeck-build.json")


class TrustedArtefact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]+\.gguf$")
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dtype: str


class QwenLlamaManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["modeldeck-qwen-llamacpp-runtime"]
    version: Literal[1]
    id: str
    status: Literal["reviewed-candidate", "experimental", "approved-local"]
    original_model_id: str
    original_model_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    artefact_model_id: str
    artefact_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    quantisation: str
    model: TrustedArtefact
    projector: TrustedArtefact | None = None
    mtp_model: TrustedArtefact | None = None
    llama_cpp_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    operating_system: Literal["linux"]
    architecture: Literal["x86_64"]
    backend: Literal["Vulkan"]
    qwen_architecture: Literal["qwen35"]
    chat_template_fingerprint: str
    context_length: int = Field(ge=8192, le=32768)
    cache_type_k: Literal["f16", "q8_0"]
    cache_type_v: Literal["f16", "q8_0"]
    mtp_draft_tokens: Literal[4] | None = None
    source_url: str
    licence: Literal["Apache-2.0"]

    @model_validator(mode="after")
    def validate_optional_companions(self) -> QwenLlamaManifest:
        if (self.projector is None) != (self.mtp_model is None):
            raise ValueError("Qwen llama.cpp projector and MTP artefacts must be declared together")
        if (self.mtp_model is None) != (self.mtp_draft_tokens is None):
            raise ValueError("Qwen llama.cpp MTP settings must match the declared MTP artefact")
        return self


class LlamaBuildReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["modeldeck-llama-build"]
    version: Literal[1]
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    executable_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    backend: Literal["Vulkan"]
    operating_system: Literal["linux"]
    architecture: Literal["x86_64"]


@dataclass(frozen=True)
class ValidatedQwenRuntime:
    manifest: QwenLlamaManifest
    executable: Path
    model: Path
    projector: Path | None
    mtp_model: Path | None
    executable_sha256: str


def manifest_path(profile: str) -> Path:
    if profile not in {
        "qwen35-4b-q8-vulkan",
        "qwen38-q8-mtp-vulkan",
        "qwen38-q4-mtp-vulkan",
    }:
        raise ValueError("Unknown allowlisted Qwen llama.cpp profile")
    return Path(__file__).with_name("registry_data") / f"{profile}.json"


def load_qwen_manifest(
    profile: str, *, data_dir: Path | None = None, candidate_id: str | None = None
) -> QwenLlamaManifest:
    if profile == "qwen35-approved-q8-vulkan":
        if data_dir is None or candidate_id is None:
            raise ValueError("The approved Qwen candidate identity is required")
        from modeldeck.qwen_candidates import load_candidate

        return load_candidate(data_dir, candidate_id)
    return QwenLlamaManifest.model_validate_json(manifest_path(profile).read_bytes())


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artefact(path: Path, expected: TrustedArtefact) -> None:
    if path.name != expected.filename or not path.is_file():
        raise ValueError(f"Trusted llama.cpp artefact is missing: {expected.filename}")
    if path.stat().st_size != expected.size:
        raise ValueError(f"Trusted llama.cpp artefact size mismatch: {expected.filename}")
    if sha256_file(path) != expected.sha256:
        raise ValueError(f"Trusted llama.cpp artefact checksum mismatch: {expected.filename}")


def validate_qwen_runtime(
    profile: str,
    snapshot: Path,
    *,
    data_dir: Path | None = None,
    candidate_id: str | None = None,
) -> ValidatedQwenRuntime:
    manifest = (
        load_qwen_manifest(profile, data_dir=data_dir, candidate_id=candidate_id)
        if profile == "qwen35-approved-q8-vulkan"
        else load_qwen_manifest(profile)
    )
    if platform.system().lower() != manifest.operating_system:
        raise ValueError("The trusted llama.cpp runtime supports Linux only")
    if platform.machine().lower() not in {manifest.architecture, "amd64"}:
        raise ValueError("The trusted llama.cpp runtime supports x86_64 only")
    if manifest.llama_cpp_commit != LLAMA_CPP_COMMIT:
        raise ValueError("The Qwen runtime manifest does not match the code-owned llama.cpp pin")

    runtime_root = LLAMA_RUNTIME_ROOT.resolve()
    executable = runtime_root / LLAMA_SERVER_RELATIVE_PATH
    receipt_path = runtime_root / LLAMA_BUILD_RECEIPT_RELATIVE_PATH
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Pinned llama.cpp Vulkan runtime is missing or not executable")
    if not receipt_path.is_file():
        raise ValueError("Pinned llama.cpp build receipt is missing")
    receipt = LlamaBuildReceipt.model_validate_json(receipt_path.read_bytes())
    if receipt.commit != LLAMA_CPP_COMMIT:
        raise ValueError("Installed llama.cpp build does not match the pinned commit")
    executable_digest = sha256_file(executable)
    if executable_digest != receipt.executable_sha256:
        raise ValueError("Installed llama-server checksum does not match its build receipt")

    snapshot = snapshot.absolute()
    model = snapshot / manifest.model.filename
    projector = snapshot / manifest.projector.filename if manifest.projector else None
    mtp_model = snapshot / manifest.mtp_model.filename if manifest.mtp_model else None
    _verify_artefact(model, manifest.model)
    if projector is not None and manifest.projector is not None:
        _verify_artefact(projector, manifest.projector)
    if mtp_model is not None and manifest.mtp_model is not None:
        _verify_artefact(mtp_model, manifest.mtp_model)
    return ValidatedQwenRuntime(
        manifest=manifest,
        executable=executable,
        model=model,
        projector=projector,
        mtp_model=mtp_model,
        executable_sha256=executable_digest,
    )


def configuration_fingerprint(runtime: ValidatedQwenRuntime, *, thinking_mode: str | None = None) -> str:
    payload = {
        "manifest": runtime.manifest.model_dump(mode="json"),
        "executable_sha256": runtime.executable_sha256,
        "thinking_mode": thinking_mode,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

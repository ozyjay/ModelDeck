from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LLAMA_CPP_COMMIT = "9d77fa17254e1dee4b9e92504c91611a60b1359f"
LLAMA_ACCEPTED_OLDER_COMMITS: frozenset[str] = frozenset()
LLAMA_ACCEPTED_ALTERNATIVE_COMMITS: frozenset[str] = frozenset()
LLAMA_REVOKED_COMMITS: frozenset[str] = frozenset()
LLAMA_RUNTIME_ROOT = Path(".runtime-tools/llama.cpp")
LLAMA_SERVER_RELATIVE_PATH = Path("bin/llama-server")
LLAMA_BUILD_RECEIPT_RELATIVE_PATH = Path("bin/modeldeck-build.json")

GPT_OSS_LLAMA_REQUIRED_FLAGS = (
    "--host",
    "--port",
    "--model",
    "--ctx-size",
    "--parallel",
    "--n-gpu-layers",
    "--flash-attn",
    "--jinja",
)
QWEN_LLAMA_REQUIRED_FLAGS = (
    "--host",
    "--port",
    "--model",
    "--ctx-size",
    "--parallel",
    "--device",
    "--gpu-layers",
    "--fit",
    "--flash-attn",
    "--cache-type-k",
    "--cache-type-v",
    "--jinja",
    "--reasoning-format",
    "--metrics",
    "--slots",
    "--offline",
    "--no-mmproj",
    "--reasoning-effort",
)
QWEN_MTP_LLAMA_REQUIRED_FLAGS = QWEN_LLAMA_REQUIRED_FLAGS + (
    "--mmproj",
    "--spec-type",
    "--spec-draft-model",
    "--spec-draft-device",
    "--spec-draft-ngl",
    "--spec-draft-n-max",
)
ALL_LLAMA_REQUIRED_FLAGS = tuple(dict.fromkeys(GPT_OSS_LLAMA_REQUIRED_FLAGS + QWEN_MTP_LLAMA_REQUIRED_FLAGS))


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


class LlamaRuntimeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_revision: str | None = None
    executable_sha256: str | None = None
    executable_size_bytes: int | None = None
    receipt_sha256: str | None = None
    receipt_version: int | None = None
    backend: str | None = None
    operating_system: str
    architecture: str
    version_output: str | None = None


class LlamaRuntimeInstallation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    installation_id: Literal["llama-cpp-vulkan"] = "llama-cpp-vulkan"
    display_name: Literal["llama.cpp Vulkan"] = "llama.cpp Vulkan"
    integrity_status: Literal[
        "verified",
        "missing",
        "receipt-missing",
        "receipt-invalid",
        "modified",
        "feature-mismatch",
        "inspection-failed",
    ]
    currency_status: Literal[
        "recommended",
        "accepted-older",
        "accepted-alternative",
        "different-unqualified",
        "newer-unqualified",
        "revoked",
        "unknown",
    ]
    start_allowed: bool
    detected: LlamaRuntimeIdentity
    recommended_source_revision: str = LLAMA_CPP_COMMIT
    required_features: tuple[str, ...]
    missing_features: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    inspected_at: datetime


@dataclass(frozen=True)
class ValidatedLlamaInstallation:
    executable: Path
    receipt: LlamaBuildReceipt
    executable_sha256: str
    receipt_sha256: str
    version_output: str


@dataclass(frozen=True)
class ValidatedQwenRuntime:
    manifest: QwenLlamaManifest
    executable: Path
    model: Path
    projector: Path | None
    mtp_model: Path | None
    executable_sha256: str
    receipt_sha256: str
    source_revision: str


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


def _bounded_command_output(executable: Path, argument: str) -> str:
    result = subprocess.run(
        [str(executable), argument],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise RuntimeError(f"llama-server {argument} exited with code {result.returncode}")
    return output[:128_000]


def inspect_llama_installation(
    *,
    runtime_root: Path | None = None,
    required_flags: tuple[str, ...] = ALL_LLAMA_REQUIRED_FLAGS,
) -> LlamaRuntimeInstallation:
    root = (runtime_root or LLAMA_RUNTIME_ROOT).resolve()
    executable = root / LLAMA_SERVER_RELATIVE_PATH
    receipt_path = root / LLAMA_BUILD_RECEIPT_RELATIVE_PATH
    os_name = platform.system().lower()
    architecture = platform.machine().lower()
    inspected_at = datetime.now(UTC)

    def result(
        integrity_status: str,
        currency_status: str,
        *,
        identity: LlamaRuntimeIdentity,
        missing_features: tuple[str, ...] = (),
        reason_codes: tuple[str, ...] = (),
    ) -> LlamaRuntimeInstallation:
        return LlamaRuntimeInstallation(
            integrity_status=integrity_status,
            currency_status=currency_status,
            start_allowed=integrity_status == "verified"
            and currency_status in {"recommended", "accepted-older", "accepted-alternative"},
            detected=identity,
            required_features=required_flags,
            missing_features=missing_features,
            reason_codes=reason_codes,
            inspected_at=inspected_at,
        )

    base_identity = LlamaRuntimeIdentity(operating_system=os_name, architecture=architecture)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        return result("missing", "unknown", identity=base_identity, reason_codes=("executable_missing",))
    try:
        executable_size = executable.stat().st_size
    except OSError:
        return result(
            "inspection-failed", "unknown", identity=base_identity, reason_codes=("executable_stat_failed",)
        )
    identity_values: dict[str, object] = {
        "operating_system": os_name,
        "architecture": architecture,
        "executable_size_bytes": executable_size,
    }
    if not receipt_path.is_file():
        return result(
            "receipt-missing",
            "unknown",
            identity=LlamaRuntimeIdentity(**identity_values),
            reason_codes=("build_receipt_missing",),
        )
    try:
        receipt_bytes = receipt_path.read_bytes()
        receipt = LlamaBuildReceipt.model_validate_json(receipt_bytes)
        identity_values.update(
            source_revision=receipt.commit,
            receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
            receipt_version=receipt.version,
            backend=receipt.backend,
        )
    except (OSError, ValueError):
        return result(
            "receipt-invalid",
            "unknown",
            identity=LlamaRuntimeIdentity(**identity_values),
            reason_codes=("build_receipt_invalid",),
        )
    if receipt.commit in LLAMA_REVOKED_COMMITS:
        currency_status = "revoked"
    elif receipt.commit == LLAMA_CPP_COMMIT:
        currency_status = "recommended"
    elif receipt.commit in LLAMA_ACCEPTED_OLDER_COMMITS:
        currency_status = "accepted-older"
    elif receipt.commit in LLAMA_ACCEPTED_ALTERNATIVE_COMMITS:
        currency_status = "accepted-alternative"
    else:
        currency_status = "different-unqualified"
    try:
        executable_digest = sha256_file(executable)
        identity_values["executable_sha256"] = executable_digest
    except OSError:
        return result(
            "inspection-failed",
            currency_status,
            identity=LlamaRuntimeIdentity(**identity_values),
            reason_codes=("executable_hash_failed",),
        )
    if executable_digest != receipt.executable_sha256:
        return result(
            "modified",
            currency_status,
            identity=LlamaRuntimeIdentity(**identity_values),
            reason_codes=("executable_checksum_mismatch",),
        )
    if os_name != receipt.operating_system or architecture not in {receipt.architecture, "amd64"}:
        return result(
            "feature-mismatch",
            currency_status,
            identity=LlamaRuntimeIdentity(**identity_values),
            reason_codes=("platform_mismatch",),
        )
    if currency_status not in {"recommended", "accepted-older", "accepted-alternative"}:
        return result(
            "verified",
            currency_status,
            identity=LlamaRuntimeIdentity(**identity_values),
            reason_codes=("source_revision_not_recommended",),
        )
    try:
        version_output = _bounded_command_output(executable, "--version")[:1024]
        help_output = _bounded_command_output(executable, "--help")
        identity_values["version_output"] = version_output
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        return result(
            "inspection-failed",
            currency_status,
            identity=LlamaRuntimeIdentity(**identity_values),
            reason_codes=("feature_inspection_failed",),
        )
    missing_features = tuple(flag for flag in required_flags if flag not in help_output)
    if missing_features:
        return result(
            "feature-mismatch",
            currency_status,
            identity=LlamaRuntimeIdentity(**identity_values),
            missing_features=missing_features,
            reason_codes=("required_feature_missing",),
        )
    return result("verified", currency_status, identity=LlamaRuntimeIdentity(**identity_values))


def validate_llama_installation(
    *, required_flags: tuple[str, ...], runtime_root: Path | None = None
) -> ValidatedLlamaInstallation:
    installation = inspect_llama_installation(runtime_root=runtime_root, required_flags=required_flags)
    if not installation.start_allowed:
        detail = ", ".join(installation.reason_codes or installation.missing_features) or "untrusted identity"
        raise ValueError(
            "Pinned llama.cpp Vulkan runtime failed installation validation: "
            f"{installation.integrity_status}; {installation.currency_status}; {detail}"
        )
    root = (runtime_root or LLAMA_RUNTIME_ROOT).resolve()
    receipt_path = root / LLAMA_BUILD_RECEIPT_RELATIVE_PATH
    receipt = LlamaBuildReceipt.model_validate_json(receipt_path.read_bytes())
    return ValidatedLlamaInstallation(
        executable=root / LLAMA_SERVER_RELATIVE_PATH,
        receipt=receipt,
        executable_sha256=installation.detected.executable_sha256 or "",
        receipt_sha256=installation.detected.receipt_sha256 or "",
        version_output=installation.detected.version_output or "",
    )


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

    required_flags = (
        QWEN_MTP_LLAMA_REQUIRED_FLAGS if manifest.mtp_model is not None else QWEN_LLAMA_REQUIRED_FLAGS
    )
    installation = validate_llama_installation(required_flags=required_flags)
    if installation.receipt.commit != manifest.llama_cpp_commit:
        raise ValueError("Installed llama.cpp build does not match the Qwen runtime manifest")

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
        executable=installation.executable,
        model=model,
        projector=projector,
        mtp_model=mtp_model,
        executable_sha256=installation.executable_sha256,
        receipt_sha256=installation.receipt_sha256,
        source_revision=installation.receipt.commit,
    )


def configuration_fingerprint(runtime: ValidatedQwenRuntime, *, thinking_mode: str | None = None) -> str:
    payload = {
        "manifest": runtime.manifest.model_dump(mode="json"),
        "executable_sha256": runtime.executable_sha256,
        "receipt_sha256": runtime.receipt_sha256,
        "source_revision": runtime.source_revision,
        "thinking_mode": thinking_mode,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from modeldeck.llama_runtime import LLAMA_CPP_COMMIT, QwenLlamaManifest, TrustedArtefact, sha256_file

TRUST_DIRECTORY_NAME = "trusted-qwen-candidates"
QWEN35_REPOSITORY = re.compile(r"^bartowski/Qwen_Qwen3\.5-(?P<size>0\.8B|2B|4B|9B)-GGUF$")
REVISION = re.compile(r"^[a-f0-9]{40}$")


class QwenCandidateInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    eligible: bool
    approved: bool = False
    candidate_id: str | None = None
    filename: str | None = None
    expected_size: int | None = None
    expected_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    reason: str


def candidate_directory(data_dir: Path) -> Path:
    return data_dir / TRUST_DIRECTORY_NAME


def candidate_path(data_dir: Path, candidate_id: str) -> Path:
    if not re.fullmatch(r"qwen35-(?:0\.8b|2b|4b|9b)-q8-[a-f0-9]{12}", candidate_id):
        raise ValueError("Invalid trusted Qwen candidate identifier")
    return candidate_directory(data_dir) / f"{candidate_id}.json"


def load_candidate(data_dir: Path, candidate_id: str) -> QwenLlamaManifest:
    manifest = QwenLlamaManifest.model_validate_json(candidate_path(data_dir, candidate_id).read_bytes())
    match = QWEN35_REPOSITORY.fullmatch(manifest.artefact_model_id)
    expected_filename = f"Qwen_Qwen3.5-{match.group('size')}-Q8_0.gguf" if match is not None else None
    if (
        manifest.id != candidate_id
        or manifest.status != "approved-local"
        or match is None
        or manifest.original_model_id != f"Qwen/Qwen3.5-{match.group('size')}"
        or manifest.original_model_revision is not None
        or manifest.quantisation != "Q8_0"
        or manifest.model.filename != expected_filename
        or manifest.model.dtype != "Q8_0"
        or not manifest.id.endswith(manifest.model.sha256[:12])
        or manifest.llama_cpp_commit != LLAMA_CPP_COMMIT
        or manifest.context_length != 8192
        or manifest.projector is not None
        or manifest.mtp_model is not None
        or manifest.mtp_draft_tokens is not None
    ):
        raise ValueError("The local Qwen candidate manifest is not approved")
    return manifest


def load_candidates(data_dir: Path | None) -> dict[tuple[str, str], QwenLlamaManifest]:
    if data_dir is None or not candidate_directory(data_dir).is_dir():
        return {}
    candidates: dict[tuple[str, str], QwenLlamaManifest] = {}
    for path in sorted(candidate_directory(data_dir).glob("qwen35-*-q8-*.json")):
        try:
            manifest = load_candidate(data_dir, path.stem)
            candidates[(manifest.artefact_model_id, manifest.artefact_revision)] = manifest
        except (OSError, ValueError):
            continue
    return candidates


def _candidate_metadata(
    repo_id: str,
    revision: str,
    snapshot: Path,
    *,
    library_root: Path | None = None,
) -> tuple[str, str, int, str]:
    match = QWEN35_REPOSITORY.fullmatch(repo_id)
    if match is None:
        raise ValueError("Only bartowski Qwen3.5 GGUF repositories are eligible")
    if REVISION.fullmatch(revision) is None:
        raise ValueError("The Qwen3.5 candidate must use an immutable commit revision")
    size_label = match.group("size")
    filename = f"Qwen_Qwen3.5-{size_label}-Q8_0.gguf"
    model_path = snapshot / filename
    if not model_path.is_file():
        raise ValueError(f"The selected Q8_0 artefact is missing: {filename}")

    root = library_root or Path.home() / ".cache/huggingfacepull/library"
    marker_path = root / repo_id.replace("/", "--") / revision / ".huggingfacepull.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("The exact HuggingFacePull completion marker is unavailable") from error
    if any(
        marker.get(name) != expected
        for name, expected in (
            ("repo_id", repo_id),
            ("expected_commit", revision),
            ("revision", revision),
            ("resolved_revision", revision),
        )
    ):
        raise ValueError("The HuggingFacePull marker does not match this exact Model revision")
    files = marker.get("files")
    selected = (
        next(
            (item for item in files if isinstance(item, dict) and item.get("path") == filename),
            None,
        )
        if isinstance(files, list)
        else None
    )
    if selected is None:
        raise ValueError("HuggingFacePull did not record the selected Q8_0 artefact")
    marker_snapshot = marker.get("snapshot_path")
    if not isinstance(marker_snapshot, str) or Path(marker_snapshot).resolve() != snapshot.resolve():
        raise ValueError("The HuggingFacePull marker points to a different snapshot")

    model_dir = snapshot.parent.parent
    tree_path = model_dir / "trees" / f"{revision}.json"
    try:
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        metadata = tree["files"][filename]
        expected_size = int(metadata["lfs_size"])
        expected_sha256 = str(metadata["lfs_sha256"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("HuggingFacePull tree metadata is incomplete for the Q8_0 artefact") from error
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("HuggingFacePull recorded an invalid Q8_0 checksum")
    if selected.get("size") != expected_size or selected.get("blob_id") != metadata.get("blob_id"):
        raise ValueError("The HuggingFacePull marker and tree metadata do not identify the same artefact")
    if model_path.stat().st_size != expected_size:
        raise ValueError("The local Q8_0 artefact size does not match HuggingFacePull metadata")
    return size_label, filename, expected_size, expected_sha256


def inspect_candidate(
    repo_id: str,
    revision: str,
    snapshot: Path,
    *,
    data_dir: Path | None = None,
    library_root: Path | None = None,
) -> QwenCandidateInspection:
    try:
        _, filename, expected_size, expected_sha256 = _candidate_metadata(
            repo_id, revision, snapshot, library_root=library_root
        )
    except ValueError as error:
        return QwenCandidateInspection(eligible=False, reason=str(error))
    approved = load_candidates(data_dir).get((repo_id, revision)) if data_dir is not None else None
    return QwenCandidateInspection(
        eligible=True,
        approved=approved is not None,
        candidate_id=approved.id if approved else None,
        filename=filename,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        reason=(
            "Approved locally with an exact checksum."
            if approved
            else "Ready for explicit local approval and full SHA-256 verification."
        ),
    )


def approve_candidate(
    repo_id: str,
    revision: str,
    snapshot: Path,
    *,
    data_dir: Path,
    library_root: Path | None = None,
) -> QwenLlamaManifest:
    size_label, filename, expected_size, expected_sha256 = _candidate_metadata(
        repo_id, revision, snapshot, library_root=library_root
    )
    actual_sha256 = sha256_file(snapshot / filename)
    if actual_sha256 != expected_sha256:
        raise ValueError("The local Q8_0 artefact checksum does not match HuggingFacePull metadata")
    candidate_id = f"qwen35-{size_label.lower()}-q8-{expected_sha256[:12]}"
    manifest = QwenLlamaManifest(
        format="modeldeck-qwen-llamacpp-runtime",
        version=1,
        id=candidate_id,
        status="approved-local",
        original_model_id=f"Qwen/Qwen3.5-{size_label}",
        original_model_revision=None,
        artefact_model_id=repo_id,
        artefact_revision=revision,
        quantisation="Q8_0",
        model=TrustedArtefact(
            filename=filename,
            size=expected_size,
            sha256=expected_sha256,
            dtype="Q8_0",
        ),
        llama_cpp_commit=LLAMA_CPP_COMMIT,
        operating_system="linux",
        architecture="x86_64",
        backend="Vulkan",
        qwen_architecture="qwen35",
        chat_template_fingerprint=f"embedded-gguf-sha256:{expected_sha256}",
        context_length=8192,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        source_url=f"https://huggingface.co/{repo_id}",
        licence="Apache-2.0",
    )
    destination = candidate_path(data_dir, candidate_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return manifest

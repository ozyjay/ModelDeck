"""Local-administrator-only benchmark capture and frozen source installations.

No web input controls paths, launch arguments or environment. A registration is an
explicit local trust action, stored separately from operator-editable Worker settings.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

from modeldeck.scenechat_experiment import digest, file_digest


def root() -> Path:
    return Path.cwd() / ".modeldeck" / "scenechat-experiments"


def registration(worker_id: str) -> dict[str, Any] | None:
    try:
        UUID(worker_id)
    except ValueError:
        return None
    path = root() / "workers" / f"{worker_id}.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if (
        set(value) != {"version", "worker_id", "bundle_sha256", "image_hashes", "diagnostic_timing"}
        or value["version"] != 1
        or value["worker_id"] != worker_id
    ):
        raise ValueError("Invalid benchmark Worker registration")
    hashes = [value["bundle_sha256"], *value["image_hashes"]]
    if not hashes or any(
        not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes
    ):
        raise ValueError("Invalid benchmark evidence digest")
    if type(value["diagnostic_timing"]) is not bool:
        raise ValueError("Invalid diagnostic timing policy")
    return value


def freeze_installation(destination: Path) -> dict[str, Any]:
    """Run with the intended ROCm interpreter; copy source and hash installed dependencies."""
    source = Path(__file__).resolve().parent
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, destination / "modeldeck", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = {
        str(p.relative_to(destination)): file_digest(p)
        for p in sorted((destination / "modeldeck").rglob("*"))
        if p.is_file()
    }
    dependencies = {}
    for distribution in importlib.metadata.distributions():
        if distribution.metadata["Name"].lower() == "modeldeck":
            continue
        for entry in distribution.files or []:
            path = Path(distribution.locate_file(entry)).resolve()
            if path.is_file() and path.suffix != ".pyc":
                dependencies[str(path)] = file_digest(path)
    receipt = {
        "version": 1,
        "files": files,
        "dependencies": dependencies,
        "python": str(Path(sys.executable).absolute()),
        "python_sha256": file_digest(Path(sys.executable)),
        "python_version": sys.version,
    }
    (destination / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def verify_bundle(bundle_sha256: str, python: Path | None = None) -> Path:
    bundle = root() / "bundles" / bundle_sha256
    receipt = json.loads((bundle / "receipt.json").read_text())
    if digest(receipt) != bundle_sha256:
        raise ValueError("Frozen SceneChat receipt changed")
    if python is not None and (
        str(python.absolute()) != receipt["python"] or file_digest(python) != receipt["python_sha256"]
    ):
        raise ValueError("Frozen SceneChat interpreter changed")
    actual = {
        str(p.relative_to(bundle))
        for p in (bundle / "modeldeck").rglob("*")
        if p.is_file() and p.suffix != ".pyc"
    }
    if actual != set(receipt["files"]):
        raise ValueError("Frozen SceneChat source inventory changed")
    for name, expected in receipt["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or file_digest(path) != expected:
            raise ValueError("Frozen SceneChat implementation changed")
    for name, expected in receipt["dependencies"].items():
        if file_digest(Path(name)) != expected:
            raise ValueError("Frozen SceneChat dependency changed")
    return bundle


def launch_environment(worker_id: str, python: Path, environment: dict[str, str]) -> None:
    value = registration(worker_id)
    if value is None:
        return
    bundle = verify_bundle(value["bundle_sha256"], python)
    environment["PYTHONPATH"] = str(bundle)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"


class BenchmarkCapture:
    def __init__(self, worker_id: str, value: dict[str, Any]) -> None:
        self.worker_id = worker_id
        self.value = value

    @classmethod
    def for_worker(cls, worker_id: str) -> BenchmarkCapture | None:
        value = registration(worker_id)
        if value:
            bundle = root() / "bundles" / value["bundle_sha256"]
            if not Path(__file__).resolve().is_relative_to(bundle.resolve()):
                raise ValueError("Benchmark Worker must load its registered frozen installation")
        return cls(worker_id, value) if value else None

    def approve(self, data_url: str | None) -> str:
        if data_url is None:
            raise ValueError("Benchmark capture requires an approved corpus image")
        image_hash = hashlib.sha256(base64.b64decode(data_url.split(",", 1)[1], validate=True)).hexdigest()
        if image_hash not in self.value["image_hashes"]:
            raise ValueError("Image is not in this Worker's approved benchmark corpus")
        return image_hash

    def write(self, request_id: str, image_hash: str, text: str, diagnostics: dict[str, Any]) -> None:
        # The digest avoids using even a validated request identifier as a path.
        directory = root() / "raw" / self.worker_id
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / f"{digest(request_id)}.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                {
                    "version": 1,
                    "request_id": request_id,
                    "image_sha256": image_hash,
                    "bundle_sha256": self.value["bundle_sha256"],
                    "text": text,
                    "diagnostics": diagnostics,
                },
                stream,
                allow_nan=False,
            )
            stream.flush()
            os.fsync(stream.fileno())

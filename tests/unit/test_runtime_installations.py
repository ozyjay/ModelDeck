from __future__ import annotations

from pathlib import Path

from modeldeck import llama_runtime
from modeldeck.llama_runtime import LlamaBuildReceipt


def _installation(
    tmp_path: Path,
    *,
    flags: tuple[str, ...] = ("--model", "--device"),
    commit: str | None = None,
) -> tuple[Path, Path]:
    root = tmp_path / "llama.cpp"
    executable = root / "bin" / "llama-server"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo \'test llama-server\'; exit 0; fi\n'
        f"printf '%s\\n' '{' '.join(flags)}'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    receipt = LlamaBuildReceipt(
        format="modeldeck-llama-build",
        version=1,
        commit=commit or llama_runtime.LLAMA_CPP_COMMIT,
        executable_sha256=llama_runtime.sha256_file(executable),
        backend="Vulkan",
        operating_system="linux",
        architecture="x86_64",
    )
    receipt_path = executable.parent / "modeldeck-build.json"
    receipt_path.write_text(receipt.model_dump_json(), encoding="utf-8")
    return root, executable


def test_llama_installation_reports_exact_receipt_and_executable_identity(tmp_path: Path) -> None:
    root, executable = _installation(tmp_path)

    installation = llama_runtime.inspect_llama_installation(
        runtime_root=root, required_flags=("--model", "--device")
    )

    assert installation.integrity_status == "verified"
    assert installation.currency_status == "recommended"
    assert installation.start_allowed is True
    assert installation.detected.executable_sha256 == llama_runtime.sha256_file(executable)
    assert installation.detected.receipt_sha256 is not None
    assert installation.detected.version_output == "test llama-server"


def test_llama_installation_distinguishes_missing_receipt_and_modified_binary(tmp_path: Path) -> None:
    root, executable = _installation(tmp_path)
    receipt = root / llama_runtime.LLAMA_BUILD_RECEIPT_RELATIVE_PATH
    receipt.unlink()

    missing = llama_runtime.inspect_llama_installation(runtime_root=root, required_flags=("--model",))
    assert missing.integrity_status == "receipt-missing"
    assert missing.start_allowed is False

    _installation(tmp_path)
    executable.write_text("#!/bin/sh\necho tampered\n", encoding="utf-8")
    modified = llama_runtime.inspect_llama_installation(runtime_root=root, required_flags=("--model",))
    assert modified.integrity_status == "modified"
    assert modified.reason_codes == ("executable_checksum_mismatch",)


def test_llama_installation_blocks_missing_features_and_unaccepted_revision(tmp_path: Path) -> None:
    root, _ = _installation(tmp_path, flags=("--model",))

    feature_mismatch = llama_runtime.inspect_llama_installation(
        runtime_root=root, required_flags=("--model", "--device")
    )
    assert feature_mismatch.integrity_status == "feature-mismatch"
    assert feature_mismatch.missing_features == ("--device",)
    assert feature_mismatch.start_allowed is False

    root, _ = _installation(tmp_path, commit="a" * 40)
    unaccepted = llama_runtime.inspect_llama_installation(runtime_root=root, required_flags=("--model",))
    assert unaccepted.integrity_status == "verified"
    assert unaccepted.currency_status == "different-unqualified"
    assert unaccepted.start_allowed is False


def test_llama_installation_can_accept_an_explicit_older_revision(monkeypatch, tmp_path: Path) -> None:
    older = "b" * 40
    root, _ = _installation(tmp_path, commit=older)
    monkeypatch.setattr(llama_runtime, "LLAMA_ACCEPTED_OLDER_COMMITS", frozenset({older}))

    installation = llama_runtime.inspect_llama_installation(
        runtime_root=root, required_flags=("--model", "--device")
    )

    assert installation.integrity_status == "verified"
    assert installation.currency_status == "accepted-older"
    assert installation.start_allowed is True

# Framework Desktop ROCm evidence

## Configured target

- Fedora 44 Workstation
- AMD Radeon 8060S Graphics, `gfx1151`
- ROCm family 7.2.x and a ROCm 7.2-compatible PyTorch build
- Python 3.12 project environment
- work SSD at `/mnt/work`

## Detected on 10 July 2026

- Fedora 44 (Forty Four), kernel `7.1.3-200.fc44.x86_64`
- system Python `3.14.6`; `python3.12` available through pyenv
- 125 GiB RAM, 8 GiB swap
- `/mnt/work` mounted with roughly 870 GiB free
- Fedora RPMs included `rocm-core-7.1.1`, `rocm-runtime-7.1.1`,
  `rocminfo-7.1.0`, and related 7.1.x libraries — **not the configured 7.2.x target**
- `rocm-smi` reported device ID `0x1586`, SKU `STRXLGEN`, and `gfx1151`, but device
  authentication was restricted in the execution environment
- `/usr/lib64/libhsa-runtime64.so.1` exists
- `/dev/kfd` and `/dev/dri` were not visible inside the restricted execution environment
- no PyTorch, Transformers, Accelerate, safetensors, or Hugging Face Hub package was
  installed in the system Python used for inspection
- Docker CLI was installed; Podman CLI was installed but its runtime directory was not
  writable in the restricted inspection environment
- no target model ports or active model processes were detected

Cached repositories included Qwen 0.5B/1.5B/3B instruct variants,
`google/diffusiongemma-26B-A4B-it`, Red Hat Gemma and GPT-OSS variants, and OpenAI
GPT-OSS. Cache presence is not compatibility evidence.

## Current online support status

AMD's ROCm 7.2 Radeon/Ryzen support matrix explicitly lists `gfx1151` and the AMD Ryzen
AI Max+ 395. It lists PyTorch 2.9, ROCm 7.2, and Python 3.12 as production-supported,
with FP16 officially validated. This confirms that ROCm 7.2 supports the Framework
Desktop's processor and GPU architecture.

The OS qualification is narrower: AMD's matrix lists Ubuntu 24.04.3, while Fedora's
Fedora 44 package catalogue still provides ROCm 7.1.0. Fedora Rawhide/Fedora 45 carries
7.2 packages. Consequently, “ROCm 7.2 supports the Framework Desktop hardware” is
confirmed, but “Fedora 44's standard ROCm packages provide 7.2” is currently false.
Community reports demonstrate working Framework Desktop 7.2.1 configurations, primarily
on Ubuntu or with separately installed ROCm/TheRock components.

Sources:

- <https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html>
- <https://packages.fedoraproject.org/pkgs/rocm/rocm/>
- <https://community.frame.work/t/step-by-step-guide-ubuntu-24-04-rocm-7-2-1-llama-cpp-on-framework-desktop/81721>

## Probe and smoke policy

Run `pwsh -NoProfile -File scripts/check_environment.ps1`. The optional physical
allocation check is:

```powershell
pwsh -NoProfile -File scripts/check_environment.ps1 --allocation-test
```

ROCm PyTorch deliberately exposes devices through the `cuda` API, so
`torch.ones((2, 2), device="cuda")` can test an AMD device. It is not evidence of an
NVIDIA GPU. Normal CI never runs this allocation.

The HSA runtime candidate may be added to a **single worker's** `LD_PRELOAD` only when
the file exists, its hardware/runtime profile enables `auto`, and smoke evidence says it
is required. ModelDeck will not change the parent process environment globally.

## Known-good and known-failed configurations

An initial physical smoke passed on 10 July 2026 with:

- Fedora 44 and kernel `7.1.3-200.fc44.x86_64`;
- Radeon 8060S / `gfx1151`;
- isolated Python 3.12 `.venv-rocm72`;
- PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`, HIP `7.2.53211-e1a6bc5663`;
- Transformers `5.13.0`;
- `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- FP16, local-files-only, trusted remote code disabled;
- no `LD_PRELOAD`, no `HSA_OVERRIDE_GFX_VERSION`, and offline Hub/Transformers mode.

The four-token compatibility smoke measured a 0.478-second cached load, 0.019-second
first output, approximately 65.6 tokens/second, 1.116 GB peak torch allocation, and 1.110 GB
steady torch allocation. These short-smoke figures are evidence for this fingerprint,
not general benchmark claims. In-flight cancellation through the stable gateway passed.
A 1,808.851-second stability run completed 343 gateway requests with zero failures, then
shut down cleanly. Multiple subsequent worker loads also passed. Process exit was
confirmed after every run; full unified-memory recovery was not directly measured by a
reliable system-wide counter.

No current negative Qwen evidence has been recorded. Official hardware support still
does not replace local evidence for other models, dtypes, revisions, or package versions.

Upgrade events create new fingerprints; they do not overwrite older positive or negative
records.

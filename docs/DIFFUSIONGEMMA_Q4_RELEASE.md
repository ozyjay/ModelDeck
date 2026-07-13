# DiffusionGemma Q4 release process

This process packages the expert-only GPTQ Q4 g32 delta without duplicating its 12+ GiB
of shards. It binds the checkpoint, evaluation evidence, ModelDeck source revision,
runtime versions, model card, licence, and notices using SHA-256.

## Release prerequisites

- The 30-layer checkpoint exists at
  `var/diffusiongemma-26b-a4b-it-gptq-q4-g32` and its `q4-manifest.json` state is
  `complete`.
- The pinned base snapshot is available locally at revision
  `52de6b914ee1749a7d4933202505ddf5b414ec43`.
- `.venv-rocm72-q4` contains the validated ROCm Q4 stack.
- The comparative release gate has just passed with 8/8 Q4 and 8/8 BF16 constraint
  checks.
- `main` is checked out at the exact source revision that will be recorded in the
  bundle.

## 1. Generate the canonical evaluation report

```powershell
./scripts/evaluate_diffusiongemma_q4.ps1
```

This writes `var/q4-quality-evaluation.json` and leaves the Q4 worker ready.

## 2. Package the release bundle

```powershell
./scripts/package_diffusiongemma_q4_release.ps1
```

The command validates the checkpoint and evaluation before writing these files beside
the existing expert shards:

- `release-manifest.json` — provenance, pinned runtime, evaluation summary, and every
  payload hash;
- `SHA256SUMS` — independent checksums for every payload plus the release manifest;
- `README.md` — Hugging Face-compatible model card containing the measured release
  evidence;
- `q4-quality-evaluation.json` — publication-safe copy of the canonical evaluation
  report, retaining prompts, outputs, versions, measurements, and gates while removing
  local endpoints, paths, process IDs, and request/job IDs;
- `LICENSE` — Apache License 2.0, matching the upstream DiffusionGemma licence;
- `THIRD_PARTY_NOTICES.md` — base-model provenance and modification notice.

The 30 `experts-layer-XX.safetensors` files and `q4-manifest.json` remain in place and
are not recopied.

## 3. Verify before use or upload

```powershell
./scripts/package_diffusiongemma_q4_release.ps1 -VerifyOnly
```

Verification streams every shard through SHA-256 and rejects missing files, size or hash
mismatches, unsafe paths, an incompatible base revision, an incomplete checkpoint, failed
release gates, non-exact deterministic replay, or anything below 8/8 constraint passes
for either Q4 or BF16.

Run verification after copying the bundle to another disk and immediately before any
external upload.

## Distribution boundary

The bundle is an expert-weight delta, not a standalone model. It does not include the
base model's BF16 non-expert weights. Users must obtain
`google/diffusiongemma-26B-A4B-it` revision
`52de6b914ee1749a7d4933202505ddf5b414ec43` separately.

The upstream model page identifies DiffusionGemma as Apache-2.0. The generated bundle
includes that licence and a prominent notice that the expert weights were modified by
GPTQ quantization. The pinned upstream snapshot has no separate `NOTICE` file. Re-check
this requirement before packaging any future base revision.

## Hugging Face publication boundary

Publish the model bundle in its own Hugging Face model repository. Do not commit the
12+ GiB weight payload to ModelDeck and do not create a model-version tag in the
ModelDeck Git repository. The software, quantized artifact, and upstream base revision
have independent version identities.

The recommended first repository is
`ozyjay/diffusiongemma-26b-a4b-it-modeldeck-gptq-q4-g32`. Keeping `modeldeck` in the
name makes the custom-loader dependency explicit. The generated model card intentionally
does not declare `library_name: transformers`: this expert delta is not directly
loadable with `transformers.AutoModel` or a generic GPTQ loader.

Create the repository privately, upload only the verified checkpoint directory, and
verify a clean download before making it public:

```powershell
hf auth whoami
$RepoId = 'ozyjay/diffusiongemma-26b-a4b-it-modeldeck-gptq-q4-g32'
hf repos create $RepoId --type model --private

$Checkpoint = 'var/diffusiongemma-26b-a4b-it-gptq-q4-g32'
$Env:HF_XET_HIGH_PERFORMANCE = '1'
hf upload $RepoId $Checkpoint . `
    --commit-message 'Publish expert-only GPTQ Q4 g32 v1.0.0'

$Verification = 'var/verification/diffusiongemma-q4-v1.0.0'
hf download $RepoId --revision main --local-dir $Verification
./scripts/package_diffusiongemma_q4_release.ps1 `
    -CheckpointDir $Verification `
    -VerifyOnly
```

Run the ModelDeck smoke test against the downloaded directory. After it succeeds, create
the artifact tag on Hugging Face and then make the repository public from its settings:

```powershell
hf repos tag create $RepoId v1.0.0 --revision main `
    --message 'Initial verified ModelDeck expert-only GPTQ Q4 g32 release'
```

The checkpoint directory contains 37 upload files: 30 expert shards, the Q4 manifest,
five generated release documents/evidence files, and `SHA256SUMS`. The verifier reports
35 payload files because `release-manifest.json` and `SHA256SUMS` describe and verify the
payload rather than being counted as payload entries themselves.

Publishing remains a separate, explicit action; the packaging command performs no
network upload or repository creation.

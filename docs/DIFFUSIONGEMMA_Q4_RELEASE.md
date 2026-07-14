# DiffusionGemma Q4 release process

This process packages the self-contained GPTQ Q4 g32/BF16 hybrid without duplicating its
weight shards. It binds the checkpoint, evaluation evidence, ModelDeck source revision,
runtime versions, model card, licence, and notices using SHA-256.

## Release prerequisites

- The 30-layer checkpoint exists at
  `var/diffusiongemma-26b-a4b-it-gptq-q4-g32` and its `q4-manifest.json` state is
  `complete`.
- For first-time materialisation only, the pinned base snapshot is available locally at
  revision `52de6b914ee1749a7d4933202505ddf5b414ec43`.
- `.venv-rocm72-q4` contains the validated ROCm Q4 stack.
- The comparative release gate has just passed with 8/8 Q4 and 8/8 BF16 constraint
  checks.
- `main` is checked out at the exact source revision that will be recorded in the
  bundle.

## 1. Materialise the self-contained checkpoint

Run this once against the existing v1 expert delta:

```powershell
./scripts/materialize_diffusiongemma_q4.ps1
```

This retains the verified Q4 expert shards and adds sharded BF16 non-expert weights plus
the configuration, processor, tokenizer, and generation metadata. The resulting v2
manifest records the upstream model as provenance rather than a runtime dependency.

Prove the resulting checkpoint loads and generates with an empty Hugging Face home and
offline mode enforced:

```powershell
./scripts/smoke_diffusiongemma_q4_offline.ps1
```

## 2. Generate the canonical evaluation report

```powershell
./scripts/evaluate_diffusiongemma_q4.ps1
```

This writes `var/q4-quality-evaluation.json` and leaves the Q4 worker ready.

## 3. Package the release bundle

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

The expert and non-expert Safetensors files and `q4-manifest.json` remain in place and
are not recopied.

## 4. Verify before use or upload

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

The v2 bundle is self-contained for the ModelDeck runtime. It includes packed Q4 expert
weights, BF16 non-expert weights, and all processor/configuration files required for
offline loading. The pinned upstream identity and revision remain mandatory provenance,
but users do not obtain that checkpoint separately.

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
does not declare `library_name: transformers`: this self-contained hybrid is not directly
loadable with `transformers.AutoModel` or a generic GPTQ loader.

Upload only the verified checkpoint directory to the existing model repository, then
verify a clean download before creating the immutable release tag:

```powershell
hf auth whoami
$RepoId = 'ozyjay/diffusiongemma-26b-a4b-it-modeldeck-gptq-q4-g32'

$Checkpoint = 'var/diffusiongemma-26b-a4b-it-gptq-q4-g32'
$Env:HF_XET_HIGH_PERFORMANCE = '1'
hf upload $RepoId $Checkpoint . `
    --commit-message 'Publish self-contained GPTQ Q4 g32 v1.1.0'

$Verification = 'var/verification/diffusiongemma-q4-v1.1.0'
hf download $RepoId --revision main --local-dir $Verification
./scripts/package_diffusiongemma_q4_release.ps1 `
    -CheckpointDir $Verification `
    -VerifyOnly
```

Run the offline ModelDeck smoke test against the downloaded directory. After it succeeds,
create the artifact tag on Hugging Face:

```powershell
./scripts/smoke_diffusiongemma_q4_offline.ps1 `
    -CheckpointDir $Verification `
    -JsonOutput var/verification/q4-hub-self-contained-smoke.json

hf repos tag create $RepoId v1.1.0 --revision main `
    --message 'Verified self-contained ModelDeck GPTQ Q4 g32 release'
```

The exact file count depends on the tokenizer metadata and the number of non-expert
shards. The verifier reports the authoritative payload file count and byte total;
`release-manifest.json` and `SHA256SUMS` describe the payload rather than counting as
payload entries themselves.

Publishing remains a separate, explicit action; the packaging command performs no
network upload or repository creation.

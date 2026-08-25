param(
    [string]$CacheRoot = "/mnt/work/models/huggingface/hub",
    [string]$DataDir = ".modeldeck",
    [ValidateSet("decode", "full")]
    [string]$Stage = "full",
    [double]$CandidateTimeoutSeconds = 120,
    [double]$SecondaryTimeoutSeconds = 15
)

$ErrorActionPreference = "Stop"
$Runtime = Join-Path $PSScriptRoot "../../.venv-rocm72/bin/python"
if (-not (Test-Path -LiteralPath $Runtime -PathType Leaf)) {
    throw "ROCm 7.2 runtime is missing; run scripts/setup/setup.ps1 first."
}

& $Runtime (Join-Path $PSScriptRoot "tune_qwen38_fp8.py") `
    --cache-root $CacheRoot `
    --data-dir $DataDir `
    --stage $Stage `
    --candidate-timeout-seconds $CandidateTimeoutSeconds `
    --secondary-timeout-seconds $SecondaryTimeoutSeconds
if ($LASTEXITCODE -ne 0) {
    throw "Qwen3.8 FP8 tuning failed with exit code $LASTEXITCODE."
}

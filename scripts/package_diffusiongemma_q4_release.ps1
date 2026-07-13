[CmdletBinding()]
param(
    [string]$CheckpointDir = 'var/diffusiongemma-26b-a4b-it-gptq-q4-g32',

    [string]$EvaluationReport = 'var/q4-quality-evaluation.json',

    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$Runtime = (Resolve-Path '.venv-rocm72-q4/bin/python' -ErrorAction SilentlyContinue).Path
if (-not $Runtime) {
    throw 'Q4 runtime missing. Create .venv-rocm72-q4 and install the Q4 requirements.'
}

$Arguments = @(
    './scripts/package_diffusiongemma_q4_release.py',
    '--checkpoint-dir', $CheckpointDir
)
if ($VerifyOnly) {
    $Arguments += '--verify-only'
}
else {
    $Arguments += @('--evaluation-report', $EvaluationReport)
}

& $Runtime @Arguments
if ($LASTEXITCODE -ne 0) {
    if ($VerifyOnly) {
        throw "Q4 release verification failed: $CheckpointDir"
    }
    throw "Q4 release packaging failed. Check $CheckpointDir and $EvaluationReport."
}

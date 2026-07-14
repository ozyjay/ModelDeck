[CmdletBinding()]
param(
    [string]$CheckpointDir = 'var/diffusiongemma-26b-a4b-it-gptq-q4-g32',

    [string]$OutputDir,

    [string]$CacheRoot = '/mnt/work/models/huggingface/hub'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$Runtime = (Resolve-Path '.venv-rocm72-q4/bin/python' -ErrorAction SilentlyContinue).Path
if (-not $Runtime) {
    throw 'Q4 runtime missing. Create .venv-rocm72-q4 and install the Q4 requirements.'
}

$Arguments = @(
    './scripts/materialize_diffusiongemma_q4.py',
    '--checkpoint-dir', $CheckpointDir,
    '--cache-root', $CacheRoot
)
if ($OutputDir) {
    $Arguments += @('--output-dir', $OutputDir)
}

& $Runtime @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Q4 self-contained materialisation failed: $CheckpointDir"
}

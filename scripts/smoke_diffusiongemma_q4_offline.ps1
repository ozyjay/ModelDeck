[CmdletBinding()]
param(
    [string]$CheckpointDir = 'var/diffusiongemma-26b-a4b-it-gptq-q4-g32',

    [string]$JsonOutput = 'var/q4-self-contained-offline-smoke.json'
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$Runtime = (Resolve-Path '.venv-rocm72-q4/bin/python' -ErrorAction SilentlyContinue).Path
if (-not $Runtime) {
    throw 'Q4 runtime missing. Create .venv-rocm72-q4 and install the Q4 requirements.'
}

$OfflineHome = Join-Path (Resolve-Path 'var') 'offline-empty-hf-home'
New-Item -ItemType Directory -Path $OfflineHome -Force | Out-Null
$PreviousHfHome = [Environment]::GetEnvironmentVariable('HF_HOME', 'Process')
$PreviousHubOffline = [Environment]::GetEnvironmentVariable('HF_HUB_OFFLINE', 'Process')
$PreviousTransformersOffline = [Environment]::GetEnvironmentVariable('TRANSFORMERS_OFFLINE', 'Process')

try {
    $Env:HF_HOME = $OfflineHome
    $Env:HF_HUB_OFFLINE = '1'
    $Env:TRANSFORMERS_OFFLINE = '1'

    & $Runtime ./scripts/q4_direct_load_smoke.py `
        --cache-root $OfflineHome `
        --checkpoint-dir $CheckpointDir `
        --json-output $JsonOutput
    if ($LASTEXITCODE -ne 0) {
        throw 'The self-contained offline Q4 smoke test failed.'
    }

    $Result = Get-Content $JsonOutput -Raw | ConvertFrom-Json
    if ($Result.base_model_runtime_dependency -ne $false) {
        throw 'The smoke result still reports a base-model runtime dependency.'
    }
    if ($Result.weight_source -ne 'checkpoint') {
        throw "Unexpected Q4 weight source: $($Result.weight_source)"
    }
    Write-Host 'Self-contained offline Q4 verification passed.'
}
finally {
    [Environment]::SetEnvironmentVariable('HF_HOME', $PreviousHfHome, 'Process')
    [Environment]::SetEnvironmentVariable('HF_HUB_OFFLINE', $PreviousHubOffline, 'Process')
    [Environment]::SetEnvironmentVariable(
        'TRANSFORMERS_OFFLINE',
        $PreviousTransformersOffline,
        'Process'
    )
}

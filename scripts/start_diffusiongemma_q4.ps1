[CmdletBinding()]
param([switch]$Smoke)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$Runtime = (Resolve-Path '.venv-rocm72-q4/bin/python' -ErrorAction SilentlyContinue).Path
if (-not $Runtime) {
    throw 'Q4 runtime missing. Create .venv-rocm72-q4 and install requirements-rocm72-q4-gptqmodel.txt.'
}
$Checkpoint = 'var/diffusiongemma-26b-a4b-it-gptq-q4-g32'
$ManifestPath = Join-Path $Checkpoint 'q4-manifest.json'
if (-not (Test-Path $ManifestPath)) {
    throw "Q4 checkpoint manifest missing: $ManifestPath"
}
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
if ($Manifest.format_version -ne 2 -or $Manifest.artifact_type -ne 'self-contained') {
    throw @"
The Q4 checkpoint is still an expert-only v1 delta. Materialise the self-contained v2
artifact before starting this profile:

    ./scripts/materialize_diffusiongemma_q4.ps1
"@
}
$Env:MODELDECK_ROCM72_Q4_PYTHON = $Runtime
$ManagementUrl = 'http://127.0.0.1:3600'

try {
    Invoke-RestMethod -Uri "$ManagementUrl/api/health" -TimeoutSec 1 | Out-Null
    $Profiles = Invoke-RestMethod -Uri "$ManagementUrl/api/profiles" -TimeoutSec 2
    if ('diffusiongemma-q4-rocm' -notin @($Profiles.id)) {
        throw 'The running management service predates the Q4 profile.'
    }
}
catch {
    & (Join-Path $PSScriptRoot 'stop.ps1')
    & (Join-Path $PSScriptRoot 'run.ps1')
    Start-Sleep -Seconds 1
}

try {
    Invoke-RestMethod -Method Post `
        -Uri "$ManagementUrl/api/workers/diffusiongemma-rocm/stop" `
        -TimeoutSec 60 | Out-Null
}
catch {
    Write-Verbose 'The BF16 worker was already stopped or unavailable.'
}

Write-Host 'Starting self-contained DiffusionGemma GPTQ Q4/BF16 hybrid (group size 32)...'
$Worker = Invoke-RestMethod -Method Post `
    -Uri "$ManagementUrl/api/workers/diffusiongemma-q4-rocm/start" `
    -TimeoutSec 900

if ($Worker.state -ne 'ready') {
    throw "Q4 worker did not become ready: $($Worker.state)"
}

$Worker | ConvertTo-Json -Depth 12
Write-Host ''
Write-Host 'Q4 metrics:'
Invoke-RestMethod http://127.0.0.1:8622/metrics |
    ConvertTo-Json -Depth 12

if ($Smoke) {
    & (Join-Path $PSScriptRoot 'q4_smoke.ps1') -Model text-diffusion
}

[CmdletBinding()]
param([switch]$Smoke, [string]$Worker, [string]$RouteName)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
Import-Module (Join-Path $PSScriptRoot 'modeldeck_helpers.psm1') -Force

$Runtime = (Resolve-Path '.venv-rocm72-q4/bin/python' -ErrorAction SilentlyContinue).Path
if (-not $Runtime) {
    throw 'Q4 runtime missing. Create .venv-rocm72-q4 and install requirements-rocm72-q4-gptqmodel.txt.'
}
$Checkpoint = '/mnt/work/models/modeldeck/diffusiongemma-26b-a4b-it-gptq-q4-g32'
$ManifestPath = Join-Path $Checkpoint 'q4-manifest.json'
if (-not (Test-Path $ManifestPath)) {
    throw "Q4 checkpoint manifest missing: $ManifestPath"
}
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
if ($Manifest.format_version -ne 2 -or $Manifest.artifact_type -ne 'self-contained') {
    throw @"
The Q4 checkpoint is still an expert-only v1 delta. Materialise the self-contained v2
artefact before starting this Worker:

    ./scripts/materialize_diffusiongemma_q4.ps1
"@
}
$Env:MODELDECK_ROCM72_Q4_PYTHON = $Runtime
$ManagementUrl = 'http://127.0.0.1:3600'

try {
    Invoke-RestMethod -Uri "$ManagementUrl/api/health" -TimeoutSec 1 | Out-Null
    Invoke-RestMethod -Uri "$ManagementUrl/api/health" -TimeoutSec 2 | Out-Null
}
catch {
    & (Join-Path $PSScriptRoot 'stop.ps1')
    & (Join-Path $PSScriptRoot 'run.ps1')
    Start-Sleep -Seconds 1
}

$SelectedWorker = Resolve-ModelDeckWorker -ManagementUrl $ManagementUrl -Worker $Worker `
    -Runtime 'text-diffusion-gptq-rocm'
$OtherWorkers = @(Invoke-RestMethod -Uri "$ManagementUrl/api/workers" -TimeoutSec 10 | Where-Object {
    $_.id -ne $SelectedWorker.id -and $_.generation_family -eq 'text-diffusion' -and $_.state -ne 'stopped'
})
foreach ($OtherWorker in $OtherWorkers) {
    Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$($OtherWorker.id)/stop" `
        -TimeoutSec 60 | Out-Null
}

Write-Host 'Starting self-contained DiffusionGemma GPTQ Q4/BF16 hybrid (group size 32)...'
$Worker = Invoke-RestMethod -Method Post `
    -Uri "$ManagementUrl/api/workers/$($SelectedWorker.id)/start" `
    -TimeoutSec 900

if ($Worker.state -ne 'ready') {
    throw "Q4 worker did not become ready: $($Worker.state)"
}

$Worker | ConvertTo-Json -Depth 12
Write-Host ''
Write-Host 'Q4 metrics:'
Invoke-RestMethod "$($Worker.endpoint)/metrics" |
    ConvertTo-Json -Depth 12

if ($Smoke) {
    $Route = Resolve-ModelDeckRoute -ManagementUrl $ManagementUrl -WorkerId $SelectedWorker.id `
        -PublicName $RouteName
    & (Join-Path $PSScriptRoot 'q4_smoke.ps1') -Model $Route.public_name
}

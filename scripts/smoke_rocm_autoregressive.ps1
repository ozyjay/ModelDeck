[CmdletBinding()]
param(
    [string]$Worker,
    [string]$ModelId = 'Qwen/Qwen2.5-0.5B-Instruct'
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
Import-Module (Join-Path $PSScriptRoot 'modeldeck_helpers.psm1') -Force
if (-not (Test-Path '.venv-rocm72/bin/python')) {
    throw 'Run pwsh -NoProfile -File scripts/setup.ps1 first.'
}
$ManagementUrl = 'http://127.0.0.1:3600'
$StartedServices = $false
$WorkerStopped = $false
try {
    try {
        Invoke-RestMethod -Uri "$ManagementUrl/api/health" -TimeoutSec 1 | Out-Null
        Invoke-RestMethod -Uri "$ManagementUrl/api/health" -TimeoutSec 2 | Out-Null
    }
    catch {
        & (Join-Path $PSScriptRoot 'stop.ps1')
        & (Join-Path $PSScriptRoot 'run.ps1')
        $StartedServices = $true
        Start-Sleep -Seconds 1
    }
    $SelectedWorker = Resolve-ModelDeckWorker -ManagementUrl $ManagementUrl -Worker $Worker `
        -ModelId $ModelId -Runtime 'transformers-rocm'
    Write-Host 'Starting the pinned Qwen ROCm worker; the first local load can take several minutes.'
    $RunningWorker = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$($SelectedWorker.id)/start" -TimeoutSec 360
    if ($RunningWorker.state -ne 'ready') { throw "Worker did not become ready: $($RunningWorker.state)" }
    $Result = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$($SelectedWorker.id)/smoke" -TimeoutSec 120
    $Result | ConvertTo-Json -Depth 12
    if (-not $Result.ok) { throw 'The autoregressive ROCm compatibility smoke failed.' }
    $Stopped = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$($SelectedWorker.id)/stop" -TimeoutSec 30
    if ($Stopped.state -ne 'stopped' -or $null -ne $Stopped.pid) {
        throw 'The ROCm worker process did not report a clean stop.'
    }
    $WorkerStopped = $true
    $Lifecycle = @{
        shutdown_result = 'success'
        memory_recovery_result = 'not-measured-process-exit-confirmed'
    } | ConvertTo-Json
    Invoke-RestMethod -Method Put -Uri "$ManagementUrl/api/compatibility/tests/$($Result.test.id)/lifecycle" `
        -ContentType 'application/json' -Body $Lifecycle -TimeoutSec 10 | Out-Null
} finally {
    if (-not $WorkerStopped) {
        try {
            if ($SelectedWorker) { Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$($SelectedWorker.id)/stop" -TimeoutSec 30 |
                Out-Null
            }
        } catch { Write-Warning "Could not request worker shutdown: $_" }
    }
    if ($StartedServices) { & (Join-Path $PSScriptRoot 'stop.ps1') }
}

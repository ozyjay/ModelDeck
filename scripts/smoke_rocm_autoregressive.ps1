$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not (Test-Path '.venv-rocm72/bin/python')) {
    throw 'Run pwsh -NoProfile -File scripts/setup_rocm72.ps1 first.'
}
$ManagementUrl = 'http://127.0.0.1:3600'
$StartedServices = $false
$WorkerStopped = $false
try {
    try {
        Invoke-RestMethod -Uri "$ManagementUrl/api/health" -TimeoutSec 1 | Out-Null
        $Profiles = Invoke-RestMethod -Uri "$ManagementUrl/api/profiles" -TimeoutSec 2
        if ('qwen-small-rocm' -notin @($Profiles.id)) {
            throw 'The running ModelDeck service predates the ROCm worker profile.'
        }
    }
    catch {
        & (Join-Path $PSScriptRoot 'stop_dev.ps1')
        & (Join-Path $PSScriptRoot 'run_dev.ps1')
        $StartedServices = $true
        Start-Sleep -Seconds 1
    }
    Write-Host 'Starting the pinned Qwen ROCm worker; the first local load can take several minutes.'
    $Worker = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/qwen-small-rocm/start" -TimeoutSec 360
    if ($Worker.state -ne 'ready') { throw "Worker did not become ready: $($Worker.state)" }
    $Result = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/qwen-small-rocm/smoke" -TimeoutSec 120
    $Result | ConvertTo-Json -Depth 12
    if (-not $Result.ok) { throw 'The autoregressive ROCm compatibility smoke failed.' }
    $Stopped = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/qwen-small-rocm/stop" -TimeoutSec 30
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
            Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/qwen-small-rocm/stop" -TimeoutSec 30 |
                Out-Null
        } catch { Write-Warning "Could not request worker shutdown: $_" }
    }
    if ($StartedServices) { & (Join-Path $PSScriptRoot 'stop_dev.ps1') }
}

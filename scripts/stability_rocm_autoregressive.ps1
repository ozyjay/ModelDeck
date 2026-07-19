param(
    [double]$DurationMinutes = 30,
    [int]$IntervalSeconds = 5,
    [string]$Worker,
    [string]$RouteName
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
Import-Module (Join-Path $PSScriptRoot 'modeldeck_helpers.psm1') -Force
if (-not (Test-Path '.venv-rocm72/bin/python')) {
    throw 'Run pwsh -NoProfile -File scripts/setup.ps1 first.'
}
if ($DurationMinutes -le 0) { throw 'DurationMinutes must be greater than zero.' }
if ($IntervalSeconds -lt 1) { throw 'IntervalSeconds must be at least one.' }

$ManagementUrl = 'http://127.0.0.1:3600'
$GatewayUrl = 'http://127.0.0.1:8600'
$StartedServices = $false
$WorkerStopped = $false
$RequestCount = 0
$Failures = 0
$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
try {
    try {
        Invoke-RestMethod -Uri "$ManagementUrl/api/health" -TimeoutSec 2 | Out-Null
    } catch {
        & (Join-Path $PSScriptRoot 'stop.ps1')
        & (Join-Path $PSScriptRoot 'run.ps1')
        $StartedServices = $true
        Start-Sleep -Seconds 1
    }
    $SelectedWorker = Resolve-ModelDeckWorker -ManagementUrl $ManagementUrl -Worker $Worker `
        -ModelId 'Qwen/Qwen2.5-0.5B-Instruct' -Runtime 'transformers-rocm'
    $Route = Resolve-ModelDeckRoute -ManagementUrl $ManagementUrl -WorkerId $SelectedWorker.id `
        -PublicName $RouteName
    Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$($SelectedWorker.id)/start" -TimeoutSec 360 |
        Out-Null
    $Evidence = Invoke-RestMethod -Method Post `
        -Uri "$ManagementUrl/api/workers/$($SelectedWorker.id)/smoke" -TimeoutSec 120
    if (-not $Evidence.ok) { throw 'Initial compatibility smoke failed.' }

    $Deadline = [datetime]::UtcNow.AddMinutes($DurationMinutes)
    Write-Host "Running Qwen stability checks for $DurationMinutes minutes."
    while ([datetime]::UtcNow -lt $Deadline) {
        $Body = @{
            model = $Route.public_name
            prompt = 'Reply with a short confirmation that the local worker is ready.'
            max_tokens = 16
            temperature = 0
            seed = 7
        } | ConvertTo-Json
        try {
            $Response = Invoke-RestMethod -Method Post -Uri "$GatewayUrl/v1/completions" `
                -ContentType 'application/json' -Body $Body -TimeoutSec 30
            if (-not $Response.choices[0].text) { throw 'Gateway completion was empty.' }
            $RequestCount += 1
        } catch {
            $Failures += 1
            Write-Warning "Stability request failed: $_"
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
    $Stopwatch.Stop()
    $Stopped = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$($SelectedWorker.id)/stop" -TimeoutSec 30
    if ($Stopped.state -ne 'stopped' -or $null -ne $Stopped.pid) {
        throw 'The ROCm worker process did not report a clean stop.'
    }
    $WorkerStopped = $true
    $Lifecycle = @{
        shutdown_result = 'success'
        memory_recovery_result = 'not-measured-process-exit-confirmed'
        stability_duration_seconds = [math]::Round($Stopwatch.Elapsed.TotalSeconds, 3)
        stability_request_count = $RequestCount
        stability_failures = $Failures
    } | ConvertTo-Json
    Invoke-RestMethod -Method Put `
        -Uri "$ManagementUrl/api/compatibility/tests/$($Evidence.test.id)/lifecycle" `
        -ContentType 'application/json' -Body $Lifecycle -TimeoutSec 10 | Out-Null
    Write-Host "Stability run complete: $RequestCount requests, $Failures failures."
    if ($Failures -gt 0) { throw 'The stability run contained failed requests.' }
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

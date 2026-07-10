param(
    [double]$DurationMinutes = 30,
    [int]$IntervalSeconds = 5
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not (Test-Path '.venv-rocm72/bin/python')) {
    throw 'Run pwsh -NoProfile -File scripts/setup_rocm72.ps1 first.'
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
        $Profiles = Invoke-RestMethod -Uri "$ManagementUrl/api/profiles" -TimeoutSec 2
        if ('qwen-small-rocm' -notin @($Profiles.id)) { throw 'Stale management service.' }
    } catch {
        & (Join-Path $PSScriptRoot 'stop_dev.ps1')
        & (Join-Path $PSScriptRoot 'run_dev.ps1')
        $StartedServices = $true
        Start-Sleep -Seconds 1
    }
    Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/qwen-small-rocm/start" -TimeoutSec 360 |
        Out-Null
    $Evidence = Invoke-RestMethod -Method Post `
        -Uri "$ManagementUrl/api/workers/qwen-small-rocm/smoke" -TimeoutSec 120
    if (-not $Evidence.ok) { throw 'Initial compatibility smoke failed.' }

    $Deadline = [datetime]::UtcNow.AddMinutes($DurationMinutes)
    Write-Host "Running Qwen stability checks for $DurationMinutes minutes."
    while ([datetime]::UtcNow -lt $Deadline) {
        $Body = @{
            model = 'token-explainer'
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
    $Stopped = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/qwen-small-rocm/stop" -TimeoutSec 30
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
            Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/qwen-small-rocm/stop" -TimeoutSec 30 |
                Out-Null
        } catch { Write-Warning "Could not request worker shutdown: $_" }
    }
    if ($StartedServices) { & (Join-Path $PSScriptRoot 'stop_dev.ps1') }
}

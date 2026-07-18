[CmdletBinding()]
param(
    [double]$DurationMinutes = 30,
    [int]$IntervalSeconds = 5,
    [string]$JsonOutput,
    [string]$MarkdownOutput,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if ($DurationMinutes -le 0) { throw 'DurationMinutes must be greater than zero.' }
if ($IntervalSeconds -lt 1) { throw 'IntervalSeconds must be at least one.' }

$ManagementUrl = 'http://127.0.0.1:3600'
$GatewayUrl = 'http://127.0.0.1:8600'
$WorkerId = 'diffusiongemma-q4-rocm'
$Alias = 'text-diffusion'
$RecoveryToleranceBytes = 1GB
$RecoveryTimeoutSeconds = 30
$StartedServices = $false
$WorkerStopped = $false
$RequestCount = 0
$Failures = 0
$FailureCategories = @{}
$Latencies = [System.Collections.Generic.List[double]]::new()
$Evidence = $null
$Metrics = $null
$PeakGtt = $null
$WorkloadStopwatch = $null
$StartedAt = [datetime]::UtcNow
$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

function Get-AmdGttUsedBytes {
    foreach ($Card in Get-ChildItem '/sys/class/drm' -Directory -ErrorAction SilentlyContinue) {
        if ($Card.Name -notmatch '^card[0-9]+$') { continue }
        $Device = Join-Path $Card.FullName 'device'
        $VendorPath = Join-Path $Device 'vendor'
        $GttPath = Join-Path $Device 'mem_info_gtt_used'
        if ((Test-Path $VendorPath) -and (Test-Path $GttPath) -and
            (Get-Content $VendorPath -Raw).Trim().ToLowerInvariant() -eq '0x1002') {
            return [int64](Get-Content $GttPath -Raw).Trim()
        }
    }
    return $null
}

function Update-PeakGtt {
    $Current = Get-AmdGttUsedBytes
    if ($null -ne $Current -and ($null -eq $script:PeakGtt -or $Current -gt $script:PeakGtt)) {
        $script:PeakGtt = $Current
    }
}

function Get-NumericSummary([System.Collections.Generic.List[double]]$Values) {
    if (-not $Values.Count) {
        return @{ minimum = $null; median = $null; p95 = $null; maximum = $null }
    }
    $Sorted = @($Values | Sort-Object)
    $Middle = [math]::Floor($Sorted.Count / 2)
    $Median = if ($Sorted.Count % 2) {
        $Sorted[$Middle]
    } else {
        ($Sorted[$Middle - 1] + $Sorted[$Middle]) / 2
    }
    $P95Index = [math]::Max(0, [math]::Ceiling(0.95 * $Sorted.Count) - 1)
    return [ordered]@{
        minimum = [math]::Round($Sorted[0], 6)
        median = [math]::Round($Median, 6)
        p95 = [math]::Round($Sorted[$P95Index], 6)
        maximum = [math]::Round($Sorted[-1], 6)
    }
}

if (-not $JsonOutput) {
    $Stamp = [datetime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $JsonOutput = "var/benchmarks/diffusiongemma-stability-$Stamp.json"
}
if (-not $MarkdownOutput) { $MarkdownOutput = [IO.Path]::ChangeExtension($JsonOutput, '.md') }
if ([IO.Path]::GetFullPath($JsonOutput) -eq [IO.Path]::GetFullPath($MarkdownOutput)) {
    throw 'JSON and Markdown outputs must use different paths.'
}
if ($ValidateOnly) {
    Write-Host 'Q4 DiffusionGemma stability configuration is valid.'
    Write-Host "Profile: $WorkerId"
    Write-Host "Duration: $DurationMinutes minutes; interval: $IntervalSeconds seconds"
    Write-Host "JSON report: $JsonOutput"
    Write-Host "Markdown report: $MarkdownOutput"
    return
}
if (-not (Test-Path '.venv-rocm72/bin/python')) {
    throw 'Run pwsh -NoProfile -File scripts/setup.ps1 first.'
}

$BaselineGtt = Get-AmdGttUsedBytes
$PeakGtt = $BaselineGtt
try {
    try {
        $Profiles = Invoke-RestMethod -Uri "$ManagementUrl/api/profiles" -TimeoutSec 2
        $Profile = $Profiles | Where-Object { $_.id -eq $WorkerId }
        if (-not $Profile -or $Profile.preferred_runtime -ne 'text-diffusion-gptq-rocm') {
            throw 'The default Q4 DiffusionGemma profile is unavailable.'
        }
    } catch {
        & (Join-Path $PSScriptRoot 'stop.ps1')
        & (Join-Path $PSScriptRoot 'run.ps1')
        $StartedServices = $true
        Start-Sleep -Seconds 1
        $Profiles = Invoke-RestMethod -Uri "$ManagementUrl/api/profiles" -TimeoutSec 5
        $Profile = $Profiles | Where-Object { $_.id -eq $WorkerId }
        if (-not $Profile -or $Profile.preferred_runtime -ne 'text-diffusion-gptq-rocm') {
            throw 'The default Q4 DiffusionGemma profile is unavailable.'
        }
    }
    $ActiveWorkers = @(
        Invoke-RestMethod -Uri "$ManagementUrl/api/workers" -TimeoutSec 5 |
            Where-Object { $_.state -ne 'stopped' }
    )
    if ($ActiveWorkers.Count) {
        throw 'Stop all managed workers before running the Q4 DiffusionGemma stability gate.'
    }

    Write-Host 'Starting Q4 DiffusionGemma for the sustained stability run.'
    $Worker = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$WorkerId/start" `
        -TimeoutSec 900
    if ($Worker.state -ne 'ready') { throw "Worker did not become ready: $($Worker.state)" }
    Update-PeakGtt
    $Evidence = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$WorkerId/smoke" `
        -TimeoutSec 600
    if (-not $Evidence.ok) { throw 'Initial Q4 DiffusionGemma compatibility smoke failed.' }
    $Models = Invoke-RestMethod -Uri "$GatewayUrl/v1/models" -TimeoutSec 10
    $GatewayModel = $Models.data | Where-Object { $_.id -eq $Alias }
    if (-not $GatewayModel -or $GatewayModel.effective_provider -ne $WorkerId) {
        throw 'The stable gateway did not select the default Q4 DiffusionGemma provider.'
    }

    $Deadline = [datetime]::UtcNow.AddMinutes($DurationMinutes)
    $WorkloadStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host "Running Q4 DiffusionGemma stability checks for $DurationMinutes minutes."
    while ([datetime]::UtcNow -lt $Deadline) {
        $Body = @{
            model = $Alias
            prompt = 'Explain why reliable local software should be tested before a demonstration.'
            max_length = 128
            block_length = 128
            denoising_steps = 24
            temperature = 0.8
            seed = 11
            stream_intermediate_frames = $false
        } | ConvertTo-Json
        $RequestTimer = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $Headers = $null
            $Queued = Invoke-RestMethod -Method Post -Uri "$GatewayUrl/v1/diffuse" `
                -ContentType 'application/json' -Body $Body -TimeoutSec 30 `
                -ResponseHeadersVariable Headers
            if ([string]$Headers['x-modeldeck-provider'] -ne $WorkerId) {
                throw 'Gateway selected an unexpected provider.'
            }
            $JobId = [string]$Queued.job_id
            if (-not $JobId) { throw 'Gateway did not return a diffusion job identifier.' }
            $JobDeadline = [datetime]::UtcNow.AddSeconds(900)
            do {
                Start-Sleep -Milliseconds 250
                Update-PeakGtt
                $Job = Invoke-RestMethod -Uri "$GatewayUrl/v1/jobs/$JobId" -TimeoutSec 10
                if ([datetime]::UtcNow -ge $JobDeadline) {
                    try {
                        Invoke-RestMethod -Method Post -Uri "$GatewayUrl/v1/jobs/$JobId/cancel" `
                            -TimeoutSec 10 | Out-Null
                    } catch {
                        Write-Warning 'Could not cancel the timed-out diffusion job.'
                    }
                    throw 'Diffusion stability request exceeded 900 seconds.'
                }
            } while ($Job.state -notin @('complete', 'failed', 'cancelled'))
            $RequestTimer.Stop()
            if ($Job.state -ne 'complete' -or -not ([string]$Job.text).Trim()) {
                throw "Diffusion stability request ended in state $($Job.state)."
            }
            $Latencies.Add($RequestTimer.Elapsed.TotalSeconds)
            $RequestCount += 1
        } catch {
            $RequestTimer.Stop()
            $Failures += 1
            $Category = $_.Exception.GetType().Name
            $FailureCategories[$Category] = 1 + [int]$FailureCategories[$Category]
            Write-Warning "Stability request failed with $Category."
        }
        try {
            $Metrics = Invoke-RestMethod -Uri "$($Worker.endpoint)/metrics" -TimeoutSec 5
            Update-PeakGtt
        } catch {
            Write-Warning 'Could not sample worker metrics.'
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
    $WorkloadStopwatch.Stop()

    $Stopped = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$WorkerId/stop" `
        -TimeoutSec 60
    if ($Stopped.state -ne 'stopped' -or $null -ne $Stopped.pid) {
        throw 'The Q4 DiffusionGemma worker process did not report a clean stop.'
    }
    $WorkerStopped = $true

    $RecoveryTimer = [System.Diagnostics.Stopwatch]::StartNew()
    $AfterStopGtt = Get-AmdGttUsedBytes
    while ($null -ne $BaselineGtt -and $null -ne $AfterStopGtt -and
        $AfterStopGtt -gt ($BaselineGtt + $RecoveryToleranceBytes) -and
        $RecoveryTimer.Elapsed.TotalSeconds -lt $RecoveryTimeoutSeconds) {
        Start-Sleep -Seconds 1
        $AfterStopGtt = Get-AmdGttUsedBytes
    }
    $RecoveryTimer.Stop()
    $Recovered = $null -ne $BaselineGtt -and $null -ne $AfterStopGtt -and
        $AfterStopGtt -le ($BaselineGtt + $RecoveryToleranceBytes)
    $MemoryRecovery = if ($null -eq $BaselineGtt -or $null -eq $AfterStopGtt) {
        'not-measured-process-exit-confirmed'
    } elseif ($Recovered) {
        'measured-recovered'
    } else {
        'measured-not-recovered'
    }
    $Lifecycle = @{
        shutdown_result = 'success'
        memory_recovery_result = $MemoryRecovery
        stability_duration_seconds = [math]::Round($WorkloadStopwatch.Elapsed.TotalSeconds, 3)
        stability_request_count = $RequestCount
        stability_failures = $Failures
    } | ConvertTo-Json
    Invoke-RestMethod -Method Put `
        -Uri "$ManagementUrl/api/compatibility/tests/$($Evidence.test.id)/lifecycle" `
        -ContentType 'application/json' -Body $Lifecycle -TimeoutSec 10 | Out-Null

    $Stopwatch.Stop()
    $Report = [ordered]@{
        format = 'modeldeck-diffusiongemma-stability'
        format_version = 1
        status = if ($RequestCount -gt 0 -and $Failures -eq 0 -and
            $MemoryRecovery -ne 'measured-not-recovered') {
            'completed'
        } else {
            'completed-with-failures'
        }
        started_at = $StartedAt.ToString('o')
        completed_at = [datetime]::UtcNow.ToString('o')
        requested_duration_minutes = $DurationMinutes
        duration_seconds = [math]::Round($WorkloadStopwatch.Elapsed.TotalSeconds, 3)
        total_seconds = [math]::Round($Stopwatch.Elapsed.TotalSeconds, 3)
        interval_seconds = $IntervalSeconds
        profile_id = $WorkerId
        model_id = $Profile.model_id
        model_revision = $Profile.revision
        runtime = $Profile.preferred_runtime
        compatibility_test_id = $Evidence.test.id
        workload = [ordered]@{
            max_length = 128
            block_length = 128
            denoising_steps = 24
            temperature = 0.8
            seed = 11
        }
        requests = [ordered]@{
            successful = $RequestCount
            failed = $Failures
            failure_categories = $FailureCategories
            wall_seconds = Get-NumericSummary $Latencies
        }
        memory = [ordered]@{
            baseline_system_gtt_used_bytes = $BaselineGtt
            peak_system_gtt_used_bytes = $PeakGtt
            after_stop_system_gtt_used_bytes = $AfterStopGtt
            recovery_tolerance_bytes = $RecoveryToleranceBytes
            recovery_wait_seconds = [math]::Round($RecoveryTimer.Elapsed.TotalSeconds, 3)
            recovery_result = $MemoryRecovery
            worker_peak_memory_allocated_bytes = $Metrics.peak_memory_allocated_bytes
            worker_peak_memory_reserved_bytes = $Metrics.peak_memory_reserved_bytes
        }
        process_exit_confirmed = $true
    }
    $OutputDirectory = Split-Path $JsonOutput -Parent
    if ($OutputDirectory) { New-Item -ItemType Directory -Force $OutputDirectory | Out-Null }
    $MarkdownDirectory = Split-Path $MarkdownOutput -Parent
    if ($MarkdownDirectory) { New-Item -ItemType Directory -Force $MarkdownDirectory | Out-Null }
    $Report | ConvertTo-Json -Depth 10 | Set-Content $JsonOutput -Encoding utf8
    $Latency = $Report.requests.wall_seconds
    $PeakGttGiB = if ($null -eq $Report.memory.peak_system_gtt_used_bytes) {
        'not measured'
    } else {
        "$([math]::Round($Report.memory.peak_system_gtt_used_bytes / 1GB, 4)) GiB"
    }
    @(
        '# Q4 DiffusionGemma stability report'
        ''
        "- Status: ``$($Report.status)``"
        "- Duration: $($Report.duration_seconds) seconds"
        "- Requests: $RequestCount successful, $Failures failed"
        "- Median request: $($Latency.median) seconds"
        "- p95 request: $($Latency.p95) seconds"
        "- Peak system GTT: $PeakGttGiB"
        "- Memory recovery: ``$MemoryRecovery``"
        '- Process exit: confirmed'
        ''
        'Prompts and generated content are not retained.'
        ''
    ) | Set-Content $MarkdownOutput -Encoding utf8
    Write-Host "Stability run complete: $RequestCount requests, $Failures failures."
    Write-Host "JSON report: $JsonOutput"
    Write-Host "Markdown report: $MarkdownOutput"
    if ($Report.status -ne 'completed') { throw 'The Q4 DiffusionGemma stability run did not pass.' }
} finally {
    if (-not $WorkerStopped) {
        try {
            Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$WorkerId/stop" `
                -TimeoutSec 60 | Out-Null
        } catch { Write-Warning 'Could not request Q4 DiffusionGemma worker shutdown.' }
    }
    if ($StartedServices) { & (Join-Path $PSScriptRoot 'stop.ps1') }
}

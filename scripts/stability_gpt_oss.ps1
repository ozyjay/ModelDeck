[CmdletBinding()]
param(
    [double]$DurationMinutes = 30,
    [int]$IntervalSeconds = 5,
    [Parameter(Mandatory)][string]$Worker,
    [Parameter(Mandatory)][string]$RouteName,
    [string]$JsonOutput,
    [string]$MarkdownOutput
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
Import-Module (Join-Path $PSScriptRoot 'modeldeck_helpers.psm1') -Force
if (-not (Test-Path '.venv/bin/python')) {
    throw 'Run pwsh -NoProfile -File scripts/setup.ps1 first.'
}
if ($DurationMinutes -le 0) { throw 'DurationMinutes must be greater than zero.' }
if ($IntervalSeconds -lt 1) { throw 'IntervalSeconds must be at least one.' }

$ManagementUrl = 'http://127.0.0.1:3600'
$GatewayUrl = 'http://127.0.0.1:8600'
$RecoveryToleranceBytes = 1GB
$StartedServices = $false
$WorkerStopped = $false
$RequestCount = 0
$Failures = 0
$FailureCategories = @{}
$Latencies = [System.Collections.Generic.List[double]]::new()
$Evidence = $null
$Metrics = $null
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
    $JsonOutput = "var/benchmarks/gpt-oss-stability-$Stamp.json"
}
if (-not $MarkdownOutput) { $MarkdownOutput = [IO.Path]::ChangeExtension($JsonOutput, '.md') }
if ([IO.Path]::GetFullPath($JsonOutput) -eq [IO.Path]::GetFullPath($MarkdownOutput)) {
    throw 'JSON and Markdown outputs must use different paths.'
}

$BaselineGtt = Get-AmdGttUsedBytes
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
        -Runtime 'llama-vulkan'
    $Route = Resolve-ModelDeckRoute -ManagementUrl $ManagementUrl -WorkerId $SelectedWorker.id `
        -PublicName $RouteName
    $WorkerId = $SelectedWorker.id

    Write-Host 'Starting GPT-OSS for the sustained stability run.'
    $Worker = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$WorkerId/start" `
        -TimeoutSec 700
    if ($Worker.state -ne 'ready') { throw "Worker did not become ready: $($Worker.state)" }
    $Evidence = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$WorkerId/smoke" `
        -TimeoutSec 120
    if (-not $Evidence.ok) { throw 'Initial GPT-OSS compatibility smoke failed.' }
    $Deadline = [datetime]::UtcNow.AddMinutes($DurationMinutes)
    $WorkloadStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host "Running GPT-OSS stability checks for $DurationMinutes minutes."
    while ([datetime]::UtcNow -lt $Deadline) {
        $Body = @{
            model = $Route.public_name
            messages = @(@{
                role = 'user'
                content = 'Reply with a short confirmation that the local worker is ready.'
            })
            max_tokens = 128
            temperature = 0
            seed = 7
            stream = $false
        } | ConvertTo-Json -Depth 6
        $RequestTimer = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $Headers = $null
            $Response = Invoke-RestMethod -Method Post -Uri "$GatewayUrl/v1/chat/completions" `
                -ContentType 'application/json' -Body $Body -TimeoutSec 120 `
                -ResponseHeadersVariable Headers
            $RequestTimer.Stop()
            if (-not ([string]$Response.choices[0].message.content).Trim()) {
                throw 'Gateway completion was empty.'
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
        $Metrics = Invoke-RestMethod -Uri "$($Worker.endpoint)/metrics" -TimeoutSec 5
        Start-Sleep -Seconds $IntervalSeconds
    }
    $WorkloadStopwatch.Stop()

    $Stopwatch.Stop()
    $Stopped = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$WorkerId/stop" `
        -TimeoutSec 60
    if ($Stopped.state -ne 'stopped' -or $null -ne $Stopped.pid) {
        throw 'The GPT-OSS worker process did not report a clean stop.'
    }
    $WorkerStopped = $true
    Start-Sleep -Seconds 2
    $AfterStopGtt = Get-AmdGttUsedBytes
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

    $Report = [ordered]@{
        format = 'modeldeck-gpt-oss-stability'
        format_version = 1
        status = if ($Failures -eq 0 -and $MemoryRecovery -ne 'measured-not-recovered') {
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
        worker_id = $WorkerId
        worker_name = $SelectedWorker.name
        route_name = $Route.public_name
        model_id = $SelectedWorker.model_id
        model_revision = $SelectedWorker.revision
        runtime = $SelectedWorker.runtime
        compatibility_test_id = $Evidence.test.id
        requests = [ordered]@{
            successful = $RequestCount
            failed = $Failures
            failure_categories = $FailureCategories
            wall_seconds = Get-NumericSummary $Latencies
        }
        memory = [ordered]@{
            baseline_system_gtt_used_bytes = $BaselineGtt
            peak_system_gtt_used_bytes = $Metrics.system_gtt_peak_used_bytes
            after_stop_system_gtt_used_bytes = $AfterStopGtt
            recovery_tolerance_bytes = $RecoveryToleranceBytes
            recovery_result = $MemoryRecovery
        }
        process_exit_confirmed = $true
    }
    New-Item -ItemType Directory -Force (Split-Path $JsonOutput -Parent) | Out-Null
    $Report | ConvertTo-Json -Depth 10 | Set-Content $JsonOutput -Encoding utf8
    $Latency = $Report.requests.wall_seconds
    @(
        '# GPT-OSS stability report'
        ''
        "- Status: ``$($Report.status)``"
        "- Duration: $($Report.duration_seconds) seconds"
        "- Requests: $RequestCount successful, $Failures failed"
        "- Median request: $($Latency.median) seconds"
        "- p95 request: $($Latency.p95) seconds"
        "- Peak system GTT: $([math]::Round($Report.memory.peak_system_gtt_used_bytes / 1GB, 4)) GiB"
        "- Memory recovery: ``$MemoryRecovery``"
        "- Process exit: confirmed"
        ''
        'Prompts and generated content are not retained.'
        ''
    ) | Set-Content $MarkdownOutput -Encoding utf8
    Write-Host "Stability run complete: $RequestCount requests, $Failures failures."
    Write-Host "JSON report: $JsonOutput"
    Write-Host "Markdown report: $MarkdownOutput"
    if ($Report.status -ne 'completed') { throw 'The GPT-OSS stability run did not pass.' }
} finally {
    if (-not $WorkerStopped) {
        try {
            Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$WorkerId/stop" `
                -TimeoutSec 60 | Out-Null
        } catch { Write-Warning 'Could not request GPT-OSS worker shutdown.' }
    }
    if ($StartedServices) { & (Join-Path $PSScriptRoot 'stop.ps1') }
}

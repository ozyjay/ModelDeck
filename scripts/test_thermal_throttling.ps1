[CmdletBinding()]
param(
    [ValidateSet('NoProtection', 'HostOnly', 'ModelDeckOnly', 'Combined')]
    [string]$Condition = 'Combined',
    [string]$ManagementUrl = 'http://127.0.0.1:3600',
    [string]$GatewayUrl = 'http://127.0.0.1:8600',
    [string]$Model,
    [string]$Prompt = 'Explain local model thermal throttling in two short sentences.',
    [ValidateRange(1, 3600)]
    [int]$DurationSeconds = 60,
    [ValidateRange(1, 60)]
    [int]$RequestIntervalSeconds = 5,
    [switch]$RunControlledWorkload,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not $OutputPath) {
    $Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    $OutputPath = "var/benchmarks/thermal-throttling-$($Condition.ToLowerInvariant())-$Stamp.json"
}
if ($RunControlledWorkload -and -not $Model) {
    throw 'Supply -Model when -RunControlledWorkload is selected.'
}

$Initial = Invoke-RestMethod -Uri "$($ManagementUrl.TrimEnd('/'))/api/thermal" -TimeoutSec 5
if (-not $Initial.enabled -and $Condition -in @('ModelDeckOnly', 'Combined')) {
    throw 'ModelDeck thermal throttling must be enabled for this validation condition.'
}
Write-Host "Condition: $Condition"
Write-Host 'This script observes external host policy only; it never changes TuneD or systemd state.'

$Samples = [System.Collections.Generic.List[object]]::new()
$Requests = [System.Collections.Generic.List[object]]::new()
$Deadline = [DateTimeOffset]::UtcNow.AddSeconds($DurationSeconds)
while ([DateTimeOffset]::UtcNow -lt $Deadline) {
    $Status = Invoke-RestMethod -Uri "$($ManagementUrl.TrimEnd('/'))/api/thermal" -TimeoutSec 5
    $Samples.Add([pscustomobject]@{
        timestamp = [DateTimeOffset]::UtcNow.ToString('o')
        state = $Status.state
        temperature_c = $Status.temperature_c
        telemetry_age_seconds = $Status.telemetry_age_seconds
        heavy_concurrency_limit = $Status.heavy_concurrency_limit
        background_paused = $Status.background_paused
        model_loading_allowed = $Status.model_loading_allowed
        reason_code = $Status.reason_code
        host_power_policy = $Status.host_power_policy
    })
    Write-Host ("{0:u} {1,-20} {2,5}°C heavy={3} reason={4}" -f `
        (Get-Date), $Status.state, $Status.temperature_c, $Status.heavy_concurrency_limit, $Status.reason_code)

    if ($Status.state -eq 'telemetry_degraded') {
        Write-Warning 'Fresh thermal telemetry disappeared; stopping the controlled test safely.'
        break
    }
    if ($Status.state -eq 'critical') {
        Write-Warning 'The critical thermal state is active; no further workload will be submitted.'
        break
    }
    if ($RunControlledWorkload) {
        $Body = @{
            model = $Model
            messages = @(@{ role = 'user'; content = $Prompt })
            max_tokens = 64
            temperature = 0
            stream = $false
            request_id = [guid]::NewGuid().ToString()
        } | ConvertTo-Json -Depth 6 -Compress
        $Started = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $Response = Invoke-WebRequest -Method Post -Uri "$($GatewayUrl.TrimEnd('/'))/v1/chat/completions" `
                -ContentType 'application/json' -Body $Body -TimeoutSec 180 -SkipHttpErrorCheck
            $Requests.Add([pscustomobject]@{
                timestamp = [DateTimeOffset]::UtcNow.ToString('o')
                status_code = [int]$Response.StatusCode
                latency_seconds = [Math]::Round($Started.Elapsed.TotalSeconds, 4)
                thermal_state = $Response.Headers['x-modeldeck-thermal-state']
                thermal_reason = $Response.Headers['x-modeldeck-thermal-reason']
            })
        }
        catch {
            $Requests.Add([pscustomobject]@{
                timestamp = [DateTimeOffset]::UtcNow.ToString('o')
                status_code = $null
                latency_seconds = [Math]::Round($Started.Elapsed.TotalSeconds, 4)
                error = $_.Exception.GetType().Name
            })
        }
    }
    Start-Sleep -Seconds $RequestIntervalSeconds
}

$Temperatures = @($Samples | Where-Object { $null -ne $_.temperature_c } | ForEach-Object temperature_c)
$Report = [ordered]@{
    format = 'modeldeck-thermal-validation'
    version = 1
    condition = $Condition
    started_with = $Initial
    configuration_note = 'Host policy state is externally configured and read-only to ModelDeck.'
    samples = $Samples
    requests = $Requests
    summary = [ordered]@{
        sample_count = $Samples.Count
        request_count = $Requests.Count
        peak_temperature_c = if ($Temperatures.Count) { ($Temperatures | Measure-Object -Maximum).Maximum } else { $null }
        mean_temperature_c = if ($Temperatures.Count) { [Math]::Round(($Temperatures | Measure-Object -Average).Average, 3) } else { $null }
        time_at_or_above_75_seconds = @($Samples | Where-Object { $_.temperature_c -ge 75 }).Count * $RequestIntervalSeconds
        time_at_or_above_80_seconds = @($Samples | Where-Object { $_.temperature_c -ge 80 }).Count * $RequestIntervalSeconds
        time_at_or_above_83_seconds = @($Samples | Where-Object { $_.temperature_c -ge 83 }).Count * $RequestIntervalSeconds
        time_at_or_above_85_seconds = @($Samples | Where-Object { $_.temperature_c -ge 85 }).Count * $RequestIntervalSeconds
    }
}
$Parent = Split-Path -Parent $OutputPath
if ($Parent) { New-Item -ItemType Directory -Force -Path $Parent | Out-Null }
$Report | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 $OutputPath
Write-Host "Thermal validation report: $OutputPath"

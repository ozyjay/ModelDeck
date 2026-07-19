param(
    [int]$MaximumTokens = 128,
    [string]$Worker,
    [string]$RouteName
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
Import-Module (Join-Path $PSScriptRoot 'modeldeck_helpers.psm1') -Force
if (-not (Test-Path '.venv-rocm72/bin/python')) {
    throw 'Run pwsh -NoProfile -File scripts/setup.ps1 first.'
}
$ManagementUrl = 'http://127.0.0.1:3600'
$GatewayUrl = 'http://127.0.0.1:8600'
$StartedServices = $false
$WorkerStopped = $false
$StreamJob = $null
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
    $RequestId = [guid]::NewGuid().ToString()
    $Body = @{
        request_id = $RequestId
        model = $Route.public_name
        prompt = 'Continue producing short numbered test tokens until the request is cancelled.'
        stream = $true
        max_tokens = $MaximumTokens
        min_tokens = $MaximumTokens
        temperature = 0.8
        seed = 19
    } | ConvertTo-Json
    $StreamJob = Start-ThreadJob -ScriptBlock {
        param($Url, $Json)
        Invoke-WebRequest -Method Post -Uri "$Url/native/autoregressive/trace" `
            -ContentType 'application/json' -Body $Json -TimeoutSec 60
    } -ArgumentList $GatewayUrl, $Body

    $Cancelled = $null
    foreach ($Attempt in 1..40) {
        Start-Sleep -Milliseconds 50
        $Cancelled = Invoke-RestMethod -Method Post `
            -Uri "$GatewayUrl/v1/requests/$RequestId/cancel" -TimeoutSec 2
        if ($Cancelled.ok) { break }
        if ($StreamJob.State -ne 'Running') { break }
    }
    $Response = Receive-Job -Job $StreamJob -Wait -AutoRemoveJob
    $StreamJob = $null
    if (-not $Cancelled.ok) { throw 'The gateway did not find the in-flight request to cancel.' }
    if ($Response.Content -notmatch 'event: cancelled') {
        throw 'The stream did not report cancellation.'
    }
    Write-Host "Cancellation passed for request $RequestId via $($Cancelled.workers -join ', ')."
    $Stopped = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$($SelectedWorker.id)/stop" -TimeoutSec 30
    if ($Stopped.state -ne 'stopped') { throw 'Worker did not stop after cancellation smoke.' }
    $WorkerStopped = $true
} finally {
    if ($StreamJob) { Stop-Job $StreamJob -ErrorAction SilentlyContinue; Remove-Job $StreamJob -Force }
    if (-not $WorkerStopped) {
        try {
            if ($SelectedWorker) { Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$($SelectedWorker.id)/stop" -TimeoutSec 30 |
                Out-Null
            }
        } catch { Write-Warning "Could not request worker shutdown: $_" }
    }
    if ($StartedServices) { & (Join-Path $PSScriptRoot 'stop.ps1') }
}

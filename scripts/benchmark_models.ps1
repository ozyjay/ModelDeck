[CmdletBinding()]
param(
    [ValidateSet('Quick', 'Standard')]
    [string]$Preset = 'Standard',

    [string[]]$Workers = @(),

    [string]$JsonOutput,
    [string]$MarkdownOutput
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

if (-not (Test-Path '.venv/bin/python')) {
    throw 'Run pwsh -NoProfile -File scripts/setup.ps1 first.'
}

$ManagementUrl = 'http://127.0.0.1:3600'
$StartedServices = $false
$BenchmarkExitCode = 0

try {
    $ManagementAvailable = $false
    try {
        Invoke-RestMethod -Uri "$ManagementUrl/api/health" -TimeoutSec 1 | Out-Null
        $ManagementAvailable = $true
    }
    catch { }

    if ($ManagementAvailable) {
        try {
            Invoke-RestMethod -Uri 'http://127.0.0.1:8600/v1/health' -TimeoutSec 1 | Out-Null
        }
        catch {
            throw 'ModelDeck management is running but the gateway is unavailable. Run scripts/stop.ps1, then retry.'
        }
    }
    else {
        Write-Host 'ModelDeck is not running; starting local services for the benchmark.'
        $StartedServices = $true
        & (Join-Path $PSScriptRoot 'run.ps1')

        $Deadline = [datetime]::UtcNow.AddSeconds(30)
        do {
            try {
                Invoke-RestMethod -Uri "$ManagementUrl/api/health" -TimeoutSec 1 | Out-Null
                Invoke-RestMethod -Uri 'http://127.0.0.1:8600/v1/health' -TimeoutSec 1 | Out-Null
                $Ready = $true
            }
            catch {
                $Ready = $false
                Start-Sleep -Milliseconds 200
            }
        } while (-not $Ready -and [datetime]::UtcNow -lt $Deadline)
        if (-not $Ready) { throw 'ModelDeck services did not become ready within 30 seconds.' }
    }

    if (-not $Workers.Count) {
        $ConfiguredWorkers = @(Invoke-RestMethod -Uri "$ManagementUrl/api/workers" -TimeoutSec 10)
        $Workers = @($ConfiguredWorkers | Where-Object { $_.runtime -ne 'mock' } | ForEach-Object { $_.id })
    }
    if (-not $Workers.Count) {
        throw 'Create at least one physical Worker, or select Workers with -Workers.'
    }
    $Arguments = @(
        './scripts/benchmark_models.py',
        '--preset', $Preset.ToLowerInvariant(),
        '--workers'
    ) + $Workers
    if ($JsonOutput) { $Arguments += @('--json-output', $JsonOutput) }
    if ($MarkdownOutput) { $Arguments += @('--markdown-output', $MarkdownOutput) }

    & .venv/bin/python @Arguments
    $BenchmarkExitCode = $LASTEXITCODE
}
finally {
    if ($StartedServices) {
        & (Join-Path $PSScriptRoot 'stop.ps1')
    }
}

if ($BenchmarkExitCode -ne 0) {
    throw "The benchmark completed with exit code $BenchmarkExitCode. Inspect the generated report."
}

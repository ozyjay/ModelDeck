$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '../..')

function Stop-ModelDeckProcess {
    param(
        [int]$ProcessId,
        [string]$Name,
        [switch]$Recovered
    )

    $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $Process) { return $false }
    $Prefix = if ($Recovered) { 'recovered untracked ' } else { '' }
    Write-Host "  $Name`: ${Prefix}stopping process $ProcessId…"
    Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue
    try { Wait-Process -Id $ProcessId -Timeout 10 -ErrorAction Stop }
    catch {
        Write-Warning "$Name did not stop gracefully; forcing process $ProcessId to exit."
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $ProcessId -Timeout 5 -ErrorAction SilentlyContinue
        $script:ForcedServices++
    }
    Write-Host "  $Name`: stopped."
    $script:StoppedServices++
    return $true
}

function Find-ModelDeckProcessIds {
    param(
        [string]$Module,
        [string]$LegacyExecutable
    )

    $Found = @()
    $PythonPath = (Join-Path (Get-Location) '.venv/bin/python')
    foreach ($ProcessDirectory in Get-ChildItem /proc -Directory -ErrorAction SilentlyContinue) {
        if ($ProcessDirectory.Name -notmatch '^\d+$') { continue }
        try {
            $Arguments = (Get-Content "$($ProcessDirectory.FullName)/cmdline" -Raw -ErrorAction Stop) `
                -split [char]0
        }
        catch { continue }
        $IsCurrentLaunch = $Arguments -contains $PythonPath -and $Arguments -contains '-m' -and $Arguments -contains $Module
        if ($IsCurrentLaunch -or $Arguments -contains $LegacyExecutable) {
            $Found += [int]$ProcessDirectory.Name
        }
    }
    return $Found
}

Write-Host '[1/4] Requesting graceful Worker shutdown…'
try {
    Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:3600/api/workers/stop-all' -TimeoutSec 15 |
        Out-Null
    Write-Host '  Workers: graceful shutdown accepted.'
}
catch { Write-Host '  Workers: management unavailable; continuing with process shutdown.' }

Write-Host '[2/4] Stopping ModelDeck services…'
$StoppedServices = 0
$AbsentServices = 0
$ForcedServices = 0
$StoppedProcessIds = @{}
$ServiceDefinitions = @{
    'management' = [pscustomobject]@{
        Module = 'modeldeck'
        LegacyExecutable = (Join-Path (Get-Location) '.venv/bin/modeldeck')
    }
    'gateway' = [pscustomobject]@{
        Module = 'modeldeck.gateway.app'
        LegacyExecutable = (Join-Path (Get-Location) '.venv/bin/modeldeck-gateway')
    }
    'gateway-docker-bridge' = (Join-Path (Get-Location) '.venv/bin/modeldeck-gateway')
}
foreach ($Name in @('gateway-docker-bridge', 'gateway', 'gateway-loopback', 'management')) {
    $Path = "var/run/$Name.pid"
    if (-not (Test-Path $Path)) {
        Write-Host "  $Name`: not running (no PID file)."
        $AbsentServices++
        continue
    }
    $ProcessId = 0
    if (-not [int]::TryParse((Get-Content $Path -Raw).Trim(), [ref]$ProcessId)) {
        Write-Warning "$Name has an invalid PID file; removing it."
        Remove-Item $Path -ErrorAction SilentlyContinue
        $AbsentServices++
        continue
    }
    if (Stop-ModelDeckProcess -ProcessId $ProcessId -Name $Name) {
        $StoppedProcessIds[$ProcessId] = $true
    } else {
        Write-Host "  $Name`: process $ProcessId has already exited."
        $AbsentServices++
    }
    Remove-Item $Path -ErrorAction SilentlyContinue
}

foreach ($Name in @('gateway', 'management')) {
    $Definition = $ServiceDefinitions[$Name]
    foreach ($ProcessId in Find-ModelDeckProcessIds -Module $Definition.Module -LegacyExecutable $Definition.LegacyExecutable) {
        if ($StoppedProcessIds.ContainsKey($ProcessId)) { continue }
        if (Stop-ModelDeckProcess -ProcessId $ProcessId -Name $Name -Recovered) {
            $StoppedProcessIds[$ProcessId] = $true
        }
    }
}

Write-Host '[3/4] Checking for stale ModelDeck Workers…'
& (Join-Path $PSScriptRoot 'stop_stale_workers.ps1') -Quiet
Write-Host '  Stale Worker check complete.'
Write-Host "[4/4] ModelDeck stopped: $StoppedServices service(s) stopped, $AbsentServices already absent, $ForcedServices forced."

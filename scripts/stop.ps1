$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')
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
foreach ($Name in @('gateway', 'management')) {
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
    $Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($Process) {
        Write-Host "  $Name`: stopping process $ProcessId…"
        Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue
        try { Wait-Process -Id $ProcessId -Timeout 10 -ErrorAction Stop }
        catch {
            Write-Warning "$Name did not stop gracefully; forcing process $ProcessId to exit."
            Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
            Wait-Process -Id $ProcessId -Timeout 5 -ErrorAction SilentlyContinue
            $ForcedServices++
        }
        Write-Host "  $Name`: stopped."
        $StoppedServices++
    }
    else {
        Write-Host "  $Name`: process $ProcessId has already exited."
        $AbsentServices++
    }
    Remove-Item $Path -ErrorAction SilentlyContinue
}

Write-Host '[3/4] Checking for stale ModelDeck Workers…'
& (Join-Path $PSScriptRoot 'stop_stale_workers.ps1') -Quiet
Write-Host '  Stale Worker check complete.'
Write-Host "[4/4] ModelDeck stopped: $StoppedServices service(s) stopped, $AbsentServices already absent, $ForcedServices forced."

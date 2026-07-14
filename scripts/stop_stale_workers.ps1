param(
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$Root = (Get-Location).Path
$WorkerModules = @(
    'modeldeck.workers.mock_worker',
    'modeldeck.workers.autoregressive_worker',
    'modeldeck.workers.text_diffusion_worker'
)
$WorkerPorts = @(8610, 8611, 8620, 8621, 8622, 8623, 8624)
$Stopped = @()

foreach ($ProcessDirectory in Get-ChildItem /proc -Directory -ErrorAction SilentlyContinue) {
    if ($ProcessDirectory.Name -notmatch '^\d+$') { continue }
    $ProcessId = [int]$ProcessDirectory.Name
    try {
        $CommandLine = (Get-Content "$($ProcessDirectory.FullName)/cmdline" -Raw -ErrorAction Stop) `
            -replace "`0", ' '
    }
    catch { continue }
    if ($CommandLine -notmatch [regex]::Escape("$Root/.venv")) { continue }
    if (-not ($WorkerModules | Where-Object { $CommandLine -match [regex]::Escape($_) })) { continue }
    $PortMatch = [regex]::Match($CommandLine, '--port\s+(\d+)')
    if (-not $PortMatch.Success) { continue }
    $Port = [int]$PortMatch.Groups[1].Value
    if ($Port -notin $WorkerPorts) { continue }

    try {
        Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/shutdown" -TimeoutSec 2 | Out-Null
        try { Wait-Process -Id $ProcessId -Timeout 5 -ErrorAction Stop }
        catch { Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue }
    }
    catch { Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue }
    try { Wait-Process -Id $ProcessId -Timeout 5 -ErrorAction Stop }
    catch { Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue }
    $Stopped += "$ProcessId ($Port)"
}

if (-not $Quiet) {
    if ($Stopped.Count) { Write-Host "Stopped stale ModelDeck workers: $($Stopped -join ', ')" }
    else { Write-Host 'No stale ModelDeck workers found.' }
}

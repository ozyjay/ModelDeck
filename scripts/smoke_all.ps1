$ErrorActionPreference = 'Stop'
$RunScript = Join-Path $PSScriptRoot 'run.ps1'
$StopScript = Join-Path $PSScriptRoot 'stop.ps1'
$Started = $false
try {
    & $RunScript
    $Started = $true
    Start-Sleep -Seconds 1
    Write-Host '--- Autoregressive mock smoke ---'
    & (Join-Path $PSScriptRoot 'smoke_autoregressive.ps1')
    Write-Host '--- Text-diffusion mock smoke ---'
    & (Join-Path $PSScriptRoot 'smoke_text_diffusion.ps1')
} finally {
    if ($Started) { & $StopScript }
}

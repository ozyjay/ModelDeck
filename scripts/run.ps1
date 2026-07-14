[CmdletBinding()]
param([switch]$OpenDay)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if ($OpenDay) {
    $Env:MODELDECK_OPEN_DAY = '1'
    $Env:MODELDECK_ALLOW_DOWNLOADS = '0'
}
if (-not (Test-Path '.venv/bin/modeldeck')) { throw 'Run scripts/setup.ps1 first.' }
& (Join-Path $PSScriptRoot 'check_ports.ps1')
New-Item -ItemType Directory -Force -Path var/log,var/run | Out-Null
$management = Start-Process .venv/bin/modeldeck -RedirectStandardOutput var/log/management.log -RedirectStandardError var/log/management-error.log -PassThru
$gateway = Start-Process .venv/bin/modeldeck-gateway -RedirectStandardOutput var/log/gateway.log -RedirectStandardError var/log/gateway-error.log -PassThru
Set-Content var/run/management.pid $management.Id
Set-Content var/run/gateway.pid $gateway.Id
Write-Host 'Management: http://127.0.0.1:3600'
Write-Host 'Gateway:    http://127.0.0.1:8600/v1/health'
Write-Host 'Workers:    mocks 8610/8611, Qwen ROCm 8620/8623/8624, DiffusionGemma BF16 8621, default Q4 8622 (stopped)'

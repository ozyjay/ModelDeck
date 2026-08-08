[CmdletBinding()]
param([Alias('OpenDay')][switch]$LockConfiguration)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
Import-Module (Join-Path $PSScriptRoot 'environment_helpers.psm1') -Force
Import-ModelDeckEnvironment -Path (Join-Path (Get-Location) '.env')
if ($LockConfiguration) {
    $Env:MODELDECK_CONFIGURATION_LOCKED = '1'
}
if (-not (Test-Path '.venv/bin/modeldeck')) { throw 'Run scripts/setup.ps1 first.' }
& (Join-Path $PSScriptRoot 'check_ports.ps1')
New-Item -ItemType Directory -Force -Path var/log,var/run | Out-Null
$management = Start-Process .venv/bin/modeldeck -RedirectStandardOutput var/log/management.log -RedirectStandardError var/log/management-error.log -PassThru
$gateway = Start-Process .venv/bin/modeldeck-gateway -RedirectStandardOutput var/log/gateway.log -RedirectStandardError var/log/gateway-error.log -PassThru
Set-Content var/run/management.pid $management.Id
Set-Content var/run/gateway.pid $gateway.Id
$ManagementHost = if ($Env:MODELDECK_HOST) { $Env:MODELDECK_HOST } else { '127.0.0.1' }
$ManagementPort = if ($Env:MODELDECK_MANAGEMENT_PORT) { $Env:MODELDECK_MANAGEMENT_PORT } else { '3600' }
$GatewayHost = if ($Env:MODELDECK_GATEWAY_HOST) { $Env:MODELDECK_GATEWAY_HOST } else { '127.0.0.1' }
$GatewayPort = if ($Env:MODELDECK_GATEWAY_PORT) { $Env:MODELDECK_GATEWAY_PORT } else { '8600' }
Write-Host "Management: http://${ManagementHost}:${ManagementPort}"
Write-Host "Gateway:    http://${GatewayHost}:${GatewayPort}/v1/health"
Write-Host 'Workers:    Managed from the ModelDeck console; no Worker instances or public Routes are seeded.'

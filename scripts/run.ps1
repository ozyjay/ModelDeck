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
# The loopback gateway now uses gateway.pid. Remove the superseded record left
# by earlier development launchers before writing the current service records.
Remove-Item var/run/gateway-loopback.pid -ErrorAction SilentlyContinue
$ManagementHost = if ($Env:MODELDECK_HOST) { $Env:MODELDECK_HOST } else { '127.0.0.1' }
$ManagementPort = if ($Env:MODELDECK_MANAGEMENT_PORT) { $Env:MODELDECK_MANAGEMENT_PORT } else { '3600' }
$GatewayHost = if ($Env:MODELDECK_GATEWAY_HOST) { $Env:MODELDECK_GATEWAY_HOST } else { '127.0.0.1' }
$GatewayPort = if ($Env:MODELDECK_GATEWAY_PORT) { $Env:MODELDECK_GATEWAY_PORT } else { '8600' }
$DockerBridgeEnabled = $Env:MODELDECK_ENABLE_DOCKER_BRIDGE -and $Env:MODELDECK_ENABLE_DOCKER_BRIDGE.Trim().ToLowerInvariant() -in @('1', 'true', 'yes', 'on')
$PythonPath = (Resolve-Path '.venv/bin/python').Path
$management = Start-Process $PythonPath -ArgumentList @('-m', 'modeldeck') -RedirectStandardOutput var/log/management.log -RedirectStandardError var/log/management-error.log -PassThru
$gateway = Start-Process $PythonPath -ArgumentList @('-m', 'modeldeck.gateway.app') -RedirectStandardOutput var/log/gateway.log -RedirectStandardError var/log/gateway-error.log -PassThru
$gatewayDockerBridge = $null
if ($DockerBridgeEnabled) {
    # The primary listener remains loopback-only. This companion preserves the
    # Docker bridge endpoint for SprintBot without changing local desktop access.
    $OriginalGatewayHost = $Env:MODELDECK_GATEWAY_HOST
    try {
        $Env:MODELDECK_GATEWAY_HOST = '172.17.0.1'
        $gatewayDockerBridge = Start-Process $PythonPath -ArgumentList @('-m', 'modeldeck.gateway.app') -RedirectStandardOutput var/log/gateway-docker-bridge.log -RedirectStandardError var/log/gateway-docker-bridge-error.log -PassThru
    }
    finally {
        if ($null -eq $OriginalGatewayHost) { Remove-Item Env:MODELDECK_GATEWAY_HOST -ErrorAction SilentlyContinue }
        else { $Env:MODELDECK_GATEWAY_HOST = $OriginalGatewayHost }
    }
}
Set-Content var/run/management.pid $management.Id
Set-Content var/run/gateway.pid $gateway.Id
if ($gatewayDockerBridge) { Set-Content var/run/gateway-docker-bridge.pid $gatewayDockerBridge.Id }
Write-Host "Management: http://${ManagementHost}:${ManagementPort}"
Write-Host "Gateway:    http://${GatewayHost}:${GatewayPort}/v1/health"
if ($gatewayDockerBridge) { Write-Host "Gateway:    http://172.17.0.1:${GatewayPort}/v1/health (Docker bridge companion)" }
Write-Host 'Workers:    Managed from the ModelDeck console; no Worker instances or public Routes are seeded.'

[CmdletBinding()]
param([Parameter(Mandatory)][string]$RouteName)

$ErrorActionPreference = 'Stop'
$ManagementUrl = if ($Env:MODELDECK_MANAGEMENT_URL) { $Env:MODELDECK_MANAGEMENT_URL.TrimEnd('/') } else { 'http://127.0.0.1:3600' }
$GatewayUrl = if ($Env:MODELDECK_GATEWAY_URL) { $Env:MODELDECK_GATEWAY_URL.TrimEnd('/') } else { 'http://127.0.0.1:8600' }
$Live = Invoke-RestMethod -Uri "$ManagementUrl/api/live" -TimeoutSec 10
$Route = $Live.routes | Where-Object { $_.public_name -eq $RouteName }
if (-not $Route) { throw "The published Event has no Route named '$RouteName'." }
if (@($Route).Count -ne 1) { throw "The public Route name '$RouteName' is ambiguous." }
$PrimaryWorkerId = $Route.worker_ids[0]
Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$PrimaryWorkerId/start" | Out-Null
$Body = @{ model = $RouteName; prompt = 'Open Day smoke test' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$GatewayUrl/v1/completions" -ContentType 'application/json' -Body $Body |
    ConvertTo-Json -Depth 8

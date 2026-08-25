$ErrorActionPreference = 'Stop'
$ManagementUrl = if ($Env:MODELDECK_MANAGEMENT_URL) { $Env:MODELDECK_MANAGEMENT_URL.TrimEnd('/') } else { 'http://127.0.0.1:3600' }
$GatewayUrl = if ($Env:MODELDECK_GATEWAY_URL) { $Env:MODELDECK_GATEWAY_URL.TrimEnd('/') } else { 'http://127.0.0.1:8600' }
Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/mock-diffusion/start" | Out-Null
$Body = @{
    model = 'text-diffusion'
    prompt = 'A robot arrives at orientation.'
    seed = 11
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$GatewayUrl/v1/refine" -ContentType 'application/json' -Body $Body |
    ConvertTo-Json -Depth 8

[CmdletBinding()]
param(
    [string]$ManagementUrl = 'http://127.0.0.1:3600',
    [string]$FastWorkerName = 'Qwen2.5 0.5B Instruct',
    [string]$DeepWorkerName = 'Qwen2.5 3B Instruct'
)

$ErrorActionPreference = 'Stop'

# This is intentionally a management-plane configuration action: it creates no Workers,
# does not change another profile's draft, and never contacts a remote service.
$workers = @(Invoke-RestMethod -Method Get -Uri "$ManagementUrl/api/workers")
if ($workers.Count -eq 1 -and $workers[0] -is [array]) { $workers = @($workers[0]) }
$fast = $workers | Where-Object { $_.name -eq $FastWorkerName -and $_.model_id -eq 'Qwen/Qwen2.5-0.5B-Instruct' } | Select-Object -First 1
$deep = $workers | Where-Object { $_.name -eq $DeepWorkerName -and $_.model_id -eq 'Qwen/Qwen2.5-3B-Instruct' } | Select-Object -First 1
if (-not $fast) { throw "No configured fast Worker named '$FastWorkerName' for Qwen/Qwen2.5-0.5B-Instruct was found." }
if (-not $deep) { throw "No configured deep Worker named '$DeepWorkerName' for Qwen/Qwen2.5-3B-Instruct was found." }

$profiles = @(Invoke-RestMethod -Method Get -Uri "$ManagementUrl/api/routing-profiles").profiles
$existing = $profiles | Where-Object { $_.definition.name -eq 'wayfinder-gate0' } | Select-Object -First 1
$profileId = if ($existing) { $existing.definition.id } else { [guid]::NewGuid().ToString() }
$definition = [ordered]@{
    id = $profileId
    name = 'wayfinder-gate0'
    description = 'WayFinder Gate 0 local OpenAI-compatible chat backends. WayFinder chooses the model ID per request.'
    qualification = 'compatible'
    capabilities = @(
        [ordered]@{
            id = if ($existing) { $existing.definition.capabilities | Where-Object { $_.public_name -eq 'fast-local' } | Select-Object -ExpandProperty id -First 1 } else { [guid]::NewGuid().ToString() }
            display_name = 'WayFinder fast local chat'
            public_name = 'fast-local'
            protocol_contract = 'openai-chat-v1'
            worker_ids = @($fast.id)
        },
        [ordered]@{
            id = if ($existing) { $existing.definition.capabilities | Where-Object { $_.public_name -eq 'deep-local' } | Select-Object -ExpandProperty id -First 1 } else { [guid]::NewGuid().ToString() }
            display_name = 'WayFinder deep local chat'
            public_name = 'deep-local'
            protocol_contract = 'openai-chat-v1'
            worker_ids = @($deep.id)
        }
    )
}
foreach ($capability in $definition.capabilities) {
    if (-not $capability.id) { $capability.id = [guid]::NewGuid().ToString() }
}
$json = $definition | ConvertTo-Json -Depth 8
if ($existing) {
    Invoke-RestMethod -Method Put -Uri "$ManagementUrl/api/routing-profiles/$profileId/draft" -ContentType 'application/json' -Body $json | Out-Null
} else {
    Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/routing-profiles" -ContentType 'application/json' -Body $json | Out-Null
}
$validation = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/routing-profiles/$profileId/validate"
if (-not $validation.valid) { throw "wayfinder-gate0 validation failed: $($validation.errors | ConvertTo-Json -Compress)" }
$published = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/routing-profiles/$profileId/publish"
Write-Host "Published wayfinder-gate0 revision $($published.revision). Existing active Routing Profiles remain active."

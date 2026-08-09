[CmdletBinding()]
param(
    [string]$ManagementUrl = 'http://127.0.0.1:3600',
    [string]$FastWorkerName = 'WayFinder Qwen2.5 0.5B Instruct',
    [string]$DeepWorkerName = 'WayFinder Qwen2.5 3B Instruct'
)

$ErrorActionPreference = 'Stop'

# This is intentionally a local management-plane action. It only creates the two named
# WayFinder Workers from already cached, allowlisted snapshots; it never alters shared
# Workers, downloads a Model, starts a Worker, or contacts a remote service.
$workers = @(Invoke-RestMethod -Method Get -Uri "$ManagementUrl/api/workers")
if ($workers.Count -eq 1 -and $workers[0] -is [array]) { $workers = @($workers[0]) }
$catalogue = @((Invoke-RestMethod -Method Get -Uri "$ManagementUrl/api/catalogue").models)

function Resolve-WayFinderWorker {
    param(
        [string]$Name,
        [string]$ModelId
    )

    $existing = $workers | Where-Object { $_.name -eq $Name -and $_.model_id -eq $ModelId } | Select-Object -First 1
    if ($existing) {
        if ($existing.settings.context_length -ne 32768 -or $existing.settings.maximum_new_tokens -ne 4096) {
            throw "Existing WayFinder Worker '$Name' must use context_length=32768 and maximum_new_tokens=4096. Create a replacement rather than changing a shared Worker."
        }
        return $existing
    }

    $model = $catalogue | Where-Object {
        $_.model_id -eq $ModelId -and $_.download_state -eq 'installed-untested'
    } | Select-Object -First 1
    if (-not $model) { throw "No complete cached snapshot for $ModelId was found." }
    if (-not $model.modeldeck_allowed) { throw "Allow the cached $ModelId snapshot in ModelDeck before configuring WayFinder." }
    $created = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers" -ContentType 'application/json' -Body (@{
        name = $Name
        model_id = $model.model_id
        revision = $model.revision
        runtime_template_id = 'autoregressive-transformers'
        context_length = 32768
        maximum_new_tokens = 4096
    } | ConvertTo-Json)
    $script:workers = @($workers) + @($created)
    return $created
}

$fast = Resolve-WayFinderWorker -Name $FastWorkerName -ModelId 'Qwen/Qwen2.5-0.5B-Instruct'
$deep = Resolve-WayFinderWorker -Name $DeepWorkerName -ModelId 'Qwen/Qwen2.5-3B-Instruct'

$profiles = @((Invoke-RestMethod -Method Get -Uri "$ManagementUrl/api/routing-profiles").profiles)
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

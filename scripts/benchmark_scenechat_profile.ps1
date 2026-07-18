[CmdletBinding()]
param(
    [string]$WorkerUrl = 'http://127.0.0.1:8682',
    [string]$ModelId = 'google/gemma-4-12B-it',
    [int]$Runs = 3
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$ImageBase64 = & .venv-rocm72/bin/python -c "import base64,io; from PIL import Image; b=io.BytesIO(); Image.new('RGB',(256,256),(70,100,130)).save(b,'PNG'); print(base64.b64encode(b.getvalue()).decode())"
if ($LASTEXITCODE -ne 0) { throw 'Could not create the synthetic benchmark image.' }
$SystemPrompt = (Get-Content 'backend/modeldeck/contracts/scenechat/scene_analysis_system.txt' -Raw).Trim()
$Prompt = "$SystemPrompt`n`nSelected curated question:`nDescribe the scene."
$Payload = @{
    model = $ModelId
    messages = @(@{
        role = 'user'
        content = @(
            @{ type = 'image_url'; image_url = @{ url = "data:image/png;base64,$ImageBase64" } },
            @{ type = 'text'; text = $Prompt }
        )
    })
    temperature = 0.1
    max_tokens = 700
    response_format = @{ type = 'json_object' }
    stream = $false
} | ConvertTo-Json -Depth 12 -Compress

$Samples = @()
for ($Index = 1; $Index -le $Runs; $Index++) {
    $Response = $null
    $Elapsed = Measure-Command {
        $Response = Invoke-RestMethod -Method Post -Uri "$WorkerUrl/v1/chat/completions" `
            -Headers @{ Authorization = 'Bearer local' } -ContentType 'application/json' `
            -Body $Payload -TimeoutSec 90
    }
    $Analysis = $Response.choices[0].message.content | ConvertFrom-Json
    if (-not $Analysis.summary) { throw "Run $Index returned no structured SceneChat summary." }
    $Samples += [ordered]@{
        run = $Index
        latency_seconds = [Math]::Round($Elapsed.TotalSeconds, 4)
        prompt_tokens = $Response.usage.prompt_tokens
        completion_tokens = $Response.usage.completion_tokens
    }
}
$Metrics = Invoke-RestMethod -Uri "$WorkerUrl/metrics" -TimeoutSec 5
[ordered]@{
    model_id = $ModelId
    runs = $Samples
    average_latency_seconds = [Math]::Round(($Samples.latency_seconds | Measure-Object -Average).Average, 4)
    peak_memory_allocated_bytes = $Metrics.peak_memory_allocated_bytes
    memory_allocated_bytes = $Metrics.memory_allocated_bytes
    runtime = [ordered]@{
        torch = $Metrics.torch_version
        rocm = $Metrics.hip_version
        transformers = $Metrics.transformers_version
        processor_class = $Metrics.processor_class
        model_class = $Metrics.model_class
        load_seconds = $Metrics.load_seconds
    }
} | ConvertTo-Json -Depth 8

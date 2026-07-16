$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
& (Join-Path $PSScriptRoot 'check_ports.ps1')
& (Join-Path $PSScriptRoot 'verify_scenechat_snapshot.ps1') | Out-Host

$ManagementUrl = 'http://127.0.0.1:3600'
$WorkerUrl = 'http://127.0.0.1:8000'
$WorkerId = 'scenechat-gemma4-e2b-rocm'
$ModelId = 'google/gemma-4-E2B-it'
$ApiKey = if ($Env:MODELDECK_SCENECHAT_API_KEY) { $Env:MODELDECK_SCENECHAT_API_KEY } else { 'local' }
$Headers = @{ Authorization = "Bearer $ApiKey" }
$StartedServices = $false
$WorkerStopped = $false

try {
    & (Join-Path $PSScriptRoot 'run.ps1') -OpenDay
    $StartedServices = $true
    Start-Sleep -Seconds 1

    Write-Host 'Starting the pinned SceneChat Gemma 4 worker; the first offline load can take several minutes.'
    $Worker = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$WorkerId/start" -TimeoutSec 780
    if ($Worker.state -ne 'ready') { throw "Worker did not become ready: $($Worker.state)" }

    $Native = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$WorkerId/smoke" `
        -TimeoutSec 60
    if (-not $Native.ok) { throw 'The native SceneChat synthetic-image smoke failed.' }

    $Models = Invoke-RestMethod -Uri "$WorkerUrl/v1/models" -Headers $Headers -TimeoutSec 5
    if ($Models.data[0].id -ne $ModelId) { throw 'The OpenAI model listing returned an unexpected model.' }

    $ImageBase64 = & .venv-rocm72/bin/python -c "import base64,io; from PIL import Image; b=io.BytesIO(); Image.new('RGB',(64,64),(70,100,130)).save(b,'PNG'); print(base64.b64encode(b.getvalue()).decode())"
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the approved non-visitor PNG fixture.' }
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
    $Completion = Invoke-RestMethod -Method Post -Uri "$WorkerUrl/v1/chat/completions" `
        -Headers $Headers -ContentType 'application/json' -Body $Payload -TimeoutSec 75
    $Analysis = $Completion.choices[0].message.content | ConvertFrom-Json
    if (-not $Analysis.summary) { throw 'The OpenAI-compatible response did not contain a scene summary.' }

    $Stopped = Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$WorkerId/stop" -TimeoutSec 30
    if ($Stopped.state -ne 'stopped' -or $null -ne $Stopped.pid) {
        throw 'The SceneChat worker process did not report a clean stop.'
    }
    $WorkerStopped = $true
    $Lifecycle = @{
        shutdown_result = 'success'
        memory_recovery_result = 'not-measured-process-exit-confirmed'
    } | ConvertTo-Json
    Invoke-RestMethod -Method Put -Uri "$ManagementUrl/api/compatibility/tests/$($Native.test.id)/lifecycle" `
        -ContentType 'application/json' -Body $Lifecycle -TimeoutSec 10 | Out-Null
    [ordered]@{
        native_smoke = 'passed'
        openai_models = 'passed'
        approved_png_completion = 'passed'
        process_exit = 'confirmed'
    } | ConvertTo-Json
}
finally {
    if (-not $WorkerStopped) {
        try {
            Invoke-RestMethod -Method Post -Uri "$ManagementUrl/api/workers/$WorkerId/stop" -TimeoutSec 30 |
                Out-Null
        }
        catch { Write-Warning "Could not request SceneChat worker shutdown: $_" }
    }
    if ($StartedServices) { & (Join-Path $PSScriptRoot 'stop.ps1') }
}
